import torch
import torch.nn as nn

from easy_tpp.model.torch_model.torch_baselayer import EncoderLayer, MultiHeadAttention, TimePositionalEncoding, ScaledSoftplus
from easy_tpp.model.torch_model.torch_basemodel import TorchBaseModel


class THP(TorchBaseModel):
    """Torch implementation of Transformer Hawkes Process, ICML 2020, https://arxiv.org/abs/2002.09291.
    Note: Part of the code is collected from https://github.com/yangalan123/anhp-andtt/tree/master/thp.
    """

    def __init__(self, model_config):
        """Initialize the model

        Args:
            model_config (EasyTPP.ModelConfig): config of model specs.
        """
        super(THP, self).__init__(model_config)
        self.d_model = model_config.hidden_size
        self.d_time = model_config.time_emb_size
        self.use_norm = model_config.use_ln

        self.n_layers = model_config.num_layers
        self.n_head = model_config.num_heads
        self.dropout = model_config.dropout_rate

        self.layer_temporal_encoding = TimePositionalEncoding(self.d_model, device=self.device)

        self.factor_intensity_base = nn.Parameter(torch.empty([1, self.num_event_types], device=self.device))
        self.factor_intensity_decay = nn.Parameter(torch.empty([1, self.num_event_types], device=self.device))
        # Use smaller initialization to prevent extreme values
        nn.init.uniform_(self.factor_intensity_base, -0.1, 0.1)
        nn.init.uniform_(self.factor_intensity_decay, -0.1, 0.1)

        # convert hidden vectors into event-type-sized vector
        self.layer_intensity_hidden = nn.Linear(self.d_model, self.num_event_types)
        self.softplus = ScaledSoftplus(self.num_event_types, threshold=10.0)   # learnable mark-specific beta with lower threshold
        
        # Add a small epsilon for numerical stability
        self.intensity_eps = 1e-8

        # Add MLP layer
        # Equation (5)
        self.feed_forward = nn.Sequential(
            nn.Linear(self.d_model, self.d_model * 2),
            nn.ReLU(),
            nn.Linear(self.d_model * 2, self.d_model)
        )

        self.stack_layers = nn.ModuleList(
            [EncoderLayer(
                self.d_model,
                MultiHeadAttention(self.n_head, self.d_model, self.d_model, self.dropout,
                                   output_linear=False),
                use_residual=False,
                feed_forward=self.feed_forward,
                dropout=self.dropout
            ) for _ in range(self.n_layers)])

    def forward(self, time_seqs, type_seqs, attention_mask):
        """Call the model

        Args:
            time_seqs (tensor): [batch_size, seq_len], timestamp seqs.
            type_seqs (tensor): [batch_size, seq_len], event type seqs.
            attention_mask (tensor): [batch_size, seq_len, hidden_size], attention masks.

        Returns:
            tensor: hidden states at event times.
        """
        # [batch_size, seq_len, hidden_size]
        tem_enc = self.layer_temporal_encoding(time_seqs)
        enc_output = self.layer_type_emb(type_seqs)

        # [batch_size, seq_len, hidden_size]
        for enc_layer in self.stack_layers:
            enc_output += tem_enc
            enc_output = enc_layer(
                enc_output,
                mask=attention_mask)

        return enc_output

    def loglike_loss(self, batch):
        """Compute the loglike loss with optional hybrid loss.

        Args:
            batch (tuple, list): batch input.

        Returns:
            tuple: loglike loss, num events.
        """
        # Handle BatchEncoding objects (convert to list of tensors)
        if hasattr(batch, 'values'):
            batch = list(batch.values())
            
        time_seqs, time_delta_seqs, type_seqs, batch_non_pad_mask, attention_mask = batch

        # 1. compute event-loglik
        # [batch_size, seq_len, hidden_size]
        enc_out = self.forward(time_seqs[:, :-1], type_seqs[:, :-1], attention_mask[:, :-1, :-1])

        # [batch_size, seq_len, num_event_types]
        # update time decay based on Equation (6)
        # [1, 1, num_event_types]
        factor_intensity_decay = self.factor_intensity_decay[None, ...]
        factor_intensity_base = self.factor_intensity_base[None, ...]

        # update time decay based on Equation (6)
        # [batch_size, seq_len, num_event_types]
        intensity_states = factor_intensity_decay * time_delta_seqs[:, 1:, None] + self.layer_intensity_hidden(
            enc_out) + factor_intensity_base

        lambda_at_event = self.softplus(intensity_states)
        # Clamp intensities to prevent numerical issues
        lambda_at_event = torch.clamp(lambda_at_event, min=self.intensity_eps, max=100.0)

        # 2. compute non-event-loglik (using MC sampling to compute integral)
        # 2.1 sample dtimes
        # [batch_size, seq_len, num_sample]
        sample_dtimes = self.make_dtime_loss_samples(time_delta_seqs[:, 1:])

        # 2.2 compute intensities at sampled times
        # [batch_size, num_times = max_len - 1, num_sample, event_num]
        state_t_sample = self.compute_states_at_sample_times(event_states=enc_out,
                                                             sample_dtimes=sample_dtimes)
        lambda_t_sample = self.softplus(state_t_sample)
        # Clamp intensities to prevent numerical issues
        lambda_t_sample = torch.clamp(lambda_t_sample, min=self.intensity_eps, max=100.0)

        # Compute standard NLL
        event_ll, non_event_ll, num_events = self.compute_loglikelihood(
            lambda_at_event=lambda_at_event,
            lambdas_loss_samples=lambda_t_sample,
            time_delta_seq=time_delta_seqs[:, 1:],
            seq_mask=batch_non_pad_mask[:, 1:],
            type_seq=type_seqs[:, 1:])
        
        # If NLL uncertainty weighting is enabled (type/time split)
        if self.use_nll_uncertainty:
            # Use STANDARD NLL formula: -(event_ll - non_event_ll)
            # Compute type and time components separately
            type_ll = event_ll.sum()  # Type component (event type prediction)
            time_ll = -non_event_ll.sum()  # Time component (non-event/intensity integral)

            # Apply uncertainty weighting to balance type and time
            nll_loss, type_weight, time_weight = self.compute_nll_with_uncertainty(type_ll, time_ll)

            # Store weights for monitoring
            if hasattr(self, 'training') and self.training:
                self._last_type_weight = type_weight
                self._last_time_weight = time_weight
                self._last_type_ll = type_ll.item()
                self._last_time_ll = time_ll.item()

            loss = nll_loss
            self._metric_nll_loss = nll_loss

        # If event-only loss is enabled (event NLL + MSE with uncertainty weighting)
        elif self.use_event_only_loss:
            # Use ONLY event term (no non-event/integral term)
            event_ll_sum = event_ll.sum()

            # Compute MSE loss on time predictions
            total_intensity = lambda_at_event.sum(dim=-1)  # [batch_size, seq_len-1]
            pred_dtime = 1.0 / (total_intensity + self.eps)
            label_dtime = time_delta_seqs[:, 1:]
            seq_mask = batch_non_pad_mask[:, 1:]

            mse_loss = self.compute_mse_loss(pred_dtime, label_dtime, seq_mask)

            # Combine event-only NLL and MSE with uncertainty weighting
            loss, event_weight, mse_weight = self.compute_event_only_hybrid_loss(event_ll_sum, mse_loss)

            # Store weights for monitoring
            if hasattr(self, 'training') and self.training:
                self._last_event_weight = event_weight
                self._last_mse_weight = mse_weight
                self._last_event_ll = event_ll_sum.item()
                self._last_mse_loss = mse_loss.item()

            # For metrics, store the event-only NLL
            self._metric_nll_loss = -event_ll_sum        # If simple hybrid is enabled (NLL + MSE without uncertainty weighting)
        elif self.use_simple_hybrid:
            # Compute standard NLL loss
            nll_loss = - (event_ll - non_event_ll).sum()

            # Compute MSE loss on time predictions
            total_intensity = lambda_at_event.sum(dim=-1)  # [batch_size, seq_len-1]
            pred_dtime = 1.0 / (total_intensity + self.eps)
            label_dtime = time_delta_seqs[:, 1:]  # Ground truth time intervals
            seq_mask = batch_non_pad_mask[:, 1:]  # Mask for valid events

            mse_loss = self.compute_mse_loss(pred_dtime, label_dtime, seq_mask)

            # Simple addition without any weighting
            loss = nll_loss + mse_loss

            # Store losses for monitoring
            if hasattr(self, 'training') and self.training:
                self._last_nll_loss = nll_loss.item()
                self._last_mse_loss = mse_loss.item()

            # For metrics, store the NLL only
            self._metric_nll_loss = nll_loss



        # If hybrid loss is enabled, compute MSE and combine with uncertainty weighting
        elif self.use_hybrid_loss:
            # Compute NLL loss (using standard formula)
            nll_loss = - (event_ll - non_event_ll).sum()
            # Compute MSE loss on time predictions
            # For THP, use intensity to predict time intervals
            total_intensity = lambda_at_event.sum(dim=-1)  # [batch_size, seq_len-1]
            pred_dtime = 1.0 / (total_intensity + self.eps)
            label_dtime = time_delta_seqs[:, 1:]
            seq_mask = batch_non_pad_mask[:, 1:]
            
            mse_loss = self.compute_mse_loss(pred_dtime, label_dtime, seq_mask)
            
            # Combine NLL and MSE with uncertainty weighting
            loss, nll_weight, mse_weight = self.compute_hybrid_loss(nll_loss, mse_loss)
            
            # Store weights for monitoring
            if hasattr(self, 'training') and self.training:
                self._last_nll_weight = nll_weight
                self._last_mse_weight = mse_weight
                self._last_nll_loss = nll_loss.item()
                self._last_mse_loss = mse_loss.item()
            
            # IMPORTANT: For metrics, we need to return NLL only (not hybrid loss)
            # Store NLL separately for metric calculation
            self._metric_nll_loss = nll_loss

        else:
            # Standard NLL loss (no uncertainty weighting, no MSE)
            # Using standard NLL formula
            nll_loss = - (event_ll - non_event_ll).sum()
            loss = nll_loss
            self._metric_nll_loss = nll_loss
            
        # Return loss and num_events
        return loss, num_events

    def compute_states_at_sample_times(self, event_states, sample_dtimes):
        """Compute the hidden states at sampled times.

        Args:
            event_states (tensor): [batch_size, seq_len, hidden_size].
            sample_dtimes (tensor): [batch_size, seq_len, num_samples].

        Returns:
            tensor: hidden state at each sampled time.
        """
        # [batch_size, seq_len, 1, hidden_size]
        event_states = event_states[:, :, None, :]

        # [batch_size, seq_len, num_samples, 1]
        sample_dtimes = sample_dtimes[..., None]

        # [1, 1, 1, num_event_types]
        factor_intensity_decay = self.factor_intensity_decay[None, None, ...]
        factor_intensity_base = self.factor_intensity_base[None, None, ...]

        # update time decay based on Equation (6)
        # [batch_size, seq_len, num_samples, num_event_types]
        intensity_states = factor_intensity_decay * sample_dtimes + self.layer_intensity_hidden(
            event_states) + factor_intensity_base

        return intensity_states

    def compute_intensities_at_sample_times(self,
                                            time_seqs,
                                            time_delta_seqs,
                                            type_seqs,
                                            sample_dtimes,
                                            **kwargs):
        """Compute hidden states at sampled times.

        Args:
            time_seqs (tensor): [batch_size, seq_len], times seqs.
            time_delta_seqs (tensor): [batch_size, seq_len], time delta seqs.
            type_seqs (tensor): [batch_size, seq_len], event type seqs.
            sample_dtimes (tensor): [batch_size, seq_len, num_samples], sampled inter-event timestamps.

        Returns:
            tensor: [batch_size, seq_len, num_samples, num_event_types], intensity at all sampled times.
        """

        attention_mask = kwargs.get('attention_mask', None)
        compute_last_step_only = kwargs.get('compute_last_step_only', False)

        if attention_mask is None:
            batch_size, seq_len = time_seqs.size()
            attention_mask = torch.triu(torch.ones(seq_len, seq_len, device=self.device), diagonal=1).unsqueeze(0)
            attention_mask = attention_mask.expand(batch_size, -1, -1).to(torch.bool)

        # [batch_size, seq_len, hidden_size] - encoder output
        enc_out = self.forward(time_seqs, type_seqs, attention_mask)

        # [batch_size, seq_len, num_samples, num_event_types]
        encoder_output = self.compute_states_at_sample_times(enc_out, sample_dtimes)

        if compute_last_step_only:
            lambdas = self.softplus(encoder_output[:, -1:, :, :])
        else:
            # [batch_size, seq_len, num_samples, num_event_types]
            lambdas = self.softplus(encoder_output)
        
        # Clamp intensities to prevent numerical issues
        lambdas = torch.clamp(lambdas, min=self.intensity_eps, max=100.0)
        return lambdas
