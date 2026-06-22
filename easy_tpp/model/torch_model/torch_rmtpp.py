import torch
from torch import nn
import math

from easy_tpp.model.torch_model.torch_basemodel import TorchBaseModel

class RMTPP(TorchBaseModel):
    """Torch implementation of Recurrent Marked Temporal Point Processes, KDD 2016.
    https://www.kdd.org/kdd2016/papers/files/rpp1081-duA.pdf
    """

    def __init__(self, model_config):
        """Initialize the model

        Args:
            model_config (EasyTPP.ModelConfig): config of model specs.
        """
        super(RMTPP, self).__init__(model_config)

        self.layer_temporal_emb = nn.Linear(1, self.hidden_size)
        self.layer_rnn = nn.RNN(input_size=self.hidden_size, hidden_size=self.hidden_size,
                                num_layers=1, batch_first=True)

        self.hidden_to_intensity_logits = nn.Linear(self.hidden_size, self.num_event_types)
        self.b_t = nn.Parameter(torch.zeros(1, self.num_event_types))
        self.w_t = nn.Parameter(torch.zeros(1, self.num_event_types))
        # Use uniform initialization instead of xavier_normal
        nn.init.uniform_(self.b_t, -0.1, 0.1)
        nn.init.uniform_(self.w_t, -0.01, 0.01)

    def evolve_and_get_intentsity(self, right_hiddens_BNH, dts_BNG):
        """
        Eq.11 that computes intensity.
        """
        past_influence_BNGM = self.hidden_to_intensity_logits(right_hiddens_BNH[..., None, :])
        intensity_BNGM = (past_influence_BNGM + self.w_t[None, None, :] * dts_BNG[..., None]
                         + self.b_t[None, None, :]).clamp(max=math.log(1e5)).exp()
        return intensity_BNGM

    def forward(self, batch):
        """
        Suppose we have inputs with original sequence length N+1
        ts: [t0, t1, ..., t_N]
        dts: [0, t1 - t0, t2 - t1, ..., t_N - t_{N-1}]
        marks: [k0, k1, ..., k_N] (k0 and kN could be padded marks if t0 and tN correspond to left and right windows)

        Return:
            left limits of *intensity* at [t_1, ..., t_N] of shape: (batch_size, seq_len - 1, hidden_dim)
            right limits of *hidden states* [t_0, ..., t_{N-1}, t_N] of shape: (batch_size, seq_len, hidden_dim)
            We need the right limit of t_N to sample continuation.
        """

        t_BN, dt_BN, marks_BN, _, _ = batch
        mark_emb_BNH = self.layer_type_emb(marks_BN)
        time_emb_BNH = self.layer_temporal_emb(t_BN[..., None])
        right_hiddens_BNH, _ = self.layer_rnn(mark_emb_BNH + time_emb_BNH)
        left_intensity_B_Nm1_M = self.evolve_and_get_intentsity(right_hiddens_BNH[:, :-1, :], dt_BN[:, 1:][...,None]).squeeze(-2)
        return left_intensity_B_Nm1_M, right_hiddens_BNH


    def loglike_loss(self, batch):
        """Compute the log-likelihood loss with optional hybrid loss.

        Args:
            batch (list): batch input.

        Returns:
            tuple: loglikelihood loss and num of events.
        """
        # Handle BatchEncoding objects (convert to list of tensors)
        if hasattr(batch, 'values'):
            batch = list(batch.values())
        
        ts_BN, dts_BN, marks_BN, batch_non_pad_mask, _ = batch

        # compute left intensity and hidden states at event time
        # left limits of intensity at [t_1, ..., t_N]
        # right limits of hidden states at [t_0, ..., t_{N-1}, t_N]
        left_intensity_B_Nm1_M, right_hiddens_BNH = self.forward((ts_BN, dts_BN, marks_BN, None, None))
        right_hiddens_B_Nm1_H = right_hiddens_BNH[..., :-1, :]  # discard right limit at t_N for logL

        dts_sample_B_Nm1_G = self.make_dtime_loss_samples(dts_BN[:, 1:])
        intensity_dts_B_Nm1_G_M = self.evolve_and_get_intentsity(right_hiddens_B_Nm1_H, dts_sample_B_Nm1_G)

        # Compute standard NLL
        event_ll, non_event_ll, num_events = self.compute_loglikelihood(
            lambda_at_event=left_intensity_B_Nm1_M,
            lambdas_loss_samples=intensity_dts_B_Nm1_G_M,
            time_delta_seq=dts_BN[:, 1:],
            seq_mask=batch_non_pad_mask[:, 1:],
            type_seq=marks_BN[:, 1:]
        )
        
        # Compute NLL loss
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
            total_intensity = left_intensity_B_Nm1_M.sum(dim=-1)  # [batch_size, seq_len-1]
            pred_dtime = 1.0 / (total_intensity + self.eps)
            label_dtime = dts_BN[:, 1:]
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
            total_intensity = left_intensity_B_Nm1_M.sum(dim=-1)  # [batch_size, seq_len-1]
            pred_dtime = 1.0 / (total_intensity + self.eps)
            label_dtime = dts_BN[:, 1:]  # Ground truth time intervals
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
            # For RMTPP, use intensity to predict time intervals
            total_intensity = left_intensity_B_Nm1_M.sum(dim=-1)  # [batch_size, seq_len-1]
            pred_dtime = 1.0 / (total_intensity + self.eps)
            label_dtime = dts_BN[:, 1:]
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



    def compute_intensities_at_sample_times(self, time_seqs, time_delta_seqs, type_seqs, sample_dtimes, **kwargs):
        """Compute the intensity at sampled times, not only event times.

        Args:
            time_seq (tensor): [batch_size, seq_len], times seqs.
            time_delta_seq (tensor): [batch_size, seq_len], time delta seqs.
            event_seq (tensor): [batch_size, seq_len], event type seqs.
            sample_dtimes (tensor): [batch_size, seq_len, num_sample], sampled inter-event timestamps.

        Returns:
            tensor: [batch_size, num_times, num_mc_sample, num_event_types],
                    intensity at each timestamp for each event type.
        """

        compute_last_step_only = kwargs.get('compute_last_step_only', False)

        _input = time_seqs, time_delta_seqs, type_seqs, None, None
        _, right_hiddens_BNH = self.forward(_input)

        if compute_last_step_only:
            sampled_intensities = self.evolve_and_get_intentsity(right_hiddens_BNH[:, -1:, :], sample_dtimes[:, -1:, :])
        else:
            sampled_intensities = self.evolve_and_get_intentsity(right_hiddens_BNH, sample_dtimes)  # shape: [B, N, G, M]
        return sampled_intensities
