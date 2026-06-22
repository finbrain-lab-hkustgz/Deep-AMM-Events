""" Base model with common functionality  """

import torch
from torch import nn
from torch.nn import functional as F

from easy_tpp.model.torch_model.torch_thinning import EventSampler
from easy_tpp.utils import set_device


class TorchBaseModel(nn.Module):
    def __init__(self, model_config):
        """Initialize the BaseModel

        Args:
            model_config (EasyTPP.ModelConfig): model spec of configs
        """
        super(TorchBaseModel, self).__init__()
        self.loss_integral_num_sample_per_step = model_config.loss_integral_num_sample_per_step
        self.hidden_size = model_config.hidden_size
        self.num_event_types = model_config.num_event_types  # not include [PAD], [BOS], [EOS]
        self.num_event_types_pad = model_config.num_event_types_pad  # include [PAD], [BOS], [EOS]
        self.pad_token_id = model_config.pad_token_id
        self.eps = torch.finfo(torch.float32).eps

        self.layer_type_emb = nn.Embedding(self.num_event_types_pad,  # have padding
                                           self.hidden_size,
                                           padding_idx=self.pad_token_id)

        self.gen_config = model_config.thinning
        self.event_sampler = None
        self.device = set_device(model_config.gpu)
        self.use_mc_samples = model_config.use_mc_samples
        
        # Loss configuration
        self.use_hybrid_loss = model_config.model_specs.get('use_hybrid_loss', False)
        self.use_nll_uncertainty = model_config.model_specs.get('use_nll_uncertainty', False)
        self.use_event_only_loss = model_config.model_specs.get('use_event_only_loss', False)
        self.use_simple_hybrid = model_config.model_specs.get('use_simple_hybrid', False)

        if self.use_hybrid_loss or self.use_event_only_loss:
            # Initialize learnable uncertainty parameters for NLL/MSE weighting
            # These will be optimized during training to balance NLL and MSE losses
            self.log_var_nll = nn.Parameter(torch.zeros(1))
            self.log_var_mse = nn.Parameter(torch.zeros(1))

            # Optional: set initial values if needed
            # Smaller log_var means higher weight (inverse relationship)
            init_log_var_nll = model_config.model_specs.get('init_log_var_nll', 0.0)
            init_log_var_mse = model_config.model_specs.get('init_log_var_mse', 0.0)
            self.log_var_nll.data.fill_(init_log_var_nll)
            self.log_var_mse.data.fill_(init_log_var_mse)

        if self.use_nll_uncertainty:
            # Initialize learnable uncertainty parameters for type/time weighting within NLL
            # These will be optimized during training to balance type and time components
            self.log_var_type = nn.Parameter(torch.zeros(1))
            self.log_var_time = nn.Parameter(torch.zeros(1))

            # Set initial values if provided
            init_log_var_type = model_config.model_specs.get('init_log_var_type', 0.0)
            init_log_var_time = model_config.model_specs.get('init_log_var_time', 0.0)
            self.log_var_type.data.fill_(init_log_var_type)
            self.log_var_time.data.fill_(init_log_var_time)

        self.to(self.device)

        if self.gen_config:
            self.event_sampler = EventSampler(num_sample=self.gen_config.num_sample,
                                              num_exp=self.gen_config.num_exp,
                                              over_sample_rate=self.gen_config.over_sample_rate,
                                              patience_counter=self.gen_config.patience_counter,
                                              num_samples_boundary=self.gen_config.num_samples_boundary,
                                              dtime_max=self.gen_config.dtime_max,
                                              device=self.device)

    @staticmethod
    def generate_model_from_config(model_config):
        """Generate the model in derived class based on model config.

        Args:
            model_config (EasyTPP.ModelConfig): config of model specs.
        """
        model_id = model_config.model_id

        for subclass in TorchBaseModel.__subclasses__():
            if subclass.__name__ == model_id:
                return subclass(model_config)

        raise RuntimeError('No model named ' + model_id)

    @staticmethod
    def get_logits_at_last_step(logits, batch_non_pad_mask, sample_len=None):
        """Retrieve the hidden states of last non-pad events.

        Args:
            logits (tensor): [batch_size, seq_len, hidden_dim], a sequence of logits
            batch_non_pad_mask (tensor): [batch_size, seq_len], a sequence of masks
            sample_len (tensor): default None, use batch_non_pad_mask to find out the last non-mask position

        ref: https://medium.com/analytics-vidhya/understanding-indexing-with-pytorch-gather-33717a84ebc4

        Returns:
            tensor: retrieve the logits of EOS event
        """

        seq_len = batch_non_pad_mask.sum(dim=1)
        select_index = seq_len - 1 if sample_len is None else seq_len - 1 - sample_len
        # [batch_size, hidden_dim]
        select_index = select_index.unsqueeze(1).repeat(1, logits.size(-1))
        # [batch_size, 1, hidden_dim]
        select_index = select_index.unsqueeze(1)
        # [batch_size, hidden_dim]
        last_logits = torch.gather(logits, dim=1, index=select_index).squeeze(1)
        return last_logits

    def compute_loglikelihood(self, time_delta_seq, lambda_at_event, lambdas_loss_samples, seq_mask, type_seq):
        """Compute the loglikelihood of the event sequence based on Equation (8) of NHP paper.

        Args:
            time_delta_seq (tensor): [batch_size, seq_len], time_delta_seq from model input.
            lambda_at_event (tensor): [batch_size, seq_len, num_event_types], unmasked intensity at
            (right after) the event.
            lambdas_loss_samples (tensor): [batch_size, seq_len, num_sample, num_event_types],
            intensity at sampling times.
            seq_mask (tensor): [batch_size, seq_len], sequence mask vector to mask the padded events.
            type_seq (tensor): [batch_size, seq_len], sequence of mark ids, with padded events having a mark of self.pad_token_id

        Returns:
            tuple: event loglike, non-event loglike, intensity at event with padding events masked
        """

        # First, add an epsilon to every marked intensity for stability
        lambda_at_event = lambda_at_event + self.eps
        lambdas_loss_samples = lambdas_loss_samples + self.eps

        log_marked_event_lambdas = lambda_at_event.log()
        total_sampled_lambdas = lambdas_loss_samples.sum(dim=-1)

        # Compute event LL - [batch_size, seq_len]
        event_ll = -F.nll_loss(
            log_marked_event_lambdas.permute(0, 2, 1),  # mark dimension needs to come second, not third to match nll_loss specs
            target=type_seq,
            ignore_index=self.pad_token_id,  # Padded events have a pad_token_id as a value
            reduction='none', # Does not aggregate, and replaces what would have been the log(marked intensity) with 0.
        )

        # Compute non-event LL [batch_size, seq_len]
        # interval_integral = length_interval * average of sampled lambda(t)
        if self.use_mc_samples:
            non_event_ll = total_sampled_lambdas.mean(dim=-1) * time_delta_seq * seq_mask
        else: # Use trapezoid rule
            non_event_ll = 0.5 * (total_sampled_lambdas[..., 1:] + total_sampled_lambdas[..., :-1]).mean(dim=-1) * time_delta_seq * seq_mask

        num_events = torch.masked_select(event_ll, event_ll.ne(0.0)).size()[0]
        return event_ll, non_event_ll, num_events
    
    def compute_loglikelihood_with_type_time_split(self, time_delta_seq, lambda_at_event, lambdas_loss_samples, seq_mask, type_seq):
        """Compute the loglikelihood split into type and time components.
        
        This returns separate components for event type prediction and time prediction
        to enable separate uncertainty weighting.
        
        Args:
            time_delta_seq (tensor): [batch_size, seq_len], time_delta_seq from model input.
            lambda_at_event (tensor): [batch_size, seq_len, num_event_types], unmasked intensity at event.
            lambdas_loss_samples (tensor): [batch_size, seq_len, num_sample, num_event_types].
            seq_mask (tensor): [batch_size, seq_len], sequence mask vector.
            type_seq (tensor): [batch_size, seq_len], sequence of mark ids.
        
        Returns:
            tuple: (type_ll, time_ll, num_events)
                - type_ll: Event type log-likelihood (scalar after summing)
                - time_ll: Time/non-event log-likelihood (scalar after summing)
                - num_events: Number of valid events
        """
        # Add epsilon for stability
        lambda_at_event = lambda_at_event + self.eps
        lambdas_loss_samples = lambdas_loss_samples + self.eps
        log_marked_event_lambdas = lambda_at_event.log()
        total_sampled_lambdas = lambdas_loss_samples.sum(dim=-1)
        
        # Type component: Event type prediction loss (positive contribution to LL)
        type_ll = -F.nll_loss(
            log_marked_event_lambdas.permute(0, 2, 1),
            target=type_seq,
            ignore_index=self.pad_token_id,
            reduction='none',
        )
        
        # Time component: Non-event integral (negative contribution to LL)
        if self.use_mc_samples:
            non_event_integral = total_sampled_lambdas.mean(dim=-1) * time_delta_seq * seq_mask
        else:
            non_event_integral = 0.5 * (total_sampled_lambdas[..., 1:] + total_sampled_lambdas[..., :-1]).mean(dim=-1) * time_delta_seq * seq_mask
        
        # Time LL is negative of the integral (since it's subtracted from event LL)
        time_ll = -non_event_integral
        
        num_events = torch.masked_select(type_ll, type_ll.ne(0.0)).size()[0]
        
        # Return summed components (both are now log-likelihood components)
        return type_ll.sum(), time_ll.sum(), num_events

    def compute_mse_loss(self, pred_dtime, label_dtime, seq_mask):
        """Compute MSE loss for time predictions.
        
        Direct MSE between predicted and label time intervals.
        
        Args:
            pred_dtime (tensor): [batch_size, seq_len] predicted time intervals
            label_dtime (tensor): [batch_size, seq_len] ground truth time intervals  
            seq_mask (tensor): [batch_size, seq_len] mask for valid events
            
        Returns:
            tensor: scalar MSE loss value
        """
        # Mask out padded values
        if seq_mask is not None:
            pred_masked = pred_dtime * seq_mask
            label_masked = label_dtime * seq_mask
            # Count valid elements
            num_valid = seq_mask.sum()
            if num_valid == 0:
                return torch.tensor(0.0, device=self.device)
        else:
            pred_masked = pred_dtime
            label_masked = label_dtime
            num_valid = pred_dtime.numel()
        
        # Compute mean squared error
        squared_diff = (pred_masked - label_masked) ** 2
        mse = squared_diff.sum() / num_valid
        
        return mse
    
    def compute_nll_with_uncertainty(self, type_ll, time_ll):
        """Compute NLL with uncertainty-based weighting for type and time components.
        
        This implements uncertainty weighting WITHIN the NLL loss:
        L_nll = 1/(2*σ_type²) * L_type + 1/(2*σ_time²) * L_time + log(σ_type) + log(σ_time)
        
        Args:
            type_ll (tensor): Event type log-likelihood loss (scalar, negative)
            time_ll (tensor): Time/non-event log-likelihood loss (scalar, negative)
            
        Returns:
            tuple: (weighted_nll, type_weight, time_weight) 
                - weighted_nll: Combined NLL with uncertainty weighting
                - type_weight: Weight applied to type loss (for monitoring)
                - time_weight: Weight applied to time loss (for monitoring)
        """
        # Initialize learnable parameters if not exists
        if not hasattr(self, 'log_var_type'):
            self.log_var_type = nn.Parameter(torch.zeros(1, device=self.device))
        if not hasattr(self, 'log_var_time'):
            self.log_var_time = nn.Parameter(torch.zeros(1, device=self.device))
        
        # Compute uncertainty-weighted NLL
        # Clamp log_var to prevent extreme values
        # Range: [-2, 2] gives weights from ~0.13 to ~3.7
        log_var_type_clamped = torch.clamp(self.log_var_type, min=-2.0, max=2.0)
        log_var_time_clamped = torch.clamp(self.log_var_time, min=-2.0, max=2.0)
        
        # precision = exp(-log_var) = 1/σ²
        precision_type = torch.exp(-log_var_type_clamped)
        precision_time = torch.exp(-log_var_time_clamped)
        
        # Note: type_ll and time_ll are log-likelihood components (both negative for typical data)
        # We need to negate them to get positive loss values for minimization
        weighted_type = 0.5 * precision_type * (-type_ll)
        weighted_time = 0.5 * precision_time * (-time_ll)
        
        # Regularization terms (use clamped values)
        reg_type = 0.5 * log_var_type_clamped
        reg_time = 0.5 * log_var_time_clamped
        
        # Total weighted NLL
        weighted_nll = weighted_type + weighted_time + reg_type + reg_time
        
        # Extract weights for monitoring
        with torch.no_grad():
            type_weight = (0.5 * precision_type).item()
            time_weight = (0.5 * precision_time).item()
            
        return weighted_nll, type_weight, time_weight
    
    def compute_hybrid_loss(self, nll_loss, mse_loss):
        """Compute hybrid loss using uncertainty-based weighting.
        
        This implements the formula:
        L_total = 1/(2*σ1²) * L_NLL + 1/(2*σ2²) * L_MSE + log(σ1) + log(σ2)
        
        Args:
            nll_loss (tensor): Negative log-likelihood loss (scalar)
            mse_loss (tensor): Mean squared error loss (scalar)
            
        Returns:
            tuple: (total_loss, nll_weight, mse_weight) where weights are for monitoring
        """
        if not self.use_hybrid_loss:
            # If hybrid loss is not enabled, just return NLL loss
            return nll_loss, 1.0, 0.0
        
        # Compute uncertainty-weighted loss
        # # precision = exp(-log_var) = 1/σ²
        # precision_nll = torch.exp(-self.log_var_nll)
        # precision_mse = torch.exp(-self.log_var_mse)

        # precision = 1/σ²
        precision_nll = 1/((self.log_var_nll)**2)
        precision_mse = 1/((self.log_var_mse)**2)
        
        # Weighted losses: L_i / (2 * σ_i²)
        weighted_nll = 0.5 * precision_nll * nll_loss
        weighted_mse = 0.5 * precision_mse * mse_loss
        
        # Regularization terms: log(σ_i)
        reg_nll = torch.log(self.log_var_nll)
        reg_mse = torch.log(self.log_var_mse)
        
        # Total loss
        total_loss = weighted_nll + weighted_mse + reg_nll + reg_mse
        
        # Extract weights for monitoring
        with torch.no_grad():
            # These are the effective weights being applied to each loss
            nll_weight = (0.5 * precision_nll).item()
            mse_weight = (0.5 * precision_mse).item()
            
        return total_loss, nll_weight, mse_weight

    def compute_event_only_hybrid_loss(self, event_ll, mse_loss):
        """Compute hybrid loss using event-only NLL (no non-event term) and MSE with uncertainty weighting.

        This implements the formula:
        L_total = 1/(2*σ1²) * L_event_only + 1/(2*σ2²) * L_MSE + log(σ1) + log(σ2)

        Args:
            event_ll (tensor): Event log-likelihood (scalar, positive - just the event term)
            mse_loss (tensor): Mean squared error loss (scalar)

        Returns:
            tuple: (total_loss, event_weight, mse_weight) where weights are for monitoring
        """
        # Convert event_ll to loss (negate it)
        event_only_nll = -event_ll

        # Compute uncertainty-weighted loss
        # precision = 1/σ²
        precision_nll = 1/((self.log_var_nll)**2)
        precision_mse = 1/((self.log_var_mse)**2)

        # Weighted losses: L_i / (2 * σ_i²)
        weighted_nll = 0.5 * precision_nll * event_only_nll
        weighted_mse = 0.5 * precision_mse * mse_loss

        # Regularization terms: log(σ_i)
        reg_nll = torch.log(self.log_var_nll)
        reg_mse = torch.log(self.log_var_mse)

        # Total loss
        total_loss = weighted_nll + weighted_mse + reg_nll + reg_mse

        # Extract weights for monitoring
        with torch.no_grad():
            # These are the effective weights being applied to each loss
            nll_weight = (0.5 * precision_nll).item()
            mse_weight = (0.5 * precision_mse).item()

        return total_loss, nll_weight, mse_weight

    # def compute_hybrid_loss(self, nll_loss, mse_loss):
    #     """Compute hybrid loss using uncertainty-based weighting.
        
    #     This implements the formula:
    #     L_total = 1/(2*σ1²) * L_NLL + 1/(2*σ2²) * L_MSE + log(σ1) + log(σ2)
        
    #     Args:
    #         nll_loss (tensor): Negative log-likelihood loss (scalar)
    #         mse_loss (tensor): Mean squared error loss (scalar)
            
    #     Returns:
    #         tuple: (total_loss, nll_weight, mse_weight) where weights are for monitoring
    #     """
    #     if not self.use_hybrid_loss:
    #         # If hybrid loss is not enabled, just return NLL loss
    #         return nll_loss, 1.0, 0.0
        
    #     # Compute uncertainty-weighted loss
    #     # # precision = exp(-log_var) = 1/σ²
    #     # precision_nll = torch.exp(-self.log_var_nll)
    #     # precision_mse = torch.exp(-self.log_var_mse)

    #     # precision = 1/σ²
    #     precision_nll = 1/((self.log_var_nll)**2)
    #     precision_mse = 1/((self.log_var_mse)**2)
        
    #     # Weighted losses: L_i / (2 * σ_i²)
    #     weighted_nll = 0.5 * precision_nll * nll_loss
    #     weighted_mse = 0.5 * precision_mse * mse_loss
        
    #     # Regularization terms: log(σ_i)
    #     reg_nll = 0.5 * self.log_var_nll
    #     reg_mse = 0.5 * self.log_var_mse
        
    #     # Total loss
    #     total_loss = weighted_nll + weighted_mse + reg_nll + reg_mse
        
    #     # Extract weights for monitoring
    #     with torch.no_grad():
    #         # These are the effective weights being applied to each loss
    #         nll_weight = (0.5 * precision_nll).item()
    #         mse_weight = (0.5 * precision_mse).item()
            
    #     return total_loss, nll_weight, mse_weight

    def make_dtime_loss_samples(self, time_delta_seq):
        """Generate the time point samples for every interval.

        Args:
            time_delta_seq (tensor): [batch_size, seq_len].

        Returns:
            tensor: [batch_size, seq_len, n_samples]
        """
        # [1, 1, n_samples]
        dtimes_ratio_sampled = torch.linspace(start=0.0,
                                              end=1.0,
                                              steps=self.loss_integral_num_sample_per_step,
                                              device=self.device)[None, None, :]

        # [batch_size, max_len, n_samples]
        sampled_dtimes = time_delta_seq[:, :, None] * dtimes_ratio_sampled

        return sampled_dtimes

    def compute_states_at_sample_times(self, **kwargs):
        raise NotImplementedError('This need to implemented in inherited class ! ')

    def predict_one_step_at_every_event(self, batch):
        """One-step prediction for every event in the sequence.

        Args:
            time_seqs (tensor): [batch_size, seq_len].
            time_delta_seqs (tensor): [batch_size, seq_len].
            type_seqs (tensor): [batch_size, seq_len].

        Returns:
            tuple: tensors of dtime and type prediction, [batch_size, seq_len].
        """
        time_seq, time_delta_seq, event_seq, batch_non_pad_mask, _ = batch

        # remove the last event, as the prediction based on the last event has no label
        # note: the first dts is 0
        # [batch_size, seq_len]
        time_seq, time_delta_seq, event_seq = time_seq[:, :-1], time_delta_seq[:, :-1], event_seq[:, :-1]

        # [batch_size, seq_len]
        dtime_boundary = torch.max(time_delta_seq * self.event_sampler.dtime_max,
                                   time_delta_seq + self.event_sampler.dtime_max)

        # [batch_size, seq_len, num_sample]
        accepted_dtimes, weights = self.event_sampler.draw_next_time_one_step(time_seq,
                                                                              time_delta_seq,
                                                                              event_seq,
                                                                              dtime_boundary,
                                                                              self.compute_intensities_at_sample_times,
                                                                              compute_last_step_only=False)  # make it explicit

        # We should condition on each accepted time to sample event mark, but not conditioned on the expected event time.
        # 1. Use all accepted_dtimes to get intensity.
        # [batch_size, seq_len, num_sample, num_marks]
        intensities_at_times = self.compute_intensities_at_sample_times(time_seq,
                                                                        time_delta_seq,
                                                                        event_seq,
                                                                        accepted_dtimes)

        # 2. Normalize the intensity over last dim and then compute the weighted sum over the `num_sample` dimension.
        # Each of the last dimension is a categorical distribution over all marks.
        # [batch_size, seq_len, num_sample, num_marks]
        intensities_normalized = intensities_at_times / intensities_at_times.sum(dim=-1, keepdim=True)

        # 3. Compute weighted sum of distributions and then take argmax.
        # [batch_size, seq_len, num_marks]
        intensities_weighted = torch.einsum('...s,...sm->...m', weights, intensities_normalized)

        # [batch_size, seq_len]
        types_pred = torch.argmax(intensities_weighted, dim=-1)

        # [batch_size, seq_len]
        dtimes_pred = torch.sum(accepted_dtimes * weights, dim=-1)  # compute the expected next event time
        return dtimes_pred, types_pred

    def predict_multi_step_since_last_event(self, batch, forward=False):
        """Multi-step prediction since last event in the sequence.

        Args:
            batch (tuple): A tuple containing:
                - time_seq_label (tensor): Timestamps of events [batch_size, seq_len].
                - time_delta_seq_label (tensor): Time intervals between events [batch_size, seq_len].
                - event_seq_label (tensor): Event types [batch_size, seq_len].
                - batch_non_pad_mask_label (tensor): Mask for non-padding elements [batch_size, seq_len].
                - attention_mask (tensor): Mask for attention [batch_size, seq_len].
            forward (bool, optional): Whether to use the entire sequence for prediction. Defaults to False.

        Returns:
            tuple: tensors of dtime and type prediction, [batch_size, seq_len].
        """
        time_seq_label, time_delta_seq_label, event_seq_label, _, _  = batch

        num_step = self.gen_config.num_step_gen

        if not forward:
            time_seq = time_seq_label[:, :-num_step]
            time_delta_seq = time_delta_seq_label[:, :-num_step]
            event_seq = event_seq_label[:, :-num_step]
        else:
            time_seq, time_delta_seq, event_seq = time_seq_label, time_delta_seq_label, event_seq_label

        for i in range(num_step):
            # [batch_size, seq_len]
            dtime_boundary = time_delta_seq + self.event_sampler.dtime_max

            # [batch_size, 1, num_sample]
            accepted_dtimes, weights = \
                self.event_sampler.draw_next_time_one_step(time_seq,
                                                           time_delta_seq,
                                                           event_seq,
                                                           dtime_boundary,
                                                           self.compute_intensities_at_sample_times,
                                                           compute_last_step_only=True)

            # [batch_size, 1]
            dtimes_pred = torch.sum(accepted_dtimes * weights, dim=-1)

            # [batch_size, seq_len, 1, event_num]
            intensities_at_times = self.compute_intensities_at_sample_times(time_seq,
                                                                            time_delta_seq,
                                                                            event_seq,
                                                                            dtimes_pred[:, :, None],
                                                                            max_steps=event_seq.size()[1])

            # [batch_size, seq_len, event_num]
            intensities_at_times = intensities_at_times.squeeze(dim=-2)

            # [batch_size, seq_len]
            types_pred = torch.argmax(intensities_at_times, dim=-1)

            # [batch_size, 1]
            types_pred_ = types_pred[:, -1:]
            dtimes_pred_ = dtimes_pred[:, -1:]
            time_pred_ = time_seq[:, -1:] + dtimes_pred_

            # concat to the prefix sequence
            time_seq = torch.cat([time_seq, time_pred_], dim=-1)
            time_delta_seq = torch.cat([time_delta_seq, dtimes_pred_], dim=-1)
            event_seq = torch.cat([event_seq, types_pred_], dim=-1)

        return time_delta_seq[:, -num_step - 1:], event_seq[:, -num_step - 1:], \
               time_delta_seq_label[:, -num_step - 1:], event_seq_label[:, -num_step - 1:]
