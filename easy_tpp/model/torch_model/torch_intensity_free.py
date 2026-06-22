import numpy as np
import torch
import torch.distributions as D
from torch import nn
from torch.distributions import Categorical, TransformedDistribution
from torch.distributions import MixtureSameFamily as TorchMixtureSameFamily
from torch.distributions import Normal as TorchNormal

from easy_tpp.model.torch_model.torch_basemodel import TorchBaseModel


def clamp_preserve_gradients(x, min_val, max_val):
    """Clamp the tensor while preserving gradients in the clamped region.

    Args:
        x (tensor): tensor to be clamped.
        min_val (float): minimum value.
        max_val (float): maximum value.
    """
    return x + (x.clamp(min_val, max_val) - x).detach()


class Normal(TorchNormal):
    """Normal distribution, redefined `log_cdf` and `log_survival_function` due to
    no numerically stable implementation of them is available for normal distribution.
    """

    def log_cdf(self, x):
        cdf = clamp_preserve_gradients(self.cdf(x), 1e-7, 1 - 1e-7)
        return cdf.log()

    def log_survival_function(self, x):
        cdf = clamp_preserve_gradients(self.cdf(x), 1e-7, 1 - 1e-7)
        return torch.log(1.0 - cdf)


class MixtureSameFamily(TorchMixtureSameFamily):
    """Mixture (same-family) distribution, redefined `log_cdf` and `log_survival_function`.
    """

    def log_cdf(self, x):
        x = self._pad(x)
        log_cdf_x = self.component_distribution.log_cdf(x)
        mix_logits = self.mixture_distribution.logits
        return torch.logsumexp(log_cdf_x + mix_logits, dim=-1)

    def log_survival_function(self, x):
        x = self._pad(x)
        log_sf_x = self.component_distribution.log_survival_function(x)
        mix_logits = self.mixture_distribution.logits
        return torch.logsumexp(log_sf_x + mix_logits, dim=-1)


class LogNormalMixtureDistribution(TransformedDistribution):
    """
    Mixture of log-normal distributions.

    Args:
        locs (tensor): [batch_size, seq_len, num_mix_components].
        log_scales (tensor): [batch_size, seq_len, num_mix_components].
        log_weights (tensor): [batch_size, seq_len, num_mix_components].
        mean_log_inter_time (float): Average log-inter-event-time.
        std_log_inter_time (float): Std of log-inter-event-times.
    """

    def __init__(self, locs, log_scales, log_weights, mean_log_inter_time, std_log_inter_time, validate_args=None):
        mixture_dist = D.Categorical(logits=log_weights)
        component_dist = Normal(loc=locs, scale=log_scales.exp())
        GMM = MixtureSameFamily(mixture_dist, component_dist)
        if mean_log_inter_time == 0.0 and std_log_inter_time == 1.0:
            transforms = []
        else:
            transforms = [D.AffineTransform(loc=mean_log_inter_time, scale=std_log_inter_time)]
        self.mean_log_inter_time = mean_log_inter_time
        self.std_log_inter_time = std_log_inter_time
        transforms.append(D.ExpTransform())

        self.transforms = transforms
        sign = 1
        for transform in self.transforms:
            sign = sign * transform.sign
        self.sign = int(sign)
        super().__init__(GMM, transforms, validate_args=validate_args)

    def log_cdf(self, x):
        for transform in self.transforms[::-1]:
            x = transform.inv(x)
        if self._validate_args:
            self.base_dist._validate_sample(x)

        if self.sign == 1:
            return self.base_dist.log_cdf(x)
        else:
            return self.base_dist.log_survival_function(x)

    def log_survival_function(self, x):
        for transform in self.transforms[::-1]:
            x = transform.inv(x)
        if self._validate_args:
            self.base_dist._validate_sample(x)

        if self.sign == 1:
            return self.base_dist.log_survival_function(x)
        else:
            return self.base_dist.log_cdf(x)


class IntensityFree(TorchBaseModel):
    """Torch implementation of Intensity-Free Learning of Temporal Point Processes, ICLR 2020.
    https://openreview.net/pdf?id=HygOjhEYDH

    reference: https://github.com/shchur/ifl-tpp
    """

    def __init__(self, model_config):
        """Initialize the model

        Args:
            model_config (EasyTPP.ModelConfig): config of model specs.

        """
        super(IntensityFree, self).__init__(model_config)

        self.num_mix_components = model_config.model_specs['num_mix_components']
        # Get normalization parameters - check if they exist at root level
        self.mean_log_inter_time = getattr(model_config, "mean_log_inter_time", 0.0)
        self.std_log_inter_time = getattr(model_config, "std_log_inter_time", 1.0)
        
        # Print for debugging
        print(f"IntensityFree: mean_log_inter_time={self.mean_log_inter_time}, std_log_inter_time={self.std_log_inter_time}")

        self.num_features = 1 + self.hidden_size

        self.layer_rnn = nn.GRU(input_size=self.num_features,
                                hidden_size=self.hidden_size,
                                num_layers=1,  # used in original paper
                                batch_first=True)

        self.mark_linear = nn.Linear(self.hidden_size, self.num_event_types_pad)
        self.linear = nn.Linear(self.hidden_size, 3 * self.num_mix_components)

    def forward(self, time_delta_seqs, type_seqs):
        """Call the model.

        Args:
            time_delta_seqs (tensor): [batch_size, seq_len], inter-event time seqs.
            type_seqs (tensor): [batch_size, seq_len], event type seqs.

        Returns:
            list: hidden states, [batch_size, seq_len, hidden_dim], states right before the event happens.
        """
        # [batch_size, seq_len, hidden_size]
        # We dont normalize inter-event time here
        temporal_seqs = torch.log(time_delta_seqs + self.eps).unsqueeze(-1)

        # [batch_size, seq_len, hidden_size]
        type_emb = self.layer_type_emb(type_seqs)

        # [batch_size, seq_len, hidden_size + 1]
        rnn_input = torch.cat([temporal_seqs, type_emb], dim=-1)

        # [batch_size, seq_len, hidden_size]
        context = self.layer_rnn(rnn_input)[0]

        return context

    def loglike_loss(self, batch):
        """Compute the loglike loss.

        Args:
            batch (list): batch input.

        Returns:
            tuple: loglikelihood loss and num of events.
        """
        time_seqs, time_delta_seqs, type_seqs, batch_non_pad_mask, _ = batch

        # [batch_size, seq_len, hidden_size]
        context = self.forward(time_delta_seqs[:, :-1], type_seqs[:, :-1])

        # [batch_size, seq_len, 3 * num_mix_components]
        raw_params = self.linear(context)
        locs = raw_params[..., :self.num_mix_components]
        log_scales = raw_params[..., self.num_mix_components: (2 * self.num_mix_components)]
        log_weights = raw_params[..., (2 * self.num_mix_components):]

        log_scales = clamp_preserve_gradients(log_scales, -5.0, 3.0)
        log_weights = torch.log_softmax(log_weights, dim=-1)
        inter_time_dist = LogNormalMixtureDistribution(
            locs=locs,
            log_scales=log_scales,
            log_weights=log_weights,
            mean_log_inter_time=self.mean_log_inter_time,
            std_log_inter_time=self.std_log_inter_time
        )

        inter_times = time_delta_seqs[:, 1:].clamp(min=1e-5)
        # [batch_size, seq_len]
        event_mask = torch.logical_and(batch_non_pad_mask[:, 1:], type_seqs[:, 1:] != self.pad_token_id)
        time_ll = inter_time_dist.log_prob(inter_times) * event_mask

        # [batch_size, seq_len, num_marks]
        mark_logits = torch.log_softmax(self.mark_linear(context), dim=-1)
        mark_dist = Categorical(logits=mark_logits)
        mark_ll = mark_dist.log_prob(type_seqs[:, 1:]) * event_mask

        log_p = time_ll + mark_ll

        # [batch_size,]
        num_events = event_mask.sum().item()

        # If NLL uncertainty weighting is enabled (type/time split)
        if self.use_nll_uncertainty:
            # Use STANDARD NLL formula with split components
            # Compute type and time components separately
            type_ll = mark_ll.sum()  # Type component (event type prediction)
            time_ll_sum = time_ll.sum()  # Time component

            # Apply uncertainty weighting to balance type and time
            nll_loss, type_weight, time_weight = self.compute_nll_with_uncertainty(type_ll, time_ll_sum)

            # Store weights for monitoring
            if hasattr(self, 'training') and self.training:
                self._last_type_weight = type_weight
                self._last_time_weight = time_weight
                self._last_type_ll = type_ll.item()
                self._last_time_ll = time_ll_sum.item()

            loss = nll_loss
            self._metric_nll_loss = nll_loss

        # If event-only loss is enabled (event NLL + MSE with uncertainty weighting)
        elif self.use_event_only_loss:
            # For IntensityFree, we already have mark_ll as the event type component
            # Note: time_ll includes the time modeling, so we only use mark_ll for event-only
            event_ll_sum = mark_ll.sum()  # Only the mark/type prediction part

            # For IntensityFree models, we use the mean of the time distribution as time predictor
            pred_dtime = inter_time_dist.sample((100,)).mean(dim=0)  # Sample and average for prediction
            label_dtime = time_delta_seqs[:, 1:]  # Ground truth time intervals
            seq_mask = event_mask  # Use the existing event mask

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
            self._metric_nll_loss = -event_ll_sum

        # If simple hybrid is enabled (NLL + MSE without uncertainty weighting)
        elif self.use_simple_hybrid:
            # Compute standard NLL loss
            nll_loss = -log_p.sum()

            # For IntensityFree models, we use the mean of the time distribution as time predictor
            pred_dtime = inter_time_dist.sample((100,)).mean(dim=0)  # Sample and average for prediction
            label_dtime = time_delta_seqs[:, 1:]  # Ground truth time intervals
            seq_mask = event_mask  # Use the existing event mask

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
            # Compute NLL loss
            nll_loss = -log_p.sum()
            # For IntensityFree models, we use the mean of the time distribution as time predictor
            pred_dtime = inter_time_dist.sample((100,)).mean(dim=0)  # Sample and average for prediction
            label_dtime = time_delta_seqs[:, 1:]  # Ground truth time intervals
            seq_mask = event_mask  # Use the existing event mask
            
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
            nll_loss = -log_p.sum()
            loss = nll_loss
            self._metric_nll_loss = nll_loss

        return loss, num_events

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
        # time_delta_seq should start from 1, because the first one is zero
        time_seq, time_delta_seq, event_seq = time_seq[:, :-1], time_delta_seq[:, :-1], event_seq[:, :-1]

        # [batch_size, seq_len, hidden_size]
        context = self.forward(time_delta_seq, event_seq)

        # [batch_size, seq_len, 3 * num_mix_components]
        raw_params = self.linear(context)
        locs = raw_params[..., :self.num_mix_components]
        log_scales = raw_params[..., self.num_mix_components: (2 * self.num_mix_components)]
        log_weights = raw_params[..., (2 * self.num_mix_components):]

        log_scales = clamp_preserve_gradients(log_scales, -5.0, 3.0)
        log_weights = torch.log_softmax(log_weights, dim=-1)
        # For prediction, use a simpler approach to avoid inf values
        # Instead of sampling from the full distribution, use the mode of the mixture
        # The mode is approximately the component with highest weight and its mean
        
        # Get the component with highest weight for each position
        weights = torch.softmax(log_weights, dim=-1)
        max_weight_idx = torch.argmax(weights, dim=-1)  # [batch_size, seq_len]
        
        # Gather the corresponding locs (means in log space)
        batch_size, seq_len, _ = locs.shape
        idx_expanded = max_weight_idx.unsqueeze(-1).expand(batch_size, seq_len, 1)
        mode_locs = torch.gather(locs, 2, idx_expanded).squeeze(-1)  # [batch_size, seq_len]
        
        # Transform from normalized log space to actual time
        if self.mean_log_inter_time != 0.0 or self.std_log_inter_time != 1.0:
            mode_locs = mode_locs * self.std_log_inter_time + self.mean_log_inter_time
        
        # Apply exp transform to get actual times
        dtimes_pred = torch.exp(mode_locs)
        
        # Clamp to reasonable range
        dtimes_pred = torch.clamp(dtimes_pred, min=0.1, max=5000.0)

        # [batch_size, seq_len, num_marks]
        mark_logits = torch.log_softmax(self.mark_linear(context), dim=-1)  # Marks are modeled conditionally independently from times
        types_pred = torch.argmax(mark_logits, dim=-1)
        return dtimes_pred, types_pred
    
    def compute_intensities_at_sample_times(self,
                                            time_seqs,
                                            time_delta_seqs,
                                            type_seqs,
                                            sample_dtimes,
                                            **kwargs):
        """Compute pseudo-intensities at sampled times for compatibility with generation.
        
        Since IntensityFree doesn't model intensity functions, we approximate them
        using the probability density of the learned distribution.
        
        Args:
            time_seqs (tensor): [batch_size, seq_len], times seqs.
            time_delta_seqs (tensor): [batch_size, seq_len], time delta seqs.
            type_seqs (tensor): [batch_size, seq_len], event type seqs.
            sample_dtimes (tensor): [batch_size, seq_len, num_samples], sampled inter-event timestamps.
        
        Returns:
            tensor: [batch_size, seq_len, num_samples, num_event_types], pseudo-intensity at sampled times.
        """
        compute_last_step_only = kwargs.get('compute_last_step_only', False)
        
        # Get hidden states
        # [batch_size, seq_len, hidden_size]
        context = self.forward(time_delta_seqs, type_seqs)
        
        if compute_last_step_only:
            context = context[:, -1:, :]
            # Also need to get just the last step of sample_dtimes
            sample_dtimes = sample_dtimes[:, -1:, :]
        
        # [batch_size, seq_len or 1, 3 * num_mix_components]
        raw_params = self.linear(context)
        locs = raw_params[..., :self.num_mix_components]
        log_scales = raw_params[..., self.num_mix_components: (2 * self.num_mix_components)]
        log_weights = raw_params[..., (2 * self.num_mix_components):]
        
        log_scales = clamp_preserve_gradients(log_scales, -5.0, 3.0)
        log_weights = torch.log_softmax(log_weights, dim=-1)
        
        # Get actual dimensions from the processed tensors
        batch_size = context.shape[0]
        seq_len = context.shape[1]
        num_samples = sample_dtimes.shape[-1]
        
        # Approximate intensity using the PDF of the distribution
        # For log-normal mixture, we use the PDF value as pseudo-intensity
        
        # Expand parameters for sample dimension
        locs_exp = locs.unsqueeze(2).expand(batch_size, seq_len, num_samples, self.num_mix_components)
        log_scales_exp = log_scales.unsqueeze(2).expand(batch_size, seq_len, num_samples, self.num_mix_components)
        log_weights_exp = log_weights.unsqueeze(2).expand(batch_size, seq_len, num_samples, self.num_mix_components)
        
        # Transform sample_dtimes to log space for PDF computation
        # [batch_size, seq_len, num_samples]
        log_sample_dtimes = torch.log(sample_dtimes.clamp(min=1e-5))
        
        # Denormalize if needed
        if self.mean_log_inter_time != 0.0 or self.std_log_inter_time != 1.0:
            log_sample_dtimes = (log_sample_dtimes - self.mean_log_inter_time) / self.std_log_inter_time
        
        # Compute log PDF for each component
        # [batch_size, seq_len, num_samples, num_mix_components]
        log_sample_dtimes_exp = log_sample_dtimes.unsqueeze(-1).expand(batch_size, seq_len, num_samples, self.num_mix_components)
        
        # Gaussian log PDF in log space
        variance = (log_scales_exp.exp() ** 2)
        log_pdf_components = -0.5 * ((log_sample_dtimes_exp - locs_exp) ** 2) / variance - 0.5 * torch.log(2 * np.pi * variance)
        
        # Combine with mixture weights
        log_pdf = torch.logsumexp(log_weights_exp + log_pdf_components, dim=-1)
        
        # Convert to pseudo-intensity (PDF * scaling factor)
        # We use exp(log_pdf) but clamp to avoid extreme values
        pseudo_intensity = torch.exp(log_pdf.clamp(max=10.0))
        
        # For marks, use the mark distribution
        # [batch_size, seq_len, num_event_types]
        mark_logits = torch.log_softmax(self.mark_linear(context), dim=-1)
        mark_probs = torch.exp(mark_logits)
        
        # Expand mark probabilities to match sample dimension
        # [batch_size, seq_len, num_samples, num_event_types_pad]
        mark_probs_expanded = mark_probs.unsqueeze(2).expand(batch_size, seq_len, num_samples, self.num_event_types_pad)
        
        # Combine time intensity with mark probabilities
        # [batch_size, seq_len, num_samples, num_event_types_pad]
        intensities = pseudo_intensity.unsqueeze(-1) * mark_probs_expanded
        
        # Ensure positive values
        intensities = intensities.clamp(min=1e-7)
        
        # Return only the valid event types (exclude padding)
        # [batch_size, seq_len, num_samples, num_event_types]
        return intensities[..., :self.num_event_types]
