import torch
from torch import nn
from torch.nn import functional as F
from torch.autograd import grad

from easy_tpp.model.torch_model.torch_basemodel import TorchBaseModel


class CumulHazardFunctionNetwork(nn.Module):
    """Cumulative Hazard Function Network
    ref: https://github.com/wassname/torch-neuralpointprocess
    """

    def __init__(self, model_config):
        super(CumulHazardFunctionNetwork, self).__init__()
        self.hidden_size = model_config.hidden_size
        self.num_mlp_layers = model_config.model_specs['num_mlp_layers']
        self.num_event_types = model_config.num_event_types
        self.proper_marked_intensities = model_config.model_specs["proper_marked_intensities"]

        # transform inter-event time embedding
        self.layer_dense_1 = nn.Linear(in_features=1, out_features=self.hidden_size)

        # concat rnn states and inter-event time embedding
        self.layer_dense_2 = nn.Linear(in_features=self.hidden_size * 2, out_features=self.hidden_size)

        # mlp layers
        self.module_list = nn.ModuleList(
            [nn.Linear(in_features=self.hidden_size, out_features=self.hidden_size) for _ in
             range(self.num_mlp_layers - 1)])

        self.layer_dense_3 = nn.Sequential(nn.Linear(in_features=self.hidden_size,
                                                     out_features=self.num_event_types),
                                           nn.Softplus())

        self.params_eps = torch.finfo(torch.float32).eps  # ensure positiveness of parameters

        self.init_weights_positive()

    def init_weights_positive(self):
        for p in self.parameters():
            p.data = torch.abs(p.data)
            p.data = torch.clamp(p.data, min=self.params_eps)

    def forward(self, hidden_states, time_delta_seqs):
        for p in self.parameters():
            p.data = torch.clamp(p.data, min=self.params_eps)

        # Check if we're in a no_grad context (evaluation mode)
        compute_gradients = torch.is_grad_enabled()
        
        if compute_gradients:
            # Enable gradients for time_delta_seqs only if we're in training mode
            time_delta_seqs = time_delta_seqs.requires_grad_(True)

        # Handle both 3D and 4D inputs (4D happens during generation/sampling)
        # 3D: [batch_size, seq_len, hidden_size] (normal training case)
        # 4D: [batch_size, seq_len, num_samples, hidden_size] (generation/sampling case)
        original_shape = hidden_states.shape
        is_4d_input = len(original_shape) == 4
        
        if is_4d_input:
            batch_size, seq_len, num_samples, hidden_size = original_shape
            # Flatten to process through the network
            hidden_states = hidden_states.reshape(batch_size * seq_len * num_samples, 1, hidden_size)
            # time_delta_seqs is [batch_size, seq_len, num_samples], reshape accordingly
            time_delta_seqs = time_delta_seqs.reshape(batch_size * seq_len * num_samples, 1)
        
        # [batch_size, seq_len, hidden_size] or [batch_size*seq_len, num_samples, hidden_size]
        t = self.layer_dense_1(time_delta_seqs.unsqueeze(dim=-1))

        # [batch_size, seq_len, hidden_size] or [batch_size*seq_len, num_samples, hidden_size]
        out = torch.tanh(self.layer_dense_2(torch.cat([hidden_states, t], dim=-1)))
        for layer in self.module_list:
            out = torch.tanh(layer(out))

        # [batch_size, seq_len, num_event_types] or [batch_size*seq_len, num_samples, num_event_types]
        integral_lambda = self.layer_dense_3(out)

        # Skip gradient computation for 4D inputs (generation case)
        if is_4d_input:
            # For generation, use numerical approximation
            eps = 1e-4
            time_delta_plus = time_delta_seqs + eps
            
            # Recompute with slightly perturbed time
            t_plus = self.layer_dense_1(time_delta_plus.unsqueeze(dim=-1))
            out_plus = torch.tanh(self.layer_dense_2(torch.cat([hidden_states, t_plus], dim=-1)))
            for layer in self.module_list:
                out_plus = torch.tanh(layer(out_plus))
            integral_lambda_plus = self.layer_dense_3(out_plus)
            
            # Finite difference approximation
            derivative_integral_lambda = (integral_lambda_plus - integral_lambda) / eps
            
            # Reshape back to 4D
            # From [batch_size * seq_len * num_samples, 1, num_event_types] to [batch_size, seq_len, num_samples, num_event_types]
            integral_lambda = integral_lambda.squeeze(1).reshape(batch_size, seq_len, num_samples, -1)
            derivative_integral_lambda = derivative_integral_lambda.squeeze(1).reshape(batch_size, seq_len, num_samples, -1)
        elif self.proper_marked_intensities and compute_gradients:
            # Only compute gradients if we're in training mode (3D case)
            derivative_integral_lambdas = []
            for i in range(integral_lambda.shape[-1]):  # iterate over marks
                derivative_integral_lambdas.append(grad(
                    integral_lambda[..., i].sum(),
                    time_delta_seqs,
                    create_graph=True, retain_graph=True)[0])
            derivative_integral_lambda = torch.stack(derivative_integral_lambdas, dim=-1)
        elif compute_gradients:
            # For multi-type events without proper_marked_intensities (training mode, 3D case)
            base_derivative = grad( 
                integral_lambda.sum(),
                time_delta_seqs,
                create_graph=True, retain_graph=True)[0]
            
            # Use the integral_lambda values directly as relative intensities for each type
            # This preserves the type-specific information from the network output
            # Normalize to ensure proper intensity distribution
            type_intensities = integral_lambda / (integral_lambda.sum(dim=-1, keepdim=True) + 1e-8)
            
            # Scale by base derivative to get proper intensity values
            derivative_integral_lambda = base_derivative.unsqueeze(-1) * type_intensities
        else:
            # Evaluation mode for 3D inputs: use numerical approximation for derivatives
            # This avoids gradient computation entirely
            eps = 1e-4
            time_delta_plus = time_delta_seqs + eps
            
            # Recompute with slightly perturbed time
            t_plus = self.layer_dense_1(time_delta_plus.unsqueeze(dim=-1))
            out_plus = torch.tanh(self.layer_dense_2(torch.cat([hidden_states, t_plus], dim=-1)))
            for layer in self.module_list:
                out_plus = torch.tanh(layer(out_plus))
            integral_lambda_plus = self.layer_dense_3(out_plus)
            
            # Finite difference approximation
            derivative_integral_lambda = (integral_lambda_plus - integral_lambda) / eps

        return integral_lambda, derivative_integral_lambda


class FullyNN(TorchBaseModel):
    """Torch implementation of
        Fully Neural Network based Model for General Temporal Point Processes, NeurIPS 2019.
        https://arxiv.org/abs/1905.09690

        ref: https://github.com/KanghoonYoon/torch-neuralpointprocess/blob/master/module.py;
            https://github.com/wassname/torch-neuralpointprocess
    """

    def __init__(self, model_config):
        """Initialize the model

        Args:
            model_config (EasyTPP.ModelConfig): config of model specs.
        """
        super(FullyNN, self).__init__(model_config)

        self.rnn_type = model_config.rnn_type
        self.rnn_list = [nn.LSTM, nn.RNN, nn.GRU]
        self.n_layers = model_config.num_layers
        self.dropout_rate = model_config.dropout_rate
        for sub_rnn_class in self.rnn_list:
            if sub_rnn_class.__name__ == self.rnn_type:
                self.layer_rnn = sub_rnn_class(input_size=1 + self.hidden_size,
                                               hidden_size=self.hidden_size,
                                               num_layers=self.n_layers,
                                               batch_first=True,
                                               dropout=self.dropout_rate)

        self.layer_intensity = CumulHazardFunctionNetwork(model_config)

    def forward(self, time_seqs, time_delta_seqs, type_seqs):
        """Call the model

        Args:
            time_seqs (tensor): [batch_size, seq_len], timestamp seqs.
            time_delta_seqs (tensor): [batch_size, seq_len], inter-event time seqs.
            type_seqs (tensor): [batch_size, seq_len], event type seqs.

        Returns:
            tensor: hidden states at event times.
        """
        # [batch_size, seq_len, hidden_size]
        type_embedding = self.layer_type_emb(type_seqs)

        # [batch_size, seq_len, hidden_size + 1]
        rnn_input = torch.cat((type_embedding, time_delta_seqs.unsqueeze(-1)), dim=-1)

        # [batch_size, seq_len, hidden_size]
        # states right after the event
        hidden_states, _ = self.layer_rnn(rnn_input)

        return hidden_states

    def loglike_loss(self, batch):
        """Compute the loglike loss.

        Args:
            batch (tuple, list): batch input.

        Returns:
            list: loglike loss, num events.
        """
        # [batch_size, seq_len]
        time_seqs, time_delta_seqs, type_seqs, batch_non_pad_mask, _ = batch

        # [batch_size, seq_len, hidden_size]
        hidden_states = self.forward(
            time_seqs[:, :-1],
            time_delta_seqs[:, :-1],
            type_seqs[:, :-1],
        )
        # [batch_size, seq_len, num_event_types]
        integral_lambda, derivative_integral_lambda = self.layer_intensity(hidden_states, time_delta_seqs[:, 1:])

        # First, add an epsilon to every marked intensity for stability
        derivative_integral_lambda += self.eps

        # Compute components for each LL term
        log_marked_event_lambdas = derivative_integral_lambda.log()

        # Compute event LL - [batch_size, seq_len]
        event_ll = -F.nll_loss(
            log_marked_event_lambdas.permute(0, 2, 1),  # mark dimension needs to come second, not third to match nll_loss specs
            target=type_seqs[:, 1:],
            ignore_index=self.pad_token_id,  # Padded events have a pad_token_id as a value
            reduction='none', # Does not aggregate, and replaces what would have been the log(marked intensity) with 0.
        )

        # [batch_size, seq_len]
        # multiplied by sequence mask
        non_event_ll = integral_lambda.sum(-1) * batch_non_pad_mask[:, 1:]
        num_events = torch.masked_select(event_ll, event_ll.ne(0.0)).size()[0]

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
            total_intensity = derivative_integral_lambda.sum(dim=-1)  # [batch_size, seq_len]
            pred_dtime = 1.0 / (total_intensity + self.eps)
            label_dtime = time_delta_seqs[:, 1:]  # Ground truth time intervals
            seq_mask = batch_non_pad_mask[:, 1:]  # Mask for valid events

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
            nll_loss = - (event_ll - non_event_ll).sum()

            # Compute MSE loss on time predictions
            total_intensity = derivative_integral_lambda.sum(dim=-1)  # [batch_size, seq_len]
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
            # Compute NLL loss
            nll_loss = - (2 * event_ll - 0.2 * non_event_ll).sum()
            # Compute MSE loss on time predictions
            # Use the reciprocal of total intensity as a time predictor
            total_intensity = derivative_integral_lambda.sum(dim=-1)  # [batch_size, seq_len]
            pred_dtime = 1.0 / (total_intensity + self.eps)
            label_dtime = time_delta_seqs[:, 1:]  # Ground truth time intervals
            seq_mask = batch_non_pad_mask[:, 1:]  # Mask for valid events
            
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
            nll_loss = - (event_ll - non_event_ll).sum()
            loss = nll_loss
            self._metric_nll_loss = nll_loss

        return loss, num_events

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
            sample_dtimes (tensor): [batch_size, seq_len, num_samples] or [batch_size, 1, num_samples], 
                                    sampled inter-event timestamps.

        Returns:
            tensor: [batch_size, seq_len, num_samples, num_event_types] or [batch_size, 1, num_samples, num_event_types],
                    intensity at all sampled times.
        """

        compute_last_step_only = kwargs.get('compute_last_step_only', False)
        max_steps = kwargs.get('max_steps', None)  # Handle max_steps parameter

        # [batch_size, seq_len, hidden_size]
        hidden_states = self.forward(
            time_seqs=time_seqs,
            time_delta_seqs=time_delta_seqs,
            type_seqs=type_seqs,
        )

        if compute_last_step_only:
            # When computing only for the last step, sample_dtimes has shape [batch_size, 1, num_samples]
            # We need only the last hidden state
            hidden_states = hidden_states[:, -1:, :]  # [batch_size, 1, hidden_size]
            # Make sure sample_dtimes matches
            if sample_dtimes.size(1) != 1:
                sample_dtimes = sample_dtimes[:, -1:, :]
            
        num_samples = sample_dtimes.size()[-1]
        batch_size, seq_len, hidden_size = hidden_states.shape

        # Ensure sample_dtimes has the same seq_len as hidden_states
        if sample_dtimes.size(1) != seq_len:
            # This can happen during generation when sequences have different actual lengths
            # Truncate or pad as needed
            if sample_dtimes.size(1) > seq_len:
                sample_dtimes = sample_dtimes[:, :seq_len, :]
            else:
                # This shouldn't happen but handle it just in case
                pad_size = seq_len - sample_dtimes.size(1)
                padding = torch.zeros(batch_size, pad_size, num_samples, device=sample_dtimes.device)
                sample_dtimes = torch.cat([sample_dtimes, padding], dim=1)

        # Expand hidden states to match sample dimensions
        hidden_states_ = hidden_states[..., None, :].expand(batch_size, seq_len, num_samples, hidden_size)
        
        # Compute intensities at sampled times
        _, derivative_integral_lambda = self.layer_intensity.forward(
            hidden_states=hidden_states_,
            time_delta_seqs=sample_dtimes,
        )

        # Return the computed lambdas
        lambdas = derivative_integral_lambda
        return lambdas
