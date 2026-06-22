"""
Enhanced metrics module that applies inverse transform to predictions and labels
before calculating metrics, ensuring RMSE is computed on the original scale.
"""

import numpy as np
from easy_tpp.utils.const import PredOutputIndex
from easy_tpp.utils.metrics import MetricsHelper
from easy_tpp.utils.inverse_transform_module import InverseTransformer


class MetricsWithInverseTransform:
    """Helper class to compute metrics with inverse transform applied."""
    
    def __init__(self, use_log_transform=True):
        """
        Initialize the metrics calculator with inverse transform support.
        
        Args:
            use_log_transform (bool): Whether the data uses log1p transform.
        """
        self.transformer = InverseTransformer(use_log_transform=use_log_transform)
    
    def apply_inverse_to_predictions(self, predictions, labels, prediction_type='intervals'):
        """
        Apply inverse transform to predictions and labels.
        
        Args:
            predictions (np.array): Model predictions in transformed scale.
            labels (np.array): Ground truth in transformed scale.
            prediction_type (str): Type of prediction - 'intervals' or 'times'.
        
        Returns:
            tuple: (predictions_original, labels_original) in original scale.
        """
        if prediction_type == 'intervals':
            pred_original = self.transformer.inverse_intervals(predictions)
            label_original = self.transformer.inverse_intervals(labels)
        elif prediction_type == 'times':
            pred_original = self.transformer.inverse_times(predictions)
            label_original = self.transformer.inverse_times(labels)
        else:
            # If type not specified, try to infer from data characteristics
            # Intervals typically have first element as 0
            if len(predictions) > 0 and predictions[0] == 0:
                pred_original = self.transformer.inverse_intervals(predictions)
                label_original = self.transformer.inverse_intervals(labels)
            else:
                pred_original = self.transformer.inverse_times(predictions)
                label_original = self.transformer.inverse_times(labels)
        
        return pred_original, label_original
    
    def compute_rmse_original_scale(self, predictions, labels, seq_mask=None):
        """
        Compute RMSE on the original scale (blocks) after inverse transform.
        
        Args:
            predictions (tuple): Model predictions containing time predictions.
            labels (tuple): Ground truth containing time labels.
            seq_mask (np.array): Mask for valid sequence positions.
        
        Returns:
            float: RMSE computed on original scale.
        """
        # Extract time predictions
        if isinstance(predictions, (tuple, list)):
            pred = predictions[PredOutputIndex.TimePredIndex]
        else:
            pred = predictions
            
        if isinstance(labels, (tuple, list)):
            label = labels[PredOutputIndex.TimePredIndex]
        else:
            label = labels
        
        # Apply mask if provided
        if seq_mask is not None and len(seq_mask) > 0:
            pred = pred[seq_mask]
            label = label[seq_mask]
        
        # Flatten arrays
        pred = np.reshape(pred, [-1])
        label = np.reshape(label, [-1])
        
        # Apply inverse transform
        pred_original, label_original = self.apply_inverse_to_predictions(
            pred, label, prediction_type='intervals'
        )
        
        # Calculate RMSE on original scale
        rmse = np.sqrt(np.mean((pred_original - label_original) ** 2))
        
        return rmse
    
    def compute_mae_original_scale(self, predictions, labels, seq_mask=None):
        """
        Compute MAE on the original scale (blocks) after inverse transform.
        
        Args:
            predictions (tuple): Model predictions containing time predictions.
            labels (tuple): Ground truth containing time labels.
            seq_mask (np.array): Mask for valid sequence positions.
        
        Returns:
            float: MAE computed on original scale.
        """
        # Extract time predictions
        if isinstance(predictions, (tuple, list)):
            pred = predictions[PredOutputIndex.TimePredIndex]
        else:
            pred = predictions
            
        if isinstance(labels, (tuple, list)):
            label = labels[PredOutputIndex.TimePredIndex]
        else:
            label = labels
        
        # Apply mask if provided
        if seq_mask is not None and len(seq_mask) > 0:
            pred = pred[seq_mask]
            label = label[seq_mask]
        
        # Flatten arrays
        pred = np.reshape(pred, [-1])
        label = np.reshape(label, [-1])
        
        # Apply inverse transform
        pred_original, label_original = self.apply_inverse_to_predictions(
            pred, label, prediction_type='intervals'
        )
        
        # Calculate MAE on original scale
        mae = np.mean(np.abs(pred_original - label_original))
        
        return mae
    
    def transform_generation_output(self, generated_sequences):
        """
        Transform generated sequences back to original scale.
        
        Args:
            generated_sequences (list): List of generated sequences in log scale.
        
        Returns:
            list: Sequences transformed back to original scale.
        """
        if not generated_sequences:
            return generated_sequences
        
        original_sequences = []
        for seq in generated_sequences:
            if isinstance(seq, dict):
                original_seq = self.transformer.inverse_sequence(seq)
            else:
                # Handle array format
                original_seq = self.transformer.inverse_intervals(seq)
            original_sequences.append(original_seq)
        
        return original_sequences


# Register enhanced metrics functions with MetricsHelper
@MetricsHelper.register(name='rmse_original', direction=MetricsHelper.MINIMIZE, overwrite=True)
def rmse_original_scale_metric(predictions, labels, **kwargs):
    """
    Compute RMSE metrics on the original scale (after inverse transform).
    
    Args:
        predictions (np.array): Model predictions in log scale.
        labels (np.array): Ground truth in log scale.
    
    Returns:
        float: RMSE computed on original scale (blocks).
    """
    calculator = MetricsWithInverseTransform(use_log_transform=True)
    return calculator.compute_rmse_original_scale(predictions, labels, kwargs.get('seq_mask'))


@MetricsHelper.register(name='mae_original', direction=MetricsHelper.MINIMIZE, overwrite=True)
def mae_original_scale_metric(predictions, labels, **kwargs):
    """
    Compute MAE metrics on the original scale (after inverse transform).
    
    Args:
        predictions (np.array): Model predictions in log scale.
        labels (np.array): Ground truth in log scale.
    
    Returns:
        float: MAE computed on original scale (blocks).
    """
    calculator = MetricsWithInverseTransform(use_log_transform=True)
    return calculator.compute_mae_original_scale(predictions, labels, kwargs.get('seq_mask'))