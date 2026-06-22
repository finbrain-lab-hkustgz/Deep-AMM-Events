"""
Standalone Inverse Transformation Module
For converting log-transformed predictions back to original scale (block numbers)

This module can be directly imported and used in your model evaluation scripts.
It handles the inverse of log1p transformation without any standardization.
"""

import numpy as np
from typing import Dict, List, Union, Optional


class InverseTransformer:
    """
    Simple inverse transformer for log1p transformed temporal point process data.
    
    Usage:
        transformer = InverseTransformer()
        
        # For individual arrays
        original_intervals = transformer.inverse_intervals(log_intervals)
        original_times = transformer.inverse_times(log_times)
        
        # For entire sequences
        original_seq = transformer.inverse_sequence(transformed_seq)
    """
    
    def __init__(self, use_log_transform: bool = True):
        """
        Initialize the inverse transformer.
        
        Args:
            use_log_transform: Whether the data was log-transformed.
                              Set to False if data is already in original scale.
        """
        self.use_log_transform = use_log_transform
    
    def inverse_intervals(self, intervals: Union[np.ndarray, List[float]]) -> np.ndarray:
        """
        Inverse transform time_since_last_event values.
        
        Args:
            intervals: Log-transformed interval values (or original if no transform)
                      First element should be 0 (representing sequence start)
        
        Returns:
            original_intervals: Original scale interval values (in blocks)
        
        Example:
            >>> log_intervals = [0.0, 2.3979, 3.4012, 1.0986]
            >>> original = transformer.inverse_intervals(log_intervals)
            >>> print(original)  # [0.0, 10.0, 29.0, 2.0]
        """
        if not self.use_log_transform:
            return np.array(intervals)
        
        intervals = np.array(intervals)
        original_intervals = np.zeros_like(intervals)
        
        # First interval always stays 0 (sequence start marker)
        original_intervals[0] = 0
        
        if len(intervals) > 1:
            # Apply inverse of log1p: expm1(x) = exp(x) - 1
            original_intervals[1:] = np.expm1(intervals[1:])
            
            # Round to nearest integer since blocks are discrete
            original_intervals[1:] = np.round(original_intervals[1:])
            
            # Ensure minimum interval of 1 block (no zero intervals except first)
            original_intervals[1:] = np.maximum(original_intervals[1:], 1)
        
        return original_intervals
    
    def inverse_times(self, times: Union[np.ndarray, List[float]]) -> np.ndarray:
        """
        Inverse transform time_since_start values.
        
        Args:
            times: Log-transformed time values (or original if no transform)
        
        Returns:
            original_times: Original scale time values (cumulative blocks from start)
        
        Example:
            >>> log_times = [0.0, 2.3979, 3.5264, 3.8286]
            >>> original = transformer.inverse_times(log_times)
            >>> print(original)  # [0.0, 10.0, 33.0, 45.0]
        """
        if not self.use_log_transform:
            return np.array(times)
        
        times = np.array(times)
        
        # Apply inverse of log1p: expm1(x) = exp(x) - 1
        original_times = np.expm1(times)
        
        # Round to nearest integer since blocks are discrete
        original_times = np.round(original_times)
        
        # Ensure non-negative and monotonically increasing
        original_times = np.maximum(original_times, 0)
        
        # Fix any non-monotonic issues (shouldn't happen with proper data)
        for i in range(1, len(original_times)):
            if original_times[i] <= original_times[i-1]:
                original_times[i] = original_times[i-1] + 1
        
        return original_times
    
    def inverse_sequence(self, seq: Dict) -> Dict:
        """
        Inverse transform an entire sequence dictionary.
        
        Args:
            seq: Transformed sequence dictionary with keys:
                 - 'time_since_last_event': list of log-transformed intervals
                 - 'time_since_start': list of log-transformed cumulative times
                 - 'type_event': list of event types (unchanged)
                 - Other keys are preserved as-is
        
        Returns:
            original_seq: Sequence in original scale (blocks)
        
        Example:
            >>> transformed_seq = {
            ...     'seq_idx': 0,
            ...     'seq_len': 3,
            ...     'time_since_last_event': [0.0, 2.3979, 1.0986],
            ...     'time_since_start': [0.0, 2.3979, 3.5264],
            ...     'type_event': [0, 5, 12]
            ... }
            >>> original = transformer.inverse_sequence(transformed_seq)
        """
        # Create a copy to avoid modifying the input
        original_seq = seq.copy()
        
        if self.use_log_transform:
            # Inverse transform intervals
            if 'time_since_last_event' in seq:
                intervals = seq['time_since_last_event']
                original_seq['time_since_last_event'] = self.inverse_intervals(intervals).tolist()
            
            # Inverse transform cumulative times
            if 'time_since_start' in seq:
                times = seq['time_since_start']
                original_seq['time_since_start'] = self.inverse_times(times).tolist()
        
        return original_seq
    
    def inverse_batch(self, sequences: List[Dict]) -> List[Dict]:
        """
        Inverse transform a batch of sequences.
        
        Args:
            sequences: List of transformed sequence dictionaries
        
        Returns:
            original_sequences: List of sequences in original scale
        """
        return [self.inverse_sequence(seq) for seq in sequences]
    
    @staticmethod
    def validate_inverse_transform(original: float, transformed: float) -> bool:
        """
        Validate that a single value transforms and inverse-transforms correctly.
        
        Args:
            original: Original value in blocks
            transformed: Log-transformed value
        
        Returns:
            is_valid: True if the transformation is reversible within tolerance
        
        Example:
            >>> is_valid = InverseTransformer.validate_inverse_transform(10, 2.3979)
            >>> print(is_valid)  # True (because log1p(10) ≈ 2.398 and expm1(2.398) ≈ 10)
        """
        expected_transform = np.log1p(original)
        expected_inverse = np.expm1(transformed)
        
        transform_error = abs(expected_transform - transformed)
        inverse_error = abs(expected_inverse - original)
        
        # Allow small numerical errors
        return transform_error < 0.01 and inverse_error < 1.0


def quick_test():
    """
    Quick test to demonstrate the inverse transformation.
    """
    print("="*60)
    print("INVERSE TRANSFORMATION MODULE TEST")
    print("="*60)
    
    # Create transformer
    transformer = InverseTransformer(use_log_transform=True)
    
    # Test data
    test_intervals = [0.0, 2.3979, 1.0986, 3.4012, 0.6931]  # log1p of [0, 10, 2, 29, 1]
    test_times = [0.0, 2.3979, 2.7081, 4.0254, 4.0943]  # log1p of [0, 10, 14, 55, 59]
    
    # Test sequence
    test_seq = {
        'seq_idx': 0,
        'seq_len': 5,
        'time_since_last_event': test_intervals,
        'time_since_start': test_times,
        'type_event': [0, 5, 12, 3, 8]
    }
    
    print("\nTest Sequence (Log-transformed):")
    print(f"  Intervals: {test_intervals}")
    print(f"  Times: {test_times}")
    
    # Inverse transform
    original_intervals = transformer.inverse_intervals(test_intervals)
    original_times = transformer.inverse_times(test_times)
    original_seq = transformer.inverse_sequence(test_seq)
    
    print("\nInverse Transformed (Original Scale):")
    print(f"  Intervals: {original_intervals.tolist()}")
    print(f"  Times: {original_times.tolist()}")
    
    print("\nFull Sequence Transformation:")
    print(f"  Original intervals: {original_seq['time_since_last_event']}")
    print(f"  Original times: {original_seq['time_since_start']}")
    
    # Validation
    print("\nValidation:")
    print(f"  Transform formula: log1p(x) = log(1 + x)")
    print(f"  Inverse formula: expm1(y) = exp(y) - 1")
    print(f"  Example: log1p(10) = {np.log1p(10):.4f}")
    print(f"  Example: expm1(2.3979) = {np.expm1(2.3979):.4f}")
    
    # Test edge cases
    print("\nEdge Cases:")
    edge_cases = [0, 1, 10, 100, 1000, 10000]
    for val in edge_cases:
        log_val = np.log1p(val)
        back = np.expm1(log_val)
        print(f"  {val} → log1p → {log_val:.4f} → expm1 → {back:.1f}")


if __name__ == "__main__":
    quick_test()
    
    print("\n" + "="*60)
    print("USAGE IN YOUR MODEL EVALUATION:")
    print("="*60)
    print("""
# Import the module
from inverse_transform_module import InverseTransformer

# Create transformer
transformer = InverseTransformer(use_log_transform=True)

# After your model makes predictions (in log scale)
predicted_intervals_log = model.predict_intervals(...)  # Your model output
predicted_times_log = model.predict_times(...)          # Your model output

# Convert back to original scale (blocks)
predicted_intervals = transformer.inverse_intervals(predicted_intervals_log)
predicted_times = transformer.inverse_times(predicted_times_log)

# Or transform entire sequences
predicted_sequence_log = model.predict_sequence(...)  # Your model output
original_scale_sequence = transformer.inverse_sequence(predicted_sequence_log)
""")