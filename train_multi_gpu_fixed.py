#!/usr/bin/env python
"""
Multi-GPU training script for EasyTPP with proper GPU selection and memory optimization.
IMPORTANT: This script sets CUDA_VISIBLE_DEVICES before importing torch.
"""

import argparse
import os
import sys

def parse_args():
    """Parse command line arguments before any torch imports."""
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--config_dir', type=str, required=False, default='config/experiment_config_multi_gpu.yaml',
                        help='Dir of configuration yaml to train and evaluate the model.')
    
    parser.add_argument('--experiment_id', type=str, required=False, default='NHP_train_multi_gpu',
                        help='Experiment id in the config file.')
    
    parser.add_argument('--use_multi_gpu', action='store_true', default=False,
                        help='Use DataParallel for multi-GPU training')
    
    parser.add_argument('--use_distributed', action='store_true', default=False,
                        help='Use DistributedDataParallel for distributed training')
    
    parser.add_argument('--gpus', type=str, default='0,1,2,3',
                        help='GPU IDs to use, separated by comma (e.g., "2,3")')
    
    parser.add_argument('--batch_size_per_gpu', type=int, default=16,
                        help='Batch size per GPU (total batch size = batch_size_per_gpu * num_gpus)')
    
    parser.add_argument('--gradient_accumulation_steps', type=int, default=2,
                        help='Number of gradient accumulation steps')
    
    parser.add_argument('--mixed_precision', action='store_true', default=False,
                        help='Use mixed precision training (fp16)')
    
    parser.add_argument('--debug', action='store_true', default=False,
                        help='Enable debug mode with memory tracking')
    
    parser.add_argument('--use_hybrid_loss', action='store_true', default=False,
                        help='Enable hybrid loss (NLL + RMSE with uncertainty weighting)')

    parser.add_argument('--use_nll_uncertainty', action='store_true', default=False,
                        help='Enable NLL uncertainty weighting (type/time split) WITHOUT MSE loss')

    parser.add_argument('--use_event_only_loss', action='store_true', default=False,
                        help='Enable event-only NLL (no non-event term) + MSE with uncertainty weighting')

    parser.add_argument('--use_simple_hybrid', action='store_true', default=False,
                        help='Enable simple NLL + MSE without uncertainty weighting')

    parser.add_argument('--monitor_hybrid_loss', action='store_true', default=False,
                        help='Monitor and log hybrid loss components during training')

    parser.add_argument('--init_log_var_nll', type=float, default=0.0,
                        help='Initial log variance for NLL loss (for hybrid loss)')

    parser.add_argument('--init_log_var_mse', type=float, default=0.0,
                        help='Initial log variance for MSE loss (for hybrid loss)')

    parser.add_argument('--init_log_var_type', type=float, default=0.0,
                        help='Initial log variance for type component (for NLL uncertainty)')

    parser.add_argument('--init_log_var_time', type=float, default=0.0,
                        help='Initial log variance for time component (for NLL uncertainty)')
    
    parser.add_argument('--seed', type=int, default=None,
                        help='Random seed for reproducibility (default: None for random)')

    return parser.parse_args()


def main():
    # Parse args BEFORE any torch imports
    args = parse_args()
    
    # Setup environment variables BEFORE importing torch
    print(f"Setting up environment for GPUs: {args.gpus}")
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpus
    os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True,max_split_size_mb:128'
    
    # Now import torch and other modules
    import torch
    import torch.nn as nn
    import torch.distributed as dist
    from torch.nn.parallel import DataParallel, DistributedDataParallel
    from easy_tpp.config_factory import Config
    from easy_tpp.runner import Runner
    from easy_tpp.utils import set_seed
    
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    print(f"Number of GPUs visible to PyTorch: {torch.cuda.device_count()}")
    
    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            print(f"GPU {i}: {torch.cuda.get_device_name(i)}")
            print(f"  Memory: {torch.cuda.get_device_properties(i).total_memory / 1e9:.2f} GB")
    
    # After setting CUDA_VISIBLE_DEVICES, GPU indices are remapped
    # If you set CUDA_VISIBLE_DEVICES=2,3 then PyTorch sees them as GPU 0,1
    num_gpus = torch.cuda.device_count()
    remapped_gpu_ids = list(range(num_gpus))
    
    # Load configuration
    config = Config.build_from_yaml_file(args.config_dir, experiment_id=args.experiment_id)
    
    # Override batch size from command line
    config.trainer_config.batch_size = args.batch_size_per_gpu
    
    # Set seed in config if specified
    if args.seed is not None:
        config.trainer_config.seed = args.seed
        print(f"\n=== Setting Random Seed: {args.seed} ===")
        set_seed(args.seed)
        print(f"Random seed set for reproducibility")
    else:
        # Use default seed from config or set a default
        if not hasattr(config.trainer_config, 'seed'):
            config.trainer_config.seed = 2023  # Default seed
        set_seed(config.trainer_config.seed)
        print(f"Using seed from config: {config.trainer_config.seed}")
    
    # Ensure model_specs exists
    if not hasattr(config.model_config, 'model_specs'):
        config.model_config.model_specs = {}

    # Configure NLL uncertainty weighting (type/time split) if requested
    if args.use_nll_uncertainty:
        print("\n=== NLL Uncertainty Weighting Configuration ===")
        print(f"Enabling NLL uncertainty weighting (type/time split)")
        print(f"MSE loss will NOT be added")

        # Enable NLL uncertainty mode
        config.model_config.model_specs['use_nll_uncertainty'] = True
        config.model_config.model_specs['init_log_var_type'] = args.init_log_var_type
        config.model_config.model_specs['init_log_var_time'] = args.init_log_var_time

        print(f"Initial log_var_type: {args.init_log_var_type}")
        print(f"Initial log_var_time: {args.init_log_var_time}")
        print(f"Using uncertainty weighting between type and time components within NLL")

    # Configure event-only loss (event NLL + MSE) if requested
    elif args.use_event_only_loss:
        print("\n=== Event-Only Loss Configuration ===")
        print(f"Enabling event-only NLL (no non-event term) + MSE with uncertainty weighting")

        # Enable event-only loss mode
        config.model_config.model_specs['use_event_only_loss'] = True
        config.model_config.model_specs['init_log_var_nll'] = args.init_log_var_nll
        config.model_config.model_specs['init_log_var_mse'] = args.init_log_var_mse

        print(f"Initial log_var_nll: {args.init_log_var_nll}")
        print(f"Initial log_var_mse: {args.init_log_var_mse}")
        print(f"Using uncertainty weighting between event-only NLL and MSE")
        print(f"Non-event term (integral) is excluded from NLL")

    # Configure simple hybrid (NLL + MSE without weighting) if requested
    elif args.use_simple_hybrid:
        print("\n=== Simple Hybrid Loss Configuration ===")
        print(f"Enabling simple hybrid loss (NLL + MSE without uncertainty weighting)")

        # Enable simple hybrid mode
        config.model_config.model_specs['use_simple_hybrid'] = True

        print(f"Using direct addition: Loss = NLL + MSE")
        print(f"No uncertainty weighting applied")

    # Configure hybrid loss if requested
    elif args.use_hybrid_loss:
        print("\n=== Hybrid Loss Configuration ===")
        print(f"Enabling hybrid loss (NLL + RMSE with uncertainty weighting)")

        # Enable hybrid loss
        config.model_config.model_specs['use_hybrid_loss'] = True
        config.model_config.model_specs['init_log_var_nll'] = args.init_log_var_nll
        config.model_config.model_specs['init_log_var_mse'] = args.init_log_var_mse

        print(f"Initial log_var_nll: {args.init_log_var_nll}")
        print(f"Initial log_var_mse: {args.init_log_var_mse}")
        print(f"Using uncertainty weighting between NLL and MSE")
        print(f"Using direct MSE loss (prediction vs label)")
    
    # Setup for different training modes
    if args.use_distributed:
        # Check if running with torchrun
        if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
            rank = int(os.environ["RANK"])
            world_size = int(os.environ["WORLD_SIZE"])
            local_rank = int(os.environ['LOCAL_RANK'])
            
            torch.cuda.set_device(local_rank)
            dist.init_process_group(backend='nccl', init_method='env://')
            
            print(f"Distributed training: Rank {rank}/{world_size}, Local GPU {local_rank}")
            config.trainer_config.gpu = local_rank
        else:
            print("Warning: Distributed mode requested but not launched with torchrun")
            print("Falling back to DataParallel")
            args.use_multi_gpu = True
            args.use_distributed = False
    
    elif args.use_multi_gpu and num_gpus > 1:
        print(f"Using DataParallel on {num_gpus} GPUs")
        config.trainer_config.gpu = 0  # Primary GPU
        # Scale batch size for multiple GPUs
        config.trainer_config.batch_size = args.batch_size_per_gpu * num_gpus
        
    else:
        print(f"Using single GPU training on GPU 0")
        config.trainer_config.gpu = 0  # After CUDA_VISIBLE_DEVICES, this is the first available GPU
    
    # Add memory optimization settings
    if hasattr(config.trainer_config, 'gradient_accumulation_steps'):
        config.trainer_config.gradient_accumulation_steps = args.gradient_accumulation_steps
    
    # Debug mode
    if args.debug:
        print("\n=== Debug Mode: Memory Tracking Enabled ===")
        torch.cuda.empty_cache()
        print(f"Initial memory allocated: {torch.cuda.memory_allocated(0)/1e9:.2f} GB")
        print(f"Initial memory reserved: {torch.cuda.memory_reserved(0)/1e9:.2f} GB")
    
    # Build model runner
    print("\nBuilding model runner...")
    model_runner = Runner.build_from_config(config)
    
    if args.debug:
        print(f"After model creation - Allocated: {torch.cuda.memory_allocated(0)/1e9:.2f} GB")
        print(f"After model creation - Reserved: {torch.cuda.memory_reserved(0)/1e9:.2f} GB")
    
    # Wrap model for multi-GPU if needed
    if args.use_distributed and 'RANK' in os.environ:
        print("Wrapping model with DistributedDataParallel")
        model_runner.model_wrapper.model = DistributedDataParallel(
            model_runner.model_wrapper.model,
            device_ids=[local_rank],
            output_device=local_rank,
            find_unused_parameters=True
        )
    elif args.use_multi_gpu and num_gpus > 1:
        print(f"Wrapping model with DataParallel across GPUs: {remapped_gpu_ids}")
        model_runner.model_wrapper.model = DataParallel(
            model_runner.model_wrapper.model,
            device_ids=remapped_gpu_ids
        )
    
    # Setup mixed precision if requested
    if args.mixed_precision:
        print("Enabling mixed precision training")
        scaler = torch.cuda.amp.GradScaler()
        model_runner.scaler = scaler
    
    # Memory debugging function
    def print_memory_usage(step=""):
        if args.debug:
            for i in range(torch.cuda.device_count()):
                allocated = torch.cuda.memory_allocated(i) / 1e9
                reserved = torch.cuda.memory_reserved(i) / 1e9
                print(f"[{step}] GPU {i}: Allocated: {allocated:.2f} GB, Reserved: {reserved:.2f} GB")
    
    # Setup hybrid loss monitoring if requested
    if args.monitor_hybrid_loss and args.use_hybrid_loss:
        print("\n=== Hybrid Loss Monitoring Enabled ===")
        print("  - Tracking uncertainty weights between NLL and MSE")
        print("  - MSE uses direct time differences (pred vs label)")
        print("  - Automatic balancing via uncertainty weighting")
        
        # Import monitoring utilities
        import json
        from collections import defaultdict
        
        # Create a monitoring dictionary
        hybrid_loss_history = defaultdict(list)
        
        # Override the run method to add monitoring
        original_run = model_runner.run
        
        def monitored_run():
            """Run training with hybrid loss monitoring."""
            # Store reference to model for monitoring
            model = model_runner.model_wrapper.model
            if isinstance(model, (DataParallel, DistributedDataParallel)):
                base_model = model.module
            else:
                base_model = model
            
            # Run original training
            result = original_run()
            
            # Save monitoring history if collected
            if hasattr(base_model, '_last_nll_loss'):
                history_file = f"{args.experiment_id}_hybrid_loss_history.json"
                print(f"\nSaving hybrid loss history to {history_file}")
                
                # Collect final statistics
                final_stats = {
                    'final_nll_weight': base_model._last_nll_weight if hasattr(base_model, '_last_nll_weight') else None,
                    'final_mse_weight': base_model._last_mse_weight if hasattr(base_model, '_last_mse_weight') else None,
                    'final_nll_loss': base_model._last_nll_loss if hasattr(base_model, '_last_nll_loss') else None,
                    'final_mse_loss': base_model._last_mse_loss if hasattr(base_model, '_last_mse_loss') else None,
                }
                
                # Calculate learned uncertainties
                if final_stats['final_nll_weight'] is not None:
                    import numpy as np
                    # Correct formula: weight = 0.5 / σ², so σ = √(0.5 / weight)
                    sigma_nll = np.sqrt(0.5 / final_stats['final_nll_weight'])
                    sigma_mse = np.sqrt(0.5 / final_stats['final_mse_weight'])
                    final_stats['final_sigma_nll'] = sigma_nll
                    final_stats['final_sigma_mse'] = sigma_mse
                    
                    print(f"\nFinal Hybrid Loss Statistics:")
                    print(f"  NLL weight: {final_stats['final_nll_weight']:.4f} (σ={sigma_nll:.4f})")
                    print(f"  MSE weight: {final_stats['final_mse_weight']:.4f} (σ={sigma_mse:.4f})")
                    print(f"  Weight ratio (NLL/MSE): {final_stats['final_nll_weight']/final_stats['final_mse_weight']:.2f}x")
                    print(f"  NLL loss: {final_stats['final_nll_loss']:.4f}")
                    print(f"  MSE loss: {final_stats['final_mse_loss']:.4f}")
                    
                    # Add interpretation
                    if final_stats['final_nll_weight'] > final_stats['final_mse_weight']:
                        print(f"  → Model prioritizes NLL (accuracy/loglike) over MSE (time prediction)")
                    else:
                        print(f"  → Model prioritizes MSE (time prediction) over NLL (accuracy/loglike)")
                
                with open(history_file, 'w') as f:
                    json.dump(final_stats, f, indent=2)
            
            return result
        
        model_runner.run = monitored_run
    
    # Run training with memory monitoring
    try:
        print("\nStarting training...")
        print(f"Batch size per GPU: {args.batch_size_per_gpu}")
        print(f"Gradient accumulation steps: {args.gradient_accumulation_steps}")
        print(f"Effective batch size: {args.batch_size_per_gpu * num_gpus * args.gradient_accumulation_steps}")
        
        if args.use_hybrid_loss:
            print(f"Hybrid loss: ENABLED")
            if args.monitor_hybrid_loss:
                print(f"Hybrid loss monitoring: ENABLED")
        
        model_runner.run()
        
    except torch.cuda.OutOfMemoryError as e:
        print(f"\n!!! CUDA Out of Memory Error !!!")
        print(f"Error details: {e}")
        print_memory_usage("ERROR")
        print("\nSuggestions:")
        print("1. Reduce batch_size_per_gpu (current: {})".format(args.batch_size_per_gpu))
        print("2. Increase gradient_accumulation_steps (current: {})".format(args.gradient_accumulation_steps))
        print("3. Use mixed precision training with --mixed_precision")
        print("4. Reduce model size in config file")
        sys.exit(1)
        
    finally:
        if args.use_distributed and 'RANK' in os.environ:
            dist.destroy_process_group()
        
        # Clear cache
        torch.cuda.empty_cache()
        print_memory_usage("Final")


if __name__ == '__main__':
    main()