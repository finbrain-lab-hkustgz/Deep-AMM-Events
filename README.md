# Towards Event-Aware Forecasting in DeFi: Insights from On-chain Automated Market Maker Protocols

![Framework](docs/assets/images/framework.png)

## Abstract

Automated Market Makers (AMMs), as a core infrastructure of decentralized finance (DeFi), uniquely drive on-chain asset pricing through a deterministic reserve ratio mechanism. Unlike traditional markets, AMM price dynamics is triggered largely by on-chain events (e.g., swap) that change the reserve ratio, rather than by continuous responses to off-chain information. This makes event-level analysis crucial for understanding price formation mechanisms in AMMs. However, existing research generally neglects the micro-structural dynamics at the AMMs level, lacking both a comprehensive dataset covering multiple protocols with fine-grained event classification and an effective framework for event-aware modeling. To fill this gap, we construct a dataset containing 8.9 million on-chain event records from four representative AMMs protocols: Pendle, Uniswap v3, Aave and Morpho, with precise annotations of transaction type and block height timestamps. Furthermore, we propose an Uncertainty Weighted Mean Squared Error (UWM) loss function, which incorporates the block interval regression term into the traditional Temporal Point Process (TPP) objective function by weighting the uncertainty with homoscedasticity. Extensive experiments on eight advanced TPP architectures across four representative DeFi protocols demonstrate that this loss function reduces the time prediction error by an average of 31.17% while maintaining the accuracy of event (transaction) type prediction, establishing a robust benchmark for event-aware prediction in the AMMs ecosystem. This work provides the necessary data foundation and methodological framework for modeling the discreteness and event-driven characteristics of on-chain price discovery.


## Overview

This repository contains the code and preprocessed datasets for our paper on event-aware forecasting in DeFi Automated Market Maker (AMM) protocols. We contribute:

1. **A large-scale on-chain event dataset** containing 8.9 million event records from four representative AMM protocols: **Pendle**, **Uniswap V3**, **Aave**, and **Morpho**, with precise annotations of transaction type and block-height timestamps.

2. **Uncertainty Weighted MSE (UWM) loss function**, which incorporates a block-interval regression term into the standard Temporal Point Process (TPP) objective by weighting NLL and MSE losses via learned homoscedastic uncertainty. UWM reduces the time prediction error by an average of **56.41%** while maintaining event type prediction accuracy.

3. **Comprehensive benchmarks** on eight state-of-the-art TPP architectures (NHP, RMTPP, SAHP, THP, AttNHP, IntensityFree, FullyNN, ODETPP).

## Datasets

### Raw Data (HuggingFace)

The complete raw on-chain event data for all four protocols is available at:

**[https://huggingface.co/datasets/Jackson668/AMM-Events](https://huggingface.co/datasets/Jackson668/AMM-Events)**

| Protocol | Events | Event Types |
|:--|--:|:--|
| Pendle | 2.0M | Mint, Burn, Swap, UpdateImpliedRate |
| Uniswap V3 | 5.7M | Mint, Burn, Swap |
| Aave | 0.9M | Supply, Borrow, Withdraw, Repay |
| Morpho | 0.3M | Supply, Borrow, Withdraw, Repay, Liquidate |

### Preprocessed Data (This Repo)

The `data/` directory contains preprocessed Pendle data ready for TPP model training:

```
data/
  pendle/              # Pendle dataset (31 event combination types)
    train.json
    dev.json
    test.json
    metadata.json
    event_combination_mapping.json
```

## Model List

Built on top of [EasyTPP](https://github.com/ant-research/EasyTemporalPointProcess), we benchmark the following TPP models:

| Model | Publication | Paper |
|:--|:--|:--|
| RMTPP | KDD 2016 | Recurrent Marked Temporal Point Processes |
| NHP | NeurIPS 2017 | The Neural Hawkes Process |
| FullyNN | NeurIPS 2019 | Fully Neural Network based Model for General TPP |
| SAHP | ICML 2020 | Self-Attentive Hawkes Process |
| THP | ICML 2020 | Transformer Hawkes Process |
| IntensityFree | ICLR 2020 | Intensity-Free Learning of Temporal Point Processes |
| ODETPP | ICLR 2021 | Neural Spatio-Temporal Point Processes |
| AttNHP | ICLR 2022 | Transformer Embeddings of Irregularly Spaced Events |

## Installation

```bash
git clone https://github.com/yosen-king/Deep-AMM-Events.git
cd Deep-AMM-Events
pip install -r requirements.txt
```

**Requirements**: Python 3.9+, PyTorch 2.0+

## Usage

### Training

Train a model with UWM loss using multi-GPU:

```bash
CUDA_VISIBLE_DEVICES=3 nohup python train_multi_gpu_fixed.py \
  --config_dir config/experiment_config_memory_optimized_all_models.yaml \
  --experiment_id IntensityFree_train_multi_gpu \
  --gpus 3 \
  --use_multi_gpu \
  --batch_size_per_gpu 32 \
  --use_hybrid_loss \
  --monitor_hybrid_loss \
  --init_log_var_nll 1 \
  --init_log_var_mse 1 \
  --seed 2022 \
  > IntensityFree_mse_hybrid_log_2022.out 2>&1 &
```

Key arguments:
- `--experiment_id`: Model config name in the YAML (e.g., `NHP_train_multi_gpu`, `RMTPP_train_multi_gpu`, etc.)
- `--use_hybrid_loss`: Enable UWM loss (NLL + MSE with uncertainty weighting)
- `--init_log_var_nll` / `--init_log_var_mse`: Initial log-variance for NLL and MSE losses
- `--seed`: Random seed for reproducibility (paper uses seeds {2019, 2020, 2021, 2022, 2023})

### Evaluation

Update the model checkpoint path in `config/experiment_config.yaml`, then run:

```bash
python main.py
```

You can specify a different config and experiment ID:

```bash
python main.py --config_dir config/experiment_config.yaml --experiment_id NHP_gen
```

### OTD Calculation

After generating predictions (saved as `pred.pkl`), compute the Optimal Transport Distance:

```bash
python otd_cal.py
```

## Configuration

Config files are in `config/`:

| File | Purpose |
|:--|:--|
| `experiment_config_memory_optimized_all_models.yaml` | Main training config for all models |
| `experiment_config.yaml` | Evaluation and generation configs |
| `experiment_config_hybrid_loss.yaml` | Hybrid loss (UWM) specific configs |
| `experiment_config_multi_gpu.yaml` | Multi-GPU training configs |
| `experiment_config_memory_optimized.yaml` | Memory-optimized configs |

## Project Structure

```
.
├── config/                  # Experiment configurations
├── data/                    # Preprocessed datasets
├── easy_tpp/                # Core TPP library (based on EasyTPP)
│   ├── config_factory/      # Configuration parsing
│   ├── model/torch_model/   # TPP model implementations
│   ├── preprocess/          # Data loading and preprocessing
│   ├── runner/              # Training and evaluation runners
│   ├── ssm/                 # State Space Model components
│   └── utils/               # Utilities and metrics
├── paper/                   # LaTeX source of the paper
├── main.py                  # Entry point for evaluation
├── train_multi_gpu_fixed.py # Multi-GPU training with UWM loss
└── otd_cal.py               # Optimal Transport Distance calculation
```

## Citation

If you find this work useful, please cite:

```bibtex
@inproceedings{jia2026towards,
  title={Towards Event-Aware Forecasting in DeFi: Insights from On-chain Automated Market Maker Protocols},
  author={Jia, Huaiyu and You, Jiehshun and Luo, Yizhi and Liu, Jingyu and Sun, Shuo},
  booktitle={Proceedings of the 32nd SIGKDD Conference on Knowledge Discovery and Data Mining},
  year={2026}
}
```

## Acknowledgments

This project builds upon [EasyTPP](https://github.com/ant-research/EasyTemporalPointProcess) (ICLR 2024). We thank the EasyTPP team for their open-source toolkit.

## License

This project is licensed under the [MIT License](LICENSE).
