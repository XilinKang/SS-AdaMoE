# SS-AdaMoE: Spatio-Spectral Adaptive Mixture of Experts

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Framework: PyTorch](https://img.shields.io/badge/Framework-PyTorch-orange.svg)](https://pytorch.org/)

This repository contains the official PyTorch implementation of the paper:  
**"SS-AdaMoE: Spatio-Spectral Adaptive Mixture of Experts with Global Structural Priors for Graph Node Classification"**.

## 🚀 Overview

**SS-AdaMoE** is a novel Graph Neural Network (GNN) framework designed to tackle heterophily and over-smoothing. It introduces:

- **Dual-Domain Expert System:** Integrating spatial aggregators with learnable spectral filters (implemented via Jacobi Polynomials in `models/Spectral_2.py`).
- **Adaptive Routing:** A gating mechanism to dynamically select optimal experts for each node.

## 📂 Project Structure

The project is organized as follows:

```text
MOE_GNN/
├── datasets/             # Directory for benchmark datasets (Cora, Citeseer, etc.)
├── logs/                 # Training logs and saved model checkpoints
├── models/               # Core model definitions
│   ├── __init__.py
│   ├── spatial_experts.py # Implementation of spatial GNN experts
│   └── Spectral_2.py      # Implementation of Spectral Filters (Jacobi Poly)
├── scripts/              # Shell scripts for batch experiments
├── utils/                # Utility functions (metrics, visualization)
├── abl_exp.py            # Main script for running Ablation Studies
├── train.py              # Main entry point for training
├── cuda.py               # GPU device management
├── data_utils.py         # Data preprocessing and loading utilities
└── requirements.txt      # Python dependencies
```

## 🛠️ Installation

1. **Clone the repository:**

   Bash

   ```
   git clone [https://github.com/BenhanZhao/Sparse-MoE-SAM.git](https://github.com/BenhanZhao/Sparse-MoE-SAM.git)
   cd MOE_GNN
   ```

2. **Install dependencies:**

   Bash

   ```
   pip install -r requirements.txt
   ```

## 🏃‍♂️ Usage

### 1. Training

To train the SS-AdaMoE model on a standard dataset (e.g., Cora), run the `train.py` script:

Bash

```
# Basic training command
python train.py --dataset cora --epochs 200 --lr 0.001
```

### 2. Ablation Studies

To reproduce the ablation results (e.g., verifying the effectiveness of spectral components), use the `abl_exp.py` script:

Bash

```
# Run ablation experiment
python abl_exp.py --exp_type no_spectral
```

*(Note: Please refer to the code in `abl_exp.py` for all supported experiment types.)*

## ⚙️ Configuration

Key hyperparameters are configured in `train.py` (via `argparse`). You can adjust them via command line:

| **Argument** | **Default** | **Description**                                    |
| ------------ | ----------- | -------------------------------------------------- |
| `--dataset`  | `cora`      | Name of the graph dataset                          |
| `--hidden`   | `64`        | Hidden dimension size                              |
| `--dropout`  | `0.5`       | Dropout rate                                       |
| `--k`        | `3`         | Order of Jacobi Polynomials (for Spectral Filters) |
| `--experts`  | `4`         | Number of experts in the MoE layer                 |

## 🤝 Citation

If you find this code useful for your research, please cite our paper:

Code snippet

```
@article{kang2026ssadamoe,
  title={SS-AdaMoE: Spatio-Spectral Adaptive Mixture of Experts with Global Structural Priors for Graph Node Classification},
  author={Kang, Xilin and Yu, Tianyue and Wang, LeTao and Guo, Yutong and Zhang, Fengjun},
  journal={Entropy},
  year={2026}
}
```

