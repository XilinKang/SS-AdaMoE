# SS-AdaMoE: Spatio-Spectral Adaptive Mixture of Experts

This repository contains the official PyTorch implementation of the paper: **"SS-AdaMoE: Spatio-Spectral Adaptive Mixture of Experts with Global Structural Priors for Graph Node Classification"**.

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Framework: PyTorch](https://img.shields.io/badge/Framework-PyTorch-orange.svg)](https://pytorch.org/)

## 🚀 Overview

**SS-AdaMoE** is a novel Graph Neural Network (GNN) framework that addresses the limitations of traditional message-passing mechanisms in heterophilic graphs. It integrates:
- **Dual-Domain Expert System:** Combining spatial aggregators with learnable spectral filters (Jacobi polynomials).
- **Hierarchical Global-Prior Gating:** Using a Linear Graph Transformer to guide expert selection with global context.

## 📂 Project Structure

The project file structure is organized as follows:

```text
MOE_GNN/
├── datasets/          # Directory for storing benchmark datasets (Cora, Citeseer, etc.)
├── logs/              # Directory for saving training logs and experiment results
├── models/            # Implementation of SS-AdaMoE model components and experts
├── scripts/           # Shell scripts for running batch experiments
├── utils/             # Utility functions for metrics, logging, and visualization
├── abl_exp.py         # Script for running ablation studies
├── cuda.py            # GPU/CUDA device selection and management
├── data1.py           # Data loader for specific graph datasets (Type 1)
├── data2.py           # Data loader for specific graph datasets (Type 2)
├── data_utils.py      # Common data preprocessing and utility functions
├── data_utils1.py     # Auxiliary data utilities
├── data_utils2.py     # Auxiliary data utilities
├── requirements.txt   # Python dependencies
└── train.py           # Main entry point for training and evaluation
