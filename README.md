# Human Activity Recognition — Data Mining Assignment 3 (Spring 2026)

**Public Leaderboard Score: 0.7964 (F1 macro)**
<img width="854" height="56" alt="image" src="https://github.com/user-attachments/assets/3ee074a1-1c13-42be-87e3-672188ea320f" />

**Kaggle Display Name:** 111550203

## Overview

Wrist-worn accelerometer readings from 100 users, aggregated into 1-second intervals over 5-minute windows. The task is to classify each window into one of 6 activity labels using the mean and standard deviation of tri-axial acceleration (mean_x, mean_y, mean_z, std_x, std_y, std_z).

The key challenge is a 33× class imbalance combined with a hard user split — train and test sets contain completely different users, so the model must generalise to unseen individuals.

## Approach

An ensemble of three models:
- **LightGBM** — gradient boosting on ~300 hand-engineered features
- **XGBoost** — gradient boosting with GPU acceleration
- **HARNet** — custom 1D ResNet-style CNN operating on raw 300-timestep sequences

Final predictions are a weighted average of out-of-fold probability estimates, with weights optimised via Nelder-Mead (LGB=0.061, XGB=0.478, CNN=0.462).

## Key Design Choices

- **GroupKFold by user** (5 folds) — prevents user leakage into validation
- **Balanced class weights** — handles 33× imbalance for all three models
- **Focal Loss (γ=2)** + label smoothing for the CNN
- **Gravity separation** — low-pass rolling mean extracts static orientation; pitch/roll angles computed from gravity vector (critical for Class 4)
- **FFT band-power ratios** — captures dominant frequency differences between classes
- **Segment-level features** — 6 × 50s windows capture temporal drift within each file

## Repo Structure

```
├── preprocessing.ipynb              # Data exploration and preliminary analysis
├── asg3-complete-79_9_best.ipynb    # Full training pipeline and submission
└── README.md
```

## How to Reproduce

1. Upload both notebooks to Kaggle
2. Attach the competition dataset
3. Run `preprocessing.ipynb` first (CPU is fine)
4. Run `asg3-complete-79_9_best.ipynb` with **GPU enabled**
5. Download `submission.csv` from the output

> All random seeds are fixed at 42. Directory paths may need adjusting depending on your Kaggle dataset mount path.

## Results

| Model | OOF Macro-F1 | Class 0 | Class 1 | Class 2 | Class 3 | Class 4 | Class 5 |
|-------|-------------|---------|---------|---------|---------|---------|---------|
| LightGBM | 0.7069 | 0.9616 | 0.8819 | 0.2623 | 0.6740 | 0.8321 | 0.6295 |
| XGBoost | 0.7137 | 0.9639 | 0.8963 | 0.2594 | 0.6814 | 0.8222 | 0.6591 |
| CNN (HARNet) | 0.7071 | 0.9568 | 0.8861 | 0.2785 | 0.6796 | 0.7472 | 0.6944 |
| **Ensemble** | **0.7332** | **0.9641** | **0.9036** | **0.2871** | **0.7145** | **0.8162** | **0.7135** |

Public leaderboard: **0.7964**

## Environment

- Python 3.10
- PyTorch (GPU)
- LightGBM, XGBoost
- scikit-learn, scipy, numpy, pandas

## Course

Data Mining, Spring 2026 — NYCU
