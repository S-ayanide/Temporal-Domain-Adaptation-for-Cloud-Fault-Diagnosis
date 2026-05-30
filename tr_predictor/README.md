# Tr-Predictor — Replication

Replication of **"Tr-Predictor: An Ensemble Transfer Learning Model for Small-Sample Cloud Workload Prediction"** (Liu et al., Entropy 2022, 24, 742).

## Files

| File | Purpose |
|------|---------|
| `similarity.py` | TWED + Transfer Entropy (Copula Entropy) for source domain selection |
| `lstm_model.py` | Weighted LSTM weak learner (sample_weight support) |
| `tr_adaboost.py` | Two-stage TrAdaBoost.R2-LSTM (Algorithm 2) |
| `preprocess.py` | GC19 (.npy) + AC18 (raw CSV) → small-sample rolling windows |
| `metrics.py` | MSE, MAE, MAPE, R² |
| `run.py` | CLI entry point |

## Setup

```bash
pip install -r requirements.txt
```

## Data

Uses the preprocessed `.npy` files from the Rossi replication (`rossi_replication/data/`):

```
GC19_a.npy … GC19_h.npy   # (T, 2) float32: [cpu_fraction, mem_fraction]
AC18.npy                   # same format
```

Optionally, per-machine AC18 traces from raw CSV can be added via `--raw_dir`.

## Run

```bash
# All datasets as targets
python run.py --data_dir ../rossi_replication/data --results_dir results/

# Single target
python run.py --target GC19_a --data_dir ../rossi_replication/data

# With per-machine AC18
python run.py --data_dir ../rossi_replication/data \
              --raw_dir ~/research/data/raw \
              --results_dir results/

# GPU
python run.py --data_dir ../rossi_replication/data --gpu 0
```

## Key implementation details

| Component | Implementation |
|-----------|---------------|
| Source selection | Combined rank: TWED (lower = better) + TE (higher = better) |
| TWED | DP recurrence (Eq. 1), λ=0.5, ν=0.001 |
| Transfer Entropy | Copula entropy via rank-normalised 2/3-D histograms |
| Weak learner | LSTM(64) → Dense(32) → output, trained with sample_weight |
| TrAdaBoost rounds | T=20 (10 stage-1, 10 stage-2) |
| Stage 1 | Freeze source weights; update target weights via AdaBoost.R2 |
| Stage 2 | Freeze target weights; decay source by β_s = 1/(1+√(2·ln(n)/T)) |
| Final ensemble | Weighted average of stage-2 hypotheses, log(1/β_t) weights |
| Small-sample target | First 72 time steps (6h @ 5-min bins) |

## Baselines included

1. **No-transfer**: LSTM trained only on target data
2. **All-source** (ZS): LSTM trained on pooled source data only
3. **Tr-Predictor**: Two-stage TrAdaBoost.R2-LSTM (this paper)
