# Cross-Cloud Workload Prediction via Domain Adaptation

This repository contains the implementation for my MSc dissertation at Trinity College Dublin on cross-cloud container workload prediction using domain adaptation. Seven transfer learning methods from the literature are replicated and evaluated within a single experimental framework, and MC-CWPDDA is introduced as a novel contribution that combines adversarial and contrastive alignment.

## Papers Implemented

| Method | Paper | File(s) |
|--------|-------|---------|
| CWPDDA | Wang et al., "Container Workload Prediction using Deep Domain Adaptation", Euro-Par 2025 | `cwpdda.py` |
| MCTL | Zuo et al., "Mixed Contrastive Transfer Learning for Few-Shot Workload Prediction", Computing 2025 | `mctl.py` |
| DeepJDOT | Damodaran et al., "DeepJDOT: Deep Joint Distribution Optimal Transport for Domain Adaptation", ECCV 2018 | `deepjdot/` |
| Tr-Predictor | Liu et al., "Tr-Predictor: An Ensemble Transfer Learning Model for Cloud Workload Prediction", Computing 2022 | `tr_predictor/` |
| N-BEATS | Oreshkin et al., "N-BEATS: Neural Basis Expansion Analysis for Interpretable Time Series Forecasting", ICLR 2020 | `nbeats.py` |
| MC-CWPDDA | **Novel contribution** — combines CWPDDA's adversarial GRL with MCTL's contrastive alignment in a three-stage curriculum | `mc_cwpdda.py` |

Non-transfer baselines (ARIMA, LSTM, GRU, CNN-LSTM) are implemented in `baselines.py`.

## Datasets

> **Data is not included in this repository.** Download instructions are below.

### Google Cluster Trace 2019

Available from Google Cloud Storage. Each cell (a–h) has multiple shards:

```
https://storage.googleapis.com/clusterdata_2019_a/instance_usage-000000000000.json.gz
https://storage.googleapis.com/clusterdata_2019_a/instance_usage-000000000001.json.gz
...
```

Replace `a` with `b`–`h` for other cells. Shards are numbered sequentially; stop downloading when you get a 404.

Use the provided script to download automatically:
```bash
bash download_data.sh                  # all cells + Alibaba
bash download_data.sh --cells a b      # only cells a and b
bash download_data.sh --max-shards 5   # first 5 shards per cell
bash download_data.sh --google         # Google only
```

Expected path: `data/raw/google/cell_a/instance_usage-*.json.gz`

### Alibaba Cluster Trace 2018

```
http://aliopentrace.oss-cn-beijing.aliyuncs.com/v2018Traces/machine_usage.tar.gz
```

The download script handles this automatically. Expected path: `data/raw/alibaba/machine_usage.csv`

**File format expected by the loader (in order of preference):**
1. `machine_usage.csv` — CPU per physical machine (primary)
2. `batch_instance.csv` — CPU per batch job instance
3. `container_usage.csv` — CPU per container

---

## Directory Structure

```
updated_research/
├── run_full_experiment.py          Full regression + classification pipeline (both directions)
├── run.py                          Single-method entry point (CWPDDA / MCTL / MC-CWPDDA / N-BEATS)
├── run_gpu.sh                      GPU server launch script (runs both transfer directions)
│
├── cwpdda.py                       CWPDDA model (Wang et al., Euro-Par 2025)
├── mc_cwpdda.py                    MC-CWPDDA — novel contribution
├── mctl.py                         MCTL model (Zuo et al., Computing 2025)
├── nbeats.py                       N-BEATS zero-shot baseline (Oreshkin et al., ICLR 2020)
├── baselines.py                    ARIMA, LSTM, GRU, CNN-LSTM baselines
│
├── deepjdot/                       DeepJDOT (Damodaran et al., ECCV 2018)
│   ├── model.py                    Optimal transport domain adaptation model
│   └── train.py                    DeepJDOT training loop
│
├── tr_predictor/                   Tr-Predictor (Liu et al., Computing 2022)
│   ├── tr_adaboost.py              TrAdaBoost.R2 instance reweighting
│   ├── similarity.py               TWED + Transfer Entropy source selection
│   ├── lstm_model.py               LSTM base learner
│   ├── preprocess.py               Tr-Predictor data preparation
│   └── metrics.py                  Evaluation metrics
│
├── train.py                        Training loops for CWPDDA, MCTL, MC-CWPDDA, N-BEATS
├── evaluate.py                     Regression + classification evaluation metrics
├── data_loader.py                  Load Google and Alibaba time series
├── preprocess.py                   Windowing, train/val/test splits, DTW pairing
│
├── generate_from_gpu_results.py    Generate dissertation figures from experiment results
├── generate_presentation_figures.py Generate presentation-quality figures
│
├── download_data.sh                Automated data download
└── requirements.txt                Python dependencies
```

---

## Installation

```bash
pip install -r requirements.txt
```

---

## Running Experiments

### Full Bidirectional Experiment (Dissertation Results)

This is the pipeline that produces the dissertation results. It runs all methods (ARIMA, LSTM, GRU, CNN-LSTM, N-BEATS, DeepJDOT, CWPDDA, MC-CWPDDA, MCTL) in both transfer directions:

```bash
# Google → Alibaba (primary transfer direction)
python run_full_experiment.py --direction google_to_alibaba \
    --google data/raw/google --alibaba data/raw/alibaba \
    --device cuda --out results/g2a

# Alibaba → Google (reverse transfer direction)
python run_full_experiment.py --direction alibaba_to_google \
    --google data/raw/google --alibaba data/raw/alibaba \
    --device cuda --out results/a2g

# Or run both directions via the GPU script
bash run_gpu.sh
```

### Single-Method Runs

```bash
# CWPDDA (Wang et al.)
python run.py --paper cwpdda \
    --google data/raw/google --alibaba data/raw/alibaba --device cuda

# MC-CWPDDA (novel contribution)
python run.py --paper mc_cwpdda \
    --google data/raw/google --alibaba data/raw/alibaba --device cuda

# MCTL (Zuo et al.)
python run.py --paper mctl \
    --google data/raw/google --alibaba data/raw/alibaba --device cuda

# N-BEATS (zero-shot)
python run.py --paper nbeats \
    --google data/raw/google --alibaba data/raw/alibaba --device cuda

# Quick smoke test (CPU, few epochs, small dataset)
python run.py --paper cwpdda --quick
```

**Save/load preprocessed cache to skip slow data loading on repeat runs:**
```bash
python run.py --paper cwpdda ... --save-cache results/preprocessed.npz
python run.py --paper cwpdda ... --load-cache results/preprocessed.npz
```

### GPU Server

```bash
# tmux-based background run
tmux new -s exp
bash run_gpu.sh
# Ctrl+B D to detach; tmux attach -t exp to reconnect
```

---

## Expected Results

**Workload Prediction — GC2019 → AC2018 (fully supervised, rescaled to 0–100%):**

| Method | Target Data | MAE | MAPE (%) | RMSE |
|--------|-------------|-----|----------|------|
| ARIMA (in-domain) | Full | 6.05 | 20.05 | 6.54 |
| LSTM (no transfer) | None | 4.96 | 16.46 | 4.92 |
| N-BEATS (zero-shot) | None | 4.91 | 16.38 | 4.87 |
| DeepJDOT | None | 4.84 | 16.12 | 4.81 |
| Tr-Predictor | ~23 windows | 4.63 | 15.71 | 4.69 |
| MCTL | ~100 windows | 3.10 | 10.83 | 3.33 |
| CWPDDA | Full | 2.42 | 8.66 | 2.59 |
| **MC-CWPDDA (ours)** | Full | **2.18** | **7.82** | **2.34** |

Results are saved to `results/` as `.json` files.

---

## Troubleshooting

**"No Google series loaded"** — check path and verify gzip integrity:
```bash
python data_loader.py data/raw/google data/raw/alibaba
```

**"No recognised Alibaba CSV found"** — rename the file to `machine_usage.csv`.

**CUDA OOM** — reduce `--batch-size` to 16 or 32.

**MCTL worse than baselines** — usually too few target training windows. Check `results/meta.json` for `tgt_train_windows`. If < 100, set `--max-target-len 200`.

**DTW too slow** — use `--no-dtw`. Results are slightly lower but still competitive.
