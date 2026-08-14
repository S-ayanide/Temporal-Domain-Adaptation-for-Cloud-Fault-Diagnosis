# Cross-Cloud Workload Prediction via Domain Adaptation

This repository contains the implementation for my dissertation on cross-cloud container workload prediction using domain adaptation. The work extends and evaluates several methods from the literature and introduces MC-CWPDDA as a novel contribution.

## Papers Implemented

| Method | Paper |
|--------|-------|
| CWPDDA | Wang et al., "Container Workload Prediction using Deep Domain Adaptation", Euro-Par 2025 |
| MCTL | Zuo et al., "Mixed Contrastive Transfer Learning for Few-Shot Workload Prediction", Computing 2025 |
| DATL | Fang & Gao, "Domain-Adversarial Transfer Learning for Fault Root Cause Identification", RAIIC 2025 |
| MC-CWPDDA | Novel contribution — multi-cloud extension of CWPDDA |
| N-BEATS | Oreshkin et al., "N-BEATS: Neural basis expansion analysis", ICLR 2020 (zero-shot baseline) |

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
├── cwpdda.py                       CWPDDA model (Wang et al.)
├── mctl.py                         MCTL model (Zuo et al.)
├── mc_cwpdda.py                    MC-CWPDDA — novel contribution
├── nbeats.py                       N-BEATS zero-shot baseline
├── tcn.py                          Temporal Convolutional Network
├── baselines.py                    ARIMA, LSTM, GRU, CNN-LSTM, etc.
├── models.py                       Classification models (DANN, CDAN, FixBi, ToAlign, DATL, TA-DATL)
│
├── train.py                        Training loops for CWPDDA, MCTL, MC-CWPDDA, N-BEATS
├── trainer.py                      Training loops for classification models
├── evaluate.py                     Regression evaluation (MAE, MAPE, RMSE)
│
├── data_loader.py                  Load Google and Alibaba time series
├── preprocess.py                   Windowing, train/val/test splits, DTW
├── alibaba_io.py                   Alibaba raw data parsing
├── google_io.py                    Google raw data parsing
├── prepare_common.py               Shared feature engineering for classification
│
├── run.py                          Main entry point — workload prediction pipeline
├── run_all.py                      Run all classification experiments sequentially
├── run_google_google.py            Within-Google transfer
├── run_google_to_alibaba.py        Google→Alibaba classification pipeline
├── run_full_experiment.py          Full regression + classification for one direction
│
├── 00_prepare_data.py              Classification data preparation (within-Alibaba)
├── 00_prepare_data_google_alibaba.py   Data prep for Google→Alibaba classification
├── 00_prepare_data_google_google.py    Data prep for Google→Google classification
├── 01_train_all_models.py          Train all classification models (Table 1)
├── 02_experiment_label_scarcity.py Label scarcity experiment (Figure 2)
├── 03_experiment_class_imbalance.py Class imbalance robustness (Figure 3)
├── 04_experiment_heterogeneous_nodes.py Node heterogeneity (Figure 4)
├── 05_ablation_study.py            Ablation study (Figure 5)
│
├── generate_domain_figures.py      Domain shift visualisations
├── generate_from_gpu_results.py    Post-process GPU results into figures
├── generate_presentation_figures.py Publication-quality figures
│
├── deepjdot/                       DeepJDOT domain adaptation baseline
├── tr_predictor/                   Tr-Predictor workload prediction baseline
│
├── download_data.sh                Automated data download
├── run_gpu.sh                      GPU server launch script
└── requirements.txt                Python dependencies
```

---

## Installation

```bash
pip install -r requirements.txt
```

---

## Running Experiments

### Workload Prediction (Regression)

This pipeline trains and evaluates CWPDDA, MCTL, MC-CWPDDA, or N-BEATS on the Google→Alibaba transfer task.

```bash
# CWPDDA (Wang et al.) — full run
python run.py --paper cwpdda \
    --google  data/raw/google \
    --alibaba data/raw/alibaba \
    --device  cuda

# MC-CWPDDA — novel contribution
python run.py --paper mc_cwpdda \
    --google  data/raw/google \
    --alibaba data/raw/alibaba \
    --device  cuda

# MCTL (Zuo et al.)
python run.py --paper mctl \
    --google  data/raw/google \
    --alibaba data/raw/alibaba \
    --device  cuda

# N-BEATS zero-shot
python run.py --paper nbeats \
    --google  data/raw/google \
    --alibaba data/raw/alibaba \
    --device  cuda

# Quick smoke test (CPU, few epochs, small dataset)
python run.py --paper cwpdda --quick
```

**Save/load preprocessed cache to skip slow reload on repeat runs:**
```bash
python run.py --paper cwpdda ... --save-cache results/preprocessed.npz
python run.py --paper cwpdda ... --load-cache results/preprocessed.npz
```

**Full bidirectional experiment (regression + classification):**
```bash
python run_full_experiment.py --direction google_to_alibaba \
    --google data/raw/google --alibaba data/raw/alibaba \
    --device cuda --out results/g2a

python run_full_experiment.py --direction alibaba_to_google \
    --google data/raw/google --alibaba data/raw/alibaba \
    --device cuda --out results/a2g
```

### Fault Classification

This pipeline trains DATL, TA-DATL, DANN, CDAN, FixBi, and ToAlign on the fault root-cause classification task.

```bash
# Prepare data (within-Alibaba split)
python 00_prepare_data.py

# Google→Alibaba transfer
python 00_prepare_data_google_alibaba.py
python 01_train_all_models.py --processed-dir data/processed_google_alibaba

# Run all classification experiments end-to-end
python run_all.py

# Or use the convenience wrappers
python run_google_to_alibaba.py
python run_google_google.py
```

### GPU Server

```bash
# tmux-based background run
tmux new -s exp
python run.py --paper mc_cwpdda --google data/raw/google \
    --alibaba data/raw/alibaba --device cuda
# Ctrl+B D to detach; tmux attach -t exp to reconnect

# Or use the provided GPU script
bash run_gpu.sh
```

---

## Expected Results

**Workload Prediction — Google→Alibaba (Table 3 of CWPDDA paper):**

```
Method        MAE       MAPE%     RMSE
ARIMA         1.260e-3  4.39%     1.742e-3
LSTM          2.363e-3  7.27%     2.726e-3
GRU           1.456e-3  3.72%     1.923e-3
N-BEATS       8.938e-4  3.05%     1.128e-3
CWPDDA        2.418e0   8.66%     2.586e0
MC-CWPDDA     < CWPDDA  < CWPDDA  < CWPDDA   (novel contribution)
```

Results are saved to `results/` as `.json` and `.txt` table files.

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
