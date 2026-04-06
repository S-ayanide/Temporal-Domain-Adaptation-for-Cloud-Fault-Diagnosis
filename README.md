# TA-DATL: Temporal Attention Domain-Adversarial Transfer Learning

**Dissertation research extending:** *"Domain Adversarial Transfer Learning for Fault Root Cause Identification"*

---

## Research Contribution

The replicated paper (DATL) treats each machine reading as an **independent point** and uses a flat MLP to extract features.  
This work argues that resource metrics are **temporal signals** — a CPU spike that lasts 30 seconds is very different from one lasting 5 minutes — and that temporal structure is crucial for accurate fault root cause identification.

### Novel Model: TA-DATL

| Component | Replicated DATL | TA-DATL (This Work) |
|-----------|----------------|---------------------|
| Feature extractor | Flat MLP | Multi-scale Gated Conv + Grouped Query Attention |
| Input | Single timestep `(F,)` | Window of T timesteps `(W, F)` |
| Domain alignment | Standard MMD | Temporal MMD (mean + variance) |
| Pseudo-labels | Raw softmax confidence | Temperature-scaled (calibrated) confidence |
| Adversarial | GRL + binary discriminator | Same |

#### 1. Temporal Feature Extractor
Inspired by the multi-scale gated convolutions in GQAT-Net:
- **Three parallel gated convolutions** (kernel sizes 3, 5, 7) capture short-, medium- and long-range temporal dependencies simultaneously.
- **Grouped Query Attention (×2)** lets different channels attend to different temporal patterns (e.g. slow memory leak vs sudden CPU spike).
- **Mean + std pooling** over the time dimension: mean captures average resource load; std captures variability (fault dynamics).

#### 2. Temporal MMD
Standard MMD aligns the mean of feature distributions between source and target. Temporal MMD additionally aligns **variance** of the feature distribution — important because different fault types exhibit different temporal variability patterns.

```
L_tmmd = MMD(μ_src, μ_tgt) + 0.5 · MMD(σ²_src, σ²_tgt)
```

#### 3. Calibrated Pseudo-Labels
Raw softmax over-confidently assigns probability 1.0 to the predicted class. A learnable temperature parameter T divides the logits before softmax, producing better-calibrated confidence scores. Only windows with calibrated confidence ≥ 0.85 are selected as pseudo-labels.

```
p_calibrated = softmax(logits / T)   where T is learnable, initialised at 1.5
```

---

## Connection to Previous Work (pivot/)

The `pivot/` directory explored **cross-platform transfer** (Google → Alibaba).  
This repo supports **two** setups:

1. **Within-Alibaba** (`00_prepare_data.py`): source and target are different **failure domains** inside Alibaba 2018 — closest to the replicated paper’s setting.
2. **Google → Alibaba** (`00_prepare_data_google_alibaba.py`): **Google Cluster Trace 2019** `instance_usage` shards as **source** (fully labeled windows for training), **Alibaba 2018** as **target** (partially labeled). This matches the pivot/ dissertation angle: *train on one cloud’s telemetry, adapt to another*.

Outputs for (2) go to `data/processed_google_alibaba/` so (1)’s `data/processed/` is unchanged.

---

## Directory Structure

```
updated_research/
├── 00_prepare_data.py                  — Within-Alibaba temporal windows → data/processed/
├── 00_prepare_data_google_alibaba.py   — Google source + Alibaba target → data/processed_google_alibaba/
├── run_google_to_alibaba.py            — Prep + Table-1 train for cross-domain
├── prepare_common.py                   — Shared windowing / labels
├── alibaba_io.py / google_io.py        — Raw trace loaders
├── 01_train_all_models.py              — Table 1: all 6 methods (`--processed-dir` …)
├── 02_experiment_label_scarcity.py   — Figure 2
├── 03_experiment_class_imbalance.py  — Figure 3 (checkpointed)
├── 04_experiment_heterogeneous_nodes.py  — Figure 4
├── 05_ablation_study.py        — Figure 5 (novel)
├── run_all.py                  — Master script
├── models.py                   — All architectures
├── trainer.py                  — Training engine
├── data/
│   ├── raw/                    — Alibaba CSVs; Google under raw/google/cell_*/…
│   ├── processed/              — Within-Alibaba artifacts
│   └── processed_google_alibaba/ — Cross-domain artifacts (optional)
├── checkpoints/                — Model weights
├── results/
│   ├── tables/                 — JSON + TXT result tables
│   └── figures/                — PNG plots
└── logs/                       — Per-script logs
```

---

## How to Run

### On the GPU machine

```bash
cd ~/path/to/updated_research

# Install dependencies (Blackwell GPU — CUDA 12.8 nightly)
/opt/conda/bin/pip install -r requirements.txt

# Place Alibaba 2018 data if available (optional — synthetic fallback exists)
# cp /path/to/machine_usage.csv   data/raw/
# cp /path/to/machine_meta.csv    data/raw/

# Full pipeline (run in tmux to survive disconnects)
tmux new -s dissertation
nohup python run_all.py > logs/run_all.log 2>&1 &

# Monitor
tail -f logs/run_all.log
```

### Individual steps

```bash
python 00_prepare_data.py
python 01_train_all_models.py
python 02_experiment_label_scarcity.py
python 03_experiment_class_imbalance.py   # checkpointed — safe to interrupt & resume
python 04_experiment_heterogeneous_nodes.py
python 05_ablation_study.py
```

### Google (source) → Alibaba (target)

Place **Google 2019** parquet shards where the pivot downloader puts them, e.g.  
`data/raw/google/cell_a/instance_usage-000000000000.parquet` (any `cell_*`, any `instance_usage*.parquet`).

```bash
# Build cross-domain tensors (does not overwrite data/processed/)
python 00_prepare_data_google_alibaba.py

# Train Table 1 on that split
python 01_train_all_models.py --processed-dir data/processed_google_alibaba

# Or prep + train in one go
python run_google_to_alibaba.py
```

Experiments 02–05 also accept `--processed-dir data/processed_google_alibaba`.

**Note:** `instance_usage` has CPU and memory columns only; `mem_gps`, `net_*`, and `disk_io_percent` are **mapped proxies** so the same 6-D model and labeling rules apply. For the thesis, state this explicitly as a trace-schema limitation.

---

## Expected Results (Table 1)

| Method | Accuracy | F1-Score | AUC |
|--------|----------|----------|-----|
| DANN | ~84% | ~81% | ~87% |
| CDAN | ~86% | ~82% | ~88% |
| FixBi | ~86% | ~83% | ~89% |
| ToAlign | ~87% | ~84% | ~90% |
| DATL (replicated) | ~90% | ~85% | ~91% |
| **TA-DATL (Ours)** | **>90%** | **>86%** | **>92%** |

TA-DATL should outperform all baselines including DATL, with the advantage most pronounced in the low-label-scarcity and heterogeneous-node experiments.

---

## Datasets

**Alibaba Cluster Trace 2018** — publicly available from the Alibaba research group.  
Six features: `cpu_util_percent`, `mem_util_percent`, `mem_gps`, `net_in`, `net_out`, `disk_io_percent`.  
Six fault classes: Normal, CPU Overload, Memory Leak, Disk I/O Fault, Network Fault, Mixed Fault.

If raw data is unavailable, `00_prepare_data.py` generates realistic synthetic data with the same schema so all scripts can still run.

---

## Key Files for the Dissertation Write-up

- `models.py:TA_DATL` — the proposed model (Section 3)
- `models.py:TemporalExtractor` — feature extractor architecture (Section 3.1)
- `models.py:temporal_mmd` — temporal MMD loss (Section 3.2)
- `models.py:TA_DATL.get_pseudo_labels` — calibrated pseudo-label selection (Section 3.3)
- `05_ablation_study.py` — component analysis (Section 5)
