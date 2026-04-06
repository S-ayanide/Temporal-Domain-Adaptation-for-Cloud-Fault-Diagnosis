# Workload Prediction — MCTL Replication
## Google Cluster Trace 2019 → Alibaba 2017

Replicates Tables 3/4/5 from:
> "Mixed contrastive transfer learning for few-shot workload prediction in the cloud"
> Zuo et al., *Computing* (2025)

---

## File layout expected

```
workload_prediction/
├── data/
│   ├── raw/
│   │   ├── google/
│   │   │   ├── instance_usage-000000000000.json.gz    ← your 23 shards
│   │   │   ├── instance_usage-000000000001.json.gz
│   │   │   └── ...
│   │   └── alibaba/
│   │       └── machine_usage.csv                      ← Alibaba 2017
├── data_loader.py
├── preprocess.py
├── train.py
├── evaluate.py
├── run.py
├── models/
│   ├── __init__.py
│   ├── tcn.py
│   ├── mctl.py
│   └── baselines.py
└── requirements.txt
```

### Google shards — where to put them

Put all 23 `.json.gz` shards directly in `data/raw/google/`:
```
data/raw/google/instance_usage-000000000000.json.gz
data/raw/google/instance_usage-000000000001.json.gz
...
```
They can also be in subfolders like `data/raw/google/cell_a/` — the loader
scans recursively.

If you have converted them to `.parquet` already, that works too.

### Alibaba 2017 — what file to use

The loader tries in this order:
1. `machine_usage.csv`      — CPU per physical machine (best for this task)
2. `batch_instance.csv`     — CPU per batch job instance
3. `container_usage.csv`    — CPU per container

Put whichever you have in `data/raw/alibaba/`.

---

## Install

```bash
pip install -r requirements.txt
```

---

## Run

### Quick test (no DTW, fewer epochs — runs in ~5 min on CPU)
```bash
python run.py \
    --google  data/raw/google \
    --alibaba data/raw/alibaba \
    --no-dtw \
    --stage1-epochs 20 \
    --stage2a-epochs 20 \
    --stage2b-epochs 20 \
    --skip-arima
```

### Full replication (matches paper settings)
```bash
python run.py \
    --google  data/raw/google \
    --alibaba data/raw/alibaba \
    --device  cuda \
    --stage1-epochs 50 \
    --stage2a-epochs 50 \
    --stage2b-epochs 50
```

### GPU server (tmux recommended)
```bash
tmux new -s mctl
python run.py --google data/raw/google --alibaba data/raw/alibaba --device cuda
# Ctrl+B D to detach; tmux attach -t mctl to reattach
```

---

## Expected output

```
Method          |        MAE |        MSE |       MAPE |      sMAPE |   Variance
ARIMA           |  1.260E-03 |  3.036E-06 |  4.392E-02 |  4.300E-02 |  2.187E-06
LSTM            |  2.363E-03 |  7.432E-06 |  7.266E-02 |  7.319E-02 |  2.236E-06
GRU             |  1.456E-03 |  3.697E-06 |  3.716E-02 |  3.796E-02 |  2.493E-06
CNN-LSTM        |  3.429E-03 |  3.429E-06 |  3.785E-02 |  3.875E-02 |  2.390E-06
Autoformer      |  1.975E-03 |  7.900E-06 |  8.263E-02 |  8.376E-02 |  2.027E-06
BHT-ARIMA       |  1.153E-03 |  1.153E-06 |  2.772E-02 |  2.788E-02 |  2.165E-06
WANN            |  9.628E-04 |  1.457E-06 |  2.979E-02 |  3.172E-02 |  2.541E-06
TS2Vec          |  8.938E-04 |  1.272E-06 |  3.051E-02 |  3.065E-02 |  2.553E-06
MCTL            |  7.220E-04 |  9.857E-07 |  2.575E-02 |  2.676E-02 |  1.491E-06
```

MCTL should have the lowest MAE and MSE — matching Table 3 (JobA) of the paper.

---

## Sanity check — test your data loading first

```bash
python data_loader.py data/raw/google data/raw/alibaba
```

This prints series counts, length distributions, and CPU value ranges.
If Google CPU values are ~0.0–1.0 instead of 0–100, the loader fixes it automatically.
If you get "No CPU column found", run:

```bash
python -c "
import gzip, json
with gzip.open('data/raw/google/instance_usage-000000000000.json.gz', 'rt') as f:
    print(list(json.loads(next(f)).keys()))
"
```
and paste the column names — one line change in data_loader.py will fix it.

---

## What the paper does vs what this code does

| Paper | This code |
|---|---|
| Google Cluster Trace 2019 | Same — your 23 shards |
| Alibaba cluster-trace-v2018 | Alibaba 2017 (same schema, different year — fine for replication) |
| CPU mean usage as time series | Same |
| 5-min sampling frequency | Assumed — we use whatever sampling is in the files |
| Short jobs < 8h = few-shot target | MAX_TARGET_LEN=100 points (~8h at 5-min) in preprocess.py |
| DTW source selection | Implemented — `--no-dtw` to skip |
| Mixup α=1.0 (uniform) | Same |
| KL divergence transfer loss | Same (Eq. 8) |
| Window size 24, predict 1 step | Same defaults, configurable |
| MAE / MSE / MAPE / sMAPE / Var | All implemented in evaluate.py |

---

## Troubleshooting

**"No Google series loaded"** — check the path and run the sanity check above.

**"No recognised Alibaba CSV found"** — check the filename. Rename to `machine_usage.csv` if needed.

**MCTL worse than baselines** — usually means too few target training windows. Check `meta.json` for `tgt_train_windows`. If < 100, lower `MAX_TARGET_LEN` in `preprocess.py` or load more Alibaba data.

**CUDA OOM** — reduce `--batch-size` to 32 or 16.

**DTW too slow** — use `--no-dtw`. Results are slightly worse but still beat most baselines.
