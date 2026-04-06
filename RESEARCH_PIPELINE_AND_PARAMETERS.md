# Research pipeline: from the DATL paper → TA-DATL → Google→Alibaba

This document ties together (1) what the first paper proposes, (2) how **unsupervised domain adaptation** is implemented on **Alibaba** in this codebase, (3) which **parameters** we use and how they map to paper notation, (4) how **TA-DATL** extends the paper’s method, and (5) what changes when the **source domain is Google** instead of a second Alibaba split.

**Primary references**

- Fang, B. & Gao, D. (2025). *Domain-Adversarial Transfer Learning for Fault Root Cause Identification in Cloud Computing Systems.* RAIIC 2025. (Replicated in `replicate/`; extended in `updated_research/`.)
- Code: `updated_research/models.py`, `trainer.py`, `prepare_common.py`, `00_prepare_data.py`, `00_prepare_data_google_alibaba.py`, `google_io.py`, `alibaba_io.py`.

---

## 1. What the first paper does (conceptual summary)

### 1.1 Task

- **Problem:** Multi-class **fault root cause identification** on cloud machines (not only “failure yes/no”).
- **Setting:** **Unsupervised domain adaptation (UDA):** a **labeled source domain** and a **target domain** where labels are **scarce or absent** for training. The goal is to train a classifier that performs well on the **target** by aligning representations across domains while using source supervision.
- **Data:** **Alibaba Cluster Trace 2018** — machine-level telemetry (CPU, memory, disk, network, etc.).

### 1.2 Proposed method: DATL

The paper’s proposed model (**DATL**) combines:

1. **Feature extractor** \(F\) mapping raw inputs to a latent feature vector.
2. **Task classifier** \(C\) predicting fault class from \(F(x)\).
3. **Domain discriminator** \(D\) (with **gradient reversal**, GRL) trying to tell source vs target features apart; the extractor is trained **adversarially** to fool \(D\) → **domain-invariant** features.
4. **MMD loss** between source and target feature batches to explicitly reduce distribution mismatch (mean-based MMD in our implementation).
5. **Pseudo-labeling:** periodically add **high-confidence** predictions on **unlabeled target** samples to the source training set (self-training under domain shift).

**Total loss (paper-style decomposition, matching `replicate/README.md`):**

\[
\mathcal{L}_{\text{total}} = \mathcal{L}_{\text{cls}} + \lambda_1 \mathcal{L}_{\text{mmd}} + \lambda_2 \mathcal{L}_{\text{adv}}
\]

- \(\mathcal{L}_{\text{cls}}\): cross-entropy on **labeled source** (and expanded set after pseudo-labels).
- \(\mathcal{L}_{\text{mmd}}\): alignment of feature distributions (source vs target).
- \(\mathcal{L}_{\text{adv}}\): binary domain classification with GRL (source → label 0, target → label 1).

### 1.3 Baselines in the paper

The paper compares DATL against several domain-adaptation baselines, including **DANN**, **CDAN**, **FixBi**, **ToAlign** (all implemented in `models.py` for fair comparison).

### 1.4 Experiments in the paper

Typical stress tests (mirrored in `02`–`04` scripts):

- **Label scarcity:** fewer labeled target samples.
- **Class imbalance:** skewed fault-class frequencies in the source.
- **Heterogeneous nodes:** different machine / workload profiles.

### 1.5 Caveat on exact reproduction

The published paper reports specific accuracy / F1 / AUC numbers. Our pipeline uses a **defined** labeling rule on Alibaba (percentile thresholds), sampling, and windowing for TA-DATL. **Absolute numbers** may not match the PDF exactly; the **ranking of methods** and **trends** across scenarios are the main scientific comparison points unless every preprocessing detail in the paper is recovered bit-for-bit.

---

## 2. How “Alibaba transfer learning” works in *this* codebase

### 2.1 Domains (within-Alibaba, default `00_prepare_data.py`)

- **Source and target are both from Alibaba 2018**, but represent **different subsets** of machines / **failure domains** (`failure_domain_1` from `machine_meta.csv` when available), analogous to “different clusters” or deployment groups under the same telemetry schema.
- If meta split is degenerate, the code **falls back** to a **random machine split** (60% / 40%) so source ≠ target.

This is **still transfer learning / UDA**: train with labeled source + mostly unlabeled target, align features, evaluate on **labeled target** holdout mask.

### 2.2 What the model sees each iteration (`trainer.py`)

For adversarial models (DANN, CDAN, DATL, ToAlign, TA-DATL):

- A **source minibatch** \((x_s, y_s)\) with true labels \(y_s\).
- A **target minibatch** \(x_t\) drawn from **unlabeled** target windows (or all target if none unlabeled).
- **Gradient reversal strength** \(\alpha\) increases over epochs via `grl_alpha(epoch, total_epochs, gamma=10)` (standard DANN schedule).

**DATL / TA-DATL additionally:**

- Every `pseudo_freq` epochs (10 for TA-DATL in `train_ta_datl`), high-confidence **pseudo-labels** on unlabeled target are merged into an expanded source set \((X'_{src}, y'_{src})\).

### 2.3 Labels and features (data side)

**Features (6 dimensions, same for flat and temporal pipelines after pooling):**

| Index | Column | Role |
|------:|--------|------|
| 1 | `cpu_util_percent` | CPU utilization |
| 2 | `mem_util_percent` | Memory utilization |
| 3 | `mem_gps` | Memory bandwidth proxy |
| 4 | `net_in` | Network in |
| 5 | `net_out` | Network out |
| 6 | `disk_io_percent` | Disk I/O |

**Multi-class fault labels (6 classes):** `Normal`, `CPU_Overload`, `Memory_Leak`, `Disk_IO_Fault`, `Network_Fault`, `Mixed_Fault` — assigned by **percentile rules** on the **concatenated source+target** frame when building thresholds (`prepare_common.compute_thresholds`), so both domains share one rule book.

**Temporal windows (`prepare_common.py`):**

- `WINDOW_SIZE = 20`, `WINDOW_STEP = 5`: sliding windows per machine (or per Google id); label = class at **last timestep** in the window.

**Target labeling mask for training:**

- ~**30%** of **target windows** randomly marked `labeled=True` for **evaluation**; the rest are **unlabeled** for adaptation (simulates scarce target annotations).

**Normalization (`01_train_all_models.py` `load()`):**

- **StandardScaler fit on source**, applied to **target** (per-feature, over all timesteps in the window tensor). This matches common UDA practice: avoid using target statistics as if they were fully labeled.

---

## 3. Parameters: full list and mapping to the paper

### 3.1 Loss weights (paper \(\lambda_1\), \(\lambda_2\))

| Symbol (paper) | Meaning | **DATL** (`models.py`) | **TA-DATL** |
|----------------|---------|------------------------|-------------|
| \(\lambda_1\) | MMD (or temporal MMD) weight | `lambda1=0.1` | `lambda1=0.1` (`temporal_mmd`) |
| \(\lambda_2\) | Adversarial (domain) weight | `lambda2=0.1` | `lambda2=0.1` |

**Baselines:** DANN/CDAN/ToAlign use a fixed **`0.1`** multiplier on \(\mathcal{L}_{\text{adv}}\) (or KL for FixBi) in code, not exposed as named \(\lambda_2\) on the class.

### 3.2 Pseudo-labeling (DATL & TA-DATL)

| Parameter | Value (code) | Notes |
|-----------|--------------|--------|
| Confidence threshold \(\delta\) | `pseudo_threshold=0.85` | Paper/replicate README cite \(p \ge 0.85\). |
| Refresh period | Every **10** epochs | `pseudo_freq=10` in `train_ta_datl`; DATL uses `epoch % 10 == 0` in `train_adversarial`. |
| Start epoch | After epoch **15** | Avoid unstable early pseudo-labels. |

**TA-DATL only:** **temperature scaling** for pseudo-label confidence:

| Parameter | Value | Meaning |
|-----------|-------|---------|
| Initial \(T\) | `temperature=1.5` → stored as `log_T` | `softmax(logits / T)`; \(T\) is **learnable**, clamped to \([0.5, 5]\) in `get_pseudo_labels`. |

**DATL:** raw `softmax(logits)` without temperature.

### 3.3 Architecture — flat models (paper-style DATL & baselines)

| Component | Parameter | Value |
|-----------|-----------|-------|
| Flat extractor / hidden | `hidden_dim` | **128** (`HIDDEN` in `01_train_all_models.py`) |
| Dropout | `dropout` | **0.3** |
| Classifier | `feat_dim // 2` hidden | **64** (when `feat_dim=128`) |
| Domain discriminator | `feat_dim // 2` hidden | **64** |
| Input dim | `input_dim` | **6** (features above) |

**MMD implementation (`mmd_loss`):** squared \(\ell_2\) distance between **batch mean** features (linear MMD / mean embedding), not a full RBF-kernel MMD.

**GRL schedule:** `gamma=10.0` in `grl_alpha`.

### 3.4 Architecture — TA-DATL (temporal)

| Component | Parameter | Value |
|-----------|-----------|-------|
| Input shape | — | `(batch, W=20, F=6)` |
| Multi-scale gated conv | `hidden_dim` | **120** (must divide by 3 and by `n_groups`) |
| GQA groups | `n_groups` | **4** |
| Dropout | `dropout` | **0.3** |
| Classifier / discriminator | — | Same head design as flat stack (`Classifier`, `DomainDiscriminator` on 120-D features) |

**Pooling:** temporal mean + temporal std over time → concat → linear projection to `hidden_dim` (`TemporalExtractor`).

**Temporal MMD (`temporal_mmd`):** mean feature MMD + **0.5 ×** squared difference of **per-dimension batch variances** (biased variance, `correction=0` for numerical stability).

### 3.5 Optimization & training loop

| Parameter | Table 1 (`01_train_all_models.py`) | Experiments 02–05 |
|-----------|-----------------------------------|-------------------|
| Epochs | **200** | **150** |
| Optimizer | Adam | Adam |
| Learning rate | **1e-3** | **1e-3** |
| Weight decay | **1e-4** | **1e-4** |
| LR schedule | Cosine annealing, `T_max=epochs` | Same |
| Batch size (flat) | **256** | **256** (where applicable) |
| Batch size (TA-DATL) | **128** | **128** |
| Gradient clip | max norm **1.0** | Same |
| Device | **CUDA** (`trainer.py`: `DEVICE = cuda`) | Same |

### 3.6 Data / sampling (Alibaba)

| Parameter | Typical value | Where |
|-----------|---------------|--------|
| Alibaba usage sample rows | **500,000** (probabilistic skip) | `load_machine_usage` in `alibaba_io.py` / `00_prepare_data.py` |
| Target labeled fraction | **30%** of windows | `00_prepare_data.py` / cross-domain prep |
| Random seeds | e.g. **42** (masks), **43** (Google subsample) | prep scripts |

---

## 4. How TA-DATL improves on the first paper (methodologically)

The **first paper’s DATL** is **instantiated in code** as:

- **Flat MLP** feature extractor on **one vector per window** (mean-pooled 6-D over time for fair baseline comparison) or, in pure replicate folder, point-level features without the temporal encoder.
- **Mean-only MMD** on features.
- **Standard softmax** pseudo-labels.

**TA-DATL** keeps the **same UDA story** (classifier + domain discriminator + alignment + pseudo-labels) but changes **what is aligned** and **how confidence is computed**:

1. **Temporal representation:** Instead of a single MLP on a static 6-D vector, TA-DATL uses **multi-scale gated 1-D convolutions + grouped query attention** over a **sequence** \((W \times F)\), then mean+std pooling. This targets **temporal evolution** of overload patterns (e.g. sustained vs spiky CPU), which the paper’s flat DATL does not model explicitly.

2. **Temporal MMD:** Adds alignment of **feature variance** across domains (in addition to mean), motivated by differing **dynamics** under shift.

3. **Calibrated pseudo-labels:** **Temperature-scaled** softmax + learnable \(T\) reduces overconfident wrong pseudo-labels on the target, which can hurt self-training under domain shift.

**Empirical comparison in this repo:** Table 1 and ablations compare **TA-DATL** to **DATL** and other baselines on the **same** preprocessed tensors; the ablation script explicitly removes the temporal encoder, \(\lambda_1\), temperature, and \(\lambda_2\) to isolate effects.

---

## 5. Switching the source to Google: what changes and what does not

### 5.1 What does **not** change (transfer learning is still there)

- **Training objective:** Still **joint** source+target training with **GRL / domain loss**, **MMD or temporal MMD**, and **pseudo-labeling** for DATL/TA-DATL.
- **Target:** Still **Alibaba** windows with the same 6-D schema and the same **percentile-based multi-class labels** (thresholds computed on **Google + Alibaba combined** so the label definition is shared).
- **Evaluation:** Still **macro-F1, accuracy, AUC** on the **labeled** target mask.
- **Model code:** `models.py` and `trainer.py` are **unchanged**; only the **npz/parquet** inputs differ (`--processed-dir data/processed_google_alibaba`).

So “transfer” is **not** “pretrain on Google then fine-tune separately”; it remains **unsupervised domain adaptation** with **Google as source** and **Alibaba as target**.

### 5.2 What **does** change (data pipeline)

| Aspect | Within-Alibaba (`data/processed/`) | Google→Alibaba (`data/processed_google_alibaba/`) |
|--------|-----------------------------------|-----------------------------------------------------|
| **Source raw trace** | Alibaba `machine_usage.csv` (+ meta) | Google **`instance_usage*.json.gz`** (or parquet) under `data/raw/google/cell_*/` |
| **Source row ID** | Alibaba `machine_id` | Prefer **`machine_id`** from Google JSON; else `collection_id_instance_index` |
| **Source CPU/memory** | Native Alibaba columns | Nested **`average_usage` / `maximum_usage`** dicts (`cpus`, `memory`) mapped to % |
| **`mem_gps`, `net_*`, `disk`** on source | Real columns (Alibaba) | **Proxies** derived from usage + noise (Google instance_usage lacks Alibaba’s full IO/net fields) |
| **Output directory** | `data/processed/` | `data/processed_google_alibaba/` (default) |
| **meta.json** | `data_source` etc. | Adds `transfer_setup: "google_to_alibaba"` |

### 5.3 Interpretation for the thesis

- **Stronger domain shift:** Different provider, collection cadence, and **partial feature emulation** on the source → harder transfer, more realistic **cross-cloud** narrative (aligned with `pivot/` motivation).
- **Honest limitation:** Target disk/net are **real Alibaba**; source side uses **engineered** channels for those dimensions. Discuss as **schema alignment** / **missing modalities** rather than claiming pixel-perfect physical equivalence.

### 5.4 Commands (reference)

```bash
# Build tensors (does not overwrite within-Alibaba processed/)
python 00_prepare_data_google_alibaba.py

# Train using cross-domain tensors
python 01_train_all_models.py --processed-dir data/processed_google_alibaba
```

### 5.5 Within-Google (recommended when Google→Alibaba metrics collapse)

When **Google→Alibaba** yields ~random accuracy and **AUC = nan**, the experimental question can still be answered on **Google-only** data: split **disjoint `machine_id` sets** into source/target (default 60/40), same windows, thresholds, λ’s, and pseudo-labeling as Alibaba→Alibaba. Artifacts go to `data/processed_google_google/` via `00_prepare_data_google_google.py`. This isolates **TA-DATL vs DATL on Google telemetry** without confounding extreme cross-provider shift.

---

## 6. Quick parameter cheat sheet (copy-paste for thesis tables)

| Category | Setting |
|----------|---------|
| Classes | 6 |
| Features | 6 (`cpu_util_percent` … `disk_io_percent`) |
| Window | \(W=20\), stride 5 |
| \(\lambda_1\) (MMD / temporal MMD) | 0.1 |
| \(\lambda_2\) (adversarial) | 0.1 |
| Pseudo threshold | 0.85 |
| Pseudo refresh | every 10 epochs, after epoch 15 |
| DATL hidden | 128 |
| TA-DATL conv/attn hidden | 120, `n_groups=4` |
| Dropout | 0.3 |
| LR | 1e-3, Adam, weight decay 1e-4 |
| Cosine T_max | number of epochs |
| Epochs (Table 1) | 200 |
| Epochs (Figures 2–5 scripts) | 150 |
| Batch (flat / temporal) | 256 / 128 |
| Scaler | StandardScaler, **fit on source only** |
| Target labels for eval | ~30% of target windows |

---

*End of document.*
