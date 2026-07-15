#!/usr/bin/env bash
# =============================================================================
# run_gpu.sh
# =============================================================================
# Runs the full dissertation experiment on a GPU server (40GB VRAM).
# Two transfer directions, all seven methods, regression + classification metrics.
#
# Usage:
#   chmod +x run_gpu.sh
#   nohup ./run_gpu.sh > logs/run_gpu.log 2>&1 &
#
# Results:
#   results/g2a/   — Google → Alibaba
#   results/a2g/   — Alibaba → Google
#
# Each output directory contains:
#   all_results.json            — full nested results
#   regression_results.json     — MAE / MAPE / RMSE / MSE
#   classification_results.json — Accuracy / Precision / Recall / F1 / MCC / G-Mean
#   preprocessed.npz            — cached windowed arrays (reuse with --load-cache)
#   checkpoints/                — saved .pt model weights
# =============================================================================

set -e   # exit on first error
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# ── Config ────────────────────────────────────────────────────────────────────
GOOGLE_DIR="data/raw/google"
ALIBABA_DIR="data/raw/alibaba"
DEVICE="cuda"
BATCH_SIZE=256    # safe for 40GB GPU; raise to 512 if memory allows
LOG_DIR="logs"

mkdir -p "$LOG_DIR" results/g2a results/a2g

echo "================================================================"
echo "  Dissertation GPU Experiment"
echo "  $(date)"
echo "  Device: $DEVICE   Batch size: $BATCH_SIZE"
echo "================================================================"

# ── Check GPU ─────────────────────────────────────────────────────────────────
python -c "import torch; print('GPU:', torch.cuda.get_device_name(0), \
  '|', torch.cuda.get_device_properties(0).total_memory//1024**3, 'GB')" \
  || { echo "ERROR: CUDA not available"; exit 1; }

# ── Install dependencies (safe to re-run) ────────────────────────────────────
echo ""
echo "Checking dependencies..."
pip install -q scikit-learn POT numpy torch matplotlib
# Optional: gluonts for DeepAR/DRP/MQF2
pip install -q "gluonts[torch]" 2>/dev/null && echo "  gluonts: OK" || echo "  gluonts: not installed (DeepAR/MQF2 will be skipped)"

# =============================================================================
# DIRECTION 1 — Google → Alibaba
# =============================================================================
echo ""
echo "================================================================"
echo "  DIRECTION 1/2: Google → Alibaba"
echo "  Started: $(date)"
echo "================================================================"

python run_full_experiment.py \
    --direction google_to_alibaba \
    --google    "$GOOGLE_DIR" \
    --alibaba   "$ALIBABA_DIR" \
    --device    "$DEVICE" \
    --batch-size "$BATCH_SIZE" \
    --out       results/g2a \
    --save-cache results/g2a/preprocessed.npz \
    --lstm-epochs     150 \
    --cwpdda-epochs   100 \
    --nbeats-epochs   100 \
    --mctl-s1-epochs   50 \
    --mctl-s2a-epochs  50 \
    --mctl-s2b-epochs  50 \
    --mc-s1-epochs     30 \
    --mc-s2-epochs     50 \
    --mc-s3-epochs    100 \
    --deepjdot-epochs  50 \
    --arima-subsample 1000 \
    2>&1 | tee "$LOG_DIR/g2a.log"

echo ""
echo "  Google → Alibaba: DONE at $(date)"
echo "  Regression:      results/g2a/regression_results.json"
echo "  Classification:  results/g2a/classification_results.json"

# =============================================================================
# DIRECTION 2 — Alibaba → Google
# =============================================================================
echo ""
echo "================================================================"
echo "  DIRECTION 2/2: Alibaba → Google"
echo "  Started: $(date)"
echo "================================================================"

python run_full_experiment.py \
    --direction alibaba_to_google \
    --google    "$GOOGLE_DIR" \
    --alibaba   "$ALIBABA_DIR" \
    --device    "$DEVICE" \
    --batch-size "$BATCH_SIZE" \
    --out       results/a2g \
    --save-cache results/a2g/preprocessed.npz \
    --lstm-epochs     150 \
    --cwpdda-epochs   100 \
    --nbeats-epochs   100 \
    --mctl-s1-epochs   50 \
    --mctl-s2a-epochs  50 \
    --mctl-s2b-epochs  50 \
    --mc-s1-epochs     30 \
    --mc-s2-epochs     50 \
    --mc-s3-epochs    100 \
    --deepjdot-epochs  50 \
    --arima-subsample 1000 \
    2>&1 | tee "$LOG_DIR/a2g.log"

echo ""
echo "  Alibaba → Google: DONE at $(date)"
echo "  Regression:      results/a2g/regression_results.json"
echo "  Classification:  results/a2g/classification_results.json"

# =============================================================================
echo ""
echo "================================================================"
echo "  ALL EXPERIMENTS COMPLETE"
echo "  $(date)"
echo "================================================================"
echo ""
echo "Results:"
echo "  results/g2a/all_results.json"
echo "  results/a2g/all_results.json"
echo ""
echo "To generate figures from these results, run on your local machine:"
echo "  python generate_from_gpu_results.py"
echo "================================================================"
