#!/usr/bin/env bash
# Download raw data for CWPDDA / MCTL replication.
#
# SAFE TO RE-RUN: wget -c resumes partial downloads, so interrupting and
# restarting will not corrupt files.
#
# Usage:
#   bash download_data.sh            # download everything
#   bash download_data.sh --google   # Google only
#   bash download_data.sh --alibaba  # Alibaba only

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
GOOGLE_DIR="$SCRIPT_DIR/data/raw/google"
ALIBABA_DIR="$SCRIPT_DIR/data/raw/alibaba"

DO_GOOGLE=true
DO_ALIBABA=true

for arg in "$@"; do
  case "$arg" in
    --google)  DO_ALIBABA=false ;;
    --alibaba) DO_GOOGLE=false  ;;
  esac
done

WGET_OPTS="-c --retry-connrefused --tries=0 --timeout=60 --show-progress"

# ─── Google Cluster Trace 2019 ────────────────────────────────────────────────
# Public bucket: gs://clusterdata-2019-a  (HTTP mirror below)
# 23 shards: instance_usage-000000000000.json.gz … instance_usage-000000000022.json.gz

if $DO_GOOGLE; then
  echo "==> Downloading Google Cluster Trace 2019 (23 shards)…"
  mkdir -p "$GOOGLE_DIR"
  BASE="https://storage.googleapis.com/clusterdata-2019-a"

  for i in $(seq 0 22); do
    SHARD=$(printf "%012d" "$i")
    FILE="instance_usage-${SHARD}.json.gz"
    DEST="$GOOGLE_DIR/$FILE"

    if [[ -f "$DEST" ]]; then
      # File exists — check it is a valid gzip (not a zero-byte partial)
      if gzip -t "$DEST" 2>/dev/null; then
        echo "  [skip] $FILE — already complete"
        continue
      else
        echo "  [resume] $FILE — previous download was incomplete"
      fi
    fi

    wget $WGET_OPTS -O "$DEST" "${BASE}/${FILE}" || {
      echo "  [warn] Failed to download $FILE — safe to retry (partial kept)"
    }
  done
  echo "==> Google download complete."
fi

# ─── Alibaba Cluster Trace 2018 (schema matches 2017) ─────────────────────────
# Only machine_usage is needed for the CPU workload task.

if $DO_ALIBABA; then
  echo "==> Downloading Alibaba machine_usage…"
  mkdir -p "$ALIBABA_DIR"
  ABASE="http://aliopentrace.oss-cn-beijing.aliyuncs.com/v2018Traces"
  TARBALL="$ALIBABA_DIR/machine_usage.tar.gz"

  if [[ -f "$ALIBABA_DIR/machine_usage.csv" ]]; then
    echo "  [skip] machine_usage.csv — already extracted"
  else
    wget $WGET_OPTS -O "$TARBALL" "${ABASE}/machine_usage.tar.gz" || {
      echo "  [warn] Failed to download machine_usage.tar.gz — safe to retry"
    }

    if gzip -t "$TARBALL" 2>/dev/null; then
      echo "  Extracting machine_usage.tar.gz…"
      tar -xzf "$TARBALL" -C "$ALIBABA_DIR"
      # Flatten any subdirectory the tarball may have created
      find "$ALIBABA_DIR" -name "machine_usage.csv" ! -path "$ALIBABA_DIR/machine_usage.csv" \
        -exec mv {} "$ALIBABA_DIR/machine_usage.csv" \;
      echo "  Extracted → $ALIBABA_DIR/machine_usage.csv"
    else
      echo "  [warn] Tarball incomplete — re-run to resume"
    fi
  fi
  echo "==> Alibaba download complete."
fi

echo ""
echo "Done. Verify with:"
echo "  python data_loader.py data/raw/google data/raw/alibaba"
