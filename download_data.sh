#!/usr/bin/env bash
# Download raw data for CWPDDA / MCTL replication.
#
# Google Cluster Trace 2019 lives at:
#   https://storage.googleapis.com/clusterdata_2019_{cell}/instance_usage-{shard}.json.gz
# where cell ∈ {a..h} and shard ∈ 000000000000..N (varies per cell, stops at first 404).
#
# SAFE TO RE-RUN: uses .part temp files + Range resumption — interrupting and
# restarting will never corrupt a file.
#
# Usage:
#   bash download_data.sh                     # all cells, shards 0+, + Alibaba
#   bash download_data.sh --cells a b         # only cells a and b
#   bash download_data.sh --max-shards 2      # at most 2 shards per cell
#   bash download_data.sh --alibaba           # Alibaba only (skip Google)
#   bash download_data.sh --google            # Google only (skip Alibaba)

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
GOOGLE_DIR="$SCRIPT_DIR/data/raw/google"
ALIBABA_DIR="$SCRIPT_DIR/data/raw/alibaba"

# ─── Defaults ─────────────────────────────────────────────────────────────────
CELLS=(a b c d e f g h)
MAX_SHARDS=999    # effectively unlimited — stop on first 404
DO_GOOGLE=true
DO_ALIBABA=true

# ─── Argument parsing ─────────────────────────────────────────────────────────
while [[ $# -gt 0 ]]; do
  case "$1" in
    --google)    DO_ALIBABA=false; shift ;;
    --alibaba)   DO_GOOGLE=false;  shift ;;
    --max-shards) MAX_SHARDS="$2"; shift 2 ;;
    --cells)
      shift
      CELLS=()
      while [[ $# -gt 0 && ! "$1" =~ ^-- ]]; do
        CELLS+=("$1"); shift
      done
      ;;
    *) echo "Unknown option: $1"; exit 1 ;;
  esac
done

# ─── Resumable wget wrapper ────────────────────────────────────────────────────
# Downloads URL to DEST using a .part temp file. Resumes on retry.
# Returns 0 on success, 1 on 404 (stops the caller's shard loop).
download_resumable() {
  local url="$1"
  local dest="$2"
  local part="${dest}.part"

  # Already complete?
  if [[ -f "$dest" ]] && gzip -t "$dest" 2>/dev/null; then
    echo "    [skip] $(basename "$dest") — already complete"
    return 0
  fi

  # wget -c: resume if .part exists; -q --show-progress: clean output
  local http_code
  http_code=$(wget -c -q --show-progress --server-response \
                   --retry-connrefused --tries=5 --timeout=60 \
                   -O "$part" "$url" 2>&1 \
               | grep "^  HTTP/" | tail -1 | awk '{print $2}')

  if [[ "$http_code" == "404" ]]; then
    rm -f "$part"
    return 1
  fi

  # Verify and promote
  if gzip -t "$part" 2>/dev/null; then
    mv "$part" "$dest"
    return 0
  else
    echo "    [warn] $(basename "$dest") — incomplete gzip, kept as .part for next resume"
    return 0   # don't stop the loop — just retry next time
  fi
}

# ─── Google Cluster Trace 2019 ────────────────────────────────────────────────
if $DO_GOOGLE; then
  echo "==> Downloading Google Cluster Trace 2019 (instance_usage)"
  echo "    Cells: ${CELLS[*]}  |  max shards per cell: $MAX_SHARDS"
  BASE="https://storage.googleapis.com"
  total=0

  for cell in "${CELLS[@]}"; do
    cell_dir="$GOOGLE_DIR/cell_${cell}"
    mkdir -p "$cell_dir"
    echo ""
    echo "  --- Cell $cell ---"

    for (( i=0; i<MAX_SHARDS; i++ )); do
      shard=$(printf "%012d" "$i")
      fname="instance_usage-${shard}.json.gz"
      url="${BASE}/clusterdata_2019_${cell}/${fname}"

      download_resumable "$url" "$cell_dir/$fname" || {
        echo "    [end] No more shards in cell $cell (stopped at shard $i)"
        break
      }
      (( total++ )) || true
    done
  done

  echo ""
  echo "==> Google done. $total shard(s) across ${#CELLS[@]} cell(s)."
fi

# ─── Alibaba Cluster Trace 2018 ───────────────────────────────────────────────
if $DO_ALIBABA; then
  echo ""
  echo "==> Downloading Alibaba machine_usage (cluster-trace-v2018)"
  mkdir -p "$ALIBABA_DIR"
  ABASE="http://aliopentrace.oss-cn-beijing.aliyuncs.com/v2018Traces"
  TARBALL="$ALIBABA_DIR/machine_usage.tar.gz"
  CSV="$ALIBABA_DIR/machine_usage.csv"

  if [[ -f "$CSV" ]]; then
    echo "    [skip] machine_usage.csv — already extracted"
  else
    wget -c -q --show-progress --retry-connrefused --tries=5 --timeout=60 \
         -O "${TARBALL}.part" "${ABASE}/machine_usage.tar.gz"

    if gzip -t "${TARBALL}.part" 2>/dev/null; then
      mv "${TARBALL}.part" "$TARBALL"
      echo "    Extracting…"
      tar -xzf "$TARBALL" -C "$ALIBABA_DIR"
      # Flatten nested dir if tarball extracted into a subdir
      found=$(find "$ALIBABA_DIR" -name "machine_usage.csv" ! -path "$CSV" 2>/dev/null | head -1)
      if [[ -n "$found" ]]; then
        mv "$found" "$CSV"
      fi
      echo "    Extracted → $CSV"
    else
      echo "    [warn] Tarball incomplete — re-run to resume"
    fi
  fi
  echo "==> Alibaba done."
fi

echo ""
echo "Verify with:"
echo "  cd updated_research && python data_loader.py data/raw/google data/raw/alibaba"
