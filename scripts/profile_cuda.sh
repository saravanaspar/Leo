#!/usr/bin/env bash
set -euo pipefail

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
cd "$ROOT"

if [[ $# -lt 3 || $# -gt 6 ]]; then
  echo "usage: $0 <model.pscls> <train.bytes> <train.idx> [workers=8] [stories=8] [report-prefix=leo-cuda-profile]" >&2
  exit 2
fi

MODEL=$1
BYTES=$2
INDEX=$3
WORKERS=${4:-8}
STORIES=${5:-8}
REPORT=${6:-leo-cuda-profile}

command -v ncu >/dev/null 2>&1 || {
  echo "Nsight Compute CLI (ncu) is required for hardware-counter profiling" >&2
  exit 2
}
[[ -x target/release/leo ]] || cargo build --release -p leo-cli

# Runtime cuda_profile JSON supplies end-to-end H2D/compute/D2H/host-wait
# timings. Nsight Compute complements it with actual SM/DRAM/warp/instruction
# counters (including atomics and branch behavior where the GPU exposes them).
# --set full intentionally lets NVIDIA choose architecture-appropriate metric
# names instead of hard-coding counters that disappear across GPU generations.
# Keep the default probe deliberately tiny because Nsight Compute replays kernels
# to collect full hardware-counter sections; override workers/stories when needed.
ncu \
  --target-processes all \
  --set full \
  --import-source yes \
  --force-overwrite \
  --export "$REPORT" \
  target/release/leo benchmark \
    --train \
    --model "$MODEL" \
    --bytes "$BYTES" \
    --index "$INDEX" \
    --workers "$WORKERS" \
    --stories "$STORIES" \
    --backend gpu

ncu --import "${REPORT}.ncu-rep" --page details --csv > "${REPORT}.csv"

echo "Nsight Compute report: ${REPORT}.ncu-rep"
echo "Nsight Compute details CSV: ${REPORT}.csv"
echo "Inspect SpeedOfLight, MemoryWorkloadAnalysis, Occupancy, WarpStateStats, and InstructionStats sections for DRAM bandwidth, achieved occupancy, warp/branch behavior, instructions, and atomic activity. Runtime cuda_profile events complement these counters with H2D/D2H/host-wait timing."
