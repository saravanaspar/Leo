#!/usr/bin/env bash
set -euo pipefail

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
cd "$ROOT"

MODEL=${1:?usage: debug_replay.sh <model> <train-bytes> <train-index> [stories] [workers] [log]}
BYTES=${2:?usage: debug_replay.sh <model> <train-bytes> <train-index> [stories] [workers] [log]}
INDEX=${3:?usage: debug_replay.sh <model> <train-bytes> <train-index> [stories] [workers] [log]}
STORIES=${4:-128}
WORKERS=${5:-16}
LOG=${6:-"$ROOT/replay-debug.log"}
LEO=${LEO_BINARY:-"$ROOT/target/release/leo"}

[[ -x "$LEO" ]] || { echo "release Leo binary not found: $LEO" >&2; exit 2; }
[[ -f "$MODEL" ]] || { echo "model not found: $MODEL" >&2; exit 2; }
[[ -f "$BYTES" ]] || { echo "training bytes not found: $BYTES" >&2; exit 2; }
[[ -f "$INDEX" ]] || { echo "training index not found: $INDEX" >&2; exit 2; }
mkdir -p "$(dirname "$LOG")"

export LEO_REPLAY_DEBUG=${LEO_REPLAY_DEBUG:-1}
export LEO_REPLAY_DEBUG_SELECTION=${LEO_REPLAY_DEBUG_SELECTION:-1}
export LEO_REPLAY_DEBUG_RANGES=${LEO_REPLAY_DEBUG_RANGES:-1}
export LEO_REPLAY_DEBUG_SEGMENTS=${LEO_REPLAY_DEBUG_SEGMENTS:-1}
export LEO_REPLAY_DEBUG_SEGMENT_STRIDE=${LEO_REPLAY_DEBUG_SEGMENT_STRIDE:-1}
export LEO_CUDA_DEBUG=${LEO_CUDA_DEBUG:-1}
export LEO_CUDA_DEBUG_MEMORY=${LEO_CUDA_DEBUG_MEMORY:-1}
export LEO_CUDA_DEBUG_CHUNKS=${LEO_CUDA_DEBUG_CHUNKS:-1}
export LEO_CUDA_DEBUG_LAUNCHES=${LEO_CUDA_DEBUG_LAUNCHES:-1}
export LEO_CUDA_DEBUG_SYNC=${LEO_CUDA_DEBUG_SYNC:-1}
export LEO_CUDA_DEBUG_TRANSFERS=${LEO_CUDA_DEBUG_TRANSFERS:-1}
export LEO_CUDA_DEBUG_STATE=${LEO_CUDA_DEBUG_STATE:-1}
export LEO_CUDA_REPLAY_PROFILE=${LEO_CUDA_REPLAY_PROFILE:-1}
export LEO_CUDA_REPLAY_PROFILE_STRIDE=${LEO_CUDA_REPLAY_PROFILE_STRIDE:-8}
export LEO_CUDA_REPLAY_COOPERATIVE=${LEO_CUDA_REPLAY_COOPERATIVE:-1}
export LEO_MULTI_GPU=${LEO_MULTI_GPU:-0}

{
  echo "replay debug start"
  echo "model=$MODEL"
  echo "bytes=$BYTES"
  echo "index=$INDEX"
  echo "stories=$STORIES"
  echo "workers=$WORKERS"
  echo "profile_stride=$LEO_CUDA_REPLAY_PROFILE_STRIDE"
  echo "replay_blocks=${LEO_CUDA_REPLAY_BLOCKS:-auto}"
  echo "frozen_blocks=${LEO_CUDA_FROZEN_BLOCKS:-auto}"
} | tee "$LOG"

"$LEO" benchmark \
  --train \
  --model "$MODEL" \
  --bytes "$BYTES" \
  --index "$INDEX" \
  --stories "$STORIES" \
  --workers "$WORKERS" \
  --backend gpu \
  2>&1 | tee -a "$LOG"
