#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 2 || $# -gt 5 ]]; then
  echo "usage: $0 <prepared-data-directory> <run-directory> [valid-stories] [train-stories] [backend]" >&2
  exit 2
fi

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
DATA_DIR=$1
RUN_DIR=$2
VALID_STORIES=${3:-100}
TRAIN_STORIES=${4:-}
BACKEND=${5:-${LEO_BACKEND:-auto}}
GENERATION_STORIES=${LEO_GENERATION_STORIES:-4}
GENERATION_BYTES=${LEO_GENERATION_BYTES:-128}
MODEL="$RUN_DIR/leo.pscls"
LEO="$ROOT/target/release/leo"

[[ -x "$LEO" ]] || cargo build --release -p leo-cli
[[ -f "$MODEL" ]] || { echo "missing model: $MODEL" >&2; exit 2; }

leo_arguments=(
  --model "$MODEL"
  --bytes "$DATA_DIR/tinystories.valid.bytes"
  --index "$DATA_DIR/tinystories.valid.idx"
  --stories "$VALID_STORIES"
  --train-bytes "$DATA_DIR/tinystories.train.bytes"
  --train-index "$DATA_DIR/tinystories.train.idx"
  --generation-stories "$GENERATION_STORIES"
  --max-bytes "$GENERATION_BYTES"
  --prompt "Once upon a time"
  --backend "$BACKEND"
)
[[ -n "$TRAIN_STORIES" ]] && leo_arguments+=(--train-stories "$TRAIN_STORIES")
"$LEO" eval "${leo_arguments[@]}" 2>&1 | tee "$RUN_DIR/eval.jsonl"

"$LEO" prompt \
  --model "$MODEL" \
  --text "Once upon a time" \
  --max-bytes "$GENERATION_BYTES" \
  --temperature 0.8 \
  --backend "$BACKEND" \
  2>&1 | tee "$RUN_DIR/story.txt"

"$LEO" benchmark \
  --model "$MODEL" \
  --bytes "$DATA_DIR/tinystories.valid.bytes" \
  --index "$DATA_DIR/tinystories.valid.idx" \
  --stories "$VALID_STORIES" \
  --backend "$BACKEND" \
  2>&1 | tee "$RUN_DIR/benchmark.jsonl"

baseline_arguments=(
  --train-bytes "$DATA_DIR/tinystories.train.bytes"
  --train-index "$DATA_DIR/tinystories.train.idx"
  --valid-bytes "$DATA_DIR/tinystories.valid.bytes"
  --valid-index "$DATA_DIR/tinystories.valid.idx"
  --valid-stories "$VALID_STORIES"
  --leo-eval "$RUN_DIR/eval.jsonl"
  --require-pass
)
[[ -n "$TRAIN_STORIES" ]] && baseline_arguments+=(--train-stories "$TRAIN_STORIES")
[[ -f "$RUN_DIR/train.jsonl" ]] && baseline_arguments+=(--train-log "$RUN_DIR/train.jsonl")
python3 "$ROOT/python/baselines.py" "${baseline_arguments[@]}" \
  | tee "$RUN_DIR/baselines.json"
