#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 2 || $# -gt 9 ]]; then
  echo "usage: $0 <prepared-data-directory> <run-directory> [config] [passes] [max-stories] [workers] [max-input-bytes] [backend] [validation-stories]" >&2
  exit 2
fi

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
DATA_DIR=$1
RUN_DIR=$2
CONFIG=${3:-"$ROOT/configs/tinystories.toml"}
PASSES=${4:-}
MAX_STORIES=${5:-}
WORKERS=${6:-}
MAX_INPUT_BYTES=${7:-}
BACKEND=${8:-${LEO_BACKEND:-auto}}
VALIDATION_STORIES=${9:-${LEO_VALIDATION_STORIES:-100}}
MODEL="$RUN_DIR/leo.pscls"

[[ "$CONFIG" = /* ]] || CONFIG="$ROOT/$CONFIG"
for file in tinystories.train.bytes tinystories.train.idx tinystories.valid.bytes tinystories.valid.idx; do
  [[ -f "$DATA_DIR/$file" ]] || { echo "missing $DATA_DIR/$file" >&2; exit 2; }
done
mkdir -p "$RUN_DIR"

cargo build --release -p leo-cli
LEO="$ROOT/target/release/leo"
if [[ ! -f "$MODEL" ]]; then
  "$LEO" init --config "$CONFIG" --output "$MODEL"
fi

arguments=(
  --model "$MODEL"
  --train-bytes "$DATA_DIR/tinystories.train.bytes"
  --train-index "$DATA_DIR/tinystories.train.idx"
  --valid-bytes "$DATA_DIR/tinystories.valid.bytes"
  --valid-index "$DATA_DIR/tinystories.valid.idx"
  --backend "$BACKEND"
)
[[ -n "$PASSES" ]] && arguments+=(--passes "$PASSES")
[[ -n "$MAX_STORIES" ]] && arguments+=(--max-stories "$MAX_STORIES")
[[ -n "$WORKERS" ]] && arguments+=(--workers "$WORKERS")
[[ -n "$MAX_INPUT_BYTES" ]] && arguments+=(--max-bytes "$MAX_INPUT_BYTES")
[[ -n "$VALIDATION_STORIES" ]] && arguments+=(--validation-stories "$VALIDATION_STORIES")
"$LEO" train "${arguments[@]}" 2>&1 | tee "$RUN_DIR/train.jsonl"
