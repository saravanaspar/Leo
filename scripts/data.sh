#!/usr/bin/env bash
set -euo pipefail

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
RAW_DIR=${1:-"$ROOT/data/raw/tinystories"}
PREPARED_DIR=${2:-"$ROOT/data/prepared"}
TRAIN_LIMIT=${3:-}
VALID_LIMIT=${4:-}
TRAIN_BYTE_LIMIT=${5:-}
VALID_BYTE_LIMIT=${6:-}
REVISION=5485261731eaac25dd8e5ebbc3839d0a9870b185
REPOSITORY=roneneldan/TinyStories

mkdir -p "$RAW_DIR" "$PREPARED_DIR"
if [[ ! -f "$RAW_DIR/TinyStories-train.txt" || ! -f "$RAW_DIR/TinyStories-valid.txt" ]]; then
  if ! command -v hf >/dev/null 2>&1; then
    echo "TinyStories files are missing and the hf CLI is unavailable." >&2
    echo "Install requirements.txt or place TinyStories-train.txt and TinyStories-valid.txt in $RAW_DIR" >&2
    exit 2
  fi
  hf download "$REPOSITORY" \
    TinyStories-train.txt TinyStories-valid.txt \
    --repo-type dataset \
    --revision "$REVISION" \
    --local-dir "$RAW_DIR"
fi

(
  cd "$RAW_DIR"
  cat <<'CHECKSUMS' | sha256sum -c -
c5cf5e22ff13614e830afbe61a99fbcbe8bcb7dd72252b989fa1117a368d401f  TinyStories-train.txt
94e431816c4cce81ff71e4408ff8d3bda9a42e8d2663986697c3954288cb38b4  TinyStories-valid.txt
CHECKSUMS
)

arguments=(
  --train-input "$RAW_DIR/TinyStories-train.txt"
  --valid-input "$RAW_DIR/TinyStories-valid.txt"
  --output "$PREPARED_DIR"
  --seed 1337
  --text-format delimited
  --source-repository "$REPOSITORY"
  --source-revision "$REVISION"
  --train-source-sha256 c5cf5e22ff13614e830afbe61a99fbcbe8bcb7dd72252b989fa1117a368d401f
  --valid-source-sha256 94e431816c4cce81ff71e4408ff8d3bda9a42e8d2663986697c3954288cb38b4
)
[[ -n "$TRAIN_LIMIT" ]] && arguments+=(--train-limit "$TRAIN_LIMIT")
[[ -n "$VALID_LIMIT" ]] && arguments+=(--valid-limit "$VALID_LIMIT")
[[ -n "$TRAIN_BYTE_LIMIT" ]] && arguments+=(--train-byte-limit "$TRAIN_BYTE_LIMIT")
[[ -n "$VALID_BYTE_LIMIT" ]] && arguments+=(--valid-byte-limit "$VALID_BYTE_LIMIT")
python3 "$ROOT/python/prepare_tinystories.py" "${arguments[@]}"
