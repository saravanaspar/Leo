#!/usr/bin/env bash
set -euo pipefail

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
cd "$ROOT"

command -v cargo >/dev/null 2>&1 || { echo "cargo is required" >&2; exit 2; }
command -v python3 >/dev/null 2>&1 || { echo "python3 is required" >&2; exit 2; }
command -v nvidia-smi >/dev/null 2>&1 || { echo "nvidia-smi is required" >&2; exit 2; }
nvidia-smi -L

# The CUDA backend compiles through NVRTC at runtime. Validate that a toolkit
# header root is visible before an expensive build/probe starts.
CUDA_INCLUDE=""
for root in "${CUDA_HOME:-}" "${CUDA_PATH:-}" /usr/local/cuda /opt/cuda; do
  [[ -n "$root" && -f "$root/include/cooperative_groups.h" ]] || continue
  CUDA_INCLUDE="$root/include"
  break
done
[[ -n "$CUDA_INCLUDE" ]] || {
  echo "CUDA toolkit headers (cooperative_groups.h) were not found; set CUDA_HOME or CUDA_PATH" >&2
  exit 2
}

./scripts/check.sh
cargo build --release -p leo-cli
LEO="$ROOT/target/release/leo"

TMP=$(mktemp -d)
trap 'rm -rf "$TMP"' EXIT

# Keep the model/data tiny but provide enough equal-width logical batches to
# complete the online execution-plan search (at most 24 candidates x 2
# observations) while also exercising physical lane chunks smaller than the
# logical batch so H2D for group N+1 can overlap compute for group N. This turns CI into a real autotuner/cache execution test.
python3 - "$TMP/train.txt" "$TMP/valid.txt" <<'PY'
from pathlib import Path
import sys

train = Path(sys.argv[1])
valid = Path(sys.argv[2])
train.write_text(
    "\n\n".join(
        f"Story {i} has a small red kite and a quiet blue pond." for i in range(1024)
    )
    + "\n",
    encoding="utf-8",
)
valid.write_text(
    "\n\n".join(
        f"Validation story {i} has a cat on a warm window sill." for i in range(16)
    )
    + "\n",
    encoding="utf-8",
)
PY

python3 python/prepare_tinystories.py \
  --train-input "$TMP/train.txt" \
  --valid-input "$TMP/valid.txt" \
  --text-format paragraph \
  --output "$TMP/data" \
  --source-repository leo-gpu-ci \
  --source-revision v1.0.0

"$LEO" init --config configs/test.toml --output "$TMP/model.pscls"

# Numerical-reference gate. This also forces real NVRTC compilation, module
# load, device allocation, frozen execution, and training execution.
"$LEO" backend --backend gpu --model "$TMP/model.pscls" --json

# Workers > 1 deliberately exercises the shared-model story-batch path,
# cooperative fused wavefront when supported, dual-stream H2D/compute overlap,
# sparse apply graph, and the multidimensional online execution tuner.
BENCH_LOG="$TMP/gpu-benchmark.log"
LEO_MULTI_GPU=0 "$LEO" benchmark \
  --train \
  --model "$TMP/model.pscls" \
  --bytes "$TMP/data/tinystories.train.bytes" \
  --index "$TMP/data/tinystories.train.idx" \
  --stories 1024 \
  --workers 16 \
  --backend gpu 2>&1 | tee "$BENCH_LOG"

python3 - "$BENCH_LOG" <<'PY'
import json
from pathlib import Path
import sys

profiles = []
autotune = []
for raw in Path(sys.argv[1]).read_text(encoding="utf-8").splitlines():
    raw = raw.strip()
    if not raw.startswith("{"):
        continue
    try:
        event = json.loads(raw)
    except json.JSONDecodeError:
        continue
    if event.get("event") == "cuda_profile":
        profiles.append(event)
    elif event.get("event") == "cuda_autotune_complete":
        autotune.append(event)

if not profiles:
    raise SystemExit("GPU CI did not observe any cuda_profile telemetry events")
if not autotune:
    raise SystemExit("GPU CI did not complete an execution autotuning cycle")

required = {
    "h2d_ms",
    "compute_ms",
    "d2h_ms",
    "host_wait_ms",
    "h2d_gib_s",
    "d2h_gib_s",
    "theoretical_occupancy",
    "fused_launches",
    "direct_phase_launches",
    "graph_launches",
}
for index, profile in enumerate(profiles):
    missing = required.difference(profile)
    if missing:
        raise SystemExit(f"cuda_profile[{index}] missing fields: {sorted(missing)}")
    for field in ("h2d_ms", "compute_ms", "d2h_ms", "host_wait_ms"):
        if float(profile[field]) < 0.0:
            raise SystemExit(f"cuda_profile[{index}] has negative {field}")

if not any(
    int(profile["fused_launches"]) + int(profile["direct_phase_launches"]) > 0
    for profile in profiles
):
    raise SystemExit("GPU CI observed neither fused nor direct wavefront launches")
if not any(int(profile["graph_launches"]) > 0 for profile in profiles):
    raise SystemExit("GPU CI did not exercise CUDA Graph replay for sparse apply/reset")

final = autotune[-1]
for field in (
    "lane_chunk",
    "sparse_blocks",
    "sparse_threads",
    "fused_blocks",
    "fused_threads",
    "candidate_count",
):
    if int(final[field]) < 1:
        raise SystemExit(f"invalid autotuned {field}: {final[field]}")
print(
    "GPU telemetry/autotune OK:",
    f"profiles={len(profiles)}",
    f"candidates={final['candidate_count']}",
    f"lane_chunk={final['lane_chunk']}",
    f"sparse={final['sparse_blocks']}x{final['sparse_threads']}",
    f"fused={final['fused_blocks']}x{final['fused_threads']}",
)
PY

if [[ -n "${LEO_CACHE_DIR:-}" ]]; then
  CACHE_ROOT="$LEO_CACHE_DIR/cuda"
elif [[ -n "${XDG_CACHE_HOME:-}" ]]; then
  CACHE_ROOT="$XDG_CACHE_HOME/leo/cuda"
elif [[ -n "${HOME:-}" ]]; then
  CACHE_ROOT="$HOME/.cache/leo/cuda"
else
  echo "no CUDA cache root is available" >&2
  exit 1
fi
[[ -d "$CACHE_ROOT" ]] || {
  echo "CUDA cache root was not created: $CACHE_ROOT" >&2
  exit 1
}
find "$CACHE_ROOT" -maxdepth 3 -type f -print
find "$CACHE_ROOT" -type f -path '*/profiles/*.toml' -print -quit | grep -q . || {
  echo "execution autotuner profile cache was not persisted" >&2
  exit 1
}
find "$CACHE_ROOT" -type f ! -path '*/profiles/*' -print -quit | grep -q . || {
  echo "NVRTC/PTX cache artifact was not persisted" >&2
  exit 1
}

# A short second process validates that the cached artifacts are readable by a
# fresh Leo process rather than only by the process that created them.
LEO_MULTI_GPU=0 "$LEO" benchmark \
  --train \
  --model "$TMP/model.pscls" \
  --bytes "$TMP/data/tinystories.train.bytes" \
  --index "$TMP/data/tinystories.train.idx" \
  --stories 32 \
  --workers 16 \
  --backend gpu
