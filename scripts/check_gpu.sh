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

bash ./scripts/check.sh
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
  --source-revision v1.0.1

"$LEO" init --config configs/test.toml --output "$TMP/model.pscls"

# Isolate CUDA/PTX and execution-plan cache state so the gate proves both
# cold-start persistence and fresh-process resume deterministically.
export LEO_CACHE_DIR="$TMP/cache"

# Numerical-reference gate. Besides the single-step probe this now executes the
# actual multi-story production fast path against the CPU reference with replay
# disabled and with the locked v1 30% replay policy.
BACKEND_LOG="$TMP/gpu-backend.log"
"$LEO" backend --backend gpu --model "$TMP/model.pscls" --json 2>&1 | tee "$BACKEND_LOG"

python3 - "$BACKEND_LOG" <<'PY'
import json
from pathlib import Path
import sys

conformance = []
for raw in Path(sys.argv[1]).read_text(encoding="utf-8").splitlines():
    raw = raw.strip()
    if not raw.startswith("{"):
        continue
    try:
        event = json.loads(raw)
    except json.JSONDecodeError:
        continue
    if event.get("event") == "cuda_story_batch_conformance":
        conformance.append(event)

by_fraction = {round(float(event["replay_fraction"]), 2): event for event in conformance}
if set(by_fraction) != {0.0, 0.3}:
    raise SystemExit(
        f"expected CPU/GPU story-batch conformance at replay 0 and 0.30, got {sorted(by_fraction)}"
    )
for fraction, event in sorted(by_fraction.items()):
    if event.get("ready") is not True:
        raise SystemExit(f"story-batch conformance was not ready for replay={fraction}")
    if int(event.get("workers", 0)) != 4:
        raise SystemExit(f"story-batch conformance used unexpected worker count: {event}")
    if float(event.get("max_parameter_delta", 1.0)) > 3.0e-3:
        raise SystemExit(f"parameter parity exceeded tolerance for replay={fraction}: {event}")
    if float(event.get("mean_loss_delta", 1.0)) > 3.0e-3:
        raise SystemExit(f"loss parity exceeded tolerance for replay={fraction}: {event}")
if int(by_fraction[0.0].get("replay_steps", -1)) != 0:
    raise SystemExit("replay-disabled conformance unexpectedly executed replay steps")
if int(by_fraction[0.3].get("replay_steps", 0)) <= 0:
    raise SystemExit("30% replay conformance did not execute any replay steps")
print("CPU/GPU logical-batch conformance OK at replay=0 and replay=0.30")
PY

# A deliberately short first process leaves the 16-lane tuner incomplete. A
# second fresh process must restore that observation instead of repeating it.
PARTIAL_ONE="$TMP/gpu-autotune-partial-1.log"
PARTIAL_TWO="$TMP/gpu-autotune-partial-2.log"
LEO_MULTI_GPU=0 "$LEO" benchmark \
  --train \
  --model "$TMP/model.pscls" \
  --bytes "$TMP/data/tinystories.train.bytes" \
  --index "$TMP/data/tinystories.train.idx" \
  --stories 16 \
  --workers 16 \
  --backend gpu 2>&1 | tee "$PARTIAL_ONE"
LEO_MULTI_GPU=0 "$LEO" benchmark \
  --train \
  --model "$TMP/model.pscls" \
  --bytes "$TMP/data/tinystories.train.bytes" \
  --index "$TMP/data/tinystories.train.idx" \
  --stories 16 \
  --workers 16 \
  --backend gpu 2>&1 | tee "$PARTIAL_TWO"
grep -q '"event":"cuda_autotune_resumed"' "$PARTIAL_TWO" || {
  echo "fresh GPU process did not resume incomplete autotuning work" >&2
  exit 1
}

# Enough equal-width batches remain to finish the bounded multidimensional
# search after the two short resume probes. Workers > 1 exercises the exact
# lane-private story trajectories, fused/direct wavefronts, transfer overlap,
# and batch-end sparse canonical merge.
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

# Exact v1 batches no longer apply a canonical mean after every byte, so the
# old sparse-apply/reset graph is not expected to launch in this path. Keep the
# telemetry field for compatibility/diagnostics, but do not require activity.
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

CACHE_ROOT="$LEO_CACHE_DIR/cuda"
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

# Sample the internal phase profiler on the production launch geometry. The
# profiled kernel may have lower cooperative occupancy than the normal kernel;
# in that case Leo must run the normal kernel and report an explicit skip rather
# than silently profiling a narrower grid.
PHASE_LOG="$TMP/gpu-phase-profile.log"
LEO_CUDA_PHASE_PROFILE=1 LEO_CUDA_PHASE_PROFILE_STRIDE=1 LEO_MULTI_GPU=0 "$LEO" benchmark \
  --train \
  --model "$TMP/model.pscls" \
  --bytes "$TMP/data/tinystories.train.bytes" \
  --index "$TMP/data/tinystories.train.idx" \
  --stories 16 \
  --workers 16 \
  --backend gpu 2>&1 | tee "$PHASE_LOG"
python3 - "$PHASE_LOG" <<'PY'
import json
from pathlib import Path
import sys

phase_events = []
for raw in Path(sys.argv[1]).read_text(encoding="utf-8").splitlines():
    raw = raw.strip()
    if not raw.startswith("{"):
        continue
    try:
        event = json.loads(raw)
    except json.JSONDecodeError:
        continue
    if event.get("event") in {"cuda_phase_profile", "cuda_phase_profile_skipped"}:
        phase_events.append(event)
if not phase_events:
    raise SystemExit("phase-profiler GPU run emitted neither a profile nor an explicit skip")
for event in phase_events:
    if event.get("event") == "cuda_phase_profile":
        normal = int(event.get("normal_capacity_blocks", 0))
        profiled = int(event.get("profiled_capacity_blocks", 0))
        sampled_grid = int(event.get("sampled_grid_blocks_max", 0))
        if sampled_grid <= 0 or normal < sampled_grid or profiled < sampled_grid:
            raise SystemExit(f"phase profile used non-comparable launch geometry: {event}")
print("CUDA phase-profiler geometry gate OK")
PY

# Exercise replay-specific diagnostics on a short real GPU path. Profiling is
# sampled only in this diagnostic process and must either use production-width
# geometry or emit an explicit skip. The benchmark itself must expose hidden
# frozen-prefix work even when diagnostics are disabled elsewhere.
REPLAY_DEBUG_LOG="$TMP/gpu-replay-debug.log"
LEO_REPLAY_DEBUG=1 \
LEO_REPLAY_DEBUG_SELECTION=1 \
LEO_REPLAY_DEBUG_SEGMENTS=1 \
LEO_REPLAY_DEBUG_SEGMENT_STRIDE=8 \
LEO_CUDA_DEBUG=1 \
LEO_CUDA_DEBUG_CHUNKS=1 \
LEO_CUDA_DEBUG_LAUNCHES=1 \
LEO_CUDA_REPLAY_PROFILE=1 \
LEO_CUDA_REPLAY_PROFILE_STRIDE=1 \
LEO_MULTI_GPU=0 "$LEO" benchmark \
  --train \
  --model "$TMP/model.pscls" \
  --bytes "$TMP/data/tinystories.train.bytes" \
  --index "$TMP/data/tinystories.train.idx" \
  --stories 16 \
  --workers 16 \
  --backend gpu 2>&1 | tee "$REPLAY_DEBUG_LOG"

python3 - "$REPLAY_DEBUG_LOG" <<'PY'
import json
from pathlib import Path
import sys

events = []
for raw in Path(sys.argv[1]).read_text(encoding="utf-8").splitlines():
    raw = raw.strip()
    if not raw.startswith("{"):
        continue
    try:
        events.append(json.loads(raw))
    except json.JSONDecodeError:
        continue

names = [event.get("event") for event in events]
for required in (
    "replay_selection_debug",
    "replay_batch_debug",
    "cuda_debug_runtime",
    "cuda_debug_launch",
    "cuda_debug_chunk",
):
    if required not in names:
        raise SystemExit(f"replay GPU diagnostic run did not emit {required}")

benchmarks = [event for event in events if event.get("event") == "training_benchmark"]
if not benchmarks:
    raise SystemExit("replay GPU diagnostic run did not emit training_benchmark")
benchmark = benchmarks[-1]
for field in ("replay_prefix_steps", "replay_execution_steps"):
    if field not in benchmark:
        raise SystemExit(f"training_benchmark missing replay accounting field {field}")
if int(benchmark.get("replay_steps", 0)) <= 0:
    raise SystemExit("replay GPU diagnostic run did not execute replay targets")
if int(benchmark.get("replay_execution_steps", 0)) < int(benchmark["replay_steps"]):
    raise SystemExit("replay execution accounting is smaller than supervised replay work")

target_profile = {
    "cuda_replay_kernel_profile",
    "cuda_replay_kernel_profile_skipped",
}
if not target_profile.intersection(names):
    raise SystemExit("replay target profiler emitted neither a sample nor an explicit skip")

if int(benchmark.get("replay_prefix_steps", 0)) > 0:
    prefix_profile = {
        "cuda_frozen_kernel_profile",
        "cuda_frozen_kernel_profile_skipped",
    }
    if not prefix_profile.intersection(names):
        raise SystemExit("replay prefix work existed but prefix profiler emitted neither a sample nor a skip")

print(
    "CUDA replay diagnostics OK:",
    f"replay_steps={benchmark['replay_steps']}",
    f"prefix_steps={benchmark['replay_prefix_steps']}",
)
PY

# A final short fresh process validates complete cache reuse rather than only
# the in-memory state of the process that completed tuning.
LEO_MULTI_GPU=0 "$LEO" benchmark \
  --train \
  --model "$TMP/model.pscls" \
  --bytes "$TMP/data/tinystories.train.bytes" \
  --index "$TMP/data/tinystories.train.idx" \
  --stories 32 \
  --workers 16 \
  --backend gpu
