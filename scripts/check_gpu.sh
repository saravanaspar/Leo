#!/usr/bin/env bash
set -euo pipefail

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
cd "$ROOT"

# The GPU acceptance gate must be hermetic.  Clear inherited Leo execution,
# experimental, geometry, debug, and profiling switches before establishing
# each case explicitly below.  CUDA_VISIBLE_DEVICES is intentionally retained
# because callers use it to choose the physical devices under test.
unset LEO_MULTI_GPU
unset LEO_CUDA_DEVICE
unset LEO_REPLAY_STREAMING
unset LEO_MULTI_GPU_PARALLEL_REPLAY
unset LEO_CUDA_REPLAY_COOPERATIVE
unset LEO_CUDA_SHARED_PERSISTENT
unset LEO_CUDA_SHARED_GROUPED
unset LEO_CUDA_DEVICE_BATCH_MERGE
unset LEO_CUDA_DEVICE_STORY_STEPS
unset LEO_CUDA_DEVICE_STORY_POSTPROCESS
unset LEO_CUDA_FULL_STEP_METRICS
unset LEO_CUDA_REPLAY_BLOCKS
unset LEO_CUDA_FROZEN_BLOCKS
while IFS='=' read -r name _; do
  case "$name" in
    LEO_CUDA_DEBUG*|LEO_CUDA_REPLAY_PROFILE*|LEO_CUDA_PHASE_PROFILE*|LEO_REPLAY_DEBUG*)
      unset "$name"
      ;;
  esac
done < <(env)

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

LEO_RELEASE_VERSION=$(python3 scripts/release.py current)

python3 python/prepare_tinystories.py \
  --train-input "$TMP/train.txt" \
  --valid-input "$TMP/valid.txt" \
  --text-format paragraph \
  --output "$TMP/data" \
  --source-repository leo-gpu-ci \
  --source-revision "v$LEO_RELEASE_VERSION"

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
    if event.get("scope") != "shared_story_batch_production":
        raise SystemExit(f"phase profiler did not report production shared-story scope: {event}")
    if event.get("event") == "cuda_phase_profile":
        if event.get("profiled_kernel") != "leo_shared_wavefront_persistent_grouped_profiled":
            raise SystemExit(f"phase profiler sampled the wrong production kernel: {event}")
        normal = int(event.get("normal_capacity_blocks", 0))
        profiled = int(event.get("profiled_capacity_blocks", 0))
        sampled_grid = int(event.get("sampled_grid_blocks_max", 0))
        if sampled_grid <= 0 or normal < sampled_grid or profiled < sampled_grid:
            raise SystemExit(f"phase profile used non-comparable launch geometry: {event}")
print("CUDA phase-profiler production-geometry gate OK")
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
    "replay_batch_debug",
    "cuda_debug_runtime",
    "cuda_debug_launch",
    "cuda_debug_chunk",
):
    if required not in names:
        raise SystemExit(f"replay GPU diagnostic run did not emit {required}")

device_selection = [
    event for event in events if event.get("event") == "replay_device_selection_debug"
]
if not device_selection:
    raise SystemExit("replay GPU diagnostic run did not emit replay_device_selection_debug")
if any(event.get("selection_source") != "device_postprocess" for event in device_selection):
    raise SystemExit("device replay selection diagnostic reported a non-device selector")
if not any(int(event.get("selected_targets", 0)) > 0 for event in device_selection):
    raise SystemExit("device replay selection diagnostics reported no selected replay targets")
if "replay_selection_debug" in names:
    raise SystemExit("optimized replay diagnostic unexpectedly fell back to host replay selection")

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
    f"device_selection_events={len(device_selection)}",
)
PY

# The production shared trainer is now chunk-persistent by default. Prove that
# the launch/metrics optimization is execution-only by comparing the complete
# training-state digest against the legacy per-step shared wavefront from the
# exact same starting model and story sequence.
PERSISTENT_LOG="$TMP/gpu-persistent-shared.log"
LEGACY_LOG="$TMP/gpu-legacy-shared.log"
LEO_CUDA_DEBUG_LAUNCHES=1 \
LEO_MULTI_GPU=0 "$LEO" benchmark \
  --train \
  --model "$TMP/model.pscls" \
  --bytes "$TMP/data/tinystories.train.bytes" \
  --index "$TMP/data/tinystories.train.idx" \
  --stories 32 \
  --workers 16 \
  --backend gpu 2>&1 | tee "$PERSISTENT_LOG"
LEO_CUDA_SHARED_PERSISTENT=0 \
LEO_CUDA_SHARED_GROUPED=0 \
LEO_CUDA_DEVICE_BATCH_MERGE=0 \
LEO_CUDA_DEVICE_STORY_STEPS=0 \
LEO_CUDA_DEVICE_STORY_POSTPROCESS=0 \
LEO_MULTI_GPU=0 "$LEO" benchmark \
  --train \
  --model "$TMP/model.pscls" \
  --bytes "$TMP/data/tinystories.train.bytes" \
  --index "$TMP/data/tinystories.train.idx" \
  --stories 32 \
  --workers 16 \
  --backend gpu 2>&1 | tee "$LEGACY_LOG"
python3 - "$PERSISTENT_LOG" "$LEGACY_LOG" <<'PY'
import json
from pathlib import Path
import sys

def final_benchmark(path):
    events = []
    for raw in Path(path).read_text(encoding="utf-8").splitlines():
        raw = raw.strip()
        if not raw.startswith("{"):
            continue
        try:
            event = json.loads(raw)
        except json.JSONDecodeError:
            continue
        if event.get("event") == "training_benchmark":
            events.append(event)
    if not events:
        raise SystemExit(f"no training_benchmark event in {path}")
    return events[-1]

persistent_path = Path(sys.argv[1])
legacy_path = Path(sys.argv[2])
persistent = final_benchmark(persistent_path)
legacy = final_benchmark(legacy_path)

observed_kernels = []
device_batch_merge_observed = False
device_story_steps_observed = False
device_story_postprocess_observed = False
for raw in persistent_path.read_text(encoding="utf-8").splitlines():
    raw = raw.strip()
    if not raw.startswith("{"):
        continue
    try:
        event = json.loads(raw)
    except json.JSONDecodeError:
        continue
    if event.get("event") != "cuda_debug_launch":
        continue
    if event.get("scope") == "shared_story_batch":
        kernel = event.get("kernel")
        if kernel and kernel not in observed_kernels:
            observed_kernels.append(kernel)
    elif (
        event.get("scope") == "device_batch_merge"
        and event.get("kernel") == "leo_merge_shared_lane_fixed_parameters"
    ):
        device_batch_merge_observed = True
    elif (
        event.get("scope") == "device_story_steps"
        and event.get("kernel") == "leo_build_shared_story_steps"
    ):
        device_story_steps_observed = True
    elif (
        event.get("scope") == "device_story_postprocess"
        and event.get("kernel") == "leo_postprocess_shared_story_records"
    ):
        device_story_postprocess_observed = True

for event, label in ((persistent, "optimized"), (legacy, "legacy")):
    if not event.get("training_state_sha256"):
        raise SystemExit(f"{label} benchmark missing training_state_sha256")
    if round(float(event.get("replay_fraction", -1.0)), 2) != 0.30:
        raise SystemExit(f"{label} benchmark did not preserve 30% replay: {event}")

if persistent["training_state_sha256"] != legacy["training_state_sha256"]:
    raise SystemExit(
        "persistent shared CUDA path changed final training state: "
        f"persistent={persistent['training_state_sha256']} "
        f"legacy={legacy['training_state_sha256']}"
    )
if not device_batch_merge_observed:
    raise SystemExit("optimized exact-state leg did not execute device batch merge")
if not device_story_steps_observed:
    raise SystemExit("optimized exact-state leg did not execute device story step builder")
if not device_story_postprocess_observed:
    raise SystemExit("optimized exact-state leg did not execute device story postprocess")
kernel_summary = ",".join(observed_kernels) if observed_kernels else "fallback/no-persistent-launch-observed"
print(
    "Optimized-vs-legacy CUDA exact-state gate OK:",
    persistent["training_state_sha256"],
    f"optimized_kernel={kernel_summary}",
    "device_batch_merge=true",
    "device_story_steps=true",
    "device_story_postprocess=true",
)
PY

# On hosts with at least two homogeneous GPUs, prove that physical placement is
# execution-only: 1-GPU and 2-GPU runs must finish with the same complete
# persistent training-state digest, not merely similar loss/counters.
if [[ -n "${CUDA_VISIBLE_DEVICES:-}" ]]; then
  if [[ "${CUDA_VISIBLE_DEVICES}" == "-1" ]]; then
    GPU_COUNT=0
  else
    GPU_COUNT=$(printf '%s\n' "${CUDA_VISIBLE_DEVICES}" | awk -F',' '{print NF}')
  fi
else
  GPU_COUNT=$(nvidia-smi --query-gpu=index --format=csv,noheader,nounits | sed '/^[[:space:]]*$/d' | wc -l)
fi
if [[ "$GPU_COUNT" -ge 2 ]]; then
  echo "Running exact 1-GPU vs 2-GPU final-state conformance gate"
  python3 scripts/benchmark_multi_gpu.py \
    --leo "$LEO" \
    --model "$TMP/model.pscls" \
    --bytes "$TMP/data/tinystories.train.bytes" \
    --index "$TMP/data/tinystories.train.idx" \
    --stories 32 \
    --workers 16 \
    --counts 1,2 \
    --max-attempts 8
else
  echo "Skipping multi-GPU final-state parity gate: only $GPU_COUNT GPU visible"
fi
