# Leo training and CUDA debugging

Leo keeps debugging separate from the learning contract. The options below do **not** change the configured replay fraction, story/range order, logical `--workers` batch, canonical mean barrier, or FP32 learning equations. Heavy CUDA phase profiling is intentionally opt-in because instrumentation adds barriers and timing overhead.

## Always-on replay accounting

Training/benchmark JSON now reports the hidden work needed to reconstruct replay state:

- `replay_steps`: supervised replay target steps.
- `replay_prefix_steps`: frozen prefix steps executed to rebuild the exact transient state before selected replay ranges.
- `replay_execution_steps`: `replay_steps + replay_prefix_steps`.
- `execution_steps_with_prefix`: reported training steps plus hidden replay-prefix steps.
- `execution_steps_per_second`: total visible + hidden state-transition work per second.
- `replay_step_fraction`: supervised replay steps divided by reported training steps.
- `replay_prefix_step_fraction`: hidden prefix steps divided by base training targets.

These counters are cheap and remain enabled in normal production runs.

## High-level replay diagnostics

Environment variables:

| Variable | Effect |
| --- | --- |
| `LEO_REPLAY_DEBUG=1` | Story/batch replay timing summaries plus selection summaries. |
| `LEO_REPLAY_DEBUG_SELECTION=1` | Selection diagnostics. Host selection reports loss statistics; device-native selection reports selected range geometry without downloading per-step losses. |
| `LEO_REPLAY_DEBUG_RANGES=1` | Emits every selected range before execution. |
| `LEO_REPLAY_DEBUG_SEGMENTS=1` | Emits detailed timing for sampled replay segments. |
| `LEO_REPLAY_DEBUG_SEGMENT_STRIDE=N` | Emit one segment event every `N` segments; default `1`. |
| `LEO_REPLAY_DEBUG_TIMING=1` | Emits story/batch timing summaries without requiring selection/range/segment verbosity. |

Events:

- `replay_selection_debug` - host replay selector summary, including loss-distribution statistics.
- `replay_device_selection_debug` - device-postprocess selector summary, including selected segments/targets and estimated prefix work while per-step losses remain GPU-resident.
- `replay_range_debug`
- `replay_segment_debug`
- `replay_story_debug`
- `replay_batch_debug`

When normal single-GPU device story postprocessing is active, replay ranges are already selected on the GPU. In that path Leo emits `replay_device_selection_debug` instead of fabricating host-only loss statistics or downloading per-step losses solely for diagnostics. `replay_batch_debug` is emitted for both host-selected and device-selected replay execution.

Segment/story/batch timing separates `begin_document`, prefix construction, frozen-prefix execution, target construction, supervised target execution, and final reset.

## CUDA host/runtime diagnostics

| Variable | Effect |
| --- | --- |
| `LEO_CUDA_DEBUG=1` | Runtime/device summary and allocation summary. |
| `LEO_CUDA_DEBUG_MEMORY=1` | Runtime and lane allocation bytes. |
| `LEO_CUDA_DEBUG_CHUNKS=1` | Per replay-prefix and replay-target chunk timings. |
| `LEO_CUDA_DEBUG_LAUNCHES=1` | Kernel name, grid, threads, step count and profiling mode for replay launches. |
| `LEO_CUDA_DEBUG_SYNC=1` | Enables chunk timing around launch+synchronization. |
| `LEO_CUDA_DEBUG_TRANSFERS=1` | Enables host/device upload/download timing fields. |
| `LEO_CUDA_DEBUG_STATE=1` | Includes tick/revision state in chunk events. |

Events include:

- `cuda_debug_runtime` — device ordinal/name/UUID/PCI bus, driver version, compute capability, SM count, VRAM, cooperative-launch capability, replay/frozen grid capacities, active profiling strides and execution overrides.
- `cuda_debug_memory`
- `cuda_debug_launch`
- `cuda_debug_chunk`

`cuda_debug_chunk` is host-visible timing. It is useful for finding synchronization and transfer stalls but is not a substitute for device phase profiling.

## Sampled replay CUDA phase profiler

Replay has two different GPU workloads, so each can be sampled independently:

| Variable | Effect |
| --- | --- |
| `LEO_CUDA_REPLAY_PROFILE=1` | Profile both supervised replay targets and frozen replay prefixes. |
| `LEO_CUDA_REPLAY_PROFILE_TARGET=1` | Profile supervised replay-target kernels only. |
| `LEO_CUDA_REPLAY_PROFILE_PREFIX=1` | Profile frozen-prefix kernels only. |
| `LEO_CUDA_REPLAY_PROFILE_STRIDE=N` | Profile one eligible launch every `N`; default `1`. Use `8` or higher for larger reproductions. |

Events:

- `cuda_replay_kernel_profile`
- `cuda_replay_kernel_profile_skipped`
- `cuda_frozen_kernel_profile`
- `cuda_frozen_kernel_profile_skipped`

The target profile separates:

1. pre/event delivery/input/context
2. model-block selection
3. global winner selection
4. surrogate cache
5. recurrent eligibility
6. input eligibility
7. post/emit
8. forward reduction
9. learning signals
10. output update
11. context update
12. recurrent-weight update
13. input-weight update
14. inhibitory homeostasis
15. threshold homeostasis
16. record capture

The frozen-prefix profile separates pre, model-block selection, global selection, and post/emit.

Instrumented kernels are launched only when their cooperative occupancy can support the **same requested grid width** as the normal kernel. Leo emits an explicit `*_profile_skipped` event rather than silently shrinking the grid and reporting incomparable numbers.

Do not use a phase-profiled run as the headline throughput benchmark. Re-run with profiling disabled after identifying the bottleneck.

## Replay execution A/B and grid experiments

These controls change execution geometry only; they do not change replay selection or learning policy:

| Variable | Effect |
| --- | --- |
| `LEO_CUDA_REPLAY_COOPERATIVE=0` | Force the legacy one-block replay trainer for A/B regression comparison. Cooperative replay is enabled by default. |
| `LEO_CUDA_REPLAY_BLOCKS=N` | Clamp the cooperative supervised replay grid to `N` blocks, bounded by occupancy/model capacity. |
| `LEO_CUDA_FROZEN_BLOCKS=N` | Clamp the cooperative frozen-prefix grid to `N` blocks, bounded by occupancy/model capacity. |

These overrides are useful for diagnosing barrier-vs-parallel-work tradeoffs on a new GPU. Do not treat them as learning hyperparameters.

## Compiled CUDA resource and device-postprocess diagnostics

`LEO_CUDA_DEBUG=1` now emits `cuda_kernel_resources` for the grouped persistent
trainer, cooperative replay trainer, and frozen-prefix kernel. Each event reports
compiled registers/thread, static and maximum dynamic shared memory, local bytes
per thread, maximum threads/block, cooperative grid capacity, resident
threads/SM, and theoretical thread occupancy. This is the preferred first check
when a source change unexpectedly changes P100 occupancy.

Normal single-GPU persistent training uploads raw story bytes and builds the exact
BEGIN/byte/END step descriptors on-device, then keeps per-step fast records on-device
and returns compact per-story summaries plus exact replay ranges. Set
`LEO_CUDA_DEVICE_STORY_STEPS=0` to restore host-built step descriptors while retaining
device postprocessing. Set `LEO_CUDA_DEVICE_STORY_POSTPROCESS=0` to force the historical per-step record
download and CPU replay-range selector. This flag changes execution placement
only; the GPU acceptance gate compares the default path with the legacy exact
path by complete training-state SHA-256. With `LEO_CUDA_DEBUG_LAUNCHES=1`, the
default path emits `scope="device_story_steps"` and
`scope="device_story_postprocess"` when both device-resident stages are actually used.

## Production shared-story profiler

The logical story-batch path has a sampled profiler that stays on the production
persistent/grouped execution path instead of switching to the legacy fused path:

- `LEO_CUDA_PHASE_PROFILE=1`
- `LEO_CUDA_PHASE_PROFILE_STRIDE=N`
- events `cuda_phase_profile` / `cuda_phase_profile_skipped`

When the grouped profiled kernel cannot sustain the same cooperative grid width, Leo leaves production execution unchanged and emits `cuda_phase_profile_skipped` rather than shrinking the grid or switching kernels. The replay profiler above covers the separate single-story replay and frozen-prefix paths.

## Recommended debugging ladder

For a suspicious training slowdown, use the least invasive layer first:

1. Run the same benchmark normally and record `training_benchmark`, including `replay_prefix_steps`.
2. Enable `LEO_REPLAY_DEBUG=1` to separate selection/prefix/target host time.
3. Enable `LEO_CUDA_DEBUG_CHUNKS=1` and `LEO_CUDA_DEBUG_LAUNCHES=1` to locate host-visible CUDA stalls.
4. Enable `LEO_CUDA_REPLAY_PROFILE=1` with a sampling stride (for example `8`) on a short reproduction.
5. If geometry is suspicious, sweep `LEO_CUDA_REPLAY_BLOCKS` while keeping the model, data, workers, replay policy and cache identity fixed.
6. Use `LEO_CUDA_REPLAY_COOPERATIVE=0` once as a legacy A/B control.
7. Disable heavy profiling and rerun the clean benchmark before accepting a speed result.

## One-command replay diagnostic

```bash
bash ./scripts/debug_replay.sh \
  runs/truebench-128/leo.pscls \
  data/bench128/tinystories.train.bytes \
  data/bench128/tinystories.train.idx \
  128 \
  16 \
  runs/truebench-128/replay-debug.log
```

The script enables the detailed host diagnostics and samples CUDA replay/prefix phases every eight launches by default. Override any environment variable before invoking it if a quieter or denser trace is needed.
