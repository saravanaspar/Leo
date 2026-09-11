# Leo scaling architecture

Leo's scaling work keeps one stable v1 learning contract while
allowing the execution layer to use more CUDA devices. The model, replay policy,
FP32 arithmetic, logical worker count, and canonical mean barrier are semantic
inputs; physical GPU placement is not. Multi-GPU production acceptance still
requires the 2+ GPU hardware gates described below.

## Data parallelism = logical story workers

Leo's story workers are the data-parallel samples. With `--workers 16` the
canonical logical batch always contains 16 independent story trajectories that
start from the same parameter revision.

| Visible GPUs | Typical placement for `--workers 16` |
| ---: | --- |
| 1 | 16 stories on GPU 0 |
| 2 | 8 + 8 |
| 4 | 4 + 4 + 4 + 4 |
| 8 | 2 stories per GPU |
| 16 | 1 story per GPU |

Placement is deterministic and contiguous but balances estimated story work
(target/byte count), not merely story count. Equal-width 16-story batches on 3
GPUs still become 6 + 5 + 5; variable-length batches may use different contiguous
ranges to reduce barrier stragglers without changing canonical story order.

Each device returns **one sparse delta per original story**. The coordinator
sorts device results by the original story position, flattens all 16 deltas, and
calls the same canonical `apply_mean_deltas` once. There is no nested
mean-of-device-means, so the denominator remains 16 independent of device count.
The globally merged sparse rows are packed once, committed to GPU 0, and
synchronized to every secondary canonical replica **and every resident story
lane**. This keeps lane revisions current between batches and avoids full-model
canonical-to-lane restoration on the next logical batch. Single-GPU native
story batching uses the same packed lane update after its canonical mean instead
of restoring every lane with a complete model copy.

Enable the path with:

```bash
CUDA_VISIBLE_DEVICES=0,1 LEO_MULTI_GPU=1 \
./target/release/leo train \
  --model runs/quality/my-run/leo.pscls \
  --train-bytes data/prepared/tinystories.train.bytes \
  --train-index data/prepared/tinystories.train.idx \
  --workers 16 \
  --backend gpu
```

`LEO_MULTI_GPU=1` uses all CUDA devices visible to the process up to the logical
worker count. The model is replicated on every participating device, like
ordinary data parallel training; VRAM is not pooled. Device 0 reuses Leo's
existing canonical CUDA runtime instead of allocating a second complete replica,
so enabling data parallelism does not double the coordinator/model allocation on
the first GPU. Only devices 1..N-1 allocate additional canonical replicas. Each
GPU also owns the private mutable story-lane state for the workers assigned to
that device, so practical VRAM is one canonical/replica image plus that device's
lane states; adding GPUs distributes those lane states instead of pooling VRAM.
An explicit multi-GPU request fails if fewer than two CUDA devices are visible
rather than silently running on one GPU. Exact mode also requires a homogeneous
device group (same reported GPU model and compute capability); use
`CUDA_VISIBLE_DEVICES` to select/reorder the group, with visible device 0 as the
canonical runtime.

## What scales today

The initial logical-story pass is distributed across devices concurrently.
Secondary CUDA runtimes live on persistent owner threads instead of being
re-created or re-threaded at every batch. Sparse canonical synchronization occurs
only at the batch barrier. GPU 0 receives the exact merged rows through a packed
sparse host-mirror commit and secondary replicas consume the same immutable
packet. The packet is also scattered into resident story lanes. Batch-end sparse
GPU-to-host snapshots compact all lane change counters into one small D2H read,
then compact the variable payload into at most three typed D2H reads (f32/u32/u64)
instead of blocking or transferring lane by lane.

The host reducer still preserves the exact logical-story accumulation order, but
uses deterministic k-way sparse merging rather than allocation-heavy tree maps.
Constraint projection and validation inspect only changed parameters at the hot
barrier; full model validation remains at artifact/checkpoint boundaries.

The CUDA executor also uses sparse worklists and exact packed-key selection.
Current sparse execution optimizations include:

- compacting positive block candidates before exact sorting;
- exact shared-memory heap k-way merge of sparse sorted block-winner runs with the dense bitonic
  path retained as a fallback;
- flattened delayed recurrent-weight snapshot copies;
- sparse learning-destination worklists;
- reuse of the exact FP32 softmax exponential instead of recomputing it.

These change work placement, not the learning equations.

## Replay scaling modes

The **default/exact** multi-GPU path keeps bounded-surprise replay canonical and
sequential in the established story/range order. A replay range changes the
canonical parameters seen by the next range, so sending ranges to separate
data-parallel replicas changes the learning policy. The default therefore keeps
serial replay and its Amdahl-law scaling limit.

An **opt-in experimental** path is available with
`LEO_MULTI_GPU_PARALLEL_REPLAY=1`. It executes local replay trajectories on the
participating replicas and merges their parameter deltas. FP32 and the configured
30% replay budget remain unchanged, but cross-story replay update visibility is
different from canonical serial replay. This mode is not exact-mode semantics and
must pass held-out quality A/B validation before being used for a quality claim.

`LEO_REPLAY_STREAMING=1` is a separate opt-in replay experiment that reduces
repeated prefix reconstruction by carrying the replay timeline forward on-device.
It also changes replay-state semantics and requires the same quality gate.

For an exact future solution with better replay scaling, the intended design is
model-parallel replay using neuron/synapse sharding and device peer collectives:
each replay timestep computes local block winners/event work, performs a small
exact global winner/logit reduction, and continues on-device. A host round-trip
per byte is deliberately undesirable. NVLink/NVSwitch is preferred for this
fine-grained model-parallel stage; ordinary PCIe is much less restrictive for
story data parallelism.

## Hardware-adaptive single-GPU scheduling

Logical story workers are semantic; physical CUDA resources are not. Leo keeps
`--workers` unchanged and asks the CUDA driver/tuner how many cooperative CTAs
can actually be resident for the grouped persistent kernel. The production
launch now consumes the complete tuner-approved CTA budget instead of rounding
it down to an exact multiple of the logical lane count. Remainder CTAs are
distributed across story lanes, while every within-story dependency is protected
by a lane-local cooperative-residency barrier. Unrelated stories therefore no
longer wait at a whole-grid barrier after every CUDA phase.

This is deliberately different from auto-increasing `--workers`: a larger GPU
may run more blocks for the same 16 logical stories, but the canonical
`1 / workers` mean, RNG inputs, FP32 equations, story order, and replay policy do
not change. Kernel-resource telemetry (`cuda_kernel_resources`) exposes
registers/thread, local/shared memory, cooperative capacity, resident
threads/SM, and theoretical thread occupancy so future tuning can be based on
the actual compiled kernel rather than a GPU-model lookup table.

For normal single-GPU persistent batches of at most 4096 targets per story, the
host uploads raw story bytes once and the GPU constructs the exact persistent
BEGIN/byte/END schedule, including deterministic context dropout and END_DOCUMENT
weighting. The compact loss/activity reduction and exact bounded-surprise replay-range
selection also run on-device. The host receives one summary plus a small range list per
story rather than one record per byte. `LEO_CUDA_DEVICE_STORY_STEPS=0` restores
host-built step descriptors for execution A/B; `LEO_CUDA_DEVICE_STORY_POSTPROCESS=0`
restores the historical per-step D2H/CPU-selection path for exact A/B testing.
Long stories, full-step diagnostics, and multi-GPU retained-delta execution keep
the established fallback. The keyed context hash-table merge remains an explicit
batch safe point because its collision semantics are order-sensitive; it is not
changed merely to claim a CPU-free path.

## Scaling targets

Targets are engineering goals, not guarantees. Measure them on the intended
hardware and dataset.

For N identical GPUs define:

```text
speedup(N) = throughput(N) / throughput(1)
efficiency(N) = speedup(N) / N
```

A useful target is >= 0.80 efficiency for the data-parallel initial pass, so two
GPUs approach >= 1.6x and ideally ~1.8x. Full 30% replay-on efficiency will be
lower until replay becomes model-parallel.

Use `scripts/benchmark_multi_gpu.py` to measure 1/N GPU throughput with identical
model, data, story count, workers, and replay policy. The utility now requires an
exact SHA-256 digest of the complete trained persistent state in addition to the
workload counters/loss. It also prints a 10-second `nvidia-smi` heartbeat while
Leo is otherwise silent. On 2+ GPU hosts, `scripts/check_gpu.sh` runs a 1-GPU vs
2-GPU final-state parity gate automatically.

`training_benchmark` also reports `replay_seconds`, `replay_sync_seconds`,
`replay_wall_fraction`, and `serial_replay_speedup_ceiling` so the remaining
Amdahl limit is measured rather than guessed.

## Resume identity

The exact data-parallel synchronization identity is:

```text
gpu_story_mean_exact_v1
```

Physical GPU count is not part of the exact synchronization identity. For
backward compatibility Leo accepts the earlier exact aliases
`gpu_shared_wavefront_mean` and `gpu_multi_device_story_mean_exact` as equivalent
to `gpu_story_mean_exact_v1` when the backend is GPU and logical workers > 1.
Experimental device-mean identities remain incompatible and require a fresh run.
