# Leo scaling architecture

Leo v1.0.1 keeps one learning contract while allowing the execution layer to
use more CUDA devices. The code path is implemented, but should be treated as
requiring real 2+ GPU hardware validation before production deployment. The model, replay policy, FP32 arithmetic, logical worker
count, and canonical mean barrier are semantic inputs; physical GPU placement is
not.

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

Uneven counts use a deterministic contiguous balanced partition, for example 16
stories on 3 GPUs become 6 + 5 + 5.

Each device returns **one sparse delta per original story**. The coordinator
sorts device results by the original story position, flattens all 16 deltas, and
calls the same canonical `apply_mean_deltas` once. There is no nested
mean-of-device-means, so the denominator remains 16 independent of device count.
The globally merged sparse rows are then synchronized to the resident replicas.

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
rather than silently running on one GPU.

## What scales today

The initial logical-story pass is distributed across devices concurrently.
Sparse canonical synchronization occurs only at the batch barrier. GPU 0
receives the exact merged rows through a sparse host-mirror commit and secondary
replicas receive those same changed rows concurrently. This is the
low-communication scaling axis and is appropriate even on PCIe-only systems.

The CUDA executor also uses sparse worklists and exact packed-key selection.
Current sparse execution optimizations include:

- compacting positive block candidates before exact sorting;
- exact k-way merge of sparse sorted block-winner runs with the dense bitonic
  path retained as a fallback;
- flattened delayed recurrent-weight snapshot copies;
- sparse learning-destination worklists;
- reuse of the exact FP32 softmax exponential instead of recomputing it.

These change work placement, not the learning equations.

## What does not scale across GPUs yet

Bounded-surprise replay remains canonical and sequential in the established
story/range order. A replay range changes the canonical parameters seen by the
next range, so simply sending ranges to separate data-parallel replicas would
change the learning policy.

This means full replay-on scaling follows Amdahl's law. If half of wall time is
serial replay, even an infinitely fast data-parallel initial pass can approach
only 2x end-to-end speedup. Near-linear 2/4/8-GPU *full training* scaling needs a
future device-side model-parallel replay executor.

The required future design is neuron/synapse sharding with device peer
collectives: each replay timestep computes local block winners/event work,
performs a small exact global winner/logit reduction, and continues on-device.
A host round-trip per byte is deliberately not implemented because it would
usually make scaling worse. NVLink/NVSwitch is preferred for this fine-grained
model-parallel stage; ordinary PCIe is much less restrictive for story data
parallelism.

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
model, data, story count, workers, and replay policy. The utility checks the
semantic workload counters and loss before reporting speedup.

## Resume identity

The exact data-parallel synchronization identity is:

```text
gpu_multi_device_story_mean_exact
```

Older experimental multi-GPU resumes used a different device-mean identity.
Leo rejects those resumes instead of silently continuing with different
reduction semantics. Start a fresh multi-GPU run when moving from the old
experimental synchronization mode.
