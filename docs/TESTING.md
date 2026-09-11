# Leo v1.0.1 testing strategy

## Purpose

Leo separates **implementation correctness**, **CPU/GPU conformance**, **performance validation**, and **model-quality evaluation**. No single test substitutes for the others.

Current public model/data testing is specifically centered on the [TinyStories dataset](https://huggingface.co/datasets/roneneldan/TinyStories).

## 1. Standard repository gate

Run from the repository root:

```bash
cargo fmt --check
cargo build
cargo test
cargo clippy -- -D warnings
bash ./scripts/check.sh
```

This covers formatting, workspace compilation, Rust behavioral tests, warning-free linting, **both Python test trees (`python/tests` and top-level `tests`)**, source architecture checks, CUDA ABI consistency, supported configuration validation, and shell/workflow sanity checks performed by the script.

A failure in any of these gates blocks a normal contribution.

## 2. GPU conformance gate

Changes affecting CUDA kernels, driver bindings, launch planning, device layouts, event ordering, pinned transfers, CUDA Graphs, PTX caching, autotuning, or GPU-visible training behavior should run:

```bash
bash ./scripts/check_gpu.sh
```

This requires a real NVIDIA GPU/driver environment. The script exercises real CUDA/NVRTC execution rather than treating source inspection as proof of GPU correctness. In particular it runs the production multi-story story-batch path against CPU reference semantics twice: once with replay disabled to isolate the batch-end mean barrier, and once with `replay.fraction = 0.30` to cover replay selection/prefix reconstruction and the resulting canonical model. The gate compares replay counts, revisions, context identity/observations, training statistics, learned-parameter tolerances, and loss.

The same GPU gate deliberately stops one 16-worker autotune process before a candidate is complete, launches a fresh process, and requires `cuda_autotune_resumed`. It also samples the internal phase profiler and requires either a comparable-geometry `cuda_phase_profile` event or an explicit `cuda_phase_profile_skipped` event; instrumentation is never allowed to silently shrink the production grid.

The GitHub GPU workflow is conditional on a configured `LEO_GPU_RUNNER`. If no GPU runner is configured, a skipped GPU job is **not** evidence that CUDA was tested.

## 3. Hardware profiling

When a PR claims a CUDA performance improvement, use runtime telemetry and, when possible, Nsight Compute:

```bash
bash ./scripts/profile_cuda.sh \
  runs/quality/my-run/leo.pscls \
  data/prepared/tinystories.train.bytes \
  data/prepared/tinystories.train.idx \
  16 256 leo-cuda-profile
```

Profiler evidence should identify the actual bottleneck being improved, such as transfer behavior, occupancy, divergence, memory traffic, atomics, or launch overhead.

Performance evidence never overrides semantic conformance. For the current P100/TinyStories reference, record `--workers 16`; worker count is a semantic experiment input, so comparisons at a different logical batch width are separate experiments.

For replay-focused measurements, `leo benchmark --train` reports `replay_fraction`, `replay_segments`, `replay_steps`, and `replay_step_fraction` alongside throughput. The v1 30% replay budget must stay enabled for production-policy measurements. Replay-disabled runs are diagnostics only and must be labeled as such.

## 4. TinyStories data used by Leo

The current preparation script pins:

```text
repository: roneneldan/TinyStories
revision:   5485261731eaac25dd8e5ebbc3839d0a9870b185
```

Files:

```text
TinyStories-train.txt
sha256 c5cf5e22ff13614e830afbe61a99fbcbe8bcb7dd72252b989fa1117a368d401f

TinyStories-valid.txt
sha256 94e431816c4cce81ff71e4408ff8d3bda9a42e8d2663986697c3954288cb38b4
```

Prepare them with:

```bash
bash ./scripts/data.sh
```

The prepared train and validation files are distinct artifacts. The validation split must remain held out from training.

## 5. Smoke versus quality evaluation

Recommended validation sizes depend on the purpose:

| Validation stories | Use |
| ---: | --- |
| 10 | fast smoke check |
| 100 | routine development check |
| 1000 | larger quality check |
| full validation set | deliberate final evaluation |

Use the same limits when comparing changes. Otherwise throughput or quality deltas can be misleading.

## 6. Model-quality evidence

Repository tests validate implementation contracts. They do not prove that a model learned well.

For a model-quality comparison, record at minimum:

- Leo commit hash;
- `Cargo.lock` hash or exact dependency lockfile;
- config file + SHA-256;
- prepared dataset ID/manifest;
- TinyStories source revision/checksums when used;
- initial model/checkpoint identity;
- backend;
- GPU model/driver/CUDA version when relevant;
- logical worker count;
- pass count;
- story/byte limits;
- validation size;
- replay configuration;
- relevant random seeds;
- final checkpoint identity;
- training and held-out metrics.

## 7. CPU/GPU semantic expectations

The CPU runtime is the reference implementation. The single-GPU CUDA path is expected to preserve the v1 learning contract within the explicit conformance thresholds used by GPU tests.

Multi-GPU data parallelism preserves the logical story-batch reduction: one sparse delta per original story is flattened in canonical story order and one `1 / workers` mean is applied. Multi-GPU hardware tests should compare 1-GPU and N-GPU semantic counters/loss within the existing CPU/CUDA numerical thresholds and report scaling efficiency separately. The default/exact replay path remains canonical/serial, so its replay-on scaling is lower than base-pass scaling. `LEO_MULTI_GPU_PARALLEL_REPLAY=1` and `LEO_REPLAY_STREAMING=1` are separate semantic experiments: they retain FP32 and the configured replay fraction, but require held-out quality A/B validation and are not covered by the exact-state acceptance claim.

## 8. Regression-test rule

A bug fix should normally include a test that fails before the fix and passes after it. Prefer behavioral tests over source-string assertions when the behavior is testable without hardware-specific dependencies.

## Replay performance diagnostics

For replay-specific performance regressions, use `scripts/debug_replay.sh` and the opt-in controls documented in [DEBUGGING.md](DEBUGGING.md). Normal benchmark JSON always includes `replay_prefix_steps`/`replay_execution_steps`; heavy CUDA `clock64()` phase profiling remains opt-in and must be disabled for final throughput measurements.
