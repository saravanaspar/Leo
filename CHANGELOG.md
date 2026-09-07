# Changelog

All notable project changes should be recorded here.

## Unreleased

- Replace experimental mean-of-device-means training with exact logical-story data parallelism: retain one sparse delta per story, restore canonical story order, and apply one flat `1 / workers` canonical mean across 2..N visible GPUs, including uneven partitions and one-story-per-device layouts.
- Make `benchmark --train` use the same multi-GPU `TrainingEngine` as real training and report `gpu_devices`, so scaling measurements cannot silently fall back to GPU 0.
- Defer redundant device-local canonical merging/copies in multi-GPU shards; only the globally merged sparse canonical state is synchronized back to resident replicas.
- Reuse the canonical CUDA runtime as the device-0 data-parallel worker instead of allocating a second full model on GPU 0; secondary devices alone own replicas, and the flat canonical mean is sparse-committed before replay.
- Fail an explicit `LEO_MULTI_GPU=1` request when fewer than two CUDA devices are visible instead of silently degrading to single-GPU execution.
- Compact finite positive block candidates before exact packed-key sorting and use an exact sparse k-way merge of already-sorted block-winner runs, retaining the dense/non-finite bitonic fallback and historical cutoff behavior.
- Flatten recurrent delayed-weight snapshot copies across CUDA threads and reuse the already-computed FP32 softmax exponential in the persistent forward path.
- Add multi-GPU scaling documentation, source/reference regression tests, and a live scaling benchmark utility while keeping FP32, 30% replay, logical workers, replay ordering, checkpoint schema, and CUDA ABI v1 unchanged.
- Replace serial persistent per-block top-k selection with an exact packed-key parallel network on supported power-of-two CUDA shapes, preserving the historical block-cutoff quirk and retaining the legacy fallback for unsupported/non-finite cases.
- Accelerate persistent global winner selection with the same exact packed key and reuse already-sorted per-block winner runs to skip completed bitonic stages without changing active-neuron order.
- Add reference tests proving packed-key ordering, historical cutoff reconstruction, and pre-sorted-run merge equivalence to Leo's v1 comparator.
- Accelerate the single-story supervised replay path with a cooperative whole-grid CUDA executor while preserving the legacy one-block kernel as an execution-only A/B fallback.
- Expose hidden frozen-prefix replay work in benchmark/training progress accounting and add opt-in replay/CUDA timing, launch, memory, hardware-identity, segment, and sampled phase diagnostics.
- Keep replay selection/order, logical worker semantics, FP32 equations, artifact formats, and v1 semantic/schema identifiers unchanged.

## v1.0.1 - 2026-09-07

### Correctness and CUDA execution

- restored CPU/CUDA logical story-batch semantics: every CUDA lane now owns its mutable learned-parameter trajectory for the full story and the canonical model is mean-merged once at the batch end;
- added sparse device snapshots and batch-end canonical synchronization without full per-worker CUDA runtime replicas;
- fixed long shared-batch chunks so odd eligibility-buffer swaps refresh the lane pointer table before the next chunk;
- added real CPU/CUDA multi-story conformance at replay 0 and the standard 30% replay policy, replacing source-string-only confidence in the production batch path;
- reduced frozen replay-prefix overhead by reusing invariant pointer-table state and distributing independent cooperative post/emit work across the grid;
- persisted incomplete CUDA autotuning observations atomically and resume them in fresh processes;
- made sampled CUDA phase profiling geometry-safe: samples use the production grid width or emit an explicit skip event;
- extended training benchmark JSON with replay-budget/overhead fields.

### Validation and packaging

- made the repository gate run both Python test trees;
- made `make check` invoke the shell gate through `bash`, avoiding archive executable-bit assumptions;
- corrected TinyStories `hf` installation guidance;
- tightened workflow permissions and pinned CI actions/toolchain setup;
- removed the three `clippy -D warnings` blockers found by the release gate (unused mutability, dead CUDA upload wrapper, and an over-wide phase-profiler helper signature);
- bumped the public software release metadata to v1.0.1 while keeping all v1 semantic/schema identifiers unchanged.

### Documentation and community

- reorganized the GitHub README around quick start, architecture, GPU execution, testing, and contribution flows;
- documented TinyStories as the current public development/testing corpus with pinned provenance handled by `scripts/data.sh`;
- added contribution, security, support, code-of-conduct, issue/PR, citation, and reproducibility guidance;
- added dependency-update configuration for explicit reviewable Cargo and GitHub Actions updates.

## v1.0.0

### Core baseline

- sparse recurrent byte-learning runtime;
- CPU reference backend;
- custom NVIDIA CUDA backend;
- FP32 v1 learning semantics;
- standard 30% bounded-surprise replay;
- strict TOML configuration;
- verified dataset artifacts and provenance identity;
- strict checkpoint format and atomic persistence;
- training resume/fresh-run lifecycle;
- held-out/training evaluation and generation tools;
- execution-only CUDA autotuning and PTX caching;
- cooperative shared-wavefront fusion with hardware fallback;
- pinned asynchronous transfer/compute streams and event ordering;
- CUDA Graph sparse apply/reset replay where supported;
- GPU verification/profiling scripts;
- optional experimental multi-GPU batch-end synchronization.
