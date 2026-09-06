# Leo v1.0.0 product requirements

## Purpose

Leo is a sparse recurrent byte-learning system with a CPU reference executor and a custom NVIDIA CUDA executor. v1.0.0 is the first clean durability baseline for long-lived training experiments.

## Required properties

1. **One learning contract.** CPU/GPU selection changes execution location, not the selected learning policy.
2. **Quality-oriented default.** Standard training uses TrainingPolicy v1 with `replay.fraction = 0.30` bounded surprise replay.
3. **Exact v1 numerical policy.** FP32 only; no hidden FP16/BF16/approximate-math training shortcut.
4. **Persistent versus transient separation.** Checkpoints contain learned model state, never document runtime scratch.
5. **Immutable experiment identity.** Datasets, checkpoints, and resume operations use complete SHA-256-bound identities.
6. **Strict persistence.** Corrupt, ambiguous, duplicated, overlapping, or semantically incompatible checkpoint sections are rejected.
7. **Standard configuration.** TOML is parsed/serialized by a standards-compliant TOML library and unknown fields fail validation.
8. **Strict CLI.** Unknown/duplicate options fail instead of silently falling back to defaults.
9. **Single CUDA ABI source.** Rust and CUDA layouts/IDs are generated from one canonical definition.
10. **Typed failure modes.** CLI behavior and exit codes use structured error categories.
11. **Safe GPU acceleration.** Device persistence, persistent kernels, safe phase fusion, device-oriented memory layout, reduced divergence, exact worklists, sparse delta application, physical lane chunking, cached PTX, pinned asynchronous staging, stable CUDA Graphs, and profiling-driven execution planning may improve throughput only when v1 learning semantics are retained.
12. **Execution-only autotuning.** The tuner may select physical CUDA launch geometry, but it must never change the logical `--workers` batch, replay budget, update barrier, arithmetic precision, or learning equations.
13. **Overlapped input pipeline.** Dataset prefetch may overlap verified story I/O with training while preserving exact story order, story boundaries, and byte-budget behavior.
14. **Reproducible metadata.** Runs log Leo release, semantic/training policy, backend, dataset ID, synchronization mode, and replay configuration.
15. **Fresh v1 baseline.** Pre-v1 experimental artifacts are not a permanent compatibility burden; v1 training begins from v1 model/data artifacts.

## Supported operations

- initialize model;
- train/resume/fresh-run;
- held-out and training-set evaluation;
- prompt generation;
- explicit teaching permissions;
- inspect/checkpoint/rollback;
- training and frozen-evaluation benchmarks;
- CPU, single-GPU, and explicitly marked multi-GPU synchronization modes.

## Non-goals for v1.0.0

- FP16/BF16 or tensor-core semantic rewrites;
- approximate learning math;
- asynchronous parameter-update semantics;
- multi-node/distributed training;
- compatibility readers for pre-v1 experimental artifacts;
- dynamic topology growth.

## Acceptance

A v1.0.0 release is acceptable when repository checks pass, v1 artifacts are self-identifying/strictly validated, replay is not backend-dependent, CUDA ABI duplication is absent, docs match the implementation, and Rust/CUDA conformance is run on a machine with the required toolchain/GPU before a production training run.
