# Changelog

All notable project changes should be recorded here.

## Unreleased

No changes yet.

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
