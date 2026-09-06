# Changelog

All notable project changes should be recorded here. Leo keeps the public software version at **v1.0.0** until the project explicitly decides to change the version.

## Unreleased

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
