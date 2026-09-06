# Contributing to Leo

Thanks for helping improve Leo.

Leo is a research-oriented sparse recurrent byte-learning system. The project treats learning semantics, artifact identity, and reproducibility as first-class compatibility boundaries. Contributions are welcome, but performance or cleanup changes must not silently change those contracts.

## Before you start

Please read:

- [README.md](README.md)
- [docs/PRD.md](docs/PRD.md)
- [docs/TDD.md](docs/TDD.md)
- [docs/SEMANTICS.md](docs/SEMANTICS.md)
- [docs/FORMATS.md](docs/FORMATS.md)
- [docs/TESTING.md](docs/TESTING.md)
- [docs/REPRODUCIBILITY.md](docs/REPRODUCIBILITY.md)

For security problems, do **not** open a normal public issue. Follow [SECURITY.md](SECURITY.md).

## Development setup

Minimum supported Rust toolchain:

```text
Rust 1.85
```

Install the formatting/lint components:

```bash
rustup component add rustfmt clippy
```

Build the workspace:

```bash
cargo build
```

Run the normal repository gate:

```bash
cargo fmt --check
cargo test
cargo clippy -- -D warnings
bash ./scripts/check.sh
```

GPU-related changes require a suitable NVIDIA/CUDA machine and should additionally run:

```bash
bash ./scripts/check_gpu.sh
```

## Current dataset used for public testing

Current public development and quality testing is specifically based on:

- <https://huggingface.co/datasets/roneneldan/TinyStories>
- paper: <https://arxiv.org/abs/2305.07759>

Prepare the pinned train/validation sources used by Leo with:

```bash
bash ./scripts/data.sh
```

The repository does not vendor TinyStories. Do not commit raw/prepared corpus data or training runs.

## Contribution workflow

1. Fork the repository and create a focused branch.
2. Keep the change scoped to one concern whenever possible.
3. Add or update tests for behavioral changes.
4. Update documentation when a command, contract, file format, execution mode, or public behavior changes.
5. Run the required validation gates.
6. Open a pull request using the repository PR template.

Suggested branch names:

```text
fix/<short-description>
feature/<short-description>
docs/<short-description>
perf/<short-description>
```

## Semantic compatibility rules

The following are part of the v1.0.0 learning/execution contract and must not change accidentally:

- FP32 training semantics
- standard bounded-surprise replay fraction of 30%
- logical `--workers` batch semantics
- canonical mean-update barrier
- dataset identity/provenance rules
- checkpoint/resume identity rules
- CPU reference behavior
- CUDA ABI source-of-truth rules

Execution optimizations may change physical launch geometry, transfer scheduling, caching, or other implementation details only when the observable learning contract is preserved.

If your proposal intentionally changes semantics, say so explicitly in the issue/PR and explain:

- why the change is necessary;
- which semantic/version boundary changes;
- migration or artifact compatibility implications;
- expected quality/reproducibility effects;
- tests that distinguish old and new behavior.

## CUDA and performance contributions

For CUDA changes, include where applicable:

- GPU model and VRAM
- NVIDIA driver version
- CUDA toolkit/NVRTC version
- exact command used
- logical worker count
- dataset/story/byte limits
- before/after throughput
- correctness/conformance result
- profiler evidence when claiming a hardware bottleneck improvement

Do not trade model/training quality for speed without an explicit semantic proposal.

## Dependency changes

Leo is an application/research system and should keep `Cargo.lock` committed.

When changing dependencies:

- make the version change explicit;
- update `Cargo.lock`;
- run the full CPU/static gate;
- run GPU validation when the dependency can affect CUDA/runtime behavior;
- explain why the new dependency or version is needed.

Avoid adding dependencies for functionality already available in the workspace or standard library without a clear maintenance benefit.

## Tests and documentation

A PR is expected to include tests when it changes behavior. Source-string guards are useful for architectural invariants, but behavioral Rust tests are preferred for correctness when practical.

Documentation should describe the code that actually ships. Historical `STAGE*.md` files are implementation-history documents; current behavior belongs in the main v1 design/semantics/testing documents.

## Commit hygiene

Do not commit:

- `target/`
- prepared datasets
- raw downloaded corpora
- training runs/checkpoints unless a maintainer explicitly requests a tiny fixture
- local CUDA/PTX/autotuner caches
- secrets, API keys, or machine-specific credentials

Prefer descriptive commits, for example:

```text
Fix CUDA event ordering for batch downloads
Document TinyStories validation workflow
Add checkpoint corruption regression test
```

## Pull request checklist

Before requesting review, confirm:

- [ ] the change is focused and explained;
- [ ] `cargo fmt --check` passes;
- [ ] `cargo test` passes;
- [ ] `cargo clippy -- -D warnings` passes;
- [ ] `bash ./scripts/check.sh` passes;
- [ ] GPU checks were run when the change touches CUDA/GPU behavior, or the PR clearly states why they could not be run;
- [ ] docs/tests were updated when needed;
- [ ] learning semantics were not changed silently;
- [ ] no external dataset or generated run artifacts were committed.

By participating, you agree to follow [CODE_OF_CONDUCT.md](CODE_OF_CONDUCT.md).
