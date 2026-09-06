# Support

Leo is research-oriented open-source software. Community support is best-effort.

## Where to ask

Use GitHub Issues for reproducible bugs, feature proposals, documentation problems, or performance regressions:

<https://github.com/saravanaspar/Leo/issues>

Before opening an issue, search existing issues and include enough environment information to reproduce the problem.

For security vulnerabilities, **do not use a normal public issue**. Follow [SECURITY.md](SECURITY.md).

## Include this information for bugs

- Leo commit/version
- operating system
- `rustc --version`
- command that failed
- CPU/GPU backend
- GPU model, NVIDIA driver, and CUDA version if applicable
- relevant error output
- whether `cargo test` and `bash ./scripts/check.sh` pass
- whether the issue reproduces with the pinned TinyStories preparation path

## Model-quality questions

Current public development/testing is specifically centered on the TinyStories dataset:

<https://huggingface.co/datasets/roneneldan/TinyStories>

When discussing quality, include the dataset split, story/byte limits, config, seed where applicable, number of passes, worker count, backend, validation size, and checkpoint identity. A repository test pass is not evidence of model quality by itself.
