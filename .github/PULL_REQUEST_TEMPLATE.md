## Summary

Describe what changed and why.

## Type of change

- [ ] Bug fix
- [ ] Feature
- [ ] Performance/CUDA optimization
- [ ] Tests
- [ ] Documentation/community
- [ ] Dependency/tooling
- [ ] Intentional semantics/version change

## Validation

- [ ] `cargo fmt --check`
- [ ] `cargo build`
- [ ] `cargo test`
- [ ] `cargo clippy -- -D warnings`
- [ ] `bash ./scripts/check.sh`
- [ ] `bash ./scripts/check_gpu.sh` when GPU behavior is affected
- [ ] Not applicable items are explained below

## Semantics and reproducibility

- [ ] This PR does **not** silently change FP32 training semantics, 30% standard replay, logical worker-batch semantics, canonical update barriers, dataset identity, or checkpoint/resume identity.
- [ ] Tests were added/updated for behavioral changes.
- [ ] Documentation was updated when public behavior/contracts changed.
- [ ] `Cargo.lock` was updated if dependencies changed.
- [ ] No dataset, training-run, cache, or secret artifacts were committed.

## Performance evidence

For performance changes, include hardware, command, before/after numbers, and profiler/telemetry evidence when available.

## Additional notes

Anything reviewers should know, including validation that could not be run locally.
