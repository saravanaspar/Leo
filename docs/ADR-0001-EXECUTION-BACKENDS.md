# ADR-0001: Backend-neutral execution boundary

## Status

Accepted for Leo v1.0.0.

## Decision

`RuntimeBackend` is the execution contract and `BackendRuntime` is its dynamic owner. The CPU `Runtime` is the reference executor. Explicit `gpu` selection constructs the CUDA executor and fails if CUDA/NVRTC/device initialization is unavailable; it never silently falls back to CPU.

Backend selection is execution metadata, not model-learning policy. TrainingPolicy v1 is backend-neutral and includes the configured bounded surprise replay. A GPU-native initial pass must return enough per-story supervised metrics for the same replay policy to run; CUDA cannot silently omit replay for throughput.

Persistent learned state remains represented by `Model` and `PSCLS100`; transient execution state belongs to the backend. CUDA uses a device-specific layout generated from the canonical CUDA ABI definition without changing the checkpoint layout.

## Consequences

- CPU remains the numerical/reference implementation.
- CUDA may optimize launches, persistent execution, phase fusion, device layout, and batching while preserving ExecutionSemantics v1.
- Model access/checkpointing explicitly synchronizes the backend.
- Worker deltas are tied to a canonical parameter revision and stale deltas are rejected.
- Multi-GPU data parallelism retains one delta per original logical story and performs one flat canonical `1 / workers` mean; physical GPU scheduling remains covered by the normal CPU/CUDA numerical-conformance contract. Replay remains canonical/serial until a device-side model-parallel replay executor exists.
- CUDA/NVRTC remain dynamically loaded so CPU-only execution has no hard CUDA runtime link dependency.
