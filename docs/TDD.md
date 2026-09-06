# Leo v1.0.0 technical design

## 1. Architectural boundary

Leo is organized around persistent model state, transient execution state, durable artifacts, and a backend-neutral training policy.

```text
Config / Model / SemanticsContract
              |
        Training policy
              |
       BackendRuntime
          /       \
   CPU Runtime   CUDA executor
              |
      Model / dataset artifacts
```

`Model` owns only persistent learned state. Per-document membrane, activation, fatigue, eligibility, delay rings, selections, and scratch remain transient executor state. A fresh executor can therefore start from a canonical model without inheriting another document's transient state.

## 2. Semantics versus execution

The current semantic versions and invariants are defined in `docs/SEMANTICS.md` and `leo-core/src/semantics.rs`. The CPU runtime is the reference implementation. The CUDA executor implements the same learning policy and observable runtime contract.

Backend selection never silently disables replay. Dataset training uses TrainingPolicy v1 with bounded surprise replay, configured at 30% in the standard configurations.

## 3. Training module

`leo-cli/src/training.rs` is the deep training boundary. Its `training/` submodules own story-pass learning, replay-range selection, synchronous worker deltas, multi-GPU batch coordination, verified-dataset prefetch, validation/evaluation, checkpoint/resume transactions, early stopping, and the end-to-end training lifecycle. `main.rs` is limited to command schemas/parsing, command adapters, user-facing presentation, and non-training commands.

For a CPU multi-worker batch:

```text
canonical model theta_t
       |
       +--> worker story pass -> sparse delta 1
       +--> worker story pass -> sparse delta 2
       +--> ...
       |
       +--> mean fixed-parameter deltas
       +--> merge context by (order, fingerprint)
       |
       +--> canonical 30% replay policy
       v
canonical model theta_t+1
```

GPU-native story batching produces equivalent policy inputs (per-story supervised losses) and canonical replay is applied afterward, so GPU execution does not omit replay.

## 4. CUDA execution

CUDA keeps persistent model buffers on device and transfers/synchronizes at explicit model boundaries. v1 makes cooperative persistent execution the preferred path when the device supports it. The required non-cooperative path remains a hardware compatibility fallback.

The shared-wavefront story batch is the normal GPU batch path. It keeps lanes in the same high-level execution phase, reducing avoidable control divergence and host launch traffic. Persistent state is represented as contiguous typed device buffers (structure-of-arrays style) separate from the checkpoint representation.

The v1 executor may optimize execution aggressively while preserving FP32 learning semantics:

- persistent model state and minimized host/device synchronization;
- cooperative persistent execution when supported;
- safe cooperative-grid fusion of the complete shared-model wavefront, with grid-wide barriers replacing the old kernel-boundary barriers and the same phase helpers used by the non-cooperative fallback;
- device-oriented contiguous model/scratch layouts;
- shared phase/wavefront execution to reduce avoidable branch divergence;
- exact per-tick touched-neuron and learning-destination worklists, with epoch arrays retained as the authoritative membership state;
- sparse changed-parameter application using the existing delta lists rather than dense parameter scans;
- physical CUDA lane chunking that leaves the logical story batch and single mean-update barrier unchanged;
- online execution tuning across physical lane width, sparse-apply block/thread geometry, and cooperative fused-grid width; profiles are keyed by GPU UUID/PCI-bus/name/ordinal/VRAM, driver version, compute capability, SM/thread capacity, model/source/semantic identity, and **logical batch width**, so profiles cannot be casually reused across different hardware or partial/full logical batches;
- SHA-256-keyed NVRTC PTX caching whose key binds kernel source, cooperative-groups header, CUDA ABI, execution semantics, NVRTC version, compile options, and compute capability;
- page-locked host staging with separate compute/transfer streams, event dependencies, and pipelined HtoD staging so the next physical lane group can transfer while the current group computes; DtoH records are similarly ordered by events instead of a whole-stream host synchronization;
- one-batch-ahead dataset prefetch using an already-verified cloned dataset handle;
- CUDA Graph capture only for the stable sparse apply/reset launch sequence, with transparent direct-launch fallback when graph APIs/capture are unavailable;
- sampled runtime telemetry for HtoD/compute/DtoH/host-wait timing, effective transfer bandwidth, launch mix, and occupancy estimates without synchronizing every production batch; `scripts/profile_cuda.sh` uses Nsight Compute for architecture-specific hardware counters such as DRAM behavior, achieved occupancy, warp/branch behavior, instructions, and atomics.

The execution planner is intentionally forbidden from changing logical `--workers`, replay fraction, update ordering, precision, or the number of canonical mean updates. It may only choose physical launch geometry for work already required by ExecutionSemantics v1.

## 5. CUDA ABI

`crates/leo-core/cuda_abi.def` is the canonical Rust/CUDA ABI definition. `build.rs` generates:

- `cuda_abi_generated.rs` for Rust;
- `leo_cuda_abi.h` prepended to NVRTC source.

It defines the CUDA config fields, persistent pointer IDs, batch-delta pointer IDs, and ABI version. `python/check_cuda_abi.py` verifies one-to-one pointer coverage and rejects reintroduced manual CUDA-side layout declarations.

## 6. Data and resume identity

Prepared datasets are immutable `LEODATA1` artifacts. Dataset opening hashes complete byte and record artifacts, verifies provenance-bound identity, and validates all record ranges. `PreparedDataset::try_clone()` reuses the verified index/identity and clones the file handle so evaluation workers do not repeatedly hash a validated artifact. On Unix, story reads use positioned I/O (`read_exact_at`) so cloned handles do not contend on or race a shared seek cursor.

Training resume uses `LEOTRAIN100` and stores `DatasetId`, not sampled first/last-file fingerprints. During training, a bounded prefetch worker reads the next logical story batch through a cloned verified handle while the current batch executes; the request carries the same order position, worker count, and remaining-byte budget as the synchronous reader.

## 7. Persistence

Checkpoints use `PSCLS100`. The loader requires exact v1 sections, rejects duplicate/overlapping descriptors, verifies SHA-256 integrity, validates tensor descriptors and model invariants, and checks the embedded `SemanticsContract` before returning a model.

Checkpoint writing remains atomic. Pre-v1 experimental checkpoint layouts are deliberately not part of the runtime reader.

## 8. Parameter revision

`parameter_revision` identifies canonical parameter-state transitions and is the stale-delta gate. Training-history counters remain in statistics. Restoring the best early-stopping model creates a new canonical revision rather than relabeling old parameters with an unrelated later revision.

## 9. Configuration and CLI

Configuration is real TOML via `serde` + `toml`, with unknown fields rejected. Serialization uses the TOML library and therefore escapes strings correctly.

The CLI uses command-specific option schemas. Unknown options, duplicate options, missing values, and options belonging to another command fail immediately. Errors are typed (`Configuration`, `Dataset`, `CheckpointCorrupt`, `CheckpointIncompatible`, `Backend`, `Cuda`, `Numerical`, `Io`, etc.); exit codes are based on error type rather than English-message substring matching.

## 10. Validation strategy

The long-term boundary tests are:

- standard-TOML round trip and validation;
- SHA-256 known vectors;
- dataset corruption/identity rejection;
- checkpoint strict decoding and semantics checks;
- train -> interrupt -> resume identity checks;
- canonical synchronous-delta tests;
- replay-policy selection tests;
- CUDA ABI generation/source-of-truth checks;
- CPU/CUDA forward/training conformance plus shared-batch CUDA smoke tests on configured GPU CI runners (`.github/workflows/gpu-ci.yml`).

Python source guards are secondary architecture lint; behavioral Rust tests remain the primary correctness evidence where a Rust toolchain is available.

## 11. Open-source validation and reference dataset

The public validation ladder is documented in `docs/TESTING.md`; experiment identity guidance is in `docs/REPRODUCIBILITY.md`. Current public model/data testing is specifically centered on the externally hosted `roneneldan/TinyStories` train/validation files pinned by `scripts/data.sh`. The repository does not vendor the corpus, and TinyStories remains subject to its own upstream license and terms.

Repository CI proves source/build/test invariants. GPU claims require the real `scripts/check_gpu.sh` hardware gate, and model-quality claims require held-out evaluation under a recorded dataset/config/checkpoint/hardware identity.
