# Leo v1.0.0 semantics contract

Leo separates **what learning means** from **where and how it executes**. A checkpoint records the semantic contract, while CPU and CUDA are implementations of that contract.

## Contract versions

| Contract | Version |
| --- | ---: |
| Leo release | 1.0.0 |
| Model schema | 1 |
| Training policy | 1 |
| Execution semantics | 1 |
| Dataset schema | 1 |
| CUDA ABI | 1 |
| Checkpoint schema | 1 |

The canonical constants live in `crates/leo-core/src/semantics.rs`. CUDA's device ABI is defined once in `crates/leo-core/cuda_abi.def`; the build script generates the Rust layout and CUDA header from that definition and checks its ABI version against the semantics contract.

## TrainingPolicy v1

The v1 policy is `bounded-surprise-replay-v1`.

For every story, Leo first performs the ordinary supervised pass. It then ranks supervised target positions by loss and replays non-overlapping target ranges until the configured replay target budget is reached. The standard configurations use:

```toml
[replay]
fraction = 0.30
```

`fraction = 0.30` means approximately 30% of the story's supervised target positions are selected for bounded surprise replay. `segment_bytes` controls the target-range width. Replay is part of the learning policy, not a CUDA optimization: CPU, single-GPU, GPU story batching, and multi-GPU training all apply the same policy.

`teaching_replays` and `max_verified_replays` apply to explicit `leo teach` operations and are separate from the 30% dataset-training policy.

## ExecutionSemantics v1

`leo-exact-fp32-v1` fixes the semantic expectations that must not be silently changed by a backend optimization:

- FP32 model math; no FP16/BF16 training mode.
- No approximate-math substitution as a hidden performance option.
- Byte-level input and exact BEGIN/END document symbols.
- UTF-8 generation state remains host-visible behavior.
- Fixed model topology for a v1 model.
- Context identity is `(order, fingerprint)`, never a raw slot index.
- Selection order and tie-breaking remain deterministic for the same execution semantics and seed.
- Permission strength controls which learning updates are legal.
- Synchronous sparse deltas are produced from one canonical parameter revision and rejected if stale.
- Fixed tensors are mean-reduced; context changes are merged by context identity and resolved into canonical slots.
- Model statistics describe the complete training trajectory; `parameter_revision` describes canonical parameter-state transitions.

CPU is the numerical/reference executor. CUDA may change launch geometry, device memory layout, phase fusion, persistent execution, worklist traversal, transfer scheduling, graph replay, or **physical** lane chunking only when the learning semantics above remain intact. The logical story batch selected by `--workers`, its batch-scale denominator, and its canonical mean-update barrier are semantic inputs and are not autotuning knobs.

## Backend versus policy

Backend answers **where computation runs**:

```text
cpu | gpu
```

Training policy answers **what computation occurs**. Selecting GPU must not silently disable replay or change the learning formula.

## Multi-GPU semantics

Multi-GPU remains explicitly distinguishable from exact single-GPU wavefront ordering. Device shards begin from one canonical revision, produce sparse deltas, and merge them synchronously at the batch boundary. The merged canonical changes are synchronized back to persistent GPU replicas. Replay is then applied to the canonical backend under the same TrainingPolicy v1 and replay changes are sparsely synchronized to replicas.

The run log records the synchronization mode so multi-GPU experiments are not confused with exact single-GPU ordering.

## Changes that require a semantic version change

Examples include changing replay selection rules, update ordering, context identity, tie-breaking, boundary symbols, delta merge rules, or the permitted numerical model. Execution-only changes that preserve these contracts do not require a training-policy or execution-semantics version change.
