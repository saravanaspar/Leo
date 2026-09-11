# Leo semantics contract

Leo separates **what learning means** from **where and how it executes**. A checkpoint records the semantic contract, while CPU and CUDA are implementations of that contract.

## Contract versions

| Contract | Version |
| --- | ---: |
| Leo release | 1.0.32 |
| Model schema | 1 |
| Training policy | 1 |
| Execution semantics | 1 |
| Dataset schema | 1 |
| CUDA ABI | 1 |
| Checkpoint schema | 1 |

The software release number above is independent from the semantic/schema contract versions below it. A software release does not require artifact conversion unless one of those contract versions or the checkpoint format changes. Existing valid v1 artifacts therefore remain valid across release bumps that preserve the v1 contract.

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

For a logical batch with more than one story, every story starts from the same canonical parameter revision and owns a private learned-parameter trajectory for the **entire story**. No worker can observe another worker's updates inside that batch. After all supervised story passes finish, fixed-parameter deltas are mean-reduced once and context changes are merged once by `(order, fingerprint)`. CUDA implements this with one canonical device model plus lane-private mutable learned tensors; physical lane chunking/fusion never inserts an intermediate canonical update. The configured replay policy then runs from the merged canonical model.

Execution-plan tuning is cacheable execution state only. Partial candidate observations are persisted atomically and may be resumed by a fresh process, but a cached or resumed plan may never change logical batch width, replay budget, arithmetic, parameter visibility, or the single batch-end mean barrier.

## Backend versus policy

Backend answers **where computation runs**:

```text
cpu | gpu
```

Training policy answers **what computation occurs**. Selecting GPU must not silently disable replay or change the learning formula.

## Multi-GPU semantics

Leo's logical story workers are the data-parallel unit. Multi-GPU execution may
place those workers on different physical CUDA devices, but every worker begins
from the same canonical parameter revision and retains its own sparse story
delta. Device-local means are not used for the cross-device result. Story
deltas are restored to canonical story order and passed once to the same
`apply_mean_deltas` barrier, so the logical denominator remains exactly
`1 / workers` even for uneven device partitions or one-story-per-device layouts.

The merged canonical sparse changes are synchronized back to persistent GPU
replicas. Replay is then applied to the canonical backend under the same
TrainingPolicy v1, in the same story/range order, and replay changes are sparsely
synchronized to replicas. Replay is not yet model-parallel across devices.

The synchronization identity `gpu_story_mean_exact_v1` records this contract and
is independent of physical GPU count. Earlier exact names
`gpu_shared_wavefront_mean` and `gpu_multi_device_story_mean_exact` are accepted
as resume aliases; experimental device-mean identities are not. Exact multi-GPU
mode requires homogeneous CUDA devices, and the 2+ GPU hardware gate compares a
SHA-256 digest of the complete persistent training state across 1-GPU and 2-GPU
runs.

## Changes that require a semantic version change

Examples include changing replay selection rules, update ordering, context identity, tie-breaking, boundary symbols, delta merge rules, or the permitted numerical model. Execution-only changes that preserve these contracts do not require a training-policy or execution-semantics version change.
