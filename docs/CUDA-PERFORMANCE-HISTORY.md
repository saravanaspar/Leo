# CUDA performance history

This file is the durable performance ledger for Leo CUDA optimization work. It records measured
hardware results separately from design targets and repository tests.

## Recording rule

Append a new snapshot after each merged performance change. Never rewrite an older snapshot to
match a newer implementation.

A snapshot is identified by all of the following:

- Leo software version from the GitHub source tree;
- exact GitHub commit SHA and, when applicable, pull request number;
- GPU model and logical worker count;
- dataset identity/checksums;
- evidence archive SHA-256.

Multiple snapshots may share the same Leo semantic version. When that happens, keep them as
separate commit/PR snapshots under the same version heading. This prevents an optimization commit
from being confused with an earlier measurement made from the same release version.

Performance acceptance remains subordinate to semantic correctness. Exact/default optimization
runs keep FP32 learning, the configured replay policy, and the reference `--workers 16` logical
batch unchanged unless a snapshot explicitly says otherwise.

## Reference experiment identity

- Dataset: TinyStories, repository-pinned Hugging Face revision
  `5485261731eaac25dd8e5ebbc3839d0a9870b185`.
- Training subset: 1,024 stories, 953,787 bytes.
- Validation subset: 128 stories, 89,461 bytes.
- Prepared train bytes SHA-256:
  `c6839426e7e15935a6c4f7b7fcc9ff4c541968c135378de5ef71ccf48d6684c0`.
- Prepared train index SHA-256:
  `8eee9e66e921776a67c0d7d01bf3453235d219b4bc525a53aac1361b46b4b6ee`.
- Prepared validation bytes SHA-256:
  `f9d12102da6869a288a23fa5bea40e4fb1e9c516a231765f7092043bd6c0df35`.
- Prepared validation index SHA-256:
  `3b04509d08c29ad2744f4829d5e94491718d56b62aecb5a1054191e3d12a9d30`.
- Model: `configs/tinystories.toml`, 32,768 neurons, FP32.
- Exact/default replay fraction: 0.30.
- Reference logical batch: 16 workers.
- Primary hardware: Tesla P100-PCIE-16GB, compute capability 6.0, 56 SMs.
- P100 driver observed in the first archive: 580.159.04; CUDA driver API reported 13.0.
- NVCC observed in the fresh post-heap run: CUDA 12.8.
- Throughput goal: at least 8,000 reported training steps/s with default 30% replay.

## Leo v1.0.2

## Snapshot A - pre-heap selector baseline

- GitHub commit: `1989e656a5360cd717b7000999168b62cd2bff54`.
- Commit label: `chore(release): v1.0.2 (#19)`.
- Evidence archive: `p100-final-evidence.tar.gz`.
- Evidence SHA-256:
  `ff998296b0117ccf9f4a8ec742c0769321c36d956b5009a74527c145ebc21318`.

### Throughput

| Measurement | Result |
| --- | ---: |
| 32K replay-off autotune prediction | 2,468.283 work units/s |
| 32K replay-off optimized, 64 stories | 2,403.340 steps/s |
| 32K replay-off legacy, 64 stories | 1,553.314 steps/s |
| Optimized / legacy replay-off | 1.547x |
| Classic/default 30% replay, 16 stories | 453.437 steps/s |
| Classic replay execution throughput including prefixes | 1,086.392 steps/s |
| Experimental streaming replay, 16 stories | 1,276.624 steps/s |

The experimental streaming result is recorded only as diagnostic evidence. Its state digest differs
from classic replay, so it is not an exact/default performance result.

### Exact/default state evidence

The 64-story replay-off optimized and legacy runs both produced:

`14894f9039c59359a81ebed48d71b563617636348fb4347f2accefc5bfcd50e9`

The 16-story classic 30% replay run produced:

`5b30b3f442f3aecf17a6a41e1b98b21b802cd27aa70fbe0c79cb1f9257527e6e`

### Replay cost

- Base training targets: 11,143.
- Reported training steps: 14,496.
- Replay target steps: 3,353.
- Hidden replay prefix steps: 20,235.
- Replay execution steps: 23,588.
- Replay time: 27.3133 s.
- Replay wall fraction: 85.44%.
- Total benchmark time: 31.9691 s.

### End-to-end training smoke

A real 128-story default-replay `leo train` smoke completed normally. The last 16-story progress
window reported 378.663 steps/s. Pass validation on 64 stories reported loss 3.82271,
5.51500 bits/byte, and 0.21706 accuracy. The final training event reported 128 processed stories,
99,756 input bytes seen, and mean training loss 4.60836.

Nsight Compute was present in the Kaggle environment but hardware counter collection was blocked by
`ERR_NVGPUCTRPERM`; the internal CUDA phase profiler is therefore the authoritative phase evidence
for these snapshots.

### Profile that motivated Snapshot B

The one-story 32K replay profile showed `select_global` as the dominant operation:

- initial supervised path: 49.84% of profiled cycles;
- supervised replay chunks: roughly 46-56%;
- frozen prefixes of 67-652 steps: roughly 93.5-93.9%.

This established exact sparse global selection as the first CUDA optimization target.

## Snapshot B - exact sparse heap selector

- GitHub commit: `e23d2bdf587cee7520d9d80e429949fdc159ef80`.
- Pull request: #20, `perf(cuda): optimize sparse global selection`.
- Leo software version remained `1.0.2`; this is intentionally a separate snapshot because the
  GitHub commit changed while the semantic version did not.
- Evidence archive: `p100-heap-round1-evidence.tar.gz`.
- Evidence SHA-256:
  `79715ab145bf4ff2a0478e5fbd3d13aa35a156f92735288b96d1312eaf6cfe51`.
- Fresh environment: Rust 1.85.0, Python 3.12.13, Tesla P100-PCIE-16GB, CUDA 12.8 toolkit.
- `scripts/check_gpu.sh`: passed; optimized-vs-legacy CUDA exact-state gate passed using
  `leo_shared_wavefront_persistent_grouped` with device batch merge enabled.

### Throughput

| Measurement | Snapshot A | Snapshot B | Change |
| --- | ---: | ---: | ---: |
| 32K replay-off autotune prediction | 2,468.283 | 3,733.630 | 1.513x |
| 32K replay-off optimized, 64 stories | 2,403.340 | 3,628.031 | 1.510x |
| 32K replay-off legacy, 64 stories | 1,553.314 | 2,000.709 | 1.288x |
| Classic/default 30% replay, 16 stories | 453.437 | 1,897.610 | 4.185x |
| Classic replay execution throughput including prefixes | 1,086.392 | 4,546.489 | 4.185x |

The optimized 64-story replay-off run and legacy run again produced the same exact state digest:

`14894f9039c59359a81ebed48d71b563617636348fb4347f2accefc5bfcd50e9`

The 16-story classic 30% replay digest also remained identical to Snapshot A:

`5b30b3f442f3aecf17a6a41e1b98b21b802cd27aa70fbe0c79cb1f9257527e6e`

### GPU acceptance

The real P100 GPU gate completed with exit 0. CPU/GPU logical-batch conformance was ready at replay
0 and replay 0.30. The optimized-vs-legacy 16-worker exact-state gate produced the same small-model
digest on both paths:

`fe42ee87c14a330a0f35fbd5e346798221aebf181914bf7807f5658ea868caca`

The observed optimized kernel was `leo_shared_wavefront_persistent_grouped`, and device batch merge
telemetry confirmed `leo_merge_shared_lane_fixed_parameters` executed.

### Replay cost after the heap selector

- Replay target/prefix counts were unchanged from Snapshot A.
- Replay time fell from 27.3133 s to 4.58293 s: 5.960x faster, an 83.22% reduction.
- Replay wall fraction fell from 85.44% to 59.99%.
- Total 16-story classic benchmark time fell from 31.9691 s to 7.63908 s.

### New production bottleneck

The production-geometry grouped persistent profile at 32K, replay off, reported:

| Phase | Share of profiled cycles |
| --- | ---: |
| learning signals | 42.02% |
| post core | 21.93% |
| selection | 14.85% |
| pre | 9.36% |
| post select | 5.09% |
| post deltas | 3.66% |
| homeostasis | 2.38% |
| cache surrogate | 0.61% |
| capture | 0.08% |

This makes the exact output-error-to-neuron learning-signal projection the next base-kernel target.

### Replay profile after the heap selector

For the one-story 32K replay profile:

- initial supervised `select_global`: 11.29%; `learning_signals`: 18.75%;
- supervised replay chunks: `select_global` roughly 10.8-13.3%, `learning_signals` roughly
  15.3-17.9%;
- long frozen prefixes of 290, 448, and 652 steps: `select_global` remained about 50.0-50.4%,
  with `pre` about 28.6-29.1% and `select_blocks` about 17.3-18.0%;
- replay-prefix execution for the profiled story was 6,173 steps/s and supervised replay-target
  execution was 1,859 steps/s.

The profiled one-story 32K replay benchmark improved from 331.679 to 1,182.577 reported steps/s.
Its replay time fell from 2.03244 s to 0.355559 s, a 5.716x replay-time improvement, while the
training-state digest remained `cde6fe3c4fa5a302e7557a91e2c9ce63ad948a1d663500b425a036cd07c084aa`.

The heap selector therefore removed the first dominant bottleneck without changing replay work or
state, but exact replay-prefix selection and the production learning-signal projection remain major
cost centers.

## Leo v1.0.3

## Snapshot A - four-row learning-signal interleave regression

- GitHub release commit:
  `a68b4bfb41b8503522aefa2e755ba4e4fb1c18ec`.
- Performance change: PR #21,
  `perf(cuda): interleave learning signal projection`.
- Release metadata: PR #22, `chore(release): v1.0.3`.
- Evidence archive: `p100-v103-round2-regression-evidence.tar.gz`.
- Evidence SHA-256:
  `4a6a703e8ff7e8d1681d252b80016ce13e26d89c342da04f4584f8e3c13f00dc`.
- Hardware: Tesla P100-PCIE-16GB, compute capability 6.0, 56 SMs.
- CUDA toolkit: 12.8.
- Rust: 1.85.0.
- Dataset and prepared-file SHA-256 values matched the reference TinyStories identity above.
- `scripts/check_gpu.sh` passed.
- Optimized-vs-legacy GPU state equality passed.

### Throughput

| Measurement | v1.0.2 heap | v1.0.3 interleave | Change |
| --- | ---: | ---: | ---: |
| 32K replay-off autotune | 3,733.630 | 2,433.395 | 0.652x |
| 32K replay-off optimized, 64 stories | 3,628.031 | 2,408.681 | 0.664x |
| 32K replay-off legacy, 64 stories | 2,000.709 | 791.037 | 0.395x |
| Classic/default 30% replay, 16 stories | 1,897.610 | 1,559.470 | 0.822x |

The replay-off optimized and legacy executions remained exactly state-identical:

`14894f9039c59359a81ebed48d71b563617636348fb4347f2accefc5bfcd50e9`

The classic 30% replay digest also remained unchanged:

`5b30b3f442f3aecf17a6a41e1b98b21b802cd27aa70fbe0c79cb1f9257527e6e`

### Regression diagnosis

The four-row learning-signal interleave is rejected for P100 production use.

The 32K production phase profile changed from 42.02% learning-signal share in the exact heap
baseline to 55.45%. Replay-off optimized throughput fell by 33.6%, and the legacy path using the
same shared learning-signal helper fell by 60.5%.

This was not explained by the autotuner selecting `lane_chunk=8`: the measured
`lane_chunk=16`, 56-block candidates were also only about 2,395 work-units/s.

Classic replay regressed less severely because frozen replay-prefix reconstruction does not execute
the learning-signal projection. Replay execution time itself remained approximately unchanged at
4.57 seconds.

The implementation is therefore reverted rather than used as the basis for further optimization.

## Next snapshot

Append the next merged optimization here. Record the version reported by the GitHub source at that
commit even when it remains `1.0.2`, and always include the exact commit SHA/PR so same-version
performance snapshots stay distinguishable.
