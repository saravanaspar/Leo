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

- GitHub release commit: `a68b4bfb41b8503522aefa2e755ba4e4fb1c18ec`.
- Performance change: PR #21, `perf(cuda): interleave learning signal projection`.
- Release metadata: PR #22, `chore(release): v1.0.3`.
- Evidence archive: `p100-v103-round2-regression-evidence.tar.gz`.
- Evidence SHA-256:
  `4a6a703e8ff7e8d1681d252b80016ce13e26d89c342da04f4584f8e3c13f00dc`.
- Hardware: Tesla P100-PCIE-16GB, compute capability 6.0, 56 SMs.
- CUDA toolkit: 12.8.
- Rust: 1.85.0.
- Dataset and prepared-file SHA-256 values matched the reference TinyStories identity above.
- `scripts/check_gpu.sh` passed, including optimized-vs-legacy GPU state equality.

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

The four-row learning-signal interleave is rejected for P100 production use. The 32K production
phase profile changed from 42.02% learning-signal share in the exact heap baseline to 55.45%.
Replay-off optimized throughput fell by 33.6%, and the legacy path using the same shared learning-
signal helper fell by 60.5%.

This was not explained by the autotuner selecting `lane_chunk=8`: measured `lane_chunk=16`,
56-block candidates were also only about 2,395 work-units/s. Classic replay regressed less severely
because frozen replay-prefix reconstruction does not execute the learning-signal projection; replay
execution time itself remained approximately unchanged at 4.57 seconds.

The implementation is therefore reverted rather than used as the basis for further optimization.

## Historical reconstruction - CUDA optimization phases after the original ledger

This section is an append-only reconstruction from the GitHub commit/PR history plus the P100
progress report. It exists because several important optimizations, correctness failures, test-harness
failures, and reverted experiments happened after the original v1.0.3 snapshot but were never added
to this ledger.

When a phase below does not have an isolated P100 before/after measurement, this document says so
rather than inventing one. Later measurements are used only when they can be tied to a known commit
or exact code state.

### Phase 0 - replay-prefix specialization and measurement infrastructure

#### PR #8 - frozen replay prefix fast path

- Merge commit: `476bb6b8ee821f75432df90241ad3c613ee2f0d2`.
- PR: #8, `Optimize bounded-surprise replay prefix execution`.
- GitHub: <https://github.com/saravanaspar/Leo/pull/8>.

What changed:

- replay prefixes stopped using the normal supervised `training_step_batch` path;
- a dedicated `advance_frozen_batch` backend operation was introduced;
- the CUDA backend gained a persistent frozen-prefix kernel;
- output-only work that cannot influence the next recurrent state was skipped during frozen prefix
  reconstruction;
- replay cleanup was deferred until the last selected range where possible.

Why this was expected to help:

- canonical replay contains far more hidden prefix steps than replay target steps;
- frozen prefix reconstruction does not need loss/output-update work;
- avoiding output-only work and repeated cleanup reduces work without changing the recurrent state.

Result/lesson:

- Git history preserves the structural optimization but not an isolated P100 before/after number for
  this commit alone.
- The architectural direction was retained and later profiling confirmed that hidden prefix work is
  a dominant part of replay cost.

#### PR #9 - cooperative frozen-prefix selection

- Merge commit: `03ead2ef3e42e4e0d555532c9104cce6a9a7a896`.
- PR: #9, `Parallelize frozen replay prefix selection`.
- GitHub: <https://github.com/saravanaspar/Leo/pull/9>.

What changed:

- the one-block frozen prefix path gained a cooperative multi-CTA implementation;
- model-block selection work was distributed across the cooperative grid;
- the legacy single-block persistent path remained as a fallback;
- the implementation preserved the exact phase/reduction ordering required by the v1 semantics.

Why this was expected to help:

- per-block selection is independent across model blocks until the exact global-selection boundary;
- P100 exposes enough cooperative capacity to spread this work across many CTAs.

Result/lesson:

- no isolated accepted P100 throughput number is preserved for this exact commit;
- later profiles show frozen prefix selection is still dominated by the exact global selection step,
  so parallelizing only block-local work cannot remove the entire prefix bottleneck.

#### PRs #10 and #11 - CUDA phase profiling

- #10 merge: `4c36d27e49b326e9a6053d7e6a1cc4149494ed1f`.
- #11 merge: `c6c1fac35a7cfbff2163ac1d0e27654ba456eae1`.

These changes were measurement infrastructure, not throughput claims. They made it possible to
separate `pre`, block/global selection, post-core, learning-signal, update, homeostasis, capture,
and replay-prefix phases without relying on unavailable Nsight hardware counters in Kaggle.

Permanent lesson:

> Measurement instrumentation is part of the optimization stack. Do not replace a measured
> bottleneck with an assumed bottleneck.

#### PR #12 - cooperative supervised replay path and replay diagnostics

- Merge commit: `8ddc6fe61c383d2b751efc053688454c28c3cdf3`.
- PR: #12, `Optimize replay CUDA path and add detailed diagnostics`.

What changed:

- supervised replay gained cooperative whole-grid execution;
- the legacy replay kernel was retained for A/B debugging;
- hidden replay-prefix steps were counted explicitly;
- replay timing, launch, transfer, synchronization, selection and sampled phase diagnostics were
  added.

Permanent lesson:

- reported replay targets are not the true replay workload;
- hidden prefix reconstruction must always be measured separately.

#### PR #13 - exact packed winner selection

- Merge commit: `a3f8f45480f69fbda12d00d8c04208f61b6583b5`.
- PR: #13, `Optimize exact CUDA winner selection`.

What changed:

- per-block and global winner comparison moved to packed exact comparison keys;
- already-sorted block winner runs were reused;
- activation ordering, rotated tie-break behavior, and historical block-cutoff behavior were
  preserved;
- unsupported/non-finite cases retained the legacy fallback.

This improved the selection foundation, but later profiling still found the sparse global merge to
be the dominant replay cost. That led to PR #20's heap merge rather than further arithmetic changes.

### Phase 1 - multi-GPU and persistent/grouped execution foundation

#### PRs #14-#16 - scaling and synchronization discipline

- PR #14 introduced the multi-GPU scaling path.
- PR #15 fixed scaling barriers/synchronization.
- PR #16 added repeated clean-run measurements and required repeated final-state hashes to agree.

The main durable lesson for single-GPU work is the same one that later mattered on the P100 grouped
kernel:

> Synchronization correctness and repeatability must be proven under the maximum intended
> concurrency, not only under a small grid.

A second durable lesson from PR #16 is that a single performance run is insufficient for small
changes. Repeated min/median/max measurements are required when differences are near noise.

#### PR #17 - persistent/grouped CUDA acceleration foundation

- Merge commit: `c5dfade3b8ba378704ea195bcaed5431b6dafd41`.
- PR: #17, `perf(cuda): accelerate training and replay`.

This was the broad architecture phase that added/combined:

- persistent/grouped CUDA execution;
- device-side merge paths;
- replay acceleration experiments;
- correctness gates;
- training/performance documentation.

The later v1.0.2 pre-heap snapshot in this ledger is the durable measured baseline for the resulting
architecture: 2,403.340 replay-off optimized steps/s and 453.437 canonical replay-30 steps/s before
the exact sparse heap selector.

### Phase 2 - exact sparse heap selector: the major accepted breakthrough

PR #20 (`e23d2bdf587cee7520d9d80e429949fdc159ef80`) replaced repeated linear sparse-global winner
scans with an exact shared-memory heap merge. The full accepted results are already recorded above,
but the engineering conclusion deserves to be permanent:

- replay-off: 2,403.340 -> 3,628.031 steps/s (`1.510x`);
- canonical replay-30: 453.437 -> 1,897.610 steps/s (`4.185x`);
- replay time: 27.3133 -> 4.58293 seconds (`5.960x` faster);
- both historical state hashes remained exact.

Why this worked:

- the profiler had identified `select_global` as approximately 93-94% of long frozen-prefix cycles;
- the optimization attacked the measured dominant algorithm rather than only changing memory
  placement or occupancy.

Permanent lesson:

> Large wins came from changing the complexity of the dominant exact algorithm, not from cosmetic
> occupancy or cache changes.

### Phase 3 - learning-signal interleave regression and revert

PR #21 / v1.0.3 interleaved four independent learning-signal rows per thread so each loaded error
could be reused across four destination rows. It preserved the FP32 accumulation order within each
row and preserved exact hashes, but it was a severe P100 regression.

Measured regression:

| Workload | Exact heap baseline | Four-row interleave | Change |
| --- | ---: | ---: | ---: |
| Replay-off optimized | 3,628.031 | 2,408.681 | -33.6% |
| Replay-off legacy | 2,000.709 | 791.037 | -60.5% |
| Canonical replay-30 | 1,897.610 | 1,559.470 | -17.8% |

The production learning-signal share increased from 42.02% to 55.45%.

Why the hypothesis looked reasonable:

- output errors are reused across many neuron rows;
- interleaving independent rows appeared to increase instruction-level parallelism and reuse each
  error load.

Why it failed in practice:

- extra live accumulators/state increased pressure on the P100 execution pipeline;
- the shared helper also hurt the legacy path, proving the regression was in the learning-signal
  implementation itself rather than the grouped scheduler;
- exactness tests alone could not reveal the regression.

PR #23 (`c31b3b5f9e97e282d820efed95399dac9d73fe15`) reverted the experiment.

Permanent rule:

> Never accept a learning-signal micro-optimization from static reasoning alone. It must beat the
> canonical 32K P100 benchmark and preserve the historical hashes.

### Phase 4 - PR #24 hardware-adaptive shared-story execution

- Merge commit: `420ba89bfca029c679679a47ee94dc05bce28237`.
- PR: #24, `perf(cuda): make story execution hardware-adaptive`.

The phase attempted to remove host and lockstep overhead by:

- consuming the tuner-approved cooperative CTA budget;
- using grouped shared-story execution;
- allowing story lanes to advance through their own story lengths;
- moving normal story-step descriptor generation onto the GPU;
- reducing per-step records into per-story summaries on-device;
- selecting exact bounded-surprise replay ranges on-device;
- retaining GPU fixed-parameter batch merge while keeping order-sensitive keyed context merge at the
  exact host safe point.

This was directionally important because it moved orchestration and postprocessing off the host.
However, the custom cross-CTA lane synchronization used by the high-occupancy grouped path was later
shown to be unsafe at full occupancy.

Permanent lesson:

> A synchronization scheme that is exact at 16 or 32 CTAs is not proven exact at 112 CTAs.

### Phase 5 - optimized representation/reporting failures were not arithmetic failures

#### PR #26 - device-postprocess result dispatch bug

- Commit: `656e85332be562acfc147e1c44a620bcb39fc454`.
- PR: #26, `fix(cuda): recognize device-postprocessed story metrics`.

P100 diagnosis showed:

- max parameter delta: `9.313226e-10`;
- training loss sum delta: `4.959106e-5`;
- mean loss delta: `5.547651448`.

The learned state and accumulated loss were effectively matching. The bug was host report dispatch:
GPU postprocessing returned one empty per-step metrics vector per logical story, so testing only
`story_metrics.is_empty()` incorrectly selected the legacy host-metrics path.

Fix:

- require the expected per-story empty-vector representation together with one device summary and
  replay-range vector per story.

Permanent lesson:

> When an optimized path changes the shape of host-visible data, validate the full representation
> contract. Do not infer the execution path from one container's outer emptiness.

#### PR #30 - diagnostics accidentally coupled to the host replay selector

- Commit: `7d5afc282791e82176856cea3d4dcad8707296cd`.
- PR: #30, `fix(cuda): restore device-native replay diagnostics`.

Root cause:

- GPU replay selection correctly bypassed the historical host `apply_replay_policy` path;
- the old debug event emitter lived inside that bypassed host path;
- the GPU gate therefore failed despite valid GPU execution.

Fix:

- emit device-native replay-selection and batch-summary diagnostics from already selected compact
  device ranges without downloading per-step losses.

Permanent lesson:

> Correctness/diagnostic gates must observe the optimized path directly; they must not require a
> fallback path merely to generate telemetry.

### Phase 6 - full-capacity grouped race, exact synchronization, and residency recovery

#### PR #31 - high-occupancy race diagnosis and correctness fix

- Commit: `b58c4d50e922adf9a399f899028de264cae0f636`.
- PR: #31, `fix(cuda): synchronize grouped story phases cooperatively`.

Isolation result:

- 16 CTAs: exact;
- 32 CTAs: exact;
- 112 CTAs run A: non-exact;
- 112 CTAs run B: non-exact with a different state hash.

The two 112-CTA executions disagreeing with each other proved a real cross-CTA synchronization race,
not deterministic FP32 arithmetic drift. The 112-block grid was evenly divisible by 16 logical
lanes, eliminating remainder distribution as the cause.

Fix:

- retire the unsafe custom cross-CTA lane barrier;
- use CUDA cooperative whole-grid `grid.sync()` boundaries;
- force every CTA to follow the same `max_step_count` schedule so no CTA can leave a cooperative
  barrier early.

Tradeoff:

- exactness returned at full occupancy;
- whole-grid synchronization and increased live state reduced practical residency/performance.

Permanent rule:

> Never replace a cooperative grid barrier with a hand-rolled cross-CTA barrier unless forward
> progress and memory ordering are proven for the full launch geometry and repeated full-capacity
> hashes agree.

#### PR #32 - stress-fixture failure was configuration, not CUDA

- Commit: `247d7cdb0cb227007d65ec69900813bf2e62a7b4`.

The new 4K grouped stress model increased `neuron_count` but left `max_active_global = 8`. Leo
correctly rejected the generated configuration because `target_activity` required 164 active
neurons. The fixture was changed to `max_active_global = 256` and the intended full-capacity CUDA
regression then ran.

Permanent lesson:

> Before diagnosing a GPU failure, prove the stress fixture itself satisfies model invariants.

#### PR #34 - recover grouped-kernel residency without weakening synchronization

- Commit: `027f9a7b07b796e144dc4a5778dfa593b6000cae`.
- PR: #34, `perf(cuda): restore grouped kernel residency`.

The fix shortened long-lived eligibility-pointer/config/grid-object state while preserving
cooperative whole-grid synchronization.

Grouped kernel capability improved from approximately:

```text
134 registers/thread
1 block/SM
56 cooperative blocks
12.5% theoretical occupancy
```

to:

```text
128 registers/thread
2 blocks/SM
112 cooperative blocks
25% theoretical occupancy
```

The repeated full-capacity gate exercised all 112 cooperative CTAs and still matched the legacy
state exactly.

However, the real 32K tuner continued to prefer 56 blocks because 112-block whole-grid barrier cost
outweighed the extra resident work. Therefore higher occupancy was a capability improvement, not an
end-to-end speed win by itself.

### Accepted main baseline before PR #35

At `027f9a7b07b796e144dc4a5778dfa593b6000cae`:

| Measurement | Result |
| --- | ---: |
| 32K replay-off optimized, 64 stories | 3,393.453279 steps/s |
| 32K replay-off legacy, 64 stories | 1,993.958446 steps/s |
| Canonical replay-30, 16 stories | 1,853.697928 steps/s |
| Canonical replay execution incl. prefixes | 4,441.279162 steps/s |
| Canonical replay time | 4.586416353 s |
| Canonical total time | 7.820044346 s |
| Replay wall fraction | 58.6495% |

Compared with the best accepted exact sparse-heap measurement (3,628.031 replay-off and 1,897.610
canonical replay), the correctness/synchronization line was approximately 6.47% lower replay-off and
2.31% lower on canonical replay.

The project accepted that correctness cost and made further work start from this exact mainline
rather than reintroducing the unsafe barrier.

---

## v1.0.40 - PR #35 FP32 overhead reduction

### Release identity

PR #35 was merged and released on 2026-09-12:

- PR: <https://github.com/saravanaspar/Leo/pull/35>.
- Base: `027f9a7b07b796e144dc4a5778dfa593b6000cae`.
- Final PR head: `fb77c1980a6617ee78f67edcdf2d42fa0f55ee2b`.
- Merge commit: `01eabbb3cda5b3420af6e735c9d7c30ed88d048e`.
- Release commit/tag: `9e393dab554302509532fb6c0c147ad166549744` / `v1.0.40`.

The P100 measurements in this section were collected immediately before the release automation,
while the branch still reported workspace version `1.0.32`. The release flow reconciled the public
software version to `1.0.40` without changing the v1 semantic/schema identifiers. The corrected
performance measurements correspond to the same shared-error-cache revert that shipped in PR #35,
but the exact `v1.0.40` release commit was not separately re-benchmarked in that Kaggle session.
Future acceptance runs should record the exact release/branch commit alongside the existing golden
hashes.

### PR #35 commit sequence

1. `7441cb85f3fed3c1cc1702db88d89fa08b80f1c8` - `perf(cuda): reduce FP32 training and replay overhead`.
2. `3b4677e561c3eca846cc2c3d1f6ebe630b98d2f7` - training-guide update on the same branch.
3. `16f705fd2111276bd668803213964964fab71d70` - `perf(cuda): remove regressing shared error cache`.
4. `fb77c1980a6617ee78f67edcdf2d42fa0f55ee2b` - expanded CUDA performance history and lessons before merge.

The third commit is essential: the initial optimization bundle contained a learning-signal
shared-memory cache that preserved exact state but caused a large P100 throughput regression.

### 2026-09-12 Kaggle P100 environment

```text
GPU:             Tesla P100-PCIE-16GB
SM count:        56
Compute cap.:    6.0
Driver:          580.159.04
CUDA toolkit:    12.8 / nvcc 12.8.93
Rust:            1.85.0
Python:          3.12.13
Logical workers: 16
FP32:            unchanged
Canonical replay: 0.30
```

The Kaggle session cloned branch head `3b4677e...` for the initial tests, then manually A/B removed
the shared-error cache. That manual revert was subsequently committed on Kubuntu as `16f705f...`.
Therefore the corrected performance numbers below correspond to the same `cuda_kernels.cu` revert
that became `16f705f`, but the performance session itself was not recloned to the exact
`16f705f` commit before measurement. Re-run the acceptance commands on the exact released commit
for the final post-release snapshot.

### Dataset identity in this Kaggle session

Repository-pinned TinyStories source revision remained:

`5485261731eaac25dd8e5ebbc3839d0a9870b185`

The full prepared dataset manifest reported:

| Artifact | Value |
| --- | --- |
| Train stories | 2,119,489 |
| Train bytes | 1,891,291,909 |
| Train bytes SHA-256 | `6f9928713c3ffb6285c032c0122ce7192f606974f1895939677aa08624db3d8f` |
| Train index-records SHA-256 | `d8ce3ec224d05c5d576d909f365eaaa1e6e4134a4b97cc49b459c1a2df720bce` |
| Train dataset id | `da4fcc7dfd6d0b3ff8080cb85cb6c2e298d1af109b715c724c049a9ffed189be` |
| Validation stories | 21,990 |
| Validation bytes | 19,105,764 |
| Validation bytes SHA-256 | `56e7c41c62fc31c92145a9320b45a044b7dc8eda10fcab763cc7c9b28bbe805e` |
| Validation index-records SHA-256 | `78350a079698f3a969f0049e15d14c57919bd225104f066159b280799bb051a6` |
| Validation dataset id | `bdd64859116c5b062888069413f2d16dc24cc785f6eda0fb470cb506c2323fad` |

The canonical 64-story replay-off benchmark consumed 45,937 input bytes. The canonical 16-story
replay-30 benchmark consumed 11,127 input bytes and generated exactly the historical 3,353 replay
targets plus 20,235 hidden prefix steps.

### What PR #35 attempted

The reviewed bundle deliberately kept FP32 and canonical semantics. It included:

- autotuner schema bump plus intermediate P100 candidate widths `64, 72, 80, 84, 96`;
- replay/frozen kernel pointer/config/grid-object lifetime shortening to reduce register pressure;
- grouped production barrier fusion only where dependencies were proven independent;
- replay reset cleanup that stops clearing payload arrays hidden by zeroed counts;
- a replay-only deferred document-reset path while preserving the public synchronous
  `begin_document()` contract;
- target-zero replay fusion so `BEGIN_DOCUMENT -> first target` and remaining ordered targets share
  one supervised batch;
- the now-rejected shared-memory error cache for learning signals.

The grouped production source was reduced from about 19 whole-grid synchronization sites per
supervised step to 17 by fusing only boundaries where the touched arrays are independent. The
recurrent-vs-input eligibility barrier and other order-sensitive boundaries were intentionally kept.

### Autotuner expansion result

The new candidate widths were actually exercised on the 32K P100 warmup. With the initial
shared-error-cache build, approximate per-candidate measured wall throughput was:

| Lane chunk | Fused blocks | Approx. mean work/s |
| ---: | ---: | ---: |
| 16 | 28 | 1,539.9 |
| 16 | 56 | 2,723.4 |
| 16 | 64 | 2,088.5 |
| 16 | 72 | 2,267.5 |
| 16 | 80 | 2,323.1 |
| 16 | 84 | 2,427.0 |
| 16 | 96 | 2,474.5 |
| 16 | 112 | 2,461.2 |
| 8 | 56 | 2,726.0 |
| 8 | 112 | 2,251.8 |

The tuner chose 56 blocks with lane chunk 8. The 56-block lane-8/lane-16 difference was below
0.1% in this noisy warmup and therefore not meaningful.

A forced lane-chunk-16 A/B later proved that this near-tie was not the regression root cause.

Permanent lesson:

> Intermediate grid candidates are useful, but a larger grid is not automatically faster. On the
> current cooperative kernel, barrier cost still makes 56 blocks faster than 64-112 on the measured
> 32K workload.

### Regression incident - shared-memory `errors[257]` cache

#### Hypothesis

The base production profile showed learning signals at roughly 40-42% of cycles. Each destination
neuron computes an FP32 dot product against the same 257-element output-error vector. It appeared
reasonable to copy that approximately 1 KiB error vector into CTA shared memory and reuse it across
rows.

The implementation preserved the exact output accumulation order, so historical hashes still
matched.

#### Initial result with the cache enabled

| Measurement | Main baseline | Shared-error-cache build | Change |
| --- | ---: | ---: | ---: |
| Replay-off, 64 stories | 3,393.453279 | 2,653.625324 | -21.80% |
| Canonical replay-30 | 1,853.697928 | 1,612.802738 | -13.00% |
| Replay time | 4.586416 s | 4.762070 s | +3.83% |
| Total canonical time | 7.820044 s | 8.988080 s | +14.94% |

The replay-off historical hash remained:

`14894f9039c59359a81ebed48d71b563617636348fb4347f2accefc5bfcd50e9`

The replay-30 historical hash remained:

`5b30b3f442f3aecf17a6a41e1b98b21b802cd27aa70fbe0c79cb1f9257527e6e`

The timing split was diagnostic:

```text
baseline non-replay time = 7.820044 - 4.586416 = 3.233628 s
bad-cache non-replay     = 8.988080 - 4.762070 = 4.226010 s
non-replay regression    = +30.69%
```

Replay itself regressed only 3.83%, so the main damage was clearly in the base grouped path.

#### False lead tested - lane chunk 8 versus 16

The warmup selected lane chunk 8 by a negligible margin. To rule out tuner noise, the saved plan was
forced to:

```text
physical_lane_chunk = 16
fused_wavefront_blocks = 56
fused_wavefront_threads = 256
```

Result:

| Measurement | Cache build, auto plan | Cache build, forced lane-16 |
| --- | ---: | ---: |
| Replay-off | 2,653.625 | 2,634.227 steps/s |
| Canonical replay-30 | 1,612.803 | 1,624.635 steps/s |
| Canonical replay time | 4.762070 | 4.763044 s |

Forcing lane chunk 16 did not recover performance. Therefore the tuner near-tie was not the root
cause.

#### Isolation test - remove only shared-error caching

Only the following experiment was reverted:

- remove the two `__shared__ float shared_errors[LEO_OUTPUTS]` arrays;
- replace cached learning-signal calls with the original exact global-read helper;
- remove the now-unused cached helper.

Everything else in PR #35 remained.

Result:

| Measurement | Main baseline | No-error-cache PR #35 state | Change |
| --- | ---: | ---: | ---: |
| Replay-off, 64 stories | 3,393.453279 | 3,406.244907 | +0.38% |
| Canonical replay-30 | 1,853.697928 | 1,853.689440 | -0.0005% |
| Replay execution incl. prefix | 4,441.279162 | 4,441.258827 | -0.0005% |
| Replay time | 4.586416 | 4.588256 s | +0.04% |
| Total canonical time | 7.820044 | 7.820080 s | +0.0005% |

Both historical hashes remained exact.

Interpretation:

- the shared-error cache caused essentially the entire observed regression;
- the corrected PR #35 bundle is end-to-end performance-neutral within measurement noise on the
  canonical replay workload and slightly above the current-main replay-off measurement;
- the remaining low-risk barrier/reset/launch/register changes therefore stay, but they must not be
  advertised as an 8K breakthrough.

Why the cache likely failed on P100:

- the error vector is tiny (~1 KiB) and read-only, so normal cache/broadcast behavior is already
  favorable;
- every participating CTA paid to copy all 257 values;
- every CTA paid a `__syncthreads()` before useful learning-signal work;
- sparse/imbalanced CTAs can pay that fixed overhead even when they have little learning-signal work;
- the optimization attacked placement of a tiny shared input instead of the much larger
  neuron-major output-weight traffic.

Permanent rule:

> Do not reintroduce a CTA-wide shared-memory copy of the 257-element error vector on P100 without a
> fresh isolated A/B that beats the original global-read implementation. The previous attempt was
> exact but approximately 22% slower replay-off.

### Resource telemetry during PR #35 validation

Before the shared-error cache was removed, the P100 GPU gate reported:

| Kernel | Registers/thread | Blocks/SM | Theoretical thread occupancy |
| --- | ---: | ---: | ---: |
| `leo_shared_wavefront_persistent_grouped` | 128 | 2 | 25% |
| `leo_train_cooperative_fast` | 144 | 1 | 12.5% |
| `leo_advance_frozen_cooperative` | 104 | 2 | 25% |

The important replay-kernel change is the register direction: the earlier roadmap measured roughly
166 registers/thread for `leo_train_cooperative_fast`; PR #35's pointer-lifetime cleanup reached 144
in this gate. That is progress, but the replay kernel is still one CTA/SM. The <=128-register target
remains open.

Because this resource snapshot was emitted before the error-cache revert, rerun the resource gate on
the exact released commit before recording final static-shared-memory numbers.

### Corrected 32K base-path phase profile

After removing shared-error caching, `LEO_CUDA_PHASE_PROFILE=1` on the 16-story replay-off workload
reported:

| Phase | Share |
| --- | ---: |
| learning signals | 40.6503% |
| post core | 21.1919% |
| selection | 18.1701% |
| pre | 9.0710% |
| post select | 4.9933% |
| post deltas | 3.0425% |
| homeostasis | 2.2538% |
| cache surrogate | 0.5518% |
| capture | 0.0755% |

The instrumented benchmark itself reported 3,606.818 steps/s, but profiling changes timing and that
number is not an acceptance throughput result. The phase shares are the useful evidence.

Compared with the earlier exact heap profile, learning signals remain the largest base-kernel phase.
The failed shared-error cache proves that the next learning-signal optimization must target the real
weight-access/computation structure rather than blindly staging the tiny error vector.

### Corrected canonical replay profile

The full 16-story replay profile accounted for every canonical replay step:

```text
replay targets:       3,353
hidden prefix steps: 20,235
```

Aggregating all 73 frozen-prefix profile records and all 77 replay-target profile records gives:

#### Frozen prefix - 20,235 sampled steps

| Phase | Weighted share of prefix cycles |
| --- | ---: |
| global selection | 50.2311% |
| pre | 28.5779% |
| block selection | 17.8542% |
| post/emit | 3.3368% |

#### Supervised replay targets - 3,353 sampled steps

| Phase | Weighted share of replay-target cycles |
| --- | ---: |
| pre | 32.2254% |
| learning signals | 16.3662% |
| global selection | 12.8814% |
| block selection | 9.1565% |
| forward | 7.6793% |
| capture | 4.8798% |
| recurrent update | 3.8852% |
| recurrent eligibility | 3.3647% |
| input eligibility | 2.8851% |
| homeostasis | 1.7839% |
| input update | 1.5166% |
| cache surrogate | 1.0821% |
| post/emit | 0.9118% |
| output update | 0.7720% |
| context update | 0.6099% |

The instrumented replay-profile benchmark reported 1,756.904 steps/s and 5.01979 seconds of replay,
but this slowdown is profiling overhead; the exact canonical hash remained unchanged.

Across profiled replay kernel cycles:

- frozen-prefix reconstruction contributed 64.68%;
- supervised replay-target execution contributed 35.32%.

Largest contributors to all profiled replay cycles were:

| Combined replay component | Share of all profiled replay cycles |
| --- | ---: |
| prefix global selection | 32.49% |
| prefix pre | 18.48% |
| prefix block selection | 11.55% |
| target pre | 11.38% |
| target learning signals | 5.78% |
| target global selection | 4.55% |
| target block selection | 3.23% |
| target forward | 2.71% |

This makes the next replay target unambiguous: exact frozen-prefix global selection is the single
largest replay kernel cost.

### What did and did not improve in PR #35

Accepted/retained engineering changes:

- replay target-zero launch fusion: removes one supervised launch/sync for replay ranges beginning at
  zero while keeping the exact ordered target sequence;
- replay-only deferred reset: overlaps reset enqueue with CPU preparation without weakening the
  public synchronous `begin_document()` contract;
- reset payload-clear reduction: stale payload arrays are hidden by zeroed counts or overwritten
  before use;
- replay/frozen pointer-lifetime cleanup: reduced observed replay kernel register pressure;
- grouped barrier fusion at proven-independent boundaries;
- wider autotuner candidate search space.

Not demonstrated as an end-to-end canonical speedup yet:

- the corrected PR as a whole is essentially equal to current main on canonical replay;
- expanded 64-96 block tuner candidates did not beat 56 blocks in this run;
- reduced replay register pressure has not yet crossed the 2-CTA/SM threshold;
- replay remains host-orchestrated per selected segment;
- frozen-prefix global selection remains approximately 50% of prefix cycles.

Rejected:

- shared-memory `errors[257]` cache: exact but strongly slower on P100.

### Reproduction commands for final release acceptance

Use a clean P100 environment and a fresh CUDA tuning cache.

Replay-off exact control:

```bash
CUDA_VISIBLE_DEVICES=0 ./target/release/leo benchmark \
  --train \
  --model runs/p100-acceptance/replay0.pscls \
  --bytes data/prepared/tinystories.train.bytes \
  --index data/prepared/tinystories.train.idx \
  --stories 64 \
  --workers 16 \
  --backend gpu
```

Required hash:

`14894f9039c59359a81ebed48d71b563617636348fb4347f2accefc5bfcd50e9`

Canonical replay-30:

```bash
CUDA_VISIBLE_DEVICES=0 ./target/release/leo benchmark \
  --train \
  --model runs/p100-acceptance/replay30.pscls \
  --bytes data/prepared/tinystories.train.bytes \
  --index data/prepared/tinystories.train.idx \
  --stories 16 \
  --workers 16 \
  --backend gpu
```

Required hash:

`5b30b3f442f3aecf17a6a41e1b98b21b802cd27aa70fbe0c79cb1f9257527e6e`

Do not enable `LEO_REPLAY_STREAMING` for canonical acceptance.

---

## Rejected or conditional ideas - do not repeat blindly

This table is intentionally explicit so future optimization rounds do not rediscover the same
failures.

| Idea | Why it looked useful | Actual result / danger | Rule going forward |
| --- | --- | --- | --- |
| Four-row learning-signal interleave | Reuse each error load across independent rows | Exact hashes, but replay-off -33.6%; learning-signal share rose to 55.45% | Rejected on P100; require isolated canonical A/B before any similar multi-row interleave |
| Shared-memory `errors[257]` cache | Error vector is reused by all rows | Exact hashes, but replay-off about -22%; fixed CTA copy + barrier dominated | Do not reintroduce without new measured evidence |
| Force higher CTA occupancy | 112 CTAs gives 25% theoretical occupancy | 112-block production candidate is slower than 56 because `grid.sync()` cost grows | Optimize steps/s, not occupancy percentage |
| Hand-rolled per-lane cross-CTA barrier | Let stories advance independently without whole-grid waiting | Nondeterministic wrong state at 112 CTAs | Use cooperative synchronization unless a replacement is proven at full capacity |
| Treat exactness as sufficient | Both failed learning-signal experiments preserved hashes | Large real throughput regressions were still possible | Every exact patch must also beat canonical throughput |
| Diagnose from one tuner choice | Lane chunk 8 barely beat 16 in a noisy warmup | Forced lane 16 did not fix the regression | A/B suspected scheduler choices before blaming them |
| Use one performance run for small deltas | Faster iteration | 1-3% can be noise; <0.1% definitely is not a decision-quality margin | Warm up and repeat; report median/range |
| Couple diagnostics to host fallback | Existing debug emitter already worked | Device-native replay bypassed it and the GPU gate falsely failed | Diagnostics must follow the optimized data path |
| Infer device-postprocess mode from outer vector emptiness | Simple predicate | One empty vector per lane was misclassified and mean-loss reporting broke | Validate the complete host-visible representation |
| Assume a GPU test fixture is valid | Kernel failure seemed likely | 4K fixture violated `max_active_global` before CUDA ran | Validate model invariants before GPU diagnosis |
| Streaming replay as canonical speedup | It was faster in early experiments | State digest differs from classic replay | Diagnostic/experimental only unless semantics are explicitly changed in a new contract |

---

## Performance experiment protocol after PR #35

Every future performance PR should record all of the following in this ledger.

### 1. Correctness wall

- `scripts/check.sh` passes;
- `scripts/check_gpu.sh` passes;
- repeated full-capacity grouped run is deterministic;
- replay-off 32K hash equals the historical replay-off hash;
- canonical replay-30 hash equals the historical replay hash;
- optimized and legacy/reference paths match where the gate requires them.

### 2. Fresh-state rule

- start benchmark comparisons from the same model file/state;
- use a fresh tuning cache after kernel-source or tuning-schema changes;
- do not compare a candidate after another candidate has mutated the same in-memory model unless the
  benchmark harness explicitly resets/restores it.

### 3. Noise rule

- warm up first;
- repeat measurements for small deltas;
- treat approximately 1-3% differences cautiously;
- never redesign around a sub-1% tuner difference without a forced A/B.

### 4. Required identity fields

Record:

```text
GPU + SM count
CUDA toolkit + driver
Rust/Python versions when relevant
commit SHA + PR
software version/tag
TinyStories source revision
prepared artifact hashes/dataset id
model config
workers
replay fraction
selected tuner plan
kernel resource telemetry
training-state hash
steps/s
replay seconds + replay wall fraction
phase profile when the patch targets a kernel phase
```

### 5. Negative-result rule

If an optimization is exact but slower, keep the negative measurement in this ledger and revert the
implementation. The four-row interleave and shared-error-cache incidents are examples. Negative
results are valuable because they prevent future contributors from repeating plausible but harmful
micro-optimizations.

---

## Current optimization frontier after PR #35

The corrected PR #35 does not materially change the canonical 1,853.7 steps/s baseline. The next
round should therefore target the measured dominant costs rather than additional tiny cache changes.

Priority order from current evidence:

1. **Frozen-prefix exact global selection** - 50.23% of prefix cycles and 32.49% of all profiled
   replay cycles.
2. **Base learning signals** - 40.65% of the corrected replay-off production profile.
3. **Base post-core + selection** - together another ~39% of base profiled cycles.
4. **Replay-target pre phase** - 32.23% of replay-target cycles.
5. **Replay kernel register pressure** - continue from the observed 144 registers/thread toward a
   profitable 2-CTA/SM point if compiler/resource evidence supports it.
6. **Device-resident exact replay segment orchestration** - eliminate per-segment host reset/prefix/
   target orchestration while preserving sequential parameter visibility and exact replay semantics.

For the current canonical 14,496-step run:

```text
current total time ~= 7.8201 s
current replay     ~= 4.5883 s
current non-replay ~= 3.2318 s
8K target time      = 1.8120 s
```

Even infinitely fast replay would leave only about 4,485 reported steps/s at the current non-replay
cost. Even infinitely fast base work would leave only about 3,159 reported steps/s at the current
replay cost. Reaching 8K therefore requires major improvements on both sides.

---

## Post-merge release fill-in

After PR #35 is merged and release metadata is finalized, append - do not rewrite the pre-release
A/B history above - the following:

```text
actual merge SHA:
actual release commit SHA:
actual tag/version:
exact source version:
final P100 check_gpu.sh result:
final replay0 steps/s + hash:
final replay30 steps/s + hash:
final replay seconds / total seconds:
final selected tuner plan:
final grouped/replay/frozen kernel resource telemetry:
```

The next optimization phase should start only after that release snapshot is filled from a clean
checkout of the exact tagged/merged commit.

---

## Post-#35 10K program - exact execution tranche 1 (unreleased, not yet P100-measured)

**Base release:** `v1.0.40` / `9e393dab554302509532fb6c0c147ad166549744` (release commit after PR #35)
**Status:** source/static validation only; do not record a throughput win until a fresh P100 A/B proves it.
**Correctness contract:** FP32 equations/order, workers=16, replay fraction=0.30, replay selection, RNG, merge semantics, checkpoint identity, and the two historical exact-state hashes remain unchanged for the default path.

This tranche starts the six-point plan derived from the corrected 32K profiles.  It deliberately separates default exact-path changes from opt-in architectural experiments so another plausible optimization cannot silently become production before a canonical P100 measurement.

### Six-point plan and current status

| # | Target | This tranche | Reason / next proof |
| --- | --- | --- | --- |
| 1 | Frozen-prefix exact global selection | **Implemented, default exact path** | Reuse the existing packed selector, omit frozen-only learning-destination bookkeeping, parallelize only independent selected-state writes. The serial sparse heap itself is intentionally unchanged until this smaller A/B is measured. |
| 2 | Frozen pre phase | **Partial exact cleanup** | Frozen event delivery/input injection now pass null eligibility worklists instead of loading four pointers that cannot be consumed with `learning_trace=false`. Grid-wide pre restructuring is deferred because changing atomic delivery order would risk exact FP32 state. |
| 3 | Replay target occupancy/register pressure | **Implemented compiler specialization** | `leo_train_cooperative_fast` is compiled with mixed frozen-schedule handling disabled; experimental streaming replay uses the general kernel. P100 resource telemetry must show whether 144 regs/thread moves toward the <=128 / 2-CTA-per-SM threshold. |
| 4 | Base learning-signal memory layout | **Not changed yet** | Prior four-row interleave and shared-error cache both regressed badly. An output-major shadow can be exact, but it costs another full output matrix and duplicate update traffic; do not add it until the first three A/Bs quantify remaining headroom. |
| 5 | Remove cross-story global lockstep | **Opt-in A/B wired** | Reuse the already-existing `leo_train_story_batch` kernel through `LEO_CUDA_STORY_LOCAL_BLOCKS=1`. It gives each story one independent block and zero cross-story grid barriers, but sacrifices intra-story CTA parallelism. Keep off by default until measured. |
| 6 | Replay algorithm v2 / prefix-amplification removal | **Existing experimental path retained only** | `LEO_REPLAY_STREAMING=1` already changes replay-state semantics. It remains non-canonical; no new golden hash or semantic contract is introduced in this tranche. |

### Exact frozen selector specialization

Measured evidence before this change showed frozen replay-prefix reconstruction spending about:

```text
select_global    50.23%
pre              28.58%
select_blocks    17.85%
post_emit         3.34%
```

The old shared selector did two pieces of work that are unnecessary during a frozen prefix:

1. append each selected neuron to `LEARNING_DESTINATION_LIST` through epoch/atomic bookkeeping;
2. write all selected-state records from thread 0 even though selected neurons are unique and the active-list order is already fixed by the packed records.

The new frozen specialization keeps the same sparse heap merge, cutoff, active count, population inhibition arithmetic, packed-key order, and selected values.  It changes only execution geometry after selection:

- no frozen learning-destination worklist population;
- `active[index]` / `active_value[index]` preserve their exact ordered indices;
- per-neuron `activation` and `selected_epoch` stores are distributed across block-0 threads;
- ordinary supervised selection still uses the historical selector and still populates learning destinations.

This is intentionally narrower than immediately replacing the sparse heap.  The previous shared-error-cache failure proved that a locally plausible CUDA change must first win the real 32K workload.

### Frozen pre cleanup

The cooperative frozen prefix previously reconstructed recurrent/input eligibility-list pointers on every prefix step even though both delivery helpers receive `learning_trace=false`.  Those pointers are only dereferenced inside the learning-trace branch.  The frozen path now passes null worklist/count pointers and retains the established block-0 atomic/event ordering.

A more aggressive whole-grid pre implementation is **not** included yet.  Event delivery and symbol injection contain atomic FP32 accumulation; distributing those operations across CTAs could change accumulation order and therefore the historical hash.  Any future pre redesign must first isolate order-independent clears from order-sensitive event arithmetic.

### Replay fast-kernel specialization

PR #35 reduced `leo_train_cooperative_fast` from the older ~166 registers/thread to 144 registers/thread, still one CTA/SM on the P100.  The canonical target-batch kernel nevertheless still carried logic for the experimental mixed replay schedule (`target_index == -2`) and a separate learning-step parity counter.

This tranche makes mixed-schedule support a compile-time template choice:

```text
canonical fast replay target kernel: MIXED_SCHEDULE=false
normal/general cooperative kernel:   MIXED_SCHEDULE=true
profiled target kernel:              MIXED_SCHEDULE=false
```

`LEO_REPLAY_STREAMING=1` is routed through the general mixed-schedule kernel.  The expected benefit is reduced live state/register pressure in the canonical fast replay kernel, but only P100 `cuda_kernel_resources` telemetry can prove whether NVRTC actually lowers the register count.

### Story-local executor A/B

The source already contained `leo_train_story_batch`, where one CUDA block owns one story and executes its dependent byte sequence without a whole-grid barrier.  It was not wired into the shared production launch selector.

This tranche exposes it only through:

```text
LEO_CUDA_STORY_LOCAL_BLOCKS=1
```

The purpose is to test the central 7-10K architecture hypothesis directly:

```text
production grouped executor:
  more intra-story CTAs
  + whole-grid synchronization
  + longest-story lockstep

story-local A/B:
  one CTA per story
  + no cross-story grid synchronization
  + independent story completion
  - much less intra-story parallelism
```

This is an exact execution A/B, not a new learning algorithm.  The same per-story persistent model replicas and canonical batch-end merge remain in use.  Acceptance/scaling scripts explicitly clear the environment variable so an inherited experiment cannot contaminate canonical results.

### Why this tranche does not claim 7K/8K/10K

The clean post-#35 canonical timing is approximately:

```text
reported training steps = 14,496
base/non-replay time     = 3.2318 s
replay time              = 4.5883 s
total                    = 7.8201 s
throughput               = 1,853.69 steps/s
```

Target total-time budgets are:

```text
7K  -> 2.071 s
8K  -> 1.812 s
10K -> 1.450 s
```

Even deleting all replay cost leaves only about 4,485 reported steps/s at the current base cost.  Even deleting all base cost leaves only about 3,159 reported steps/s at the current replay cost.  Therefore no single selector, register, cache, or barrier tweak can reach 7-10K by itself.

The canonical replay run also executes 20,235 hidden frozen-prefix steps for only 3,353 replay targets - about 6.03 prefix advances per replay target.  Removing that amplification would require a replay semantic/algorithm change unless an exact state-reconstruction method can preserve sequential parameter visibility.  That is why algorithm-v2 work remains explicitly separated from exact execution optimization.

### Required P100 A/B before accepting this tranche

Use a clean checkout and fresh PTX/tuning cache.  Keep `LEO_REPLAY_STREAMING` off for canonical acceptance.

1. Run `scripts/check_gpu.sh` with all experimental variables cleared.
2. Record `cuda_kernel_resources` for:
   - `leo_shared_wavefront_persistent_grouped`;
   - `leo_train_cooperative_fast`;
   - `leo_advance_frozen_cooperative`.
3. Run fresh replay-off 64-story and replay-30 16-story canonical benchmarks and verify both historical hashes.
4. Run replay profiling and compare frozen `select_global` / `pre` cycles against the post-#35 profile.
5. A/B story-local execution separately:

```bash
LEO_CUDA_STORY_LOCAL_BLOCKS=1 \
CUDA_VISIBLE_DEVICES=0 ./target/release/leo benchmark \
  --train \
  --model runs/p100-acceptance/replay0.pscls \
  --bytes data/prepared/tinystories.train.bytes \
  --index data/prepared/tinystories.train.idx \
  --stories 64 \
  --workers 16 \
  --backend gpu
```

Reject the story-local path if the hash differs or if removing cross-story barriers does not compensate for one-CTA-per-story underutilization.  A negative result belongs in this ledger rather than being silently removed.

## Formula-v2 experiment — reduce required work per token

Base: `perf/p100-10k-tranche1` at `4b8ae914cb942da315c269cbed5f547d1a3db972`.

This experiment intentionally moves beyond the historical v1 exact-state hashes.  The v1.0.40/tranche-1 results remain the reference implementation; Formula v2 must establish its own deterministic hashes and must pass quality gates before it can replace the v1 formula.

### Evidence motivating a formula change

The first P100 tranche improved the clean 32K replay-off path only marginally, from roughly `3406.245` to `3413.783` steps/s, while the best clean canonical replay run improved from `1853.689` to `1938.301` steps/s with a 56-CTA replay grid.  The canonical run still spent about `4.2514 s` of `7.4787 s` in replay.  A 112-CTA replay grid was slower (`1919.132` steps/s), despite the replay kernel falling from 144 to 128 registers/thread and becoming capable of two resident CTAs/SM.  This is evidence that synchronization/serial work, not raw occupancy, dominates the current ceiling.

The full 32K frozen replay-prefix profile remained dominated by global selection: approximately `44.49% select_global`, `31.86% pre`, `19.90% select_blocks`, and `3.75% post_emit`.  TinyStories has `128` blocks, `8` local winners per block, and `max_active_global = 1024`; therefore `128 * 8 == 1024`.  After local block competition the global cap cannot remove a winner, so globally ranking those local winners is algorithmically redundant for this geometry.

The canonical replay policy also selected only `3353` replay targets while executing `20235` hidden prefix steps.  The repeated reconstruction cost is part of the formula, not host overhead, and cannot be eliminated by occupancy tuning alone.

### Formula-v2 changes in this experiment

1. **Non-binding global TopK elision.**  When `max_active_global >= block_count * max_active_per_block`, CPU and CUDA keep the exact per-block winner sets and deterministic local rank order, set global clipped count/cutoff to zero, and skip the global heap/bitonic winner ranking.  Binding-cap geometries retain the established global selector.
2. **Stateful batched replay.**  `replay.stateful_batch = true` reuses Leo's existing mixed frozen/supervised replay trajectory: reset once, advance recurrent/context state through gaps, supervise selected ranges, then reset once.  This preserves the configured replay target budget but changes state/parameter visibility compared with rebuilding every range from byte zero.  It is a batched/stateful replay implementation, not serialized snapshot restoration.
3. **Four-step delayed recurrent/input plasticity.**  `learning.plasticity_window = 4` keeps local eligibility traces active every learning tick and keeps output/context supervision immediate.  Recurrent/input learning-signal construction and weight commits occur only on the fourth supervised learning step; frozen replay gaps do not consume the window, and end-document forces a partial-window commit.  The commit strength is scaled by the number of ticks represented by the consolidation pulse.  Inhibitory and threshold/population homeostasis remain per-step.

### Acceptance contract for Formula v2

Do **not** compare Formula-v2 state hashes to the historical v1 hashes.  Acceptance requires: repeatable Formula-v2 hashes across repeated runs; CPU/GPU agreement for the same Formula-v2 config where the existing conformance harness applies; no regression in validation bits-per-byte/generation/recurrent-memory probes; and a material P100 throughput gain.  No speed or quality improvement is claimed until those P100 and training-quality measurements exist.

### Formula-v2 window sweep and W8 lock — P100 evidence

After Formula-v2 stabilized at a clean W4 canonical result near `3507.07 steps/s`,
the same 16-story benchmark produced `3843.52 steps/s` at W8 and `4038.44
steps/s` at W16. The longer 1024-story quality run reversed the W8/W16 short-run
ranking: W8 completed train+validation in `568.38 s` versus `577.53 s` for W16.
W8 also produced the better held-out likelihood (`3.32556 bits/byte`) versus W4
`3.33133` and W16 `3.34375`; the v1 reference was `3.31282`. W8 is therefore the
locked TinyStories consolidation window for the next formula experiment.

At the end of the 1024-story run W8 averaged about `40.85` active neurons and
`1963.26` recurrent events/byte on held-out evaluation, while W16 averaged about
`45.45` active neurons and `2183.83` recurrent events/byte. That extra mature
activity helps explain why W16's lower consolidation frequency did not translate
into a lower full-run wall time.

## Formula-v3 experiment — surprise-gated consolidation

Formula v3 keeps W8 but stops treating every scheduled consolidation boundary as
equally informative. TinyStories sets `learning.plasticity_confidence_threshold =
0.50`. On a W8 boundary, recurrent/input learning-signal construction and weight
updates execute only if the just-computed target probability is below 50%;
end-document remains unconditional. Eligibility traces, direct output/context
supervision, replay selection, and homeostasis retain their Formula-v2 behavior.

The CPU reference uses the already-normalized target probability. CUDA carries
the same threshold in the persistent step descriptor and evaluates it after the
forward softmax, before learning-signal construction, so the production path does
not require a host probability round-trip. Non-persistent CUDA reference paths
read the target probability only at scheduled consolidation boundaries.
`test.toml` and `probe.toml` use threshold `1.00`, which preserves the existing
Formula-v2 consolidation schedule for exact/conformance gates.

This change intentionally creates a new training-state hash. Acceptance requires
repeatability, CPU/GPU agreement for the same gated config, a P100 throughput
improvement over the locked W8 reference, and held-out/generation quality that
meets the project bar. Adaptive replay and a tighter activity-budget formula are
not part of this tranche; isolate the confidence gate first.
