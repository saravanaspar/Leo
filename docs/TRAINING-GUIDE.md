# Leo optimized training guide

This guide is the end-to-end runbook for building, testing, benchmarking, training,
resuming, evaluating, profiling, and rolling back the optimized Leo CUDA training
path.

The reference TinyStories configuration is `configs/tinystories.toml`:

- 32,768 neurons;
- FP32 learning;
- bounded-surprise replay fraction `0.30`;
- replay segment length 48 bytes;
- 16 logical workers for the reference GPU experiment;
- validation every 25,000,000 input bytes;
- checkpoint every 50,000,000 input bytes.

## Quick start: safe/default path

If the overlay is already applied and TinyStories is prepared, this is the shortest
safe sequence before a real run:

```bash
cargo build --release -p leo-cli
bash ./scripts/check.sh
bash ./scripts/check_gpu.sh

unset LEO_REPLAY_STREAMING
unset LEO_MULTI_GPU_PARALLEL_REPLAY
unset LEO_CUDA_FULL_STEP_METRICS

CUDA_VISIBLE_DEVICES=0 LEO_MULTI_GPU=0 \
  bash ./scripts/train.sh data/prepared runs/quality/leo-32k \
  configs/tinystories.toml 1 "" 16 "" gpu 100

CUDA_VISIBLE_DEVICES=0 \
  bash ./scripts/evaluate.sh data/prepared runs/quality/leo-32k 1000 "" gpu
```

Do not enable the experimental replay flags merely because they benchmark faster.
Sections 15-18 show how to benchmark and quality-gate them before production use.

## 1. Important execution modes

There are two classes of optimization in this branch.

### Default exact/execution optimizations

These are enabled by default on the CUDA path. They are intended to change work
placement and overhead, not the learning policy:

- persistent multi-step shared CUDA execution;
- grouped multi-CTA work per logical story;
- multi-CTA forward, eligibility, update, and homeostasis work;
- compact production training-step metrics;
- diagnostic-only atomic removal from the fast path;
- device-side fixed-parameter logical-worker merge;
- exact power-of-two indexing fast paths;
- exact short-decay fast paths.

The GPU gate compares this default path against the legacy path and requires the
same complete `training_state_sha256` on its exact-state benchmark.

### Experimental replay optimizations

These are **off by default** because they change replay execution semantics even
though FP32 and the configured 30% replay fraction remain unchanged:

- `LEO_REPLAY_STREAMING=1`: processes the selected replay schedule as one
  device-resident timeline instead of rebuilding every selected segment from
  byte zero;
- `LEO_MULTI_GPU_PARALLEL_REPLAY=1`: on multi-GPU runs, executes independent
  replay trajectories on participating devices and merges their parameter
  deltas instead of using canonical serial replay.

Do not enable either experimental replay mode for a production-quality training
run until the A/B quality procedure in this guide passes on the intended data,
hardware, and training budget.

## 2. Requirements

Required for the repository:

- Linux or another environment supported by the repository scripts;
- Rust 1.85 or newer;
- Python 3.12 or newer.

Required for CUDA training:

- an NVIDIA driver visible to `nvidia-smi`;
- CUDA toolkit headers, including `cooperative_groups.h`;
- enough VRAM for the model and logical story-lane state;
- homogeneous GPUs for the exact multi-GPU path.

Required only when `scripts/data.sh` must download TinyStories:

- Hugging Face `hf` CLI.

Check the machine before doing anything expensive:

```bash
rustc --version
cargo --version
python3 --version
nvidia-smi -L
```

If CUDA is installed outside `/usr/local/cuda` or `/opt/cuda`, set one of:

```bash
export CUDA_HOME=/path/to/cuda
# or
export CUDA_PATH=/path/to/cuda
```

Verify the cooperative-groups header:

```bash
test -f "${CUDA_HOME:-/usr/local/cuda}/include/cooperative_groups.h" && echo CUDA_HEADERS_OK
```

## 3. Apply the changed-files overlay

Create a branch before applying the overlay:

```bash
git switch -c perf/leo-training-speedup
```

If the ZIP is in `~/Downloads` and the current directory is the Leo repository
root:

```bash
unzip -o ~/Downloads/Leo-speedup-full-overlay-v2.zip -d .
```

Verify what changed:

```bash
git status --short
git diff --check
git diff --stat
```

Do not commit generated data, run directories, CUDA caches, model checkpoints, or
benchmark logs unless the repository policy explicitly requires them.

## 4. Build

Build the release CLI used for all performance measurements:

```bash
cargo build --release -p leo-cli
```

Confirm the binary exists:

```bash
test -x target/release/leo && echo LEO_RELEASE_OK
```

The CUDA backend uses NVRTC at runtime, so a successful Rust build alone does not
prove that CUDA kernels compile or execute on the target GPU. Run the GPU gate in
the next section.

## 5. Mandatory tests before training or pushing

Run the normal repository gate:

```bash
bash ./scripts/check.sh
```

On an NVIDIA machine, run the real CUDA/NVRTC gate:

```bash
bash ./scripts/check_gpu.sh
```

The GPU gate checks the production CUDA path, CPU/CUDA conformance, phase/profile
plumbing, CUDA autotuning/cache behavior, and the optimized-vs-legacy final
training-state digest. On a host with at least two visible GPUs it also runs the
1-GPU vs 2-GPU exact-state scaling gate.

For an explicit pre-push sequence:

```bash
cargo fmt --check
cargo build --workspace
cargo test --workspace
cargo clippy --workspace --all-targets -- -D warnings
bash ./scripts/check.sh
bash ./scripts/check_gpu.sh
git diff --check
git status --short
```

Do **not** push a CUDA performance change if `scripts/check_gpu.sh` has not passed
on real NVIDIA hardware.

## 6. Prepare TinyStories data

The standard helper downloads the repository-pinned TinyStories revision when the
raw files are absent, verifies the source SHA-256 values, and creates Leo's
prepared byte/index files:

```bash
bash ./scripts/data.sh
```

Default prepared output:

```text
data/prepared/tinystories.train.bytes
data/prepared/tinystories.train.idx
data/prepared/tinystories.valid.bytes
data/prepared/tinystories.valid.idx
```

For a small smoke dataset while setting up a machine:

```bash
bash ./scripts/data.sh \
  data/raw/tinystories \
  data/smoke \
  1024 \
  128
```

The third and fourth arguments limit training and validation story counts.

## 7. Inspect the reference configuration

Confirm the two non-negotiable settings before a real run:

```bash
grep -nE 'neuron_count|\[replay\]|fraction' configs/tinystories.toml
```

The standard configuration should contain:

```text
neuron_count = 32768
[replay]
fraction = 0.30
```

FP32 is the learning arithmetic contract of this code path; do not add fast-math,
FP16, BF16, or mixed-precision compiler/runtime overrides when comparing these
results.

## 8. Recommended safe single-GPU training

Before the experimental replay modes have passed the quality A/B, use the default
optimized path with the replay experiments explicitly disabled:

```bash
unset LEO_REPLAY_STREAMING
unset LEO_MULTI_GPU_PARALLEL_REPLAY
unset LEO_CUDA_FULL_STEP_METRICS

CUDA_VISIBLE_DEVICES=0 \
LEO_MULTI_GPU=0 \
bash ./scripts/train.sh \
  data/prepared \
  runs/quality/leo-32k \
  configs/tinystories.toml \
  1 \
  "" \
  16 \
  "" \
  gpu \
  100
```

Argument order for `scripts/train.sh` is:

```text
./scripts/train.sh \
  <prepared-data-directory> \
  <run-directory> \
  [config] \
  [passes] \
  [max-stories] \
  [workers] \
  [max-input-bytes] \
  [backend] \
  [validation-stories]
```

The script builds the release binary, creates the model if it does not exist,
trains it, and writes the console output to `RUN_DIR/train.jsonl`.

For the reference 32K/P100-style comparison keep `--workers 16` so results remain
comparable to the existing experiment history.

## 9. Direct CLI training

Use the CLI directly when you need full control.

Initialize a model:

```bash
mkdir -p runs/quality/leo-32k

target/release/leo init \
  --config configs/tinystories.toml \
  --output runs/quality/leo-32k/leo.pscls
```

Start a deliberately fresh single-GPU training operation:

```bash
CUDA_VISIBLE_DEVICES=0 \
LEO_MULTI_GPU=0 \
target/release/leo train \
  --model runs/quality/leo-32k/leo.pscls \
  --train-bytes data/prepared/tinystories.train.bytes \
  --train-index data/prepared/tinystories.train.idx \
  --valid-bytes data/prepared/tinystories.valid.bytes \
  --valid-index data/prepared/tinystories.valid.idx \
  --validation-stories 100 \
  --passes 1 \
  --workers 16 \
  --backend gpu \
  --fresh-run
```

`--fresh-run` intentionally discards compatible prior resume state for that
model. Do not use it when resuming an interrupted training operation.

## 10. Resume training

Resume by repeating the same compatible training command **without**
`--fresh-run`.

For scripted runs, simply repeat the same `scripts/train.sh` invocation with the
same model, prepared dataset, logical worker count, and training identity.

Do not bypass Leo's dataset/resume identity checks to combine incompatible runs.

## 11. Limit a smoke or benchmark training run

Create a dedicated smoke model first so this command cannot modify a real run:

```bash
mkdir -p runs/smoke
target/release/leo init \
  --config configs/tinystories.toml \
  --output runs/smoke/leo.pscls
```

Then limit by story count:

```bash
CUDA_VISIBLE_DEVICES=0 \
LEO_MULTI_GPU=0 \
target/release/leo train \
  --model runs/smoke/leo.pscls \
  --train-bytes data/smoke/tinystories.train.bytes \
  --train-index data/smoke/tinystories.train.idx \
  --passes 1 \
  --max-stories 256 \
  --workers 16 \
  --backend gpu \
  --fresh-run
```

A limited run is useful for compilation/runtime checks. It is not enough to prove
long-run model quality.

## 12. Benchmark the optimized path safely

`leo benchmark --train` loads the model, trains an in-memory runtime for the
measurement, reports the resulting state hash, and does not save that benchmark
state back to the input model file. It is therefore the preferred throughput A/B
command.

### Optimized default path

```bash
unset LEO_REPLAY_STREAMING
unset LEO_MULTI_GPU_PARALLEL_REPLAY
unset LEO_CUDA_FULL_STEP_METRICS

CUDA_VISIBLE_DEVICES=0 \
LEO_MULTI_GPU=0 \
target/release/leo benchmark \
  --train \
  --model runs/quality/leo-32k/leo.pscls \
  --bytes data/prepared/tinystories.train.bytes \
  --index data/prepared/tinystories.train.idx \
  --stories 1024 \
  --workers 16 \
  --backend gpu \
  2>&1 | tee benchmark-optimized.jsonl
```

### Legacy execution A/B

This disables the new persistent/grouped/device-merge execution features:

```bash
CUDA_VISIBLE_DEVICES=0 \
LEO_MULTI_GPU=0 \
LEO_CUDA_SHARED_PERSISTENT=0 \
LEO_CUDA_SHARED_GROUPED=0 \
LEO_CUDA_DEVICE_BATCH_MERGE=0 \
target/release/leo benchmark \
  --train \
  --model runs/quality/leo-32k/leo.pscls \
  --bytes data/prepared/tinystories.train.bytes \
  --index data/prepared/tinystories.train.idx \
  --stories 1024 \
  --workers 16 \
  --backend gpu \
  2>&1 | tee benchmark-legacy.jsonl
```

For exact execution-only acceptance, compare the final `training_benchmark`
events and require the same `training_state_sha256`.

Useful benchmark fields include:

- `steps_per_second`: end-to-end training steps per second;
- `execution_steps_per_second`: includes replay prefix execution work;
- `replay_seconds`;
- `replay_sync_seconds`;
- `replay_wall_fraction`;
- `serial_replay_speedup_ceiling`;
- `replay_steps`;
- `replay_prefix_steps`;
- `mean_loss` and `bits_per_byte`;
- `training_state_sha256`.

When comparing against the remembered 32K baseline, use `steps_per_second` from a
replay-on run. Do not substitute an execution-only or replay-disabled number.

Extract the final benchmark records with:

```bash
grep '"event":"training_benchmark"' benchmark-optimized.jsonl | tail -n 1
grep '"event":"training_benchmark"' benchmark-legacy.jsonl | tail -n 1
```

For an exact default-path A/B, the two `training_state_sha256` values must match.

## 13. Production metrics vs full diagnostics

Normal optimized training uses compact step records. Do not enable full per-step
diagnostics for throughput measurements.

Normal fast mode:

```bash
unset LEO_CUDA_FULL_STEP_METRICS
```

Full diagnostic mode for debugging only:

```bash
export LEO_CUDA_FULL_STEP_METRICS=1
```

Full diagnostics intentionally add work and can reduce throughput.

## 14. Exact feature rollback and isolation

All default execution optimizations can be isolated for A/B diagnosis.

Disable persistent shared execution:

```bash
export LEO_CUDA_SHARED_PERSISTENT=0
```

Disable grouped multi-CTA execution:

```bash
export LEO_CUDA_SHARED_GROUPED=0
```

Disable device-side fixed-parameter batch merge:

```bash
export LEO_CUDA_DEVICE_BATCH_MERGE=0
```

Restore full training-step metrics:

```bash
export LEO_CUDA_FULL_STEP_METRICS=1
```

Return to the normal optimized defaults:

```bash
unset LEO_CUDA_SHARED_PERSISTENT
unset LEO_CUDA_SHARED_GROUPED
unset LEO_CUDA_DEVICE_BATCH_MERGE
unset LEO_CUDA_FULL_STEP_METRICS
```

If an exact-state comparison fails, disable one optimization at a time in the
order above to isolate the first divergent path.

## 15. Experimental streaming replay

The streaming replay path keeps FP32 and the configured 30% selection budget but
changes how hidden/recurrent state is carried between selected replay regions.
It must pass the quality A/B before a long production run.

Benchmark it without modifying the checkpoint:

```bash
CUDA_VISIBLE_DEVICES=0 \
LEO_MULTI_GPU=0 \
LEO_REPLAY_STREAMING=1 \
target/release/leo benchmark \
  --train \
  --model runs/quality/leo-32k/leo.pscls \
  --bytes data/prepared/tinystories.train.bytes \
  --index data/prepared/tinystories.train.idx \
  --stories 1024 \
  --workers 16 \
  --backend gpu \
  2>&1 | tee benchmark-streaming-replay.jsonl
```

Disable it with:

```bash
unset LEO_REPLAY_STREAMING
```

## 16. Exact/default multi-GPU training

The default multi-GPU path distributes the logical story batch while retaining
canonical serial replay.

Example with two GPUs:

```bash
unset LEO_MULTI_GPU_PARALLEL_REPLAY
unset LEO_REPLAY_STREAMING

CUDA_VISIBLE_DEVICES=0,1 \
LEO_MULTI_GPU=1 \
bash ./scripts/train.sh \
  data/prepared \
  runs/quality/leo-32k-2gpu \
  configs/tinystories.toml \
  1 \
  "" \
  16 \
  "" \
  gpu \
  100
```

Rules:

- expose at least two GPUs;
- use a homogeneous GPU group for exact mode;
- visible device 0 is the canonical runtime;
- use `CUDA_VISIBLE_DEVICES` to select/reorder the group;
- VRAM is not pooled;
- keep the logical worker count fixed when comparing scaling.

## 17. Experimental parallel multi-GPU replay

This mode keeps FP32 and the configured 30% replay budget but changes cross-story
replay update visibility. It is intentionally opt-in.

Benchmark it first:

```bash
CUDA_VISIBLE_DEVICES=0,1 \
LEO_MULTI_GPU=1 \
LEO_MULTI_GPU_PARALLEL_REPLAY=1 \
target/release/leo benchmark \
  --train \
  --model runs/quality/leo-32k/leo.pscls \
  --bytes data/prepared/tinystories.train.bytes \
  --index data/prepared/tinystories.train.idx \
  --stories 1024 \
  --workers 16 \
  --backend gpu \
  2>&1 | tee benchmark-parallel-replay-2gpu.jsonl
```

Streaming and parallel replay can be combined only after each mode has been
validated independently:

```bash
CUDA_VISIBLE_DEVICES=0,1 \
LEO_MULTI_GPU=1 \
LEO_REPLAY_STREAMING=1 \
LEO_MULTI_GPU_PARALLEL_REPLAY=1 \
target/release/leo benchmark \
  --train \
  --model runs/quality/leo-32k/leo.pscls \
  --bytes data/prepared/tinystories.train.bytes \
  --index data/prepared/tinystories.train.idx \
  --stories 1024 \
  --workers 16 \
  --backend gpu
```

Do not promote the combined mode from experimental based on throughput alone.

## 18. Required A/B quality test for semantic replay changes

The exact-state hash is the correct gate for execution-only optimizations. It is
**not** the correct acceptance criterion for streaming/parallel replay because
those modes intentionally change update/state ordering.

Use identical starting checkpoints, data, story budgets, validation sets, FP32,
30% replay, and logical workers.

### Create one common starting checkpoint

```bash
rm -rf runs/ab-quality
mkdir -p runs/ab-quality/seed runs/ab-quality/baseline runs/ab-quality/streaming

target/release/leo init \
  --config configs/tinystories.toml \
  --output runs/ab-quality/seed/leo.pscls

cp runs/ab-quality/seed/leo.pscls runs/ab-quality/baseline/leo.pscls
cp runs/ab-quality/seed/leo.pscls runs/ab-quality/streaming/leo.pscls
```

### Train the baseline

Use a meaningful budget. The example below uses 10,000 stories as a screening
run; a production decision should also be confirmed at a longer representative
budget.

```bash
CUDA_VISIBLE_DEVICES=0 \
LEO_MULTI_GPU=0 \
target/release/leo train \
  --model runs/ab-quality/baseline/leo.pscls \
  --train-bytes data/prepared/tinystories.train.bytes \
  --train-index data/prepared/tinystories.train.idx \
  --valid-bytes data/prepared/tinystories.valid.bytes \
  --valid-index data/prepared/tinystories.valid.idx \
  --validation-stories 1000 \
  --passes 1 \
  --max-stories 10000 \
  --workers 16 \
  --backend gpu \
  --fresh-run \
  2>&1 | tee runs/ab-quality/baseline/train.jsonl
```

### Train streaming replay from the same starting checkpoint

```bash
CUDA_VISIBLE_DEVICES=0 \
LEO_MULTI_GPU=0 \
LEO_REPLAY_STREAMING=1 \
target/release/leo train \
  --model runs/ab-quality/streaming/leo.pscls \
  --train-bytes data/prepared/tinystories.train.bytes \
  --train-index data/prepared/tinystories.train.idx \
  --valid-bytes data/prepared/tinystories.valid.bytes \
  --valid-index data/prepared/tinystories.valid.idx \
  --validation-stories 1000 \
  --passes 1 \
  --max-stories 10000 \
  --workers 16 \
  --backend gpu \
  --fresh-run \
  2>&1 | tee runs/ab-quality/streaming/train.jsonl
```

### Evaluate both with exactly the same held-out set

```bash
CUDA_VISIBLE_DEVICES=0 bash ./scripts/evaluate.sh \
  data/prepared runs/ab-quality/baseline 1000 "" gpu

CUDA_VISIBLE_DEVICES=0 bash ./scripts/evaluate.sh \
  data/prepared runs/ab-quality/streaming 1000 "" gpu
```

Extract the held-out summaries:

```bash
grep '"event":"held_out_evaluation"' runs/ab-quality/baseline/eval.jsonl | tail -n 1
grep '"event":"held_out_evaluation"' runs/ab-quality/streaming/eval.jsonl | tail -n 1
```

Also inspect:

```text
runs/ab-quality/baseline/baselines.json
runs/ab-quality/streaming/baselines.json
runs/ab-quality/baseline/story.txt
runs/ab-quality/streaming/story.txt
```

Acceptance rule for an experimental mode: held-out loss/bits-per-byte must not
regress materially, held-out accuracy must not regress materially, baseline
quality checks must continue to pass, training must remain numerically stable,
and the result must be reproduced over more than one meaningful run before the
mode is treated as quality-safe.

Repeat the same procedure for `LEO_MULTI_GPU_PARALLEL_REPLAY=1`, comparing it
against the default serial-replay multi-GPU mode on the same visible GPU group.

## 19. Evaluate a trained model

Standard evaluation helper:

```bash
CUDA_VISIBLE_DEVICES=0 \
bash ./scripts/evaluate.sh \
  data/prepared \
  runs/quality/leo-32k \
  1000 \
  "" \
  gpu
```

It writes evaluation, generation, frozen benchmark, and baseline-comparison
artifacts into the run directory and invokes the repository baseline quality
checks with `--require-pass`.

Direct held-out evaluation is also available through `target/release/leo eval`;
use `scripts/evaluate.sh` for the standard reproducible bundle.

## 20. Checkpoint, inspect, and rollback

Inspect the current model:

```bash
target/release/leo inspect \
  --model runs/quality/leo-32k/leo.pscls \
  --json
```

Create a checkpoint:

```bash
target/release/leo checkpoint \
  --model runs/quality/leo-32k/leo.pscls
```

Rollback to a known generation:

```bash
target/release/leo rollback \
  --model runs/quality/leo-32k/leo.pscls \
  --generation 3
```

Always inspect available model/checkpoint state before selecting a rollback
generation for an important run.

## 21. CUDA phase profiling

For lightweight internal phase samples:

```bash
LEO_CUDA_PHASE_PROFILE=1 \
LEO_CUDA_PHASE_PROFILE_STRIDE=64 \
LEO_MULTI_GPU=0 \
target/release/leo benchmark \
  --train \
  --model runs/quality/leo-32k/leo.pscls \
  --bytes data/prepared/tinystories.train.bytes \
  --index data/prepared/tinystories.train.idx \
  --stories 256 \
  --workers 16 \
  --backend gpu \
  2>&1 | tee cuda-phase-profile.jsonl
```

Disable profiling for final throughput numbers:

```bash
unset LEO_CUDA_PHASE_PROFILE
unset LEO_CUDA_PHASE_PROFILE_STRIDE
```

## 22. Nsight Compute profiling

If `ncu` is installed:

```bash
bash ./scripts/profile_cuda.sh \
  runs/quality/leo-32k/leo.pscls \
  data/prepared/tinystories.train.bytes \
  data/prepared/tinystories.train.idx \
  16 \
  64 \
  leo-cuda-profile
```

This produces:

```text
leo-cuda-profile.ncu-rep
leo-cuda-profile.csv
```

Inspect occupancy, SpeedOfLight, memory workload, warp state, instruction stats,
and atomic behavior. Nsight instrumentation is not a throughput benchmark.

## 23. Replay debugging

For detailed replay diagnostics:

```bash
bash ./scripts/debug_replay.sh \
  runs/quality/leo-32k/leo.pscls \
  data/prepared/tinystories.train.bytes \
  data/prepared/tinystories.train.idx \
  128 \
  16 \
  replay-debug.log
```

This deliberately enables heavy replay/CUDA diagnostics. Do not compare its
steps/s with a clean production benchmark.

## 24. Multi-GPU scaling benchmark

Use the repository scaling utility on the same model/data/workload for all GPU
counts. See its built-in help for all options:

```bash
python3 scripts/benchmark_multi_gpu.py --help
```

A typical exact-path comparison should keep replay policy, model, story count,
and `--workers 16` fixed and vary only physical GPU count.

The repository GPU gate automatically invokes this utility for a 1-vs-2 GPU
exact-state test when at least two GPUs are visible.

## 25. Clean benchmark discipline

For trustworthy performance numbers:

1. use `target/release/leo`;
2. use the same checkpoint/model configuration;
3. use the same dataset/index and story count;
4. use the same logical worker count;
5. keep replay at 30%;
6. do not enable debug/full-metric/phase/Nsight instrumentation;
7. record GPU model, driver, CUDA version, and visible GPU count;
8. run repeated clean measurements rather than trusting one sample;
9. report `steps_per_second` and `replay_wall_fraction` together;
10. retain the `training_state_sha256` for exact-path comparisons.

Useful machine capture:

```bash
nvidia-smi
nvcc --version || true
rustc --version
cargo --version
python3 --version
git rev-parse HEAD
git status --short
```

## 26. Performance target tracking

The remembered 32K baseline is approximately 800 steps/s without replay and
approximately 600 steps/s with 30% replay. Treat those as historical estimates,
not acceptance measurements.

For the first clean run record the actual current numbers and use them as the
new baseline. The primary target discussed for 32K + 30% replay is at least
6,000 end-to-end `steps_per_second` without a model-quality regression.

Record at minimum:

```text
hardware:
gpu count:
CUDA/driver:
git commit:
model/config:
workers:
stories:
replay fraction:
steps_per_second:
execution_steps_per_second:
replay_wall_fraction:
replay_prefix_steps:
mean_loss:
training_state_sha256:
held-out bits_per_byte:
held-out accuracy:
```

## 27. Troubleshooting

### `cargo` is missing

Install/use Rust 1.85+ and verify `cargo --version`. Do not treat the source-only
Python checks as a substitute for compilation.

### CUDA headers are not found

Set `CUDA_HOME` or `CUDA_PATH` to a toolkit containing:

```text
include/cooperative_groups.h
```

Then rerun `bash ./scripts/check_gpu.sh`.

### NVRTC/kernel compilation fails

Capture the complete stderr from:

```bash
bash ./scripts/check_gpu.sh 2>&1 | tee check-gpu.log
```

Do not continue to performance benchmarking until the gate passes.

### Optimized path hash differs from legacy

Isolate features one at a time:

```bash
LEO_CUDA_SHARED_GROUPED=0 ...
LEO_CUDA_DEVICE_BATCH_MERGE=0 ...
LEO_CUDA_SHARED_PERSISTENT=0 ...
```

Start from identical model/data each time. The exact-path acceptance condition is
not "close loss"; use the repository's exact state-hash gate where it applies.

### Multi-GPU mode says too few devices are visible

Check:

```bash
nvidia-smi -L
echo "${CUDA_VISIBLE_DEVICES:-<not-set>}"
```

Then select the intended group, for example:

```bash
export CUDA_VISIBLE_DEVICES=0,1
```

### Multi-GPU exact mode rejects the device group

Use homogeneous GPUs with matching reported GPU model and compute capability.
Use `CUDA_VISIBLE_DEVICES` to exclude a different GPU.

### Training is slower than expected

Check, in this order:

1. release build, not debug build;
2. no `LEO_CUDA_FULL_STEP_METRICS`;
3. no phase/replay debug environment variables;
4. no Nsight profiler attached;
5. actual `replay_wall_fraction` and `replay_prefix_steps`;
6. GPU utilization/occupancy with `nvidia-smi` and then Nsight;
7. optimized-vs-legacy benchmark with identical input;
8. whether multi-GPU replay is still the default serial mode.

### Out of memory

Multi-GPU mode replicates the model; it does not pool VRAM. Reduce the size of a
smoke workload first. Treat changes to logical workers/model configuration as a
new quality experiment rather than silently changing them in a performance A/B.

### Experimental replay is faster but quality drops

Disable it:

```bash
unset LEO_REPLAY_STREAMING
unset LEO_MULTI_GPU_PARALLEL_REPLAY
```

The speed result is not accepted if held-out quality regresses.

## 28. Pre-push checklist

Run this immediately before pushing:

```bash
cargo fmt --check
cargo build --workspace
cargo test --workspace
cargo clippy --workspace --all-targets -- -D warnings
bash ./scripts/check.sh
bash ./scripts/check_gpu.sh
git diff --check
git status --short
```

Then review the actual patch:

```bash
git diff --stat
git diff
```

Do not include model/data/cache/run artifacts in the commit.

## 29. Recommended rollout sequence

1. Apply overlay and build release.
2. Pass `scripts/check.sh`.
3. Pass `scripts/check_gpu.sh` on the target NVIDIA GPU.
4. Run optimized-vs-legacy 32K benchmark with experimental replay off.
5. Record the real replay-on `steps_per_second` and state hash.
6. Run a longer safe/default training and `scripts/evaluate.sh`.
7. Benchmark `LEO_REPLAY_STREAMING=1` separately.
8. Run the full quality A/B before accepting streaming replay.
9. If using multiple GPUs, benchmark exact/default multi-GPU first.
10. Benchmark `LEO_MULTI_GPU_PARALLEL_REPLAY=1` separately and run its own
    quality A/B before accepting it.
11. Only combine experimental replay modes after each is independently stable.
12. Keep the exact/default path available as the production rollback baseline.

## 30. Recommended production state today

Until target-GPU compilation, exact-state gates, and full quality A/B results are
available, the recommended state is:

```bash
unset LEO_REPLAY_STREAMING
unset LEO_MULTI_GPU_PARALLEL_REPLAY
unset LEO_CUDA_FULL_STEP_METRICS
```

Use the default persistent/grouped/device-merge CUDA optimizations. Enable
`LEO_MULTI_GPU=1` only when you intentionally want the exact/default multi-GPU
story-batch path and have a homogeneous visible GPU group.

The experimental replay flags should graduate to production only after measured
throughput gains and held-out quality both satisfy the project acceptance bar.
