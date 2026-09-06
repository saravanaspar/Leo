# Leo v1.0.0

Leo is a sparse recurrent byte-learning system implemented in Rust with a CPU reference backend and a custom CUDA backend. **v1.0.0 is a clean training/artifact baseline**: prepare v1 data and initialize a fresh v1 model before the reference training run.

The v1 training policy keeps bounded surprise replay enabled at **30%** in the shipped configs, uses FP32 exact semantics, and records independent model/training/execution/dataset/CUDA-ABI contract versions. Backend selection changes execution, not learning policy.

Durability contracts: [semantics](docs/SEMANTICS.md), [artifact formats](docs/FORMATS.md), and [technical design](docs/TDD.md).

This repository provides:

- single-GPU CUDA training
- experimental multi-GPU CUDA training
- persistent per-GPU model replicas
- sparse multi-GPU parameter synchronization
- automatic multi-GPU frozen evaluation
- capped validation during scripted training
- CPU reference backend
- checkpoint/resume support
- held-out and training-set evaluation
- prompt generation, teaching, inspection, rollback, and benchmarking

## Automatic safe GPU execution acceleration

The CUDA backend automatically optimizes **execution**, not the learning policy. v1.0.0 keeps FP32 and 30% bounded-surprise replay unchanged while using exact worklists, sparse delta application, adaptive physical lane chunks, cooperative-grid wavefront fusion, online multi-dimensional GPU execution tuning, SHA-256-keyed NVRTC PTX caching, separate pinned transfer/compute streams, one-batch-ahead dataset prefetch, and CUDA Graph replay for the stable sparse apply/reset sequence when the installed driver supports it.

The logical `--workers` value is never autotuned because it defines the story batch whose sparse deltas are mean-reduced. The tuner searches execution-only lane, sparse-apply, and fused-grid geometry, and its cache identity includes the concrete GPU/driver/model/semantics identity plus logical batch width. Profiles live under `${LEO_CACHE_DIR}/cuda`, `${XDG_CACHE_HOME}/leo/cuda`, or `~/.cache/leo/cuda`. Delete that cache at any time to force PTX recompilation and execution-plan retuning; it contains no learned model state. Runtime `cuda_profile` events report sampled transfer/compute/wait timing; use `scripts/profile_cuda.sh` with Nsight Compute for real DRAM, occupancy, warp/branch, instruction, and atomic hardware counters.

## Multi-GPU note

Enable multi-GPU training with:

```bash
CUDA_VISIBLE_DEVICES=0,1 LEO_MULTI_GPU=1
```

The current training synchronization mode is:

```text
gpu_multi_device_batch_mean_experimental
```

The two GPU shards are merged at the end of each story batch. Sparse changed parameters are synchronized back to the persistent GPU replicas.

The implementation reports:

```text
exact_single_gpu_wavefront_equivalence=false
```

because multi-GPU training uses batch-end device averaging rather than reproducing the exact single-GPU byte-wavefront update order.

Frozen evaluation is read-only and can be distributed across all visible GPUs without changing the model.

---

# Repository layout

```text
Leo/
├── Cargo.toml
├── configs/
├── crates/
├── data/
│   └── prepared/
├── docs/
├── python/
├── runs/
│   └── quality/
├── scripts/
├── tests/
└── README.md
```

The clean archive keeps all of `data/` and only permanent runs under `runs/quality/`.

Temporary benchmark runs, build output, caches, and patch backups are intentionally excluded.

---

# 1. Kaggle setup

Enter the project:

```bash
cd /kaggle/working/Leo
```

Install Rust if needed:

```bash
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs \
  | sh -s -- -y --profile minimal

export PATH="$HOME/.cargo/bin:$PATH"
```

Optional development components:

```bash
rustup component add rustfmt clippy
```

Build Leo:

```bash
cargo build --release -p leo-cli
```

The source archive intentionally does not carry the pre-v1 `Cargo.lock`. The first v1 build generates a lockfile for the pinned direct dependencies; keep that generated `Cargo.lock` with the exact training environment you use for the reference run.

The binary is:

```text
target/release/leo
```

Check GPUs:

```bash
nvidia-smi -L
```

Show CLI help:

```bash
./target/release/leo --help
```

---

# 2. Prepared data

The normal prepared TinyStories dataset is:

```text
data/prepared/
├── tinystories.train.bytes
├── tinystories.train.idx
├── tinystories.valid.bytes
├── tinystories.valid.idx
└── manifest.json
```

The validation set is held out. Do not train on `tinystories.valid.*`.

## Prepare or rebuild data

General script syntax:

```bash
./scripts/data.sh \
  [raw-directory] \
  [prepared-directory] \
  [train-story-limit] \
  [valid-story-limit] \
  [train-byte-limit] \
  [valid-byte-limit]
```

Default:

```bash
./scripts/data.sh
```

The script downloads the pinned TinyStories train/validation files if they are missing, verifies their checksums, and writes the prepared byte/index files.

---

# 3. Initialize a model

```bash
mkdir -p runs/quality/my-run

./target/release/leo init \
  --config configs/tinystories.toml \
  --output runs/quality/my-run/leo.pscls
```

---

# 4. Recommended two-GPU training

`scripts/train.sh` accepts:

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

Recommended example:

```bash
CUDA_VISIBLE_DEVICES=0,1 \
LEO_MULTI_GPU=1 \
./scripts/train.sh \
  data/prepared \
  runs/quality/my-run \
  configs/tinystories.toml \
  1 \
  100000 \
  64 \
  "" \
  gpu \
  100
```

Meaning:

```text
passes               1
max training stories 100000
workers               64 total
max input bytes       unlimited
backend               gpu
validation stories    100
```

The script builds Leo, initializes the model if it does not exist, trains, and writes `train.jsonl` in the run directory.

The script caps validation instead of silently evaluating the complete validation corpus after every pass.

For development:

```text
10 validation stories    smoke check
100 validation stories   routine check
1000 validation stories  larger quality check
full validation set      deliberate final evaluation only
```

---

# 5. Direct two-GPU training

Use the direct CLI when you want complete control.

```bash
CUDA_VISIBLE_DEVICES=0,1 \
LEO_MULTI_GPU=1 \
./target/release/leo train \
  --model runs/quality/my-run/leo.pscls \
  --train-bytes data/prepared/tinystories.train.bytes \
  --train-index data/prepared/tinystories.train.idx \
  --valid-bytes data/prepared/tinystories.valid.bytes \
  --valid-index data/prepared/tinystories.valid.idx \
  --validation-stories 100 \
  --passes 1 \
  --workers 64 \
  --backend gpu \
  --fresh-run
```

Use `--fresh-run` only when intentionally starting a fresh training operation and discarding any previous resume state for that model.

---

# 6. Resume training

Leo maintains training resume state beside the model checkpoint.

To resume an interrupted direct training run, use the same command again without `--fresh-run`.

For normal scripted training, simply rerun the same `scripts/train.sh` command.

Example:

```bash
CUDA_VISIBLE_DEVICES=0,1 \
LEO_MULTI_GPU=1 \
./scripts/train.sh \
  data/prepared \
  runs/quality/my-run \
  configs/tinystories.toml \
  1 \
  100000 \
  64 \
  "" \
  gpu \
  100
```

---

# 7. Training without validation

Useful for a pure speed benchmark or controlled experiment:

```bash
CUDA_VISIBLE_DEVICES=0,1 \
LEO_MULTI_GPU=1 \
./target/release/leo train \
  --model runs/quality/my-run/leo.pscls \
  --train-bytes data/prepared/tinystories.train.bytes \
  --train-index data/prepared/tinystories.train.idx \
  --passes 1 \
  --max-stories 1024 \
  --workers 64 \
  --backend gpu \
  --fresh-run
```

No `--valid-bytes` or `--valid-index` means no post-pass validation.

---

# 8. Limit training by story count

```bash
CUDA_VISIBLE_DEVICES=0,1 \
LEO_MULTI_GPU=1 \
./target/release/leo train \
  --model runs/quality/my-run/leo.pscls \
  --train-bytes data/prepared/tinystories.train.bytes \
  --train-index data/prepared/tinystories.train.idx \
  --passes 1 \
  --max-stories 10000 \
  --workers 64 \
  --backend gpu
```

---

# 9. Limit training by input bytes

```bash
CUDA_VISIBLE_DEVICES=0,1 \
LEO_MULTI_GPU=1 \
./target/release/leo train \
  --model runs/quality/my-run/leo.pscls \
  --train-bytes data/prepared/tinystories.train.bytes \
  --train-index data/prepared/tinystories.train.idx \
  --passes 1 \
  --max-bytes 100000000 \
  --workers 64 \
  --backend gpu
```

---

# 10. Single-GPU training

Expose only one GPU and do not set `LEO_MULTI_GPU=1`:

```bash
CUDA_VISIBLE_DEVICES=0 \
./target/release/leo train \
  --model runs/quality/my-run/leo.pscls \
  --train-bytes data/prepared/tinystories.train.bytes \
  --train-index data/prepared/tinystories.train.idx \
  --valid-bytes data/prepared/tinystories.valid.bytes \
  --valid-index data/prepared/tinystories.valid.idx \
  --validation-stories 100 \
  --passes 1 \
  --workers 64 \
  --backend gpu
```

---

# 11. CPU training

CPU is primarily the reference backend:

```bash
./target/release/leo train \
  --model runs/quality/my-run/leo.pscls \
  --train-bytes data/prepared/tinystories.train.bytes \
  --train-index data/prepared/tinystories.train.idx \
  --passes 1 \
  --workers 1 \
  --backend cpu
```

---

# 12. Fast frozen held-out benchmark

Frozen benchmark evaluation automatically uses all visible GPUs.

```bash
CUDA_VISIBLE_DEVICES=0,1 \
./target/release/leo benchmark \
  --model runs/quality/my-run/leo.pscls \
  --bytes data/prepared/tinystories.valid.bytes \
  --index data/prepared/tinystories.valid.idx \
  --stories 100 \
  --backend gpu
```

During multi-GPU evaluation you should see progress similar to:

```text
{"event":"evaluation_progress","shard":0,"shards":2,...}
{"event":"evaluation_progress","shard":1,"shards":2,...}
```

---

# 13. Held-out evaluation

```bash
CUDA_VISIBLE_DEVICES=0,1 \
./target/release/leo eval \
  --model runs/quality/my-run/leo.pscls \
  --bytes data/prepared/tinystories.valid.bytes \
  --index data/prepared/tinystories.valid.idx \
  --stories 100 \
  --generation-stories 4 \
  --prompt "Once upon a time" \
  --backend gpu
```

This evaluates the model on held-out validation stories and runs generation probes.

---

# 14. Held-out plus training-set evaluation

This compares performance on unseen validation stories with performance on training stories:

```bash
CUDA_VISIBLE_DEVICES=0,1 \
./target/release/leo eval \
  --model runs/quality/my-run/leo.pscls \
  --bytes data/prepared/tinystories.valid.bytes \
  --index data/prepared/tinystories.valid.idx \
  --stories 100 \
  --train-bytes data/prepared/tinystories.train.bytes \
  --train-index data/prepared/tinystories.train.idx \
  --train-stories 100 \
  --generation-stories 4 \
  --prompt "Once upon a time" \
  --backend gpu
```

---

# 15. Complete evaluation script

General syntax:

```text
./scripts/evaluate.sh \
  <prepared-data-directory> \
  <run-directory> \
  [valid-stories] \
  [train-stories] \
  [backend]
```

Example:

```bash
CUDA_VISIBLE_DEVICES=0,1 \
./scripts/evaluate.sh \
  data/prepared \
  runs/quality/my-run \
  100 \
  100 \
  gpu
```

The script performs:

1. held-out evaluation
2. optional training-set comparison
3. generation probe
4. prompt generation
5. frozen benchmark
6. baseline comparison

Typical run outputs include:

```text
train.jsonl
eval.jsonl
story.txt
benchmark.jsonl
baselines.json
```

---

# 16. All-in-one train then evaluate

```bash
CUDA_VISIBLE_DEVICES=0,1 \
LEO_MULTI_GPU=1 \
./scripts/train.sh \
  data/prepared \
  runs/quality/my-run \
  configs/tinystories.toml \
  1 \
  100000 \
  64 \
  "" \
  gpu \
  100 \
&& \
CUDA_VISIBLE_DEVICES=0,1 \
./scripts/evaluate.sh \
  data/prepared \
  runs/quality/my-run \
  100 \
  100 \
  gpu
```

Evaluation starts only if training exits successfully.

---

# 17. Training benchmark

Benchmark training without committing the benchmark run as a normal training job:

```bash
CUDA_VISIBLE_DEVICES=0,1 \
LEO_MULTI_GPU=1 \
./target/release/leo benchmark \
  --train \
  --model runs/quality/my-run/leo.pscls \
  --bytes data/prepared/tinystories.train.bytes \
  --index data/prepared/tinystories.train.idx \
  --stories 1024 \
  --workers 64 \
  --backend gpu
```

Byte-limited benchmark:

```bash
CUDA_VISIBLE_DEVICES=0,1 \
LEO_MULTI_GPU=1 \
./target/release/leo benchmark \
  --train \
  --model runs/quality/my-run/leo.pscls \
  --bytes data/prepared/tinystories.train.bytes \
  --index data/prepared/tinystories.train.idx \
  --max-bytes 10000000 \
  --workers 64 \
  --backend gpu
```

---

# GPU verification and hardware profiling

Run the real CUDA/NVRTC/conformance smoke locally on a GPU machine:

```bash
./scripts/check_gpu.sh
```

For a hardware-counter profile of a representative training benchmark:

```bash
./scripts/profile_cuda.sh \
  runs/quality/my-run/leo.pscls \
  data/prepared/tinystories.train.bytes \
  data/prepared/tinystories.train.idx \
  64 256 leo-cuda-profile
```

The optional `.github/workflows/gpu-ci.yml` runs the same GPU smoke weekly and on manual dispatch when the repository/org variable `LEO_GPU_RUNNER` is set to a configured GitHub GPU larger-runner name (GitHub Team/Enterprise Cloud) or a custom label on a self-hosted GPU runner. Without that variable the job is skipped rather than pretending GPU validation occurred.

---

# 18. Generate text

```bash
./target/release/leo prompt \
  --model runs/quality/my-run/leo.pscls \
  --text "Once upon a time" \
  --max-bytes 500 \
  --temperature 0.8 \
  --backend gpu
```

JSON output:

```bash
./target/release/leo prompt \
  --model runs/quality/my-run/leo.pscls \
  --text "Once upon a time" \
  --max-bytes 500 \
  --temperature 0.8 \
  --backend gpu \
  --json
```

---

# 19. Teach text directly

Provisional:

```bash
./target/release/leo teach \
  --model runs/quality/my-run/leo.pscls \
  --permission provisional \
  --text "Text to teach." \
  --backend gpu
```

Training permission:

```bash
./target/release/leo teach \
  --model runs/quality/my-run/leo.pscls \
  --permission training \
  --text "Text to teach." \
  --backend gpu
```

Verified:

```bash
./target/release/leo teach \
  --model runs/quality/my-run/leo.pscls \
  --permission verified \
  --text "Text to teach." \
  --backend gpu
```

---

# 20. Inspect a checkpoint

Human-readable:

```bash
./target/release/leo inspect \
  --model runs/quality/my-run/leo.pscls
```

JSON:

```bash
./target/release/leo inspect \
  --model runs/quality/my-run/leo.pscls \
  --json
```

---

# 21. Create a checkpoint generation

```bash
./target/release/leo checkpoint \
  --model runs/quality/my-run/leo.pscls
```

---

# 22. Roll back to an older generation

```bash
./target/release/leo rollback \
  --model runs/quality/my-run/leo.pscls \
  --generation 3
```

Use a generation number that exists in retained checkpoint history.

---

# 23. Backend information and probes

CPU backend:

```bash
./target/release/leo backend \
  --backend cpu \
  --json
```

GPU backend:

```bash
./target/release/leo backend \
  --backend gpu \
  --json
```

GPU backend with a model probe:

```bash
./target/release/leo backend \
  --backend gpu \
  --model runs/quality/my-run/leo.pscls \
  --json
```

---

# 24. Development checks

Compile check:

```bash
cargo check --release -p leo-cli
```

Build:

```bash
cargo build --release -p leo-cli
```

Format:

```bash
cargo fmt --all
```

Rust delimiter scanner:

```bash
python3 python/check_rust_delimiters.py
```

Python tests:

```bash
python3 -m pytest -q python/tests tests
```

Complete project check:

```bash
./scripts/check.sh
```

---

# 25. CLI command reference

The main CLI commands are:

```text
init
train
eval
prompt
teach
inspect
checkpoint
rollback
benchmark
backend
```

Current CLI help syntax:

```text
leo init --config <FILE> --output <MODEL.pscls>

leo train --model <MODEL.pscls>
  --train-index <FILE>
  --train-bytes <FILE>
  [--valid-index <FILE> --valid-bytes <FILE>]
  [--passes N]
  [--max-stories N]
  [--max-bytes N]
  [--workers N]
  [--backend auto|cpu|gpu]
  [--fresh-run]

leo eval --model <MODEL.pscls>
  --index <FILE>
  --bytes <FILE>
  [--stories N]
  [--train-index <FILE> --train-bytes <FILE> --train-stories N]
  [--generation-stories N]
  [--prompt "Once upon a time"]
  [--backend auto|cpu|gpu]

leo prompt --model <MODEL.pscls>
  --text <STORY_PREFIX>
  [--max-bytes N]
  [--temperature F]
  [--seed N]
  [--backend auto|cpu|gpu]
  [--json]

leo teach --model <MODEL.pscls>
  --permission provisional|training|verified
  --text <LESSON>
  [--backend auto|cpu|gpu]

leo inspect --model <MODEL.pscls> [--json]

leo checkpoint --model <MODEL.pscls>

leo rollback --model <MODEL.pscls> --generation <N>

leo benchmark --model <MODEL.pscls>
  --index <FILE>
  --bytes <FILE>
  [--stories N]
  [--train --max-bytes N --workers N]
  [--backend auto|cpu|gpu]

leo backend
  [--backend auto|cpu|gpu]
  [--model <MODEL.pscls>]
  [--json]
```

Leo supports capped training validation with `--validation-stories N` and opt-in multi-GPU training through `LEO_MULTI_GPU=1`.

---

# 26. Clean archival policy

Keep:

```text
Cargo.toml
LICENSE
Makefile
README.md
configs/
crates/
data/
docs/
python/
runs/quality/
scripts/
tests/
```

Remove before making a canonical archive:

```text
target/
.pytest_cache/
.mypy_cache/
.ruff_cache/
__pycache__/
.ipynb_checkpoints/
.leo_binary_path
runs/* except runs/quality/
```

Recommended archive command from `/kaggle/working`:

```bash
tar -cf - Leo | pigz -1 -p "$(nproc)" > Leo_clean_mgpu.tar.gz
```
