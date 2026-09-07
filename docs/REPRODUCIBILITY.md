# Leo v1.0.1 reproducibility guide

## Goal

Leo's long-lived experiment baseline depends on preserving enough identity to answer: **what code, dependencies, data, configuration, hardware, and starting model produced this checkpoint?**

## Source identity

Record the exact Git commit:

```bash
git rev-parse HEAD
```

Do not describe an experiment only as "v1.0.1" when multiple commits may exist under the same software version during active development.

## Dependency identity

`Cargo.lock` should be committed for the v1.0.1 application baseline and retained with reference training environments.

Record:

```bash
sha256sum Cargo.lock
rustc --version
cargo --version
```

Dependency updates should be explicit PRs rather than implicit re-resolution during a training run.

## Dataset identity

Prepared data uses Leo's verified dataset artifact path. Keep the generated `manifest.json` and the dataset IDs emitted by the tooling/logs.

Current public testing uses TinyStories prepared by `scripts/data.sh`, which pins an upstream repository revision and source SHA-256 values.

Do not replace a dataset file in place while reusing an old manifest/run identity.

## Configuration identity

Keep the exact config used for training and record its checksum:

```bash
sha256sum configs/tinystories.toml
```

If you modify a config for an experiment, copy it into the experiment metadata/run record rather than relying on a later working-tree version.

## Model/checkpoint identity

Record the initial and final checkpoint hashes and the model generation/revision metadata exposed by Leo.

Training resume should use the same dataset and compatible training identity. Do not bypass resume identity checks to force incompatible runs together.

## GPU identity

For CUDA runs, capture:

```bash
nvidia-smi
```

At minimum retain:

- GPU model;
- GPU UUID when available;
- VRAM;
- NVIDIA driver version;
- CUDA/NVRTC environment;
- compute capability if relevant;
- Leo CUDA execution-profile identity/logs.

Leo's CUDA autotuner cache is hardware/model/semantic/logical-batch specific but remains **execution cache**, not learned state. Both complete winners and incomplete candidate observations are persisted atomically; a fresh process may emit `cuda_autotune_resumed` and continue the same search. Deleting the cache is allowed; doing so may change warmup/tuning timing and selected execution geometry without changing the intended learning policy.

## TinyStories reference provenance

Current source pin:

```text
repository: roneneldan/TinyStories
revision:   5485261731eaac25dd8e5ebbc3839d0a9870b185
```

The dataset is external to Leo and has separate upstream licensing/terms.

## Recommended experiment record

Store or log something equivalent to:

```text
leo_git_commit=
cargo_lock_sha256=
rust_version=
config_sha256=
dataset_id=
source_dataset_repository=
source_dataset_revision=
initial_checkpoint_sha256=
final_checkpoint_sha256=
backend=
gpu_model=
gpu_driver=
cuda_version=
workers=
passes=
max_stories=
max_bytes=
validation_stories=
replay_fraction=
replay_segments=
replay_steps=
cuda_autotune_profile_key=
```

The runtime already records many of these semantic/training fields; the surrounding experiment environment should retain the rest.
