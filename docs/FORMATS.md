# Leo v1.0.0 artifact formats

v1.0.0 intentionally establishes a clean artifact baseline. Pre-v1 experimental artifacts are not read by the v1 runtime; fresh training is the supported migration path for this release.

## Checkpoint: PSCLS100

`leo-format` writes checkpoint magic `PSCLS100`, checkpoint schema version 1.

Properties:

- atomic temp-file -> file sync -> rename -> directory sync commit flow;
- exact required section set;
- duplicate section rejection;
- non-overlapping, aligned section ranges;
- exact dtype/rank/shape validation;
- bounded configuration payload and checkpoint size;
- SHA-256 per-section integrity plus SHA-256 header integrity;
- embedded `SemanticsContract` checked when loading;
- standard TOML configuration payload.

Checkpoint identity returned by `checkpoint_hash()` is SHA-256, not FNV.

## Dataset: LEODATA1

Prepared `.idx` files use `LEODATA1`, dataset schema version 1. The header binds:

- record count;
- `.bytes` length;
- complete `.bytes` SHA-256;
- complete record-table SHA-256;
- provenance SHA-256;
- immutable dataset ID.

Opening a dataset validates the complete artifacts and every record range before training begins. Resume therefore cannot mistake a same-length middle-of-file mutation for the original dataset.

`python/prepare_tinystories.py` writes the v1 index and a manifest containing source/preparer provenance and artifact digests. The preparation script accepts an explicit input text format; `scripts/data.sh` uses the pinned TinyStories delimiter format.

## Training resume: LEOTRAIN100

Training resume state uses `LEOTRAIN100`. It binds the operation to:

- checkpoint generation/revision;
- complete training `DatasetId`;
- optional validation `DatasetId`;
- story/pass/worker/backend/synchronization settings;
- byte limit and validation limits;
- progress and early-stopping state;
- best-checkpoint SHA-256 when present.

Resume state is transactionally committed with checkpoints. A state that does not match the saved model or immutable dataset identity is rejected instead of guessed.

## Hash algorithm

Leo v1 uses SHA-256 for durable artifact identity and integrity. The algorithm is explicit in `leo-core::artifact`; FNV is not used as an immutable experiment/checkpoint/dataset identity gate.
