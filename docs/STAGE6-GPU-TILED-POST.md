# STAGE6 GPU TILED POST

> Historical design record. This file is retained only to explain the evolution of the CUDA executor. It is **not** the Leo v1.0.0 runtime contract.

The authoritative current behavior is documented in [TDD.md](TDD.md), [SEMANTICS.md](SEMANTICS.md), and [FORMATS.md](FORMATS.md). Leo v1.0.0 uses the shared-wavefront GPU batch path, prefers cooperative persistent execution when supported, applies the same 30% TrainingPolicy v1 replay regardless of backend, and uses `PSCLS100` checkpoints. Development-only environment switches from earlier GPU experiments are not part of v1.0.0.

Any future optimization originating from this design record must preserve ExecutionSemantics v1 or explicitly introduce a new semantic contract version.
