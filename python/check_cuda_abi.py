from __future__ import annotations

import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SPEC = ROOT / "crates/leo-core/cuda_abi.def"
CUDA_RS = ROOT / "crates/leo-core/src/cuda.rs"
KERNELS = ROOT / "crates/leo-core/src/cuda_kernels.cu"


def pascal(name: str) -> str:
    return "".join(part[:1].upper() + part[1:].lower() for part in name.split("_") if part)


def parse_spec() -> tuple[int, list[str], list[str], list[str]]:
    version: int | None = None
    section = ""
    config: list[str] = []
    persistent: list[str] = []
    delta: list[str] = []
    for raw in SPEC.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("abi_version "):
            if version is not None:
                raise AssertionError("duplicate abi_version")
            version = int(line.split()[1])
            continue
        if line.startswith("[") and line.endswith("]"):
            section = line[1:-1]
            continue
        if section == "config":
            ty, name = line.split()
            if ty not in {"u32", "f32"}:
                raise AssertionError(f"unsupported config type: {ty}")
            config.append(name)
        elif section == "persistent":
            persistent.append(line)
        elif section == "delta":
            delta.append(line)
        else:
            raise AssertionError(f"entry outside known section: {line}")
    if version is None:
        raise AssertionError("missing abi_version")
    return version, config, persistent, delta


def assert_unique(label: str, items: list[str]) -> None:
    if len(items) != len(set(items)):
        raise AssertionError(f"duplicate {label} entries")


def main() -> int:
    version, config, persistent, delta = parse_spec()
    assert_unique("config", config)
    assert_unique("persistent", persistent)
    assert_unique("delta", delta)

    cuda = CUDA_RS.read_text(encoding="utf-8")
    kernels = KERNELS.read_text(encoding="utf-8")
    if "struct CudaConfig" in cuda or "struct LeoConfig" in kernels:
        raise AssertionError("CUDA config ABI must be generated, not duplicated")
    if "enum LeoPersistentPointerIndex" in kernels or "enum LeoBatchDeltaPointerIndex" in kernels:
        raise AssertionError("CUDA pointer ABI must be generated, not duplicated")

    persistent_table = cuda.split("fn persistent_pointer_table", 1)[1].split("fn ", 1)[0]
    delta_table = cuda.split("fn batch_delta_pointer_table", 1)[1].split("fn ", 1)[0]
    for name in persistent:
        token = f"PersistentPointer::{pascal(name)}.index()"
        count = persistent_table.count(token)
        if count != 1:
            raise AssertionError(f"{token} must appear exactly once in the persistent table, found {count}")
    for name in delta:
        token = f"BatchDeltaPointer::{pascal(name)}.index()"
        count = delta_table.count(token)
        if count != 1:
            raise AssertionError(f"{token} must appear exactly once in the delta table, found {count}")

    semantics = (ROOT / "crates/leo-core/src/semantics.rs").read_text(encoding="utf-8")
    match = re.search(r"pub const CUDA_ABI_VERSION: u32 = (\d+);", semantics)
    if not match or int(match.group(1)) != version:
        raise AssertionError("semantics CUDA_ABI_VERSION must match cuda_abi.def")

    print(
        f"CUDA ABI OK: v{version}, {len(config)} config fields, "
        f"{len(persistent)} persistent pointers, {len(delta)} delta pointers"
    )
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except AssertionError as exc:
        print(f"CUDA ABI ERROR: {exc}", file=sys.stderr)
        raise SystemExit(1)
