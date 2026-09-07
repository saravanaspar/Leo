#!/usr/bin/env bash
set -euo pipefail
ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
cd "$ROOT"
export PYTHONDONTWRITEBYTECODE=1

python3 -m unittest discover -s python/tests -v
python3 -m unittest discover -s tests -v
python3 python/check_rust_delimiters.py crates
python3 python/check_cuda_abi.py
python3 - <<'PY'
from pathlib import Path
import tomllib
for path in sorted(Path("configs").glob("*.toml")):
    with path.open("rb") as handle:
        tomllib.load(handle)
    print(f"TOML OK: {path}")
PY
for script in scripts/*.sh; do
  bash -n "$script"
done

if command -v cargo >/dev/null 2>&1; then
  cargo fmt --all -- --check
  cargo test --workspace
  cargo clippy --workspace --all-targets -- -D warnings
else
  if [[ "${CI:-}" == "1" || "${CI:-}" == "true" ]]; then
    echo "cargo not found: CI requires Rust formatting, compilation, tests, and Clippy" >&2
    exit 1
  fi
  echo "cargo not found: Rust formatting, compilation, tests, and Clippy were skipped locally" >&2
fi
