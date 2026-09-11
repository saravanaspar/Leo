#!/usr/bin/env python3
"""Prepare and verify Leo software releases.

The software release version is intentionally independent from Leo's semantic,
artifact, checkpoint, dataset, and CUDA ABI contract versions.
"""

from __future__ import annotations

import argparse
import datetime as dt
import re
import sys
import tomllib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
WORKSPACE_PACKAGES = ("leo-cli", "leo-core", "leo-data", "leo-format")
SEMVER_RE = re.compile(r"^(?:v)?(0|[1-9]\d*)\.(0|[1-9]\d*)\.(0|[1-9]\d*)$")


class ReleaseError(RuntimeError):
    pass


def normalize_version(value: str) -> str:
    match = SEMVER_RE.fullmatch(value.strip())
    if match is None:
        raise ReleaseError("version must be stable SemVer X.Y.Z (an optional leading v is accepted)")
    return ".".join(match.groups())


def version_tuple(value: str) -> tuple[int, int, int]:
    normalized = normalize_version(value)
    return tuple(int(part) for part in normalized.split("."))  # type: ignore[return-value]


def current_version(root: Path = ROOT) -> str:
    with (root / "Cargo.toml").open("rb") as handle:
        workspace = tomllib.load(handle)
    return normalize_version(str(workspace["workspace"]["package"]["version"]))


def replace_workspace_version(text: str, version: str) -> str:
    lines = text.splitlines(keepends=True)
    in_workspace_package = False
    replaced = False
    for index, line in enumerate(lines):
        stripped = line.strip()
        if stripped.startswith("[") and stripped.endswith("]"):
            in_workspace_package = stripped == "[workspace.package]"
            continue
        if in_workspace_package and re.match(r"^\s*version\s*=", line):
            newline = "\n" if line.endswith("\n") else ""
            indent = line[: len(line) - len(line.lstrip())]
            lines[index] = f'{indent}version = "{version}"{newline}'
            replaced = True
            break
    if not replaced:
        raise ReleaseError("could not find workspace.package version in Cargo.toml")
    return "".join(lines)


def update_citation(text: str, version: str, release_date: str) -> str:
    updated, count = re.subn(
        r'(?m)^version:\s*"[^"]+"\s*$',
        f'version: "{version}"',
        text,
        count=1,
    )
    if count != 1:
        raise ReleaseError("CITATION.cff must contain exactly one version field")

    if re.search(r"(?m)^date-released:\s*", updated):
        updated, count = re.subn(
            r'(?m)^date-released:\s*"?[0-9]{4}-[0-9]{2}-[0-9]{2}"?\s*$',
            f'date-released: "{release_date}"',
            updated,
            count=1,
        )
        if count != 1:
            raise ReleaseError("could not update date-released in CITATION.cff")
    else:
        updated = updated.rstrip() + f'\ndate-released: "{release_date}"\n'
    return updated


def update_semantics_release(text: str, version: str) -> str:
    updated, count = re.subn(
        r"(?m)^\| Leo release \| [^|]+ \|$",
        f"| Leo release | {version} |",
        text,
        count=1,
    )
    if count != 1:
        raise ReleaseError("docs/SEMANTICS.md is missing the Leo release row")
    return updated


def release_changelog(text: str, version: str, release_date: str) -> str:
    marker = "## Unreleased"
    start = text.find(marker)
    if start < 0:
        raise ReleaseError("CHANGELOG.md is missing '## Unreleased'")

    body_start = start + len(marker)
    next_heading = text.find("\n## ", body_start)
    if next_heading < 0:
        next_heading = len(text)

    body = text[body_start:next_heading].strip()
    if not body:
        raise ReleaseError("CHANGELOG.md Unreleased section is empty")

    target_heading = f"## v{version} - {release_date}"
    if re.search(rf"(?m)^## v{re.escape(version)}(?:\s|-|$)", text):
        raise ReleaseError(f"CHANGELOG.md already contains a v{version} release section")

    prefix = text[:start]
    suffix = text[next_heading:].lstrip("\n")
    return (
        f"{prefix}## Unreleased\n\n"
        f"{target_heading}\n\n{body}\n\n"
        f"{suffix}"
    ).rstrip() + "\n"


def changelog_notes(text: str, version: str) -> str:
    pattern = re.compile(
        rf"(?ms)^## v{re.escape(version)}(?:\s+-\s+[^\n]+)?\n\n?(.*?)(?=^## |\Z)"
    )
    match = pattern.search(text)
    if match is None:
        raise ReleaseError(f"CHANGELOG.md has no v{version} section")
    notes = match.group(1).strip()
    if not notes:
        raise ReleaseError(f"CHANGELOG.md v{version} section is empty")
    return notes + "\n"


def prepare(root: Path, version: str, release_date: str) -> None:
    version = normalize_version(version)
    current = current_version(root)
    if version_tuple(version) <= version_tuple(current):
        raise ReleaseError(
            f"target version {version} must be greater than current version {current}"
        )

    cargo_path = root / "Cargo.toml"
    cargo_path.write_text(
        replace_workspace_version(cargo_path.read_text(encoding="utf-8"), version),
        encoding="utf-8",
    )

    citation_path = root / "CITATION.cff"
    citation_path.write_text(
        update_citation(citation_path.read_text(encoding="utf-8"), version, release_date),
        encoding="utf-8",
    )

    semantics_path = root / "docs" / "SEMANTICS.md"
    semantics_path.write_text(
        update_semantics_release(semantics_path.read_text(encoding="utf-8"), version),
        encoding="utf-8",
    )

    changelog_path = root / "CHANGELOG.md"
    changelog_path.write_text(
        release_changelog(
            changelog_path.read_text(encoding="utf-8"), version, release_date
        ),
        encoding="utf-8",
    )


def verify(root: Path, version: str) -> None:
    version = normalize_version(version)
    errors: list[str] = []

    if current_version(root) != version:
        errors.append(f"Cargo.toml workspace version is not {version}")

    lock = (root / "Cargo.lock").read_text(encoding="utf-8")
    for package in WORKSPACE_PACKAGES:
        if f'name = "{package}"\nversion = "{version}"' not in lock:
            errors.append(f"Cargo.lock package {package} is not {version}")

    citation = (root / "CITATION.cff").read_text(encoding="utf-8")
    if f'version: "{version}"' not in citation:
        errors.append(f"CITATION.cff version is not {version}")

    semantics = (root / "docs" / "SEMANTICS.md").read_text(encoding="utf-8")
    if f"| Leo release | {version} |" not in semantics:
        errors.append(f"docs/SEMANTICS.md release row is not {version}")

    changelog = (root / "CHANGELOG.md").read_text(encoding="utf-8")
    if not re.search(rf"(?m)^## v{re.escape(version)}(?:\s+-\s+[^\n]+)?$", changelog):
        errors.append(f"CHANGELOG.md has no v{version} release section")

    readme = (root / "README.md").read_text(encoding="utf-8")
    first_line = readme.splitlines()[0] if readme.splitlines() else ""
    if first_line != "# Leo":
        errors.append("README.md title must remain version-neutral: '# Leo'")
    if re.search(r"(?m)^# Leo v\d+\.\d+\.\d+", readme):
        errors.append("README.md must not hardcode the software release in its title")

    if errors:
        raise ReleaseError("\n".join(errors))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    normalize = subparsers.add_parser("normalize", help="normalize a release version")
    normalize.add_argument("version")

    subparsers.add_parser("current", help="print the current workspace release version")

    prepare_parser = subparsers.add_parser("prepare", help="prepare release metadata")
    prepare_parser.add_argument("--version", required=True)
    prepare_parser.add_argument(
        "--date",
        default=dt.date.today().isoformat(),
        help="release date in YYYY-MM-DD (defaults to today)",
    )

    verify_parser = subparsers.add_parser("verify", help="verify release metadata")
    verify_parser.add_argument("--version", required=True)

    notes_parser = subparsers.add_parser("notes", help="print release notes from CHANGELOG")
    notes_parser.add_argument("--version", required=True)

    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    try:
        if args.command == "normalize":
            print(normalize_version(args.version))
        elif args.command == "current":
            print(current_version())
        elif args.command == "prepare":
            prepare(ROOT, args.version, args.date)
        elif args.command == "verify":
            verify(ROOT, args.version)
        elif args.command == "notes":
            print(changelog_notes((ROOT / "CHANGELOG.md").read_text(encoding="utf-8"), normalize_version(args.version)), end="")
        else:
            parser.error(f"unknown command: {args.command}")
    except ReleaseError as exc:
        print(f"release error: {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
