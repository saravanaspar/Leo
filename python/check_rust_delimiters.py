#!/usr/bin/env python3
"""Lexically verify balanced Rust delimiters when a Rust toolchain is unavailable."""

from __future__ import annotations

import argparse
from pathlib import Path

OPEN_TO_CLOSE = {"(": ")", "[": "]", "{": "}"}
CLOSE_TO_OPEN = {value: key for key, value in OPEN_TO_CLOSE.items()}


def scan(path: Path) -> list[str]:
    text = path.read_text(encoding="utf-8")
    stack: list[tuple[str, int]] = []
    errors: list[str] = []
    index = 0
    line = 1
    state = "code"
    block_depth = 0
    raw_hashes = 0

    while index < len(text):
        char = text[index]
        following = text[index + 1] if index + 1 < len(text) else ""
        if char == "\n":
            line += 1

        if state == "code":
            if char == "/" and following == "/":
                state = "line-comment"
                index += 2
                continue
            if char == "/" and following == "*":
                state = "block-comment"
                block_depth = 1
                index += 2
                continue
            if char == '"':
                state = "string"
                index += 1
                continue
            if char == "'":
                closing = text.find("'", index + 1, min(len(text), index + 6))
                if closing != -1:
                    state = "character"
                    index += 1
                    continue
            if char == "r":
                cursor = index + 1
                while cursor < len(text) and text[cursor] == "#":
                    cursor += 1
                if cursor < len(text) and text[cursor] == '"':
                    raw_hashes = cursor - index - 1
                    state = "raw-string"
                    index = cursor + 1
                    continue
            if char in OPEN_TO_CLOSE:
                stack.append((char, line))
            elif char in CLOSE_TO_OPEN:
                if not stack or stack[-1][0] != CLOSE_TO_OPEN[char]:
                    errors.append(f"{path}:{line}: unexpected {char}")
                else:
                    stack.pop()
            index += 1
            continue

        if state == "line-comment":
            if char == "\n":
                state = "code"
            index += 1
            continue

        if state == "block-comment":
            if char == "/" and following == "*":
                block_depth += 1
                index += 2
                continue
            if char == "*" and following == "/":
                block_depth -= 1
                index += 2
                if block_depth == 0:
                    state = "code"
                continue
            index += 1
            continue

        if state in {"string", "character"}:
            if char == "\\":
                index += 2
                continue
            terminator = '"' if state == "string" else "'"
            if char == terminator:
                state = "code"
            index += 1
            continue

        if state == "raw-string":
            if char == '"' and text.startswith("#" * raw_hashes, index + 1):
                index += raw_hashes + 1
                state = "code"
            index += 1
            continue

    errors.extend(f"{path}:{opening_line}: unclosed {opening}" for opening, opening_line in stack)
    if state in {"string", "character", "raw-string", "block-comment"}:
        errors.append(f"{path}:{line}: unterminated {state}")
    return errors


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", nargs="?", type=Path, default=Path("crates"))
    args = parser.parse_args()
    paths = sorted(args.root.rglob("*.rs"))
    errors = [error for path in paths for error in scan(path)]
    if errors:
        print("\n".join(errors))
        return 1
    print(f"Rust delimiter scan OK: {len(paths)} files")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
