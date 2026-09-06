#!/usr/bin/env python3
"""Prepare exact UTF-8 story records into Leo v1 dataset artifacts."""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import struct
from collections.abc import Iterable, Iterator
from pathlib import Path

INDEX_MAGIC = b"LEODATA1"
DATASET_SCHEMA_VERSION = 1
INDEX_HEADER_SIZE = 192
INDEX = struct.Struct("<QII")
TEXT_KEYS = ("text", "story", "content")
END_OF_TEXT = b"<|endoftext|>"
STREAM_CHUNK_BYTES = 1 << 20
PREPARER_VERSION = "1.0.0"


def sha256_bytes(payload: bytes) -> bytes:
    return hashlib.sha256(payload).digest()


def canonical_json_bytes(payload: object) -> bytes:
    return json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    ).encode("utf-8")


def source_tree_digest(source: Path) -> str:
    hasher = hashlib.sha256()
    root = source if source.is_dir() else source.parent
    for path in iter_input_files(source):
        relative = path.relative_to(root).as_posix().encode("utf-8")
        hasher.update(len(relative).to_bytes(8, "little"))
        hasher.update(relative)
        with path.open("rb") as handle:
            while chunk := handle.read(STREAM_CHUNK_BYTES):
                hasher.update(chunk)
    return hasher.hexdigest()


def iter_input_files(source: Path) -> Iterator[Path]:
    if source.is_file():
        yield source
        return
    for path in sorted(source.rglob("*")):
        if path.is_file() and path.suffix.lower() in {".json", ".jsonl", ".txt"}:
            yield path


def extract_text(record: object, source: Path) -> str:
    if isinstance(record, str):
        return record
    if isinstance(record, dict):
        for key in TEXT_KEYS:
            value = record.get(key)
            if isinstance(value, str):
                return value
    raise ValueError(f"No text/story/content string in record from {source}")


def validated_story(payload: bytes, source: Path) -> bytes | None:
    if not payload:
        return None
    try:
        payload.decode("utf-8", errors="strict")
    except UnicodeDecodeError as error:
        raise ValueError(f"Invalid UTF-8 story in {source}: {error}") from error
    return payload


def iter_delimited_text(handle, source: Path) -> Iterator[bytes]:
    buffer = bytearray()
    while chunk := handle.read(STREAM_CHUNK_BYTES):
        buffer.extend(chunk)
        while True:
            separator = buffer.find(END_OF_TEXT)
            if separator < 0:
                break
            raw = bytes(buffer[:separator]).strip(b"\r\n")
            del buffer[: separator + len(END_OF_TEXT)]
            story = validated_story(raw, source)
            if story is not None:
                yield story
    story = validated_story(bytes(buffer).strip(b"\r\n"), source)
    if story is not None:
        yield story


def iter_paragraph_text(handle, source: Path) -> Iterator[bytes]:
    current: list[bytes] = []
    for line in handle:
        if line in {b"\n", b"\r\n"}:
            raw = b"".join(current).rstrip(b"\r\n")
            current.clear()
            story = validated_story(raw, source)
            if story is not None:
                yield story
        else:
            current.append(line)
    story = validated_story(b"".join(current).rstrip(b"\r\n"), source)
    if story is not None:
        yield story


def detect_text_format(path: Path) -> str:
    with path.open("rb") as handle:
        while chunk := handle.read(STREAM_CHUNK_BYTES):
            if END_OF_TEXT in chunk:
                return "delimited"
    return "paragraph"


def iter_text_records(path: Path, text_format: str) -> Iterator[bytes]:
    resolved = detect_text_format(path) if text_format == "auto" else text_format
    with path.open("rb") as handle:
        if resolved == "delimited":
            yield from iter_delimited_text(handle, path)
        elif resolved == "paragraph":
            yield from iter_paragraph_text(handle, path)
        else:
            raise ValueError(f"Unsupported text format: {resolved}")


def iter_records(path: Path, text_format: str) -> Iterator[bytes]:
    suffix = path.suffix.lower()
    if suffix == ".txt":
        yield from iter_text_records(path, text_format)
        return
    if suffix == ".jsonl":
        with path.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, 1):
                if not line.strip():
                    continue
                try:
                    text = extract_text(json.loads(line), path)
                except (json.JSONDecodeError, ValueError) as error:
                    raise ValueError(f"{path}:{line_number}: {error}") from error
                story = validated_story(text.encode("utf-8", errors="strict"), path)
                if story is not None:
                    yield story
        return

    payload = json.loads(path.read_text(encoding="utf-8"))
    records: Iterable[object]
    if isinstance(payload, list):
        records = payload
    elif isinstance(payload, dict):
        candidate = payload.get("stories") or payload.get("data")
        records = candidate if isinstance(candidate, list) else [payload]
    else:
        records = [payload]
    for record in records:
        text = extract_text(record, path)
        story = validated_story(text.encode("utf-8", errors="strict"), path)
        if story is not None:
            yield story


def iter_source_stories(
    source: Path,
    limit: int | None = None,
    byte_limit: int | None = None,
    text_format: str = "auto",
) -> Iterator[bytes]:
    emitted = 0
    emitted_bytes = 0
    for path in iter_input_files(source):
        for story in iter_records(path, text_format):
            if limit is not None and emitted >= limit:
                return
            if byte_limit is not None and emitted_bytes + len(story) > byte_limit:
                if emitted == 0:
                    raise ValueError(
                        f"First story under {source} exceeds the byte limit {byte_limit}"
                    )
                return
            yield story
            emitted += 1
            emitted_bytes += len(story)
    if emitted == 0:
        raise ValueError(f"No non-empty stories found under {source}")


def collect_stories(
    source: Path, limit: int | None = None, text_format: str = "auto"
) -> list[bytes]:
    return list(iter_source_stories(source, limit, text_format=text_format))


def compute_dataset_id(
    record_count: int,
    bytes_length: int,
    bytes_digest: bytes,
    records_digest: bytes,
    provenance_digest: bytes,
) -> bytes:
    hasher = hashlib.sha256()
    hasher.update(b"LEO-DATASET-ID\0")
    hasher.update(DATASET_SCHEMA_VERSION.to_bytes(4, "little"))
    hasher.update(record_count.to_bytes(8, "little"))
    hasher.update(bytes_length.to_bytes(8, "little"))
    hasher.update(bytes_digest)
    hasher.update(records_digest)
    hasher.update(provenance_digest)
    return hasher.digest()


def write_split(
    output_dir: Path,
    split: str,
    stories: Iterable[bytes],
    provenance: dict[str, object],
) -> dict[str, object]:
    bytes_path = output_dir / f"tinystories.{split}.bytes"
    index_path = output_dir / f"tinystories.{split}.idx"
    offset = 0
    story_count = 0
    records = bytearray()
    bytes_hasher = hashlib.sha256()
    with bytes_path.open("wb") as byte_file:
        for story in stories:
            if len(story) > 0xFFFF_FFFF:
                raise ValueError("A story exceeds the uint32 index length")
            byte_file.write(story)
            bytes_hasher.update(story)
            records.extend(INDEX.pack(offset, len(story), 0))
            offset += len(story)
            story_count += 1
        byte_file.flush()

    bytes_digest = bytes_hasher.digest()
    records_digest = sha256_bytes(bytes(records))
    provenance_digest = sha256_bytes(canonical_json_bytes(provenance))
    dataset_id = compute_dataset_id(
        story_count,
        offset,
        bytes_digest,
        records_digest,
        provenance_digest,
    )

    header = bytearray(INDEX_HEADER_SIZE)
    header[0:8] = INDEX_MAGIC
    header[8:12] = DATASET_SCHEMA_VERSION.to_bytes(4, "little")
    header[12:16] = INDEX_HEADER_SIZE.to_bytes(4, "little")
    header[16:24] = story_count.to_bytes(8, "little")
    header[24:32] = offset.to_bytes(8, "little")
    header[32:64] = bytes_digest
    header[64:96] = records_digest
    header[96:128] = provenance_digest
    header[128:160] = dataset_id
    header[160:192] = sha256_bytes(bytes(header))
    with index_path.open("wb") as index_file:
        index_file.write(header)
        index_file.write(records)
        index_file.flush()

    return {
        "stories": story_count,
        "bytes": offset,
        "dataset_id": dataset_id.hex(),
        "bytes_sha256": bytes_digest.hex(),
        "records_sha256": records_digest.hex(),
        "provenance_sha256": provenance_digest.hex(),
    }


def non_negative_limit(
    parser: argparse.ArgumentParser, value: int | None, name: str
) -> int | None:
    if value is not None and value < 0:
        parser.error(f"{name} must be non-negative")
    return value


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        type=Path,
        help="JSON/JSONL/TXT source to deterministically divide into train/validation",
    )
    parser.add_argument(
        "--train-input",
        type=Path,
        help="Official or pre-existing training split (use with --valid-input)",
    )
    parser.add_argument(
        "--valid-input",
        type=Path,
        help="Official or pre-existing validation split (use with --train-input)",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--validation-fraction", type=float, default=0.02)
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument(
        "--text-format",
        choices=("auto", "delimited", "paragraph"),
        default="auto",
        help="TXT record format. Use explicit delimited/paragraph for reproducible ingestion.",
    )
    parser.add_argument("--source-repository")
    parser.add_argument("--source-revision")
    parser.add_argument("--input-source-sha256")
    parser.add_argument("--train-source-sha256")
    parser.add_argument("--valid-source-sha256")
    parser.add_argument("--input-limit", type=int)
    parser.add_argument("--train-limit", type=int)
    parser.add_argument("--valid-limit", type=int)
    parser.add_argument("--train-byte-limit", type=int)
    parser.add_argument("--valid-byte-limit", type=int)
    args = parser.parse_args()

    input_source = args.input if isinstance(args.input, Path) else None
    train_source = args.train_input if isinstance(args.train_input, Path) else None
    valid_source = args.valid_input if isinstance(args.valid_input, Path) else None
    output_dir = args.output if isinstance(args.output, Path) else None
    if output_dir is None:
        parser.error("--output must be a valid path")

    input_limit = non_negative_limit(parser, args.input_limit, "--input-limit")
    train_limit = non_negative_limit(parser, args.train_limit, "--train-limit")
    valid_limit = non_negative_limit(parser, args.valid_limit, "--valid-limit")
    train_byte_limit = non_negative_limit(parser, args.train_byte_limit, "--train-byte-limit")
    valid_byte_limit = non_negative_limit(parser, args.valid_byte_limit, "--valid-byte-limit")
    if train_byte_limit == 0 or valid_byte_limit == 0:
        parser.error("byte limits must be positive when supplied")

    derived_split = input_source is not None
    provided_split = train_source is not None or valid_source is not None
    if derived_split == provided_split:
        parser.error("provide either --input, or both --train-input and --valid-input")
    if provided_split and (train_source is None or valid_source is None):
        parser.error("--train-input and --valid-input must be supplied together")
    if not 0.0 <= args.validation_fraction < 1.0:
        parser.error("--validation-fraction must be in [0, 1)")
    if derived_split and any(
        value is not None
        for value in (train_limit, valid_limit, train_byte_limit, valid_byte_limit)
    ):
        parser.error("split-specific record/byte limits require provided train and validation splits")
    if provided_split and input_limit is not None:
        parser.error("--input-limit is only valid with --input")

    preparer_sha256 = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    output_dir.mkdir(parents=True, exist_ok=True)

    common_provenance: dict[str, object] = {
        "preparer": "python/prepare_tinystories.py",
        "preparer_version": PREPARER_VERSION,
        "preparer_sha256": preparer_sha256,
        "dataset_schema_version": DATASET_SCHEMA_VERSION,
        "source_repository": args.source_repository,
        "source_revision": args.source_revision,
        "seed": args.seed,
        "text_format": args.text_format,
    }

    if input_source is not None:
        source_sha256 = args.input_source_sha256 or source_tree_digest(input_source)
        stories = collect_stories(input_source, input_limit, args.text_format)
        random.Random(args.seed).shuffle(stories)
        validation_count = int(round(len(stories) * args.validation_fraction))
        if args.validation_fraction > 0 and validation_count == 0 and len(stories) > 1:
            validation_count = 1
        valid = stories[:validation_count]
        train = stories[validation_count:]
        split_source = "deterministic-derived-split"
        train_provenance = {
            **common_provenance,
            "split": "train",
            "split_source": split_source,
            "validation_fraction": args.validation_fraction,
            "input_limit": input_limit,
            "source_sha256": source_sha256,
        }
        valid_provenance = {**train_provenance, "split": "valid"}
        train_artifact = write_split(output_dir, "train", train, train_provenance)
        valid_artifact = write_split(output_dir, "valid", valid, valid_provenance)
    else:
        assert train_source is not None and valid_source is not None
        train_source_sha256 = args.train_source_sha256 or source_tree_digest(train_source)
        valid_source_sha256 = args.valid_source_sha256 or source_tree_digest(valid_source)
        split_source = "provided-train-validation-splits"
        train_provenance = {
            **common_provenance,
            "split": "train",
            "split_source": split_source,
            "source_sha256": train_source_sha256,
            "record_limit": train_limit,
            "byte_limit": train_byte_limit,
        }
        valid_provenance = {
            **common_provenance,
            "split": "valid",
            "split_source": split_source,
            "source_sha256": valid_source_sha256,
            "record_limit": valid_limit,
            "byte_limit": valid_byte_limit,
        }
        train_artifact = write_split(
            output_dir,
            "train",
            iter_source_stories(
                train_source,
                train_limit,
                train_byte_limit,
                args.text_format,
            ),
            train_provenance,
        )
        valid_artifact = write_split(
            output_dir,
            "valid",
            iter_source_stories(
                valid_source,
                valid_limit,
                valid_byte_limit,
                args.text_format,
            ),
            valid_provenance,
        )

    manifest = {
        "dataset_schema_version": DATASET_SCHEMA_VERSION,
        "preparer_version": PREPARER_VERSION,
        "preparer_sha256": preparer_sha256,
        "source_repository": args.source_repository,
        "source_revision": args.source_revision,
        "seed": args.seed,
        "validation_fraction": args.validation_fraction if derived_split else None,
        "split_source": split_source,
        "text_format": args.text_format,
        "train_limit": train_limit,
        "valid_limit": valid_limit,
        "train_byte_limit": train_byte_limit,
        "valid_byte_limit": valid_byte_limit,
        "input_limit": input_limit,
        "train_stories": train_artifact["stories"],
        "train_bytes": train_artifact["bytes"],
        "valid_stories": valid_artifact["stories"],
        "valid_bytes": valid_artifact["bytes"],
        "train_dataset_id": train_artifact["dataset_id"],
        "valid_dataset_id": valid_artifact["dataset_id"],
        "train_bytes_sha256": train_artifact["bytes_sha256"],
        "valid_bytes_sha256": valid_artifact["bytes_sha256"],
        "train_index_records_sha256": train_artifact["records_sha256"],
        "valid_index_records_sha256": valid_artifact["records_sha256"],
        "train_provenance_sha256": train_artifact["provenance_sha256"],
        "valid_provenance_sha256": valid_artifact["provenance_sha256"],
        "primitive_encoding": "exact UTF-8 bytes",
        "text_record_separator": END_OF_TEXT.decode("ascii"),
        "index_format": "LEODATA1 header + little-endian <QII records",
        "digest_algorithm": "sha256",
    }
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(manifest, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
