from __future__ import annotations

import json
import struct
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "python" / "prepare_tinystories.py"
INDEX = struct.Struct("<QII")
INDEX_HEADER_SIZE = 192


def read_prepared(output: Path, split: str) -> list[str]:
    payload = (output / f"tinystories.{split}.bytes").read_bytes()
    raw_index = (output / f"tinystories.{split}.idx").read_bytes()
    decoded = []
    for position in range(INDEX_HEADER_SIZE, len(raw_index), INDEX.size):
        story_offset, length, _ = INDEX.unpack_from(raw_index, position)
        decoded.append(payload[story_offset : story_offset + length].decode("utf-8"))
    return decoded


class PrepareTinyStoriesTests(unittest.TestCase):
    def test_jsonl_is_preserved_as_exact_utf8(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            base = Path(temp)
            source = base / "stories.jsonl"
            output = base / "prepared"
            stories = ["Leo runs.\n", "مرحبا يا Leo."]
            source.write_text(
                "".join(json.dumps({"text": story}, ensure_ascii=False) + "\n" for story in stories),
                encoding="utf-8",
            )
            subprocess.run(
                [
                    sys.executable,
                    str(SCRIPT),
                    "--input",
                    str(source),
                    "--output",
                    str(output),
                    "--validation-fraction",
                    "0",
                ],
                check=True,
                capture_output=True,
                text=True,
            )
            self.assertCountEqual(read_prepared(output, "train"), stories)


    def test_byte_limit_keeps_complete_stories_within_budget(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            base = Path(temp)
            train = base / "train.txt"
            valid = base / "valid.txt"
            output = base / "prepared"
            train.write_text(
                "1234<|endoftext|>5678<|endoftext|>90<|endoftext|>",
                encoding="utf-8",
            )
            valid.write_text("ok<|endoftext|>", encoding="utf-8")
            subprocess.run(
                [
                    sys.executable,
                    str(SCRIPT),
                    "--train-input",
                    str(train),
                    "--valid-input",
                    str(valid),
                    "--train-byte-limit",
                    "8",
                    "--output",
                    str(output),
                ],
                check=True,
                capture_output=True,
                text=True,
            )
            self.assertEqual(read_prepared(output, "train"), ["1234", "5678"])
            manifest = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
            self.assertEqual(manifest["train_bytes"], 8)
            self.assertEqual(manifest["train_byte_limit"], 8)

    def test_official_delimiter_is_streamed_and_blank_paragraphs_are_preserved(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            base = Path(temp)
            train = base / "TinyStories-train.txt"
            valid = base / "TinyStories-valid.txt"
            output = base / "prepared"
            train.write_text(
                "First paragraph.\n\nSecond paragraph.\n<|endoftext|>\n"
                "Second story.\n<|endoftext|>\n"
                "Third story must be excluded.\n<|endoftext|>\n",
                encoding="utf-8",
            )
            valid.write_text(
                "Validation one.\n<|endoftext|>\nValidation two.\n<|endoftext|>\n",
                encoding="utf-8",
            )
            subprocess.run(
                [
                    sys.executable,
                    str(SCRIPT),
                    "--train-input",
                    str(train),
                    "--valid-input",
                    str(valid),
                    "--train-limit",
                    "2",
                    "--valid-limit",
                    "1",
                    "--output",
                    str(output),
                ],
                check=True,
                capture_output=True,
                text=True,
            )
            self.assertEqual(
                read_prepared(output, "train"),
                ["First paragraph.\n\nSecond paragraph.", "Second story."],
            )
            self.assertEqual(read_prepared(output, "valid"), ["Validation one."])


if __name__ == "__main__":
    unittest.main()
