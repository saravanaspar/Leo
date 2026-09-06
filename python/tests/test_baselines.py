from __future__ import annotations

import importlib.util
import json
import tempfile
import unittest
from pathlib import Path

MODULE_PATH = Path(__file__).resolve().parents[1] / "baselines.py"
SPEC = importlib.util.spec_from_file_location("leo_baselines", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
baselines = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(baselines)


class BaselineTests(unittest.TestCase):
    def test_optimized_metrics_match_expected_target_count(self) -> None:
        train = [b"abab", b"abac"]
        valid = [b"abab"]
        frequency = baselines.train_frequency(train)
        ngram = baselines.train_ngram(train, 3)
        frequency_metrics = baselines.evaluate_frequency(valid, frequency)
        ngram_metrics = baselines.evaluate_ngram(valid, ngram, 3)
        self.assertEqual(frequency_metrics["targets"], 5)
        self.assertEqual(ngram_metrics["targets"], 5)
        self.assertGreaterEqual(frequency_metrics["bits_per_byte"], 0.0)
        self.assertGreaterEqual(ngram_metrics["bits_per_byte"], 0.0)

    def test_story_loading_honors_limit(self) -> None:
        import struct

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            payload = b"onetwothree"
            (root / "data.bytes").write_bytes(payload)
            index = b"".join(
                struct.pack("<QII", offset, length, 0)
                for offset, length in ((0, 3), (3, 3), (6, 5))
            )
            (root / "data.idx").write_bytes(index)
            stories = baselines.load_stories(
                root / "data.bytes", root / "data.idx", 2
            )
            self.assertEqual(stories, [b"one", b"two"])

    def test_acceptance_requires_sequence_generation_and_compute_gates(self) -> None:
        leo_events = [
            {
                "event": "held_out_evaluation",
                "bits_per_byte": 1.5,
                "neural_only_bits_per_byte": 2.0,
                "accuracy": 0.8,
                "context_gain_bits_per_byte": 0.4,
            },
            {
                "event": "free_running_evaluation",
                "byte_accuracy": 0.2,
                "mean_repetition_rate": 0.1,
                "natural_end_fraction": 0.5,
            },
            {
                "event": "prompt_probe",
                "valid_utf8": True,
                "repetition_rate": 0.1,
            },
        ]
        training_events = [
            {
                "event": "training_progress",
                "recurrent_events_per_step": 100.0 + index,
                "context_probes_per_step": 20.0,
                "output_madds_per_step": 200.0 + index,
                "steps_per_second": 1000.0 - index,
                "context_use_fraction": 0.5,
            }
            for index in range(20)
        ]
        acceptance = baselines.build_acceptance(
            leo_events,
            {"bits_per_byte": 2.5, "next_byte_accuracy": 0.4},
            {"bits_per_byte": 2.0, "next_byte_accuracy": 0.7},
            training_events,
        )
        self.assertTrue(acceptance["prediction"]["passed"])
        self.assertTrue(acceptance["generation"]["passed"])
        self.assertTrue(acceptance["compute"]["context_dropout_observed"])
        self.assertTrue(acceptance["compute"]["stable"])
        self.assertTrue(acceptance["passed"])

    def test_acceptance_rejects_throughput_collapse(self) -> None:
        leo_events = [
            {
                "event": "held_out_evaluation",
                "bits_per_byte": 1.5,
                "neural_only_bits_per_byte": 2.0,
                "accuracy": 0.8,
                "context_gain_bits_per_byte": 0.4,
            },
            {
                "event": "free_running_evaluation",
                "byte_accuracy": 0.2,
                "mean_repetition_rate": 0.1,
                "natural_end_fraction": 0.5,
            },
            {
                "event": "prompt_probe",
                "valid_utf8": True,
                "repetition_rate": 0.1,
            },
        ]
        training_events = [
            {
                "event": "training_progress",
                "recurrent_events_per_step": 100.0,
                "context_probes_per_step": 20.0,
                "output_madds_per_step": 200.0,
                "steps_per_second": 1000.0 if index < 16 else 500.0,
                "context_use_fraction": 0.5,
            }
            for index in range(20)
        ]
        acceptance = baselines.build_acceptance(
            leo_events,
            {"bits_per_byte": 2.5, "next_byte_accuracy": 0.4},
            {"bits_per_byte": 2.0, "next_byte_accuracy": 0.7},
            training_events,
        )
        self.assertFalse(acceptance["compute"]["stable"])
        self.assertFalse(acceptance["passed"])

    def test_acceptance_rejects_missing_context_dropout(self) -> None:
        leo_events = [
            {
                "event": "held_out_evaluation",
                "bits_per_byte": 1.5,
                "neural_only_bits_per_byte": 2.0,
                "accuracy": 0.8,
                "context_gain_bits_per_byte": 0.4,
            },
            {
                "event": "free_running_evaluation",
                "byte_accuracy": 0.2,
                "mean_repetition_rate": 0.1,
                "natural_end_fraction": 0.5,
            },
            {
                "event": "prompt_probe",
                "valid_utf8": True,
                "repetition_rate": 0.1,
            },
        ]
        training_events = [
            {
                "event": "training_progress",
                "recurrent_events_per_step": 100.0,
                "context_probes_per_step": 20.0,
                "output_madds_per_step": 200.0,
                "steps_per_second": 1000.0,
                "context_use_fraction": 1.0,
            }
            for _ in range(20)
        ]
        acceptance = baselines.build_acceptance(
            leo_events,
            {"bits_per_byte": 2.5, "next_byte_accuracy": 0.4},
            {"bits_per_byte": 2.0, "next_byte_accuracy": 0.7},
            training_events,
        )
        self.assertFalse(acceptance["compute"]["context_dropout_observed"])
        self.assertFalse(acceptance["compute"]["stable"])

    def test_json_event_loading_ignores_non_json_output(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "events.jsonl"
            path.write_text(
                "compile message\n" + json.dumps({"event": "evaluation"}) + "\n",
                encoding="utf-8",
            )
            self.assertEqual(
                baselines.load_json_events(path), [{"event": "evaluation"}]
            )


if __name__ == "__main__":
    unittest.main()
