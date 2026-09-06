#!/usr/bin/env python3
"""Compare Leo with byte baselines and enforce prediction, generation, and compute gates."""

from __future__ import annotations

import argparse
import collections
import json
import math
import statistics
import struct
from pathlib import Path
from typing import Any

INDEX = struct.Struct("<QII")
END_DOCUMENT = 256
VOCAB = 257
MIN_FREE_RUNNING_ACCURACY = 0.05
MAX_REPETITION_RATE = 0.35
MIN_NEURAL_FREQUENCY_GAIN_BITS = 0.40
MIN_CONTEXT_USE_FRACTION = 0.35
MAX_CONTEXT_USE_FRACTION = 0.65
MAX_COMPUTE_DRIFT_RATIO = 1.25
MIN_THROUGHPUT_RATIO = 0.80
FrozenNgram = tuple[collections.Counter[int], int, int]
JsonObject = dict[str, Any]


def load_stories(
    bytes_path: Path, index_path: Path, limit: int | None = None
) -> list[bytes]:
    payload = bytes_path.read_bytes()
    raw_index = index_path.read_bytes()
    if len(raw_index) % INDEX.size:
        raise ValueError("Index size is not a multiple of 16")
    stories = []
    entry_count = len(raw_index) // INDEX.size
    if limit is not None:
        if limit < 1:
            raise ValueError("story limit must be at least 1")
        entry_count = min(entry_count, limit)
    for entry in range(entry_count):
        position = entry * INDEX.size
        offset, length, _ = INDEX.unpack_from(raw_index, position)
        story = payload[offset : offset + length]
        story.decode("utf-8", errors="strict")
        stories.append(story)
    return stories


def train_frequency(stories: list[bytes]) -> list[int]:
    counts = [1] * VOCAB
    for story in stories:
        for target in [*story, END_DOCUMENT]:
            counts[target] += 1
    return counts


def train_ngram(stories: list[bytes], order: int) -> list[dict[tuple[int, ...], FrozenNgram]]:
    mutable = [collections.defaultdict(collections.Counter) for _ in range(order)]
    for story in stories:
        sequence = [*story, END_DOCUMENT]
        for position, target in enumerate(sequence):
            for context_length in range(order):
                context = tuple(sequence[max(0, position - context_length) : position])
                mutable[context_length][context][target] += 1

    frozen: list[dict[tuple[int, ...], FrozenNgram]] = []
    for table in mutable:
        frozen_table: dict[tuple[int, ...], FrozenNgram] = {}
        for context, counts in table.items():
            maximum = max(counts.values())
            predicted = min(symbol for symbol, count in counts.items() if count == maximum)
            frozen_table[context] = (counts, sum(counts.values()), predicted)
        frozen.append(frozen_table)
    return frozen


def metrics(loss_sum: float, correct: int, targets: int) -> dict[str, float | int]:
    mean_loss = loss_sum / max(targets, 1)
    return {
        "negative_log_likelihood": mean_loss,
        "bits_per_byte": mean_loss / math.log(2),
        "next_byte_accuracy": correct / max(targets, 1),
        "targets": targets,
    }


def evaluate_frequency(stories: list[bytes], counts: list[int]) -> dict[str, float | int]:
    total = sum(counts)
    maximum = max(counts)
    predicted = counts.index(maximum)
    loss_sum = 0.0
    correct = 0
    targets = 0
    for story in stories:
        for target in [*story, END_DOCUMENT]:
            loss_sum -= math.log(max(counts[target] / total, 1e-300))
            correct += int(predicted == target)
            targets += 1
    return metrics(loss_sum, correct, targets)


def evaluate_ngram(
    stories: list[bytes],
    tables: list[dict[tuple[int, ...], FrozenNgram]],
    order: int,
) -> dict[str, float | int]:
    loss_sum = 0.0
    correct = 0
    targets = 0
    for story in stories:
        history: list[int] = []
        for target in [*story, END_DOCUMENT]:
            selected: FrozenNgram | None = None
            for context_length in range(order - 1, -1, -1):
                context = tuple(history[-context_length:]) if context_length else ()
                selected = tables[context_length].get(context)
                if selected is not None:
                    break
            if selected is None:
                probability = 1.0 / VOCAB
                predicted = 0
            else:
                counts, total, predicted = selected
                probability = (counts[target] + 1) / (total + VOCAB)
            loss_sum -= math.log(max(probability, 1e-300))
            correct += int(predicted == target)
            targets += 1
            if target != END_DOCUMENT:
                history.append(target)
    return metrics(loss_sum, correct, targets)


def load_json_events(path: Path) -> list[JsonObject]:
    events: list[JsonObject] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        stripped = line.strip()
        if not stripped.startswith("{"):
            continue
        try:
            value = json.loads(stripped)
        except json.JSONDecodeError as error:
            raise ValueError(f"invalid JSON in {path}:{line_number}: {error}") from error
        if isinstance(value, dict):
            events.append(value)
    return events


def event_by_name(events: list[JsonObject], name: str) -> JsonObject:
    for event in reversed(events):
        if event.get("event") == name:
            return event
    raise ValueError(f"missing {name!r} event")


def median_training_value(events: list[JsonObject], field: str) -> float | None:
    values = [
        float(event[field])
        for event in events
        if event.get("event") == "training_progress" and field in event
    ]
    return statistics.median(values) if values else None


def median_tail_ratio(events: list[JsonObject], field: str) -> float | None:
    values = [
        float(event[field])
        for event in events
        if event.get("event") == "training_progress" and field in event
    ]
    if len(values) < 8:
        return None
    window = max(4, len(values) // 5)
    previous = statistics.median(values[-2 * window : -window])
    current = statistics.median(values[-window:])
    if previous <= 0.0:
        return None
    return current / previous


def build_acceptance(
    leo_events: list[JsonObject],
    frequency_metrics: dict[str, float | int],
    ngram_metrics: dict[str, float | int],
    training_events: list[JsonObject] | None = None,
) -> JsonObject:
    held_out = event_by_name(leo_events, "held_out_evaluation")
    free_running = event_by_name(leo_events, "free_running_evaluation")
    prompt = event_by_name(leo_events, "prompt_probe")

    prediction_checks = {
        "neural_state_beats_frequency_by_margin": (
            float(frequency_metrics["bits_per_byte"])
            - float(held_out["neural_only_bits_per_byte"])
        )
        >= MIN_NEURAL_FREQUENCY_GAIN_BITS,
        "beats_ngram_bits_per_byte": float(held_out["bits_per_byte"])
        < float(ngram_metrics["bits_per_byte"]),
        "beats_ngram_accuracy": float(held_out["accuracy"])
        > float(ngram_metrics["next_byte_accuracy"]),
        "context_reduces_loss": float(held_out["context_gain_bits_per_byte"]) > 0.0,
    }
    generation_checks = {
        "free_running_byte_accuracy": float(free_running["byte_accuracy"])
        >= MIN_FREE_RUNNING_ACCURACY,
        "repetition_is_controlled": float(free_running["mean_repetition_rate"])
        <= MAX_REPETITION_RATE,
        "natural_end_observed": float(free_running["natural_end_fraction"]) > 0.0,
        "prompt_is_valid_utf8": bool(prompt.get("valid_utf8")),
        "prompt_repetition_is_controlled": float(prompt.get("repetition_rate", 1.0))
        <= MAX_REPETITION_RATE,
    }

    compute: JsonObject = {
        "fixed_recurrent_topology": True,
        "recurrent_event_drift_ratio": None,
        "context_probe_drift_ratio": None,
        "output_madd_drift_ratio": None,
        "training_throughput_ratio": None,
        "context_use_fraction": None,
        "context_dropout_observed": False,
        "stable": False,
    }
    if training_events is not None:
        recurrent_ratio = median_tail_ratio(training_events, "recurrent_events_per_step")
        context_probe_ratio = median_tail_ratio(
            training_events, "context_probes_per_step"
        )
        output_ratio = median_tail_ratio(training_events, "output_madds_per_step")
        throughput_ratio = median_tail_ratio(training_events, "steps_per_second")
        context_use_fraction = median_training_value(
            training_events, "context_use_fraction"
        )
        context_dropout_observed = (
            context_use_fraction is not None
            and MIN_CONTEXT_USE_FRACTION
            <= context_use_fraction
            <= MAX_CONTEXT_USE_FRACTION
        )
        compute.update(
            recurrent_event_drift_ratio=recurrent_ratio,
            context_probe_drift_ratio=context_probe_ratio,
            output_madd_drift_ratio=output_ratio,
            training_throughput_ratio=throughput_ratio,
            context_use_fraction=context_use_fraction,
            context_dropout_observed=context_dropout_observed,
            stable=context_dropout_observed
            and (recurrent_ratio is None or recurrent_ratio <= MAX_COMPUTE_DRIFT_RATIO)
            and (
                context_probe_ratio is None
                or context_probe_ratio <= MAX_COMPUTE_DRIFT_RATIO
            )
            and (output_ratio is None or output_ratio <= MAX_COMPUTE_DRIFT_RATIO)
            and (throughput_ratio is None or throughput_ratio >= MIN_THROUGHPUT_RATIO),
        )

    prediction_passed = all(prediction_checks.values())
    generation_passed = all(generation_checks.values())
    return {
        "prediction": {**prediction_checks, "passed": prediction_passed},
        "generation": {**generation_checks, "passed": generation_passed},
        "compute": compute,
        "passed": prediction_passed and generation_passed and bool(compute["stable"]),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-bytes", type=Path, required=True)
    parser.add_argument("--train-index", type=Path, required=True)
    parser.add_argument("--valid-bytes", type=Path, required=True)
    parser.add_argument("--valid-index", type=Path, required=True)
    parser.add_argument("--order", type=int, default=5)
    parser.add_argument("--train-stories", type=int)
    parser.add_argument("--valid-stories", type=int)
    parser.add_argument("--leo-eval", type=Path)
    parser.add_argument("--train-log", type=Path)
    parser.add_argument("--require-pass", action="store_true")
    args = parser.parse_args()
    if args.order < 1:
        parser.error("--order must be at least 1")

    train = load_stories(args.train_bytes, args.train_index, args.train_stories)
    valid = load_stories(args.valid_bytes, args.valid_index, args.valid_stories)
    frequency = train_frequency(train)
    ngram = train_ngram(train, args.order)
    frequency_metrics = evaluate_frequency(valid, frequency)
    ngram_metrics = evaluate_ngram(valid, ngram, args.order)
    report: JsonObject = {
        "byte_frequency": frequency_metrics,
        f"byte_{args.order}_gram": ngram_metrics,
    }
    if args.leo_eval is not None:
        leo_events = load_json_events(args.leo_eval)
        training_events = (
            load_json_events(args.train_log)
            if args.train_log is not None and args.train_log.exists()
            else None
        )
        report["acceptance"] = build_acceptance(
            leo_events, frequency_metrics, ngram_metrics, training_events
        )
    print(json.dumps(report, indent=2, sort_keys=True))
    if args.require_pass and not bool(report.get("acceptance", {}).get("passed")):
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
