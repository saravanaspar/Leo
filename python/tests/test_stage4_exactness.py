import random
import struct
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]


def rotated_key(neuron: int, rotation: int, neuron_count: int) -> int:
    return (neuron + neuron_count - rotation) % neuron_count


def better(left, right, rotation: int, neuron_count: int) -> bool:
    lv, ln = left
    rv, rn = right
    if lv > rv:
        return True
    if lv < rv:
        return False
    return rotated_key(ln, rotation, neuron_count) < rotated_key(rn, rotation, neuron_count)


def stage3_block_select(values, keep, rotation, neuron_count):
    top = [(-1.0, 0) for _ in range(keep + 1)]
    positive = 0
    for value, neuron in values:
        if value <= 0.0:
            continue
        positive += 1
        position = keep
        for scan in range(keep + 1):
            if top[scan][0] < 0.0 or better((value, neuron), top[scan], rotation, neuron_count):
                position = scan
                break
        for move in range(keep, position, -1):
            top[move] = top[move - 1]
        top[position] = (value, neuron)
    winner_count = min(positive, keep)
    winners = top[:winner_count]
    cutoff = top[keep][0] if positive > keep else 0.0
    return winners, cutoff




def f32(value: float) -> float:
    return struct.unpack("<f", struct.pack("<f", value))[0]


def selection_record(value: float, neuron: int, rotation: int, neuron_count: int) -> int:
    value = f32(value)
    if value <= 0.0:
        return 0
    bits = struct.unpack("<I", struct.pack("<f", value))[0]
    tie = 0xFFFFFFFF - rotated_key(neuron, rotation, neuron_count)
    return (bits << 32) | tie


def selection_record_value(record: int) -> float:
    if record == 0:
        return -1.0
    return struct.unpack("<f", struct.pack("<I", record >> 32))[0]


def selection_record_neuron(record: int, rotation: int, neuron_count: int) -> int:
    if record == 0:
        return 0
    rotated = 0xFFFFFFFF - (record & 0xFFFFFFFF)
    return (rotated + rotation) % neuron_count


def packed_block_select(values, keep, rotation, neuron_count):
    records = sorted(
        (selection_record(value, neuron, rotation, neuron_count) for value, neuron in values),
        reverse=True,
    )
    positive = sum(record != 0 for record in records)
    winner_count = min(positive, keep)
    winners = [
        (
            selection_record_value(record),
            selection_record_neuron(record, rotation, neuron_count),
        )
        for record in records[:winner_count]
    ]
    if positive <= keep:
        return winners, 0.0

    last_positive = next((item for item in reversed(values) if item[0] > 0.0), None)
    assert last_positive is not None
    last_is_winner = any(neuron == last_positive[1] for _, neuron in winners)
    cutoff = selection_record_value(records[keep]) if last_is_winner else last_positive[0]
    return winners, cutoff


def bitonic_sort_keys(records, first_width=2):
    records = list(records)
    width = first_width
    while width <= len(records):
        stride = width >> 1
        while stride:
            for index in range(len(records)):
                other = index ^ stride
                if other <= index:
                    continue
                better_first = (index & width) == 0
                left = records[index]
                right = records[other]
                should_swap = left < right if better_first else left > right
                if should_swap:
                    records[index], records[other] = right, left
            stride >>= 1
        width <<= 1
    return records

def stage4_exact_select(values, keep, rotation, neuron_count):
    positive = [(value, neuron) for value, neuron in values if value > 0.0]
    # Total ordering corresponding to leo_better.
    positive.sort(key=lambda item: (-item[0], rotated_key(item[1], rotation, neuron_count)))
    winners = positive[:keep]
    if len(positive) <= keep:
        return winners, 0.0

    # Stage 3's insertion loop has a historical cutoff quirk: a later positive
    # non-winner overwrites slot `keep`. Stage 4 reconstructs it exactly from
    # the final top-K plus the last positive in original ascending-neuron order.
    last_positive = max((item for item in values if item[0] > 0.0), key=lambda item: item[1])
    last_is_winner = any(neuron == last_positive[1] for _, neuron in winners)
    cutoff = positive[keep][0] if last_is_winner else last_positive[0]
    return winners, cutoff


class Stage4ExactnessTests(unittest.TestCase):
    def test_parallel_selection_reconstructs_stage3_winners_and_cutoff(self):
        rng = random.Random(1337)
        neuron_count = 32768
        keep = 8
        for _ in range(1000):
            block_start = rng.randrange(0, neuron_count - 256)
            rotation = rng.randrange(neuron_count)
            values = []
            for local in range(256):
                value = rng.choice((0.0, 0.0, 0.1, 0.2, round(rng.random(), 4)))
                values.append((value, block_start + local))
            self.assertEqual(
                stage3_block_select(values, keep, rotation, neuron_count),
                stage4_exact_select(values, keep, rotation, neuron_count),
            )

    def test_packed_selection_matches_serial_winners_and_cutoff(self):
        rng = random.Random(424242)
        neuron_count = 32768
        keep = 8
        for _ in range(1000):
            block_start = rng.randrange(0, neuron_count - 256)
            rotation = rng.randrange(neuron_count)
            values = []
            for local in range(256):
                value = f32(rng.choice((0.0, 0.0, 0.1, 0.2, rng.random())))
                values.append((value, block_start + local))
            self.assertEqual(
                stage3_block_select(values, keep, rotation, neuron_count),
                packed_block_select(values, keep, rotation, neuron_count),
            )

    def test_presorted_block_runs_can_skip_completed_bitonic_widths(self):
        rng = random.Random(9001)
        neuron_count = 32768
        rotation = 1337
        run = 8
        runs = []
        for block in range(128):
            records = []
            for local in range(run):
                neuron = block * 256 + local
                value = f32(rng.choice((0.0, 0.1, 0.2, rng.random())))
                records.append(selection_record(value, neuron, rotation, neuron_count))
            records.sort(reverse=True)
            if block & 1:
                records.reverse()
            runs.extend(records)

        merged = bitonic_sort_keys(runs, first_width=run * 2)
        self.assertEqual(merged, sorted(runs, reverse=True))

    def test_cuda_source_reuses_exact_packed_selection_helpers(self):
        cuda = (ROOT / "crates/leo-core/src/cuda_kernels.cu").read_text()
        self.assertIn("leo_selection_record(", cuda)
        self.assertIn("leo_bitonic_sort_selection_keys(", cuda)
        self.assertIn("presorted_runs", cuda)
        self.assertIn("last_positive_neuron", cuda)
        self.assertIn("Exact legacy fallback for unsupported shapes", cuda)
        self.assertIn("leo_p_select_model_block(p, tick, model_block, shared_selection_keys)", cuda)
        self.assertNotIn("float* shared_values,\n    unsigned int* shared_neurons", cuda)

    def test_historical_gpu_docs_point_to_current_v1_contract(self):
        main = (ROOT / "crates/leo-cli/src/main.rs").read_text()
        semantics = (ROOT / "docs/SEMANTICS.md").read_text()
        tdd = (ROOT / "docs/TDD.md").read_text()

        for stage in (4, 6):
            matches = list((ROOT / "docs").glob(f"STAGE{stage}-GPU-*.md"))
            self.assertEqual(len(matches), 1)
            text = matches[0].read_text()
            self.assertIn("Historical design record", text)
            self.assertIn("not** the Leo v1.0.1 runtime contract", text)
            self.assertIn("PSCLS100", text)

        self.assertIn("cuda_shared_model_tiled_post_wavefront_learning", main)
        self.assertIn("FP32", semantics)
        self.assertIn("shared-wavefront", tdd)


if __name__ == "__main__":
    unittest.main()
