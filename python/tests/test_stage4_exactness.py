import random
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

    def test_historical_gpu_docs_point_to_current_v1_contract(self):
        main = (ROOT / "crates/leo-cli/src/main.rs").read_text()
        semantics = (ROOT / "docs/SEMANTICS.md").read_text()
        tdd = (ROOT / "docs/TDD.md").read_text()

        for stage in (4, 6):
            matches = list((ROOT / "docs").glob(f"STAGE{stage}-GPU-*.md"))
            self.assertEqual(len(matches), 1)
            text = matches[0].read_text()
            self.assertIn("Historical design record", text)
            self.assertIn("not** the Leo v1.0.0 runtime contract", text)
            self.assertIn("PSCLS100", text)

        self.assertIn("cuda_shared_model_tiled_post_wavefront_learning", main)
        self.assertIn("FP32", semantics)
        self.assertIn("shared-wavefront", tdd)


if __name__ == "__main__":
    unittest.main()
