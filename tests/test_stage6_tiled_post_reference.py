from __future__ import annotations

import math
import unittest


def tiled_indices(neuron_count: int, threads: int = 256):
    tiles = max(1, math.ceil(neuron_count / threads))
    stride = tiles * threads
    seen = []
    for tile in range(tiles):
        for lane in range(threads):
            thread = tile * threads + lane
            neuron = thread
            while neuron < neuron_count:
                seen.append(neuron)
                neuron += stride
    return seen


class Stage6TiledPostReferenceTests(unittest.TestCase):
    def test_stage6_tiled_scan_covers_every_neuron_once(self):
        for neuron_count in [1, 255, 256, 257, 1024, 32768, 33001]:
            seen = tiled_indices(neuron_count)
            self.assertEqual(len(seen), neuron_count)
            self.assertEqual(sorted(seen), list(range(neuron_count)))

    def test_stage6_production_grid_is_story_times_128_neuron_tiles(self):
        neurons = 32768
        threads = 256
        stories = 64
        tiles = math.ceil(neurons / threads)
        self.assertEqual(tiles, 128)
        self.assertEqual(stories * tiles, 8192)


if __name__ == "__main__":
    unittest.main()
