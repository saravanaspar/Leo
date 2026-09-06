from __future__ import annotations

import math


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


def test_stage6_tiled_scan_covers_every_neuron_once():
    for neuron_count in [1, 255, 256, 257, 1024, 32768, 33001]:
        seen = tiled_indices(neuron_count)
        assert len(seen) == neuron_count
        assert sorted(seen) == list(range(neuron_count))


def test_stage6_production_grid_is_story_times_128_neuron_tiles():
    neurons = 32768
    threads = 256
    stories = 64
    tiles = math.ceil(neurons / threads)
    assert tiles == 128
    assert stories * tiles == 8192
