from functools import cmp_to_key
import random
import unittest


def better(left, right, rotation, neuron_count):
    lv, ln = left
    rv, rn = right
    if lv > rv:
        return True
    if lv < rv:
        return False
    return (ln + neuron_count - rotation) % neuron_count < (rn + neuron_count - rotation) % neuron_count


def heap_merge(runs, rotation, neuron_count, limit=None):
    heads = [0] * len(runs)
    heap = []

    def candidate(block):
        return runs[block][heads[block]]

    for block in range(len(runs)):
        if runs[block][0][0] <= 0:
            continue
        position = len(heap)
        heap.append(block)
        while position:
            parent = (position - 1) // 2
            if not better(candidate(block), candidate(heap[parent]), rotation, neuron_count):
                break
            heap[position] = heap[parent]
            position = parent
        heap[position] = block

    result = []
    keep = len(runs[0])
    total = sum(sum(value > 0 for value, _ in run) for run in runs)
    needed = total if limit is None else min(total, limit)
    while heap and len(result) < needed:
        best = heap[0]
        result.append(candidate(best))
        next_head = heads[best] + 1
        if next_head < keep and runs[best][next_head][0] > 0:
            heads[best] = next_head
            replacement = best
        else:
            replacement = heap.pop()
            if not heap:
                break

        position = 0
        while True:
            left = position * 2 + 1
            if left >= len(heap):
                break
            right = left + 1
            child = left
            if right < len(heap) and better(candidate(heap[right]), candidate(heap[left]), rotation, neuron_count):
                child = right
            if better(candidate(replacement), candidate(heap[child]), rotation, neuron_count):
                break
            heap[position] = heap[child]
            position = child
        heap[position] = replacement
    return result


class Stage5MergeReferenceTests(unittest.TestCase):
    def test_heap_merge_matches_total_leo_order(self):
        rng = random.Random(20260807)
        neuron_count = 32768
        keep = 8
        for _ in range(500):
            rotation = rng.randrange(neuron_count)
            block_count = rng.randint(1, 128)
            ids = iter(rng.sample(range(neuron_count), block_count * keep))

            def cmp(left, right):
                if better(left, right, rotation, neuron_count):
                    return -1
                if better(right, left, rotation, neuron_count):
                    return 1
                return 0

            runs = []
            for _block in range(block_count):
                positive = rng.randrange(keep + 1)
                row = [
                    (rng.choice((0.1, 0.2, 0.3, rng.random())), next(ids))
                    for _ in range(positive)
                ]
                for _ in range(keep - positive):
                    next(ids)
                row.sort(key=cmp_to_key(cmp))
                row.extend([(-1.0, 0)] * (keep - positive))
                runs.append(row)

            expected = [item for row in runs for item in row if item[0] > 0]
            expected.sort(key=cmp_to_key(cmp))
            self.assertEqual(heap_merge(runs, rotation, neuron_count), expected)
            self.assertEqual(heap_merge(runs, rotation, neuron_count, 17), expected[:17])


if __name__ == "__main__":
    unittest.main()
