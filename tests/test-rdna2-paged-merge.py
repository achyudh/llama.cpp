"""Deterministic host-side oracle for merging disjoint paged-attention outputs.

This checks the LSE/normalization formula used by the RDNA2 tile fallback.
It does NOT validate the HIP kernel, stream ordering or GPU correctness.
"""

import math
import random


def attend(scores, values):
    finite = [(s, v) for s, v in zip(scores, values) if s != -math.inf]
    if not finite:
        return 0.0, -math.inf
    m = max(s for s, _ in finite)
    weights = [math.exp(s - m) for s, _ in finite]
    total = sum(weights)
    return sum(w * v for w, (_, v) in zip(weights, finite)) / total, m + math.log(total)


def merge(acc, lse_acc, part, lse_part):
    if lse_acc == -math.inf:
        return (part, lse_part) if lse_part != -math.inf else (0.0, -math.inf)
    if lse_part == -math.inf:
        return acc, lse_acc
    m = max(lse_acc, lse_part)
    wa = math.exp(lse_acc - m)
    wb = math.exp(lse_part - m)
    return (wa * acc + wb * part) / (wa + wb), m + math.log(wa + wb)


def main():
    rng = random.Random(0x6950)
    for case in range(1024):
        n = rng.randrange(1, 1900)
        scores = [rng.uniform(-120.0, 120.0) for _ in range(n)]
        values = [rng.uniform(-8, 8) for _ in range(n)]
        if case % 11 == 0:
            scores[:] = [-math.inf] * n
        else:
            for i in range(n):
                if rng.random() < 0.4:
                    scores[i] = -math.inf
        expected, expected_lse = attend(scores, values)
        acc, lse = 0.0, -math.inf
        start = 0
        while start < n:
            end = min(n, start + rng.randint(1, 100))
            part, part_lse = attend(scores[start:end], values[start:end])
            acc, lse = merge(acc, lse, part, part_lse)
            start = end
        if expected_lse == -math.inf:
            assert lse == -math.inf and acc == 0.0, case
        else:
            assert math.isclose(acc, expected, rel_tol=1e-11, abs_tol=1e-11), case
            assert math.isclose(lse, expected_lse, rel_tol=1e-12, abs_tol=1e-12), case
    print("PASS: 1024 deterministic attention/LSE page-merge cases, masked and extreme logits")


if __name__ == "__main__":
    main()
