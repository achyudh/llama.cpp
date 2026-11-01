#!/usr/bin/env python3
"""Show the CPU model rejects wrong layouts, loop termination and unsafe load waits."""
import argparse
import json
from pathlib import Path
import numpy as np
from check_layout import simulate

p = argparse.ArgumentParser(description=__doc__)
p.add_argument('--build', type=Path, required=True)
args = p.parse_args()
source = (args.build / 'pipeline.s').read_text()
changes = {
    'wrong_b_fragment': ('s[8:11], s27 offen offset:32', 's[8:11], s27 offen offset:16'),
    'missing_load_wait': ('s_wait_loadcnt 8', 's_wait_loadcnt 12'),
    'wrong_termination': ('s_cmp_eq_u32 s28, 0', 's_cmp_eq_u32 s28, 1'),
    'wrong_k_wrap': ('s_cselect_b32 s30, s20, 0', 's_cselect_b32 s30, 0, 0'),
}
rng = np.random.default_rng(1301)
x = rng.integers(-8, 9, (129, 256)).astype(np.float16)
w = rng.integers(-8, 9, (16, 256)).astype(np.float16)
expected = x.astype(np.float64) @ w.astype(np.float64).T
results = {}
for name, (before, after) in changes.items():
    assert before in source and before != after
    mutated = source.replace(before, after)
    out = np.full((129, 16), np.nan, dtype=np.float32)
    written = np.zeros(129*16, dtype=np.int32)
    try:
        for gy in range(2):
            simulate(mutated, 129, 16, 256, 0, gy, 0, x, w, out, written)
        assert np.array_equal(out, expected), 'numeric mismatch'
        assert np.all(written == 1), 'ownership mismatch'
    except AssertionError as error:
        results[name] = {'rejected': True, 'reason': str(error)}
    else:
        raise AssertionError(f'mutation escaped model: {name}')
(args.build / 'cpu-mutations.json').write_text(json.dumps(results, indent=2) + '\n')
print(f'CPU mutation checks: {len(results)} rejected as expected')
