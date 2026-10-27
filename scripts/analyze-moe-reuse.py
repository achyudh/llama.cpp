#!/usr/bin/env python3
"""Summarize CPU expert reuse alongside target/draft graph markers."""
import argparse
import bisect
import collections
import csv
import json
import re
from pathlib import Path

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('session', type=Path)
args = parser.parse_args()
markers = []
pattern = re.compile(r'moe_trace_graph: us=(\d+) role=(\w+) tokens=(\d+) outputs=(\d+) pos0=(-?\d+)')
phase = 'unknown'
for line in (args.session / 'server.log').read_text().splitlines():
    if 'new prompt, n_ctx_slot' in line:
        phase = 'prefill'
    elif 'init_sampler:' in line and 'init sampler' in line:
        phase = 'generation'
    elif 'stop processing:' in line:
        phase = 'idle'
    match = pattern.search(line)
    if match:
        us, role, tokens, outputs, pos = match.groups()
        markers.append((int(us), role, int(tokens), int(outputs), int(pos), phase))
markers.sort()
times = [m[0] for m in markers]
primary = re.compile(r'blk\.(\d+)\.ffn_(?:up|gate_up)_exps\.weight$')
routing = collections.defaultdict(lambda: dict(records=0, unique=0, selected=0, histogram=[0]*9))
dispatch = collections.defaultdict(lambda: collections.Counter())
classes = collections.Counter()
invalid = 0
with (args.session / 'expert-reuse.csv').open(newline='') as file:
    for row in csv.DictReader(file):
        try:
            us = int(row['us'])
            nq = int(row['nq'])
            hist = [int(row[f'h{i}']) for i in range(1, 9)] + [int(row['h_gt8'])]
            unique, selected = int(row['unique']), int(row['selected'])
            paths = {k: int(row[k]) for k in ('cols', 'decode_once', 'tiled', 'stock')}
            if sum(hist) != unique or sum(paths.values()) != unique or selected != nq*int(row['n_used']):
                raise ValueError('inconsistent counters')
        except (ValueError, TypeError, KeyError):
            invalid += 1
            continue
        index = bisect.bisect_right(times, us) - 1
        if index < 0 or markers[index][2] != nq:
            category = 'unmatched'
        elif markers[index][1] == 'draft':
            category = 'draft'
        elif nq == 1:
            category = 'target_single_token'
        elif nq <= 8 and markers[index][3] == nq and markers[index][5] == 'generation':
            category = 'target_verification'
        else:
            category = 'target_prefill_or_other'
        classes[category] += 1
        if category != 'target_verification':
            continue
        key = f"nq={nq},type={row['type']},k={row['k']},m={row['m']}"
        dispatch[key].update(paths)
        match = primary.search(row['weight'])
        if not match:
            continue
        for key in (f'nq={nq}', f'nq={nq},layer={match[1]}'):
            result = routing[key]
            result['records'] += 1
            result['unique'] += unique
            result['selected'] += selected
            result['histogram'] = [a+b for a, b in zip(result['histogram'], hist)]
for result in routing.values():
    result['selection_to_unique_ratio'] = result['selected']/result['unique'] if result['unique'] else None
    result['five_row_fraction'] = 5*result['histogram'][4]/result['selected'] if result['selected'] else None
print(json.dumps(dict(
    graph_markers=len(markers),
    record_classes=dict(classes),
    invalid_or_partial_rows=invalid,
    routing=dict(sorted(routing.items())),
    expert_dispatches={key: dict(value) for key, value in sorted(dispatch.items())},
    notes=[
        'Routing counts use only up or merged gate/up weights to avoid counting each selection three times.',
        'Dispatch counters count selected experts per CPU matrix operation, not threads or GPU launches.',
        'Verification requires all-output target batches of 2-8 tokens after the server init_sampler event.',
        'Single-token target graphs are separate because prompt tails and decode are ambiguous.',
        'CPU tracing does not observe GPU expert execution. Incomplete buffered data is possible during a live run.',
    ],
), indent=2))
