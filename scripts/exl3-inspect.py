#!/usr/bin/env python3
"""Inventory EXL3 checkpoint layouts without reading tensor payloads."""

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import struct

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('model', type=Path)
parser.add_argument('--output', type=Path, required=True)
args = parser.parse_args()
index_path = args.model / 'model.safetensors.index.json'
index = json.loads(index_path.read_text())['weight_map']
headers, identities = {}, {}
for shard in sorted(set(index.values())):
    assert Path(shard).name == shard and shard.endswith('.safetensors'), shard
    path = args.model / shard
    with path.open('rb') as f:
        size = path.stat().st_size
        raw = f.read(8)
        assert len(raw) == 8, shard
        length, = struct.unpack('<Q', raw)
        assert 0 < length <= min(128 * 1024 * 1024, size - 8), shard
        raw = f.read(length)
        assert len(raw) == length
    headers[shard] = json.loads(raw)
    identities[shard] = dict(bytes=size, header_bytes=8 + length, header_sha256=hashlib.sha256(raw).hexdigest())

widths = {'F64': 8, 'F32': 4, 'F16': 2, 'BF16': 2, 'I64': 8, 'I32': 4, 'I16': 2, 'I8': 1, 'U8': 1, 'BOOL': 1}
counts, payloads, expert_layers, examples = Counter(), Counter(), {}, {}
for name, shard in index.items():
    t = headers[shard][name]
    shape = t['shape']
    assert all(type(v) is int and v >= 0 for v in shape), name
    elements = 1
    for v in shape:
        elements *= v
    begin, end = t['data_offsets']
    assert type(begin) is int and type(end) is int, name
    header_bytes = identities[shard]['header_bytes']
    assert 0 <= begin <= end <= identities[shard]['bytes'] - header_bytes, name
    assert t['dtype'] in widths and end - begin == elements * widths[t['dtype']], name
    category = 'mtp' if '.mtp.' in name or name.startswith('mtp.') else ('vision' if '.visual.' in name or name.startswith('visual.') else ('experts' if '.experts.' in name else 'main-other'))
    payloads[category] += end - begin
    if not name.endswith('.trellis'):
        continue
    assert t['dtype'] == 'I16' and len(shape) == 3 and all(shape), name
    words = shape[-1]
    assert words % 8 == 0, name
    rate = words // 8
    assert rate in (2, 3, 4, 5, 6, 7, 8, 10, 12, 14, 16), (name, shape)
    prefix = name.removesuffix('.trellis')
    assert prefix + '.mul1' in index and prefix + '.mcg' not in index, name
    for suffix, length in [('.suh', shape[0] * 16), ('.svh', shape[1] * 16)]:
        aux = headers[index[prefix + suffix]][prefix + suffix]
        assert aux['dtype'] == 'F16' and aux['shape'] == [length], (name, suffix)
    group = f'{category}:K={rate / 2:g}'
    counts[group] += 1
    examples.setdefault(group, dict(name=name, packed_shape=shape, logical_kn=[16 * shape[0], 16 * shape[1]]))
    if category == 'experts':
        layer = name.split('.layers.')[1].split('.')[0]
        expert_layers.setdefault(layer, set()).add(rate)

assert all(len(rates) == 1 for rates in expert_layers.values())
layers = Counter(next(iter(rates)) for rates in expert_layers.values())
result = dict(model=str(args.model.resolve()), index_sha256=hashlib.sha256(index_path.read_bytes()).hexdigest(),
    indexed_tensors=len(index), matrix_counts=dict(sorted(counts.items())), payload_bytes=dict(payloads),
    expert_layers_by_bits_x2=dict(sorted(layers.items())), examples=examples, shards=identities,
    scope='Header/layout inventory, not full payload checksum or runtime memory qualification. Separate n-gram file excluded.')
args.output.write_text(json.dumps(result, indent=2) + '\n')
print(json.dumps(result, indent=2))
