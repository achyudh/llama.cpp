#!/usr/bin/env python3
"""Losslessly import an EXL3 projection bank into a versioned GGUF bundle."""

import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
from safetensors import safe_open

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'gguf-py'))
from gguf import GGUFReader, GGUFWriter


def export(model, prefixes, output, first=0, columns=None):
    if output.exists() or first < 0 or first % 128 or (columns is not None and (columns <= 0 or columns % 128)):
        raise ValueError('Output must be new; output slices must contain complete 128-column blocks')
    index_path = model / 'model.safetensors.index.json'
    weight_map = json.loads(index_path.read_text())['weight_map']

    def read(key, selection=None):
        shard = weight_map[key]
        if Path(shard).name != shard:
            raise ValueError('Shard path must be a file name')
        with safe_open(model / shard, framework='numpy') as handle:
            return handle.get_tensor(key) if selection is None else handle.get_slice(key)[selection]

    packed, inputs, outputs = [], [], []
    layout = None
    for prefix in prefixes:
        if prefix + '.mul1' not in weight_map or prefix + '.mcg' in weight_map:
            raise ValueError('Only mul1 matrices are supported')
        marker = read(prefix + '.mul1')
        if marker.dtype != np.int32 or marker.size != 1 or int(marker.reshape(-1)[0]) & 0xffffffff != 0x83dcd12d:
            raise ValueError('Unrecognized mul1 codebook marker')
        if prefix + '.bias' in weight_map or prefix + '.scale' in weight_map:
            raise ValueError('Separate bias or legacy scale tensors are outside projection bundle version 1')
        shard = weight_map[prefix + '.trellis']
        if Path(shard).name != shard:
            raise ValueError('Shard path must be a file name')
        with safe_open(model / shard, framework='numpy') as handle:
            shape = handle.get_slice(prefix + '.trellis').get_shape()
        if len(shape) != 3 or shape[2] % 8:
            raise ValueError('Invalid packed matrix shape')
        k, full_n, rate = shape[0] * 16, shape[1] * 16, shape[2] // 8
        n = full_n - first if columns is None else columns
        if rate not in (2, 3, 4, 5, 6, 7, 8, 10, 12, 14, 16) or k % 128 or n <= 0 or n % 128 or first + n > full_n:
            raise ValueError('Invalid rate, shape or output slice')
        if layout is not None and layout != (k, n, rate):
            raise ValueError('All bank members must share K/N/bitrate')
        layout = k, n, rate
        p = read(prefix + '.trellis', (slice(None), slice(first // 16, (first + n) // 16), slice(None)))
        su, sv = read(prefix + '.suh'), read(prefix + '.svh', slice(first, first+n))
        if p.dtype != np.int16 or su.dtype != np.float16 or sv.dtype != np.float16 or su.shape != (k,) or sv.shape != (n,):
            raise ValueError('Invalid packed/auxiliary dtype or shape')
        packed.append(p.reshape(-1))
        inputs.append(su)
        outputs.append(sv)
    if not packed:
        raise ValueError('At least one projection is required')
    arrays = {'exl3.packed': np.stack(packed)[:, None, :], 'exl3.suh': np.stack(inputs), 'exl3.svh': np.stack(outputs)}
    temporary = output.with_name(output.name + '.part')
    # Reserve our staging name; failed imports must not overwrite another writer's file.
    with temporary.open('xb'):
        pass
    try:
        writer = GGUFWriter(temporary, 'exl3_projection')
        writer.add_uint32('exl3.version', 1)
        writer.add_string('exl3.codebook', 'mul1')
        writer.add_uint32('exl3.bits_x2', rate)
        writer.add_uint64('exl3.input_features', k)
        writer.add_uint64('exl3.output_features', n)
        writer.add_uint32('exl3.expert_count', len(prefixes))
        writer.add_uint64('exl3.output_start', first)
        writer.add_array('exl3.sources', prefixes)
        writer.add_string('exl3.source_index_sha256', hashlib.sha256(index_path.read_bytes()).hexdigest())
        for name, data in arrays.items():
            writer.add_tensor(name, data)
        writer.write_header_to_file()
        writer.write_kv_data_to_file()
        writer.write_tensors_to_file(progress=False)
        writer.close()
        reader = GGUFReader(temporary)
        for tensor in reader.tensors:
            original = arrays[tensor.name]
            if tensor.data.shape != original.shape or tensor.data.tobytes() != original.tobytes():
                raise ValueError('GGUF round-trip changed ' + tensor.name)
        if len(reader.tensors) != 3 or output.exists():
            raise ValueError('Unexpected tensor count or output already exists')
        del reader
        temporary.rename(output)
    finally:
        if temporary.exists():
            temporary.unlink()
    return dict(output=str(output), k=k, n=n, bits_x2=rate, experts=len(prefixes), sources=prefixes,
        output_start=first, index_sha256=hashlib.sha256(index_path.read_bytes()).hexdigest(),
        payload_sha256={name: hashlib.sha256(data.tobytes()).hexdigest() for name, data in arrays.items()},
        file_sha256=hashlib.sha256(output.read_bytes()).hexdigest(), round_trip='all payload bytes identical')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', type=Path, required=True)
    parser.add_argument('--prefix', action='append', required=True, help='Bank-local expert order follows argument order')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--start-column', type=int, default=0)
    parser.add_argument('--columns', type=int)
    args = parser.parse_args()
    print(json.dumps(export(args.model, args.prefix, args.output, args.start_column, args.columns), indent=2))
