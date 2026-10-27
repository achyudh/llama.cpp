from __future__ import annotations

from dataclasses import dataclass
from functools import partial
from hashlib import sha256
import json
import math
import os
from pathlib import Path
import re
import struct
import tempfile
from typing import Any

import numpy as np
import gguf


CHUNK_BYTES = 8 * 1024 * 1024
WIDTHS = {'I16': 2, 'I32': 4, 'I64': 8, 'F16': 2, 'BF16': 2, 'F32': 4}


@dataclass(frozen=True)
class Source:
    name: str
    file: Path
    offset: int
    nbytes: int
    dtype: str
    shape: tuple[int, ...]


@dataclass
class Payload:
    name: str
    shape: tuple[int, ...]
    dtype: str
    parts: list[tuple[Source, int, int]]

    def array(self, converter: Exl3Converter) -> Any:
        chunks = []
        for source, offset, size in self.parts:
            for start in range(0, size, CHUNK_BYTES):
                chunks.append(partial(converter.read, source, offset + start, min(CHUNK_BYTES, size - start)))
        return gguf.LazyChunkedTensor(chunks, self.shape, np.int16 if self.dtype == 'I16' else np.float16)


class Exl3Converter:
    def __init__(self, model: Any):
        self.model = model
        self.sources: dict[str, Source] = {}
        self.aliases: dict[str, Source] = {}
        self.account: dict[str, dict[str, Any]] = {}
        self.files: dict[Path, tuple[int, int, int]] = {}
        self.header_hashes: dict[str, str] = {}
        self.payloads: list[Payload] = []
        self.rates: dict[str, int] = {}
        self.constants: dict[str, list[int]] = {}
        self.ordinary: dict[str, Source] = {}
        self.ordinary_hashes: dict[str, dict[str, Any]] = {}
        self.table: Payload | None = None
        self.head_dim = 0

        if model.is_big_endian or not model.lazy or model.use_temp_file or model.remote_hf_model_id:
            raise ValueError('EXL3 conversion requires local, lazy, little-endian input without temporary tensor staging')
        if model.fuse_gate_up_exps or model.fuse_qkv or model._fp8_as_q8:
            raise ValueError('EXL3 conversion does not support fusion or requantization options')
        if model.ftype not in (gguf.LlamaFileType.ALL_F32, gguf.LlamaFileType.MOSTLY_F16, gguf.LlamaFileType.MOSTLY_BF16):
            raise ValueError('EXL3 conversion requires an ordinary floating-point output type')
        if model.hparams['quantization_config'].get('codebook') != 'mul1':
            raise ValueError('Unsupported EXL3 codebook')
        index = json.loads((model.dir_model / 'model.safetensors.index.json').read_text())['weight_map']
        for filename in sorted(set(index.values())):
            tensors, _ = self.header(filename)
            if set(tensors) != {name for name, shard in index.items() if shard == filename}:
                raise ValueError(f'Safetensors index/header mismatch: {filename}')
            for name, source in tensors.items():
                filtered = model.filter_tensors((name, lambda: None))
                if filtered is None:
                    if name.startswith(('visual.', 'model.visual.')):
                        reason = 'vision'
                    elif model.no_mtp and name.startswith(('mtp.', 'model.mtp.')):
                        reason = 'mtp_disabled'
                    elif model.mtp_only and not name.startswith(('mtp.', 'model.mtp.')):
                        reason = 'trunk_excluded'
                    else:
                        raise ValueError(f'Unexpected filtered source tensor: {name}')
                    self.use(source, 'excluded', [], reason=reason)
                else:
                    alias = filtered[0]
                    if alias in self.aliases:
                        raise ValueError(f'Duplicate normalized tensor name: {alias}')
                    self.aliases[alias] = source
        if set(self.aliases) != set(model.model_tensors):
            raise ValueError('EXL3 inventory differs from converter tensor index')

    def header(self, filename: str) -> tuple[dict[str, Source], dict[str, str]]:
        if Path(filename).name != filename or not filename.endswith('.safetensors'):
            raise ValueError(f'Invalid safetensors filename: {filename}')
        path = self.model.dir_model / filename
        stat = path.stat()
        self.files[path] = (stat.st_ino, stat.st_size, stat.st_mtime_ns)
        with path.open('rb') as f:
            prefix = f.read(8)
            if len(prefix) != 8:
                raise ValueError(f'Truncated safetensors header: {filename}')
            length, = struct.unpack('<Q', prefix)
            if not 0 < length <= min(128 * 1024 * 1024, stat.st_size - 8):
                raise ValueError(f'Invalid safetensors header length: {filename}')
            raw = f.read(length)
        header = json.loads(raw)
        self.header_hashes[filename] = sha256(raw).hexdigest()
        result = {}
        ranges = []
        for name, info in header.items():
            if name == '__metadata__':
                continue
            shape = tuple(info['shape'])
            begin, end = info['data_offsets']
            dtype = info['dtype']
            if dtype not in WIDTHS or any(type(n) is not int or n <= 0 for n in shape):
                raise ValueError(f'Invalid source type/shape: {name}')
            if type(begin) is not int or type(end) is not int or not 0 <= begin < end <= stat.st_size - 8 - length:
                raise ValueError(f'Invalid source byte range: {name}')
            if end - begin != math.prod(shape) * WIDTHS[dtype] or name in self.sources:
                raise ValueError(f'Invalid or duplicate source tensor: {name}')
            source = Source(name, path, 8 + length + begin, end - begin, dtype, shape)
            result[name] = self.sources[name] = source
            ranges.append((begin, end))
        ranges.sort()
        if any(a[1] > b[0] for a, b in zip(ranges, ranges[1:])):
            raise ValueError(f'Overlapping source tensors: {filename}')
        return result, header.get('__metadata__', {})

    def read(self, source: Source, offset: int = 0, size: int | None = None) -> np.ndarray:
        size = source.nbytes if size is None else size
        if offset < 0 or size <= 0 or size > CHUNK_BYTES or offset + size > source.nbytes:
            raise ValueError(f'Invalid bounded read: {source.name}')
        with source.file.open('rb') as f:
            stat = os.fstat(f.fileno())
            if (stat.st_ino, stat.st_size, stat.st_mtime_ns) != self.files[source.file]:
                raise ValueError(f'Source changed during conversion: {source.file}')
            f.seek(source.offset + offset)
            data = f.read(size)
        if len(data) != size:
            raise ValueError(f'Truncated source tensor: {source.name}')
        dtype = {'I16': '<i2', 'I32': '<i4', 'I64': '<i8', 'F16': '<f2', 'BF16': '<u2', 'F32': '<f4'}[source.dtype]
        return np.frombuffer(data, dtype=dtype)

    def use(self, source: Source, kind: str, outputs: list[str], **extra: Any) -> None:
        if source.name in self.account:
            raise ValueError(f'Source tensor consumed twice: {source.name}')
        self.account[source.name] = dict(kind=kind, outputs=outputs, **extra)

    def projection(self, prefix: str) -> tuple[Source, Source, Source, int]:
        try:
            packed, su, sv, marker = [self.aliases[prefix + suffix] for suffix in ('.trellis', '.suh', '.svh', '.mul1')]
        except KeyError as e:
            raise ValueError(f'Incomplete EXL3 projection: {prefix}') from e
        if marker.dtype != 'I32' or math.prod(marker.shape) != 1 or int(self.read(marker)[0]) & 0xffffffff != 0x83dcd12d:
            raise ValueError(f'Invalid EXL3 mul1 marker: {prefix}')
        if len(packed.shape) != 3 or packed.dtype != 'I16' or packed.shape[2] % 8:
            raise ValueError(f'Invalid EXL3 packed shape/type: {prefix}')
        k, n, rate = packed.shape[0] * 16, packed.shape[1] * 16, packed.shape[2] // 8
        if rate not in (2, 3, 4, 5, 6, 7, 8, 10, 12, 14, 16) or k % 128 or n % 128:
            raise ValueError(f'Unsupported EXL3 projection dimensions/rate: {prefix}')
        if su.dtype != 'F16' or sv.dtype != 'F16' or su.shape != (k,) or sv.shape != (n,):
            raise ValueError(f'Invalid EXL3 auxiliary shape/type: {prefix}')
        return packed, su, sv, rate

    def plan(self) -> None:
        model = self.model
        groups: dict[str, list[tuple[int, str]]] = {}
        for alias in self.aliases:
            if not alias.endswith('.trellis'):
                continue
            prefix = alias.removesuffix('.trellis')
            match = re.fullmatch(r'(model\.layers\.\d+\.mlp\.experts)\.(\d+)\.(down_proj|gate_proj|up_proj)', prefix)
            if match:
                bank = match[1] + '.' + match[3]
                groups.setdefault(bank, []).append((int(match[2]), prefix))
            else:
                groups[prefix] = [(0, prefix)]
        for bank, members in sorted(groups.items()):
            members.sort()
            expert_bank = '.mlp.experts.' in bank
            if expert_bank and (len(members) != model.hparams['num_experts'] or any(i != member[0] for i, member in enumerate(members))):
                raise ValueError(f'Missing or duplicate expert IDs: {bank}')
            projections = [self.projection(prefix) for _, prefix in members]
            prototype = projections[0]
            if any((p.shape, su.shape, sv.shape, rate) != (prototype[0].shape, prototype[1].shape, prototype[2].shape, prototype[3]) for p, su, sv, rate in projections):
                raise ValueError(f'Mixed projection geometry/rate in bank: {bank}')
            k, n, rate = prototype[1].shape[0], prototype[2].shape[0], prototype[3]
            bid = int(bank.split('.')[2]) if bank.startswith('model.layers.') else None
            if '.indexer.index_qk_proj' in bank:
                q = model.hparams['indexer_n_heads'] * model.hparams['indexer_head_dim']
                if q % 128 or n - q != model.hparams['indexer_head_dim']:
                    raise ValueError(f'Invalid indexer Q/K split: {bank}')
                outputs = [(model.format_tensor_name(gguf.MODEL_TENSOR.INDEXER_Q_PROJ, bid), 0, q),
                           (model.format_tensor_name(gguf.MODEL_TENSOR.INDEXER_K_PROJ, bid), q, n)]
            elif bank.endswith(('.nextn_fc_embedding', '.nextn_fc_hidden')):
                part = 'embedding' if bank.endswith('_embedding') else 'hidden'
                outputs = [(model.format_tensor_name(gguf.MODEL_TENSOR.NEXTN_EH_PROJ, bid, '.' + part + '.weight'), 0, n)]
            else:
                outputs = [(model.map_tensor_name(bank + '.weight'), 0, n)]
            for name, first, end in outputs:
                packed_parts, su_parts, sv_parts = [], [], []
                for p, su, sv, _ in projections:
                    if first == 0 and end == n:
                        packed_parts.append((p, 0, p.nbytes))
                    else:
                        # Output slices retain complete Hadamard blocks, in native K-tile order.
                        stride = p.shape[1] * p.shape[2] * 2
                        for i in range(p.shape[0]):
                            packed_parts.append((p, i * stride + first // 16 * p.shape[2] * 2, (end - first) // 16 * p.shape[2] * 2))
                    su_parts.append((su, 0, su.nbytes))
                    sv_parts.append((sv, first * 2, (end - first) * 2))
                e = len(members)
                self.payloads.extend([Payload(name, (e, 1, k * (end - first) * rate // 32), 'I16', packed_parts),
                                      Payload(name + '.suh', (e, 1, k), 'F16', su_parts),
                                      Payload(name + '.svh', (e, 1, end - first), 'F16', sv_parts)])
                self.rates[name] = rate
            for _, prefix in members:
                for suffix in ('.trellis', '.suh', '.svh', '.mul1'):
                    self.use(self.aliases[prefix + suffix], 'codebook' if suffix == '.mul1' else 'projection',
                             ['exl3.codebook'] if suffix == '.mul1' else [name + (suffix if suffix != '.trellis' else '') for name, _, _ in outputs])
        for alias, source in self.aliases.items():
            if source.name in self.account:
                continue
            if source.dtype not in ('F16', 'BF16', 'F32') or alias.endswith(('.suh', '.svh', '.mul1', '.mcg', '.scale')):
                raise ValueError(f'Unknown EXL3 source tensor: {source.name}')
            self.ordinary[alias] = source
        self.plan_ngram()

    def plan_ngram(self) -> None:
        hp = self.model.hparams
        if not hp.get('ple_layer_ids') or self.model.mtp_only:
            if (self.model.dir_model / 'ngram_embedding.safetensors').exists():
                tensors, _ = self.header('ngram_embedding.safetensors')
                for source in tensors.values():
                    self.use(source, 'excluded', [], reason='ngram_not_used')
            return
        tensors, meta = self.header('ngram_embedding.safetensors')
        heads = (hp['ngram_size'] - 1) * hp['heads_per_ngram']
        self.head_dim, remainder = divmod(hp['ple_embed_dim'], heads)
        bits = int(meta.get('K', '0'))
        rows = int(meta.get('rows', '0'))
        if (meta.get('format'), meta.get('version'), meta.get('codebook')) != ('exl3_ngram_trellis', '1', 'mul1') or not 1 <= bits <= 8:
            raise ValueError('Invalid EXL3 n-gram codec metadata')
        if remainder or self.head_dim % 16 or not 16 <= self.head_dim <= 256 or int(meta['row_dim']) != self.head_dim or not 0 < rows <= 0x7fffffff:
            raise ValueError('Invalid EXL3 n-gram dimensions')
        def find(suffix: str) -> Source:
            matches = [s for n, s in tensors.items() if n.endswith('.' + suffix)]
            if len(matches) != 1:
                raise ValueError(f'Missing/duplicate n-gram tensor: {suffix}')
            return matches[0]
        for suffix, native, count, key in [('ple_embedding.layer_multipliers', 'layer_multipliers', hp['ngram_size'], 'layer_multipliers'),
                                           ('ple_embedding.ngram_heads_offsets', 'head_offsets', heads, 'head_offsets'),
                                           ('ple_embedding.ngram_heads_vocab_sizes', 'head_vocab_sizes', heads, 'head_vocab_sizes')]:
            source = find(native)
            if source.dtype != 'I64' or source.shape != (count,):
                raise ValueError(f'Invalid n-gram integer constants: {native}')
            self.constants[suffix] = [int(n) for n in self.read(source)]
            self.use(source, 'metadata', ['qwen4exp.ple.' + key])
        offsets = self.constants['ple_embedding.ngram_heads_offsets']
        sizes = self.constants['ple_embedding.ngram_heads_vocab_sizes']
        if any(n < 0 for n in self.constants['ple_embedding.layer_multipliers']) or any(o < 0 or s <= 0 or o + s > rows for o, s in zip(offsets, sizes)) or any(offsets[i] < offsets[i-1] + sizes[i-1] for i in range(1, heads)):
            raise ValueError('Invalid n-gram hash ranges/constants')
        bias = find('head_bias')
        if bias.dtype != 'F16' or bias.shape != (heads, self.head_dim):
            raise ValueError('Invalid n-gram head bias')
        self.payloads.append(Payload('per_layer_token_embd.bias', bias.shape, 'F16', [(bias, 0, bias.nbytes)]))
        self.use(bias, 'ngram', ['per_layer_token_embd.bias'])
        shards = {}
        width = 1 + self.head_dim * bits // 16
        for name, source in tensors.items():
            if name in self.account:
                continue
            match = re.search(r'\.ngram_embedding\.shard_(\d+)\.trellis$', name)
            if not match or int(match[1]) in shards or source.dtype != 'I16' or len(source.shape) != 2 or source.shape[1] != width:
                raise ValueError(f'Invalid n-gram shard: {name}')
            shards[int(match[1])] = source
        if len(shards) != hp['split_ngram_parts'] or any(i != key for i, key in enumerate(sorted(shards))) or sum(s.shape[0] for s in shards.values()) != rows:
            raise ValueError('Incomplete n-gram row shards')
        parts = [(shards[i], 0, shards[i].nbytes) for i in sorted(shards)]
        self.table = Payload('per_layer_token_embd.weight', (rows, width), 'I16', parts)
        for source, _, _ in parts:
            self.use(source, 'ngram', [self.table.name])
        self.ngram_bits = bits

    def add_tensors(self) -> None:
        writer = self.model.gguf_writer
        writer.add_custom_alignment(4096)
        writer.add_uint32('exl3.version', 1)
        writer.add_string('exl3.codebook', 'mul1')
        for name, rate in self.rates.items():
            writer.add_uint32('exl3.' + name + '.bits_x2', rate)
        for payload in self.payloads:
            writer.add_tensor(payload.name, payload.array(self))
        if self.table:
            writer.add_uint32('exl3.ngram.version', 1)
            writer.add_string('exl3.ngram.codebook', 'mul1')
            writer.add_uint32('exl3.ngram.bits', self.ngram_bits)
            if writer.tensors[-1]:
                writer.tensors.append({})
            writer.add_tensor(self.table.name, self.table.array(self))
            self.payloads.append(self.table)
        missing = self.sources.keys() - self.account.keys()
        if missing:
            raise ValueError(f'Unconsumed source tensors: {sorted(missing)[:20]}')

    def ordinary_outputs(self, alias: str, names: list[str]) -> None:
        if alias not in self.ordinary or not names:
            raise ValueError(f'Unaccounted ordinary tensor: {alias}')
        self.use(self.ordinary[alias], 'ordinary', names, transform='existing Qwen tensor rules; source matrix precision, F32 norms/small parameters')

    def verify_ordinary(self, alias: str, source: Source, tensor: Any, file: Any) -> dict[str, Any]:
        import torch

        name = tensor.name
        norm = alias.endswith('norm.weight') and not alias.endswith('linear_attn.norm.weight')
        norm = norm or alias.endswith(('.ple.norm_key.weight', '.ple.norm_query.weight', '.ple.norm_conv.weight'))
        exponent = alias.endswith('.A_log')
        linear = '.linear_attn.' in alias
        head_parameter = linear and any(s in alias for s in ('.in_proj_a.', '.in_proj_b.', '.A_log', '.dt_bias', '.dt_proj'))
        convolution = '.conv1d.' in alias
        shape = tuple(n for n in source.shape if n != 1) if convolution else source.shape
        qtype = {'F16': gguf.GGMLQuantizationType.F16, 'BF16': gguf.GGMLQuantizationType.BF16,
                 'F32': gguf.GGMLQuantizationType.F32}[source.dtype]
        if norm or exponent or len(shape) <= 1 or name.endswith('_norm.weight') or name.endswith(('.ssm_conv1d.weight', '.ffn_gate_inp.weight')):
            qtype = gguf.GGMLQuantizationType.F32
        if tensor.tensor_type != qtype or tuple(tensor.shape) != tuple(reversed(shape)):
            raise ValueError(f'Invalid ordinary output shape/type: {name}')

        def f32(raw: np.ndarray) -> np.ndarray:
            return (raw.astype(np.uint32) << 16).view(np.float32) if source.dtype == 'BF16' else raw.astype(np.float32)

        def tiled(array: np.ndarray, dim: int, head_dim: int) -> np.ndarray:
            hp = self.model.hparams
            nk, nv = hp['linear_num_key_heads'], hp['linear_num_value_heads']
            moved = np.moveaxis(array, dim, 0)
            reordered = moved.reshape(nk, nv // nk, head_dim, *moved.shape[1:]).swapaxes(0, 1).reshape(moved.shape)
            return np.moveaxis(reordered, 0, dim)

        source_hash, output_hash = sha256(), sha256()
        position = 0
        file.seek(tensor.data_offset)

        def compare(expected: bytes) -> None:
            nonlocal position
            if file.read(len(expected)) != expected:
                raise ValueError(f'Ordinary output differs from FP32 source transform: {name}')
            output_hash.update(expected)
            position += len(expected)

        if norm or exponent or head_parameter or (linear and convolution):
            if source.nbytes > 8 * 1024 * 1024:
                raise ValueError(f'Ordinary transform exceeds verification workspace: {name}')
            raw = np.concatenate([self.read(source, start, min(CHUNK_BYTES, source.nbytes - start))
                                  for start in range(0, source.nbytes, CHUNK_BYTES)])
            source_hash.update(raw.tobytes())
            array = f32(raw).reshape(source.shape)
            if head_parameter and self.model.hparams['linear_num_key_heads'] != self.model.hparams['linear_num_value_heads']:
                dim = -1 if '.dt_proj' in alias and array.ndim > 1 else 0
                array = tiled(array, dim, 1)
            if convolution:
                array = array.squeeze()
                if linear and self.model.hparams['linear_num_key_heads'] != self.model.hparams['linear_num_value_heads']:
                    qk = 2 * self.model.hparams['linear_num_key_heads'] * self.model.hparams['linear_key_head_dim']
                    array = np.concatenate([array[:qk], tiled(array[qk:], 0, self.model.hparams['linear_value_head_dim'])])
            if norm:
                array = array + np.float32(1)
            elif exponent:
                array = -torch.exp(torch.from_numpy(array.copy())).numpy()
            compare(np.ascontiguousarray(array, dtype=np.float16 if qtype == gguf.GGMLQuantizationType.F16 else np.float32).tobytes())
        else:
            for start in range(0, source.nbytes, CHUNK_BYTES):
                raw = self.read(source, start, min(CHUNK_BYTES, source.nbytes - start))
                source_hash.update(raw.tobytes())
                compare(f32(raw).tobytes() if qtype == gguf.GGMLQuantizationType.F32 else raw.tobytes())
        if position != tensor.n_bytes:
            raise ValueError(f'Ordinary output length mismatch: {name}')
        return dict(source=source.name, source_sha256=source_hash.hexdigest(), sha256=output_hash.hexdigest(),
                    bytes=int(tensor.n_bytes), type=qtype.name,
                    transform='f32_add_one' if norm else 'f32_neg_exp' if exponent else 'head_permutation' if head_parameter or linear and convolution else 'identity')

    def verify(self, paths: list[Path]) -> dict[str, Any]:
        wanted = {p.name: p for p in self.payloads}
        ordinary = {}
        for alias, source in self.ordinary.items():
            names = self.account[source.name]['outputs']
            if len(names) != 1 or names[0] in ordinary:
                raise ValueError(f'Unsupported ordinary tensor mapping: {alias}')
            ordinary[names[0]] = alias, source
        self.ordinary_hashes = {}
        hashes = {}
        for path in paths:
            reader = gguf.GGUFReader(path)
            if reader.alignment != 4096:
                raise ValueError(f'Invalid split alignment: {path}')
            with path.open('rb') as f:
                for tensor in reader.tensors:
                    if tensor.data_offset % 4096:
                        raise ValueError(f'Unaligned tensor: {tensor.name}')
                    if tensor.name in ordinary:
                        alias, source = ordinary.pop(tensor.name)
                        self.ordinary_hashes[tensor.name] = self.verify_ordinary(alias, source, tensor, f)
                        continue
                    if tensor.name not in wanted:
                        raise ValueError(f'Unexpected output tensor: {tensor.name}')
                    payload = wanted.pop(tensor.name)
                    dtype = gguf.GGMLQuantizationType.I16 if payload.dtype == 'I16' else gguf.GGMLQuantizationType.F16
                    if tensor.tensor_type != dtype or tuple(tensor.shape) != tuple(reversed(payload.shape)):
                        raise ValueError(f'Invalid output shape/type: {tensor.name}')
                    expected, actual = sha256(), sha256()
                    f.seek(tensor.data_offset)
                    source_hashes = []
                    for source, offset, size in payload.parts:
                        part_hash = sha256()
                        for start in range(0, size, CHUNK_BYTES):
                            data = self.read(source, offset + start, min(CHUNK_BYTES, size - start)).tobytes()
                            output = f.read(len(data))
                            part_hash.update(data)
                            expected.update(data)
                            actual.update(output)
                            if data != output:
                                raise ValueError(f'Output payload differs from source: {tensor.name}/{source.name}')
                        source_hashes.append(dict(source=source.name, offset=offset, bytes=size, sha256=part_hash.hexdigest()))
                    if expected.digest() != actual.digest() or sum(part[2] for part in payload.parts) != tensor.n_bytes:
                        raise ValueError(f'Output payload length/hash mismatch: {tensor.name}')
                    hashes[tensor.name] = dict(sha256=actual.hexdigest(), file=path.name, offset=int(tensor.data_offset),
                                               bytes=int(tensor.n_bytes), sources=source_hashes)
            del reader
        if wanted or ordinary:
            raise ValueError(f'Missing output payloads: {sorted(wanted.keys() | ordinary.keys())}')
        return hashes

    def report(self) -> dict[str, Any]:
        return dict(version=1, source=str(self.model.dir_model.resolve()), header_sha256=self.header_hashes,
                    chunk_bytes=CHUNK_BYTES, alignment=4096, rates=self.rates,
                    source_tensors={name: dict(file=s.file.name, offset=s.offset, bytes=s.nbytes, dtype=s.dtype,
                                              shape=s.shape, **self.account[name]) for name, s in self.sources.items()})

    def split_plan(self, paths: list[Path]) -> list[dict[str, Any]]:
        from gguf.gguf_writer import GGUFValue

        writer = self.model.gguf_writer
        result = []
        count = len(paths)
        total_tensors = sum(len(tensors) for tensors in writer.tensors)
        for i, (path, tensors) in enumerate(zip(paths, writer.tensors)):
            kv = dict(writer.kv_data[0]) if i == 0 else {'general.alignment': writer.kv_data[0]['general.alignment']}
            if count > 1:
                kv.update({'split.no': GGUFValue(i, gguf.GGUFValueType.UINT16),
                           'split.count': GGUFValue(count, gguf.GGUFValueType.UINT16),
                           'split.tensors.count': GGUFValue(total_tensors, gguf.GGUFValueType.INT32)})
            kv_size = sum(len(writer._pack_val(key, gguf.GGUFValueType.STRING, add_vtype=False)) +
                          len(writer._pack_val(value.value, value.type, add_vtype=True, sub_type=value.sub_type)) for key, value in kv.items())
            info_size = sum(8 + len(name.encode('utf-8')) + 4 + 8 * len(info.shape) + 4 + 8 for name, info in tensors.items())
            data_offset = writer.ggml_pad(24 + kv_size + info_size, 4096)
            result.append(dict(file=path.name, tensors=len(tensors), payload_bytes=sum(t.nbytes for t in tensors.values()),
                               data_offset=data_offset, file_bytes=data_offset + sum(writer.ggml_pad(t.nbytes, 4096) for t in tensors.values())))
        return result

    def write(self) -> None:
        writer = self.model.gguf_writer
        if not self.model.fname_out.name.endswith('.gguf'):
            self.model.fname_out = self.model.fname_out.with_name(self.model.fname_out.name + '.gguf')
        paths = writer.format_shard_names(self.model.fname_out)
        report_path = self.model.fname_out.with_suffix('.exl3-plan.json' if self.model.dry_run else '.exl3.json')
        if any(p.exists() for p in [*paths, report_path]):
            raise ValueError('EXL3 output or accounting report already exists')
        report = self.report()
        report['splits'] = self.split_plan(paths)
        if self.model.dry_run:
            report['status'] = 'planned'
            report_path.write_text(json.dumps(report, indent=2) + '\n')
            return
        published = []
        with tempfile.TemporaryDirectory(prefix='.exl3-', dir=self.model.fname_out.parent) as directory:
            staging = Path(directory) / self.model.fname_out.name
            try:
                writer.write_header_to_file(staging)
                writer.write_kv_data_to_file()
                writer.write_tensors_to_file(progress=True)
                writer.close()
                temporary_paths = writer.format_shard_names(staging)
                if any(path.stat().st_size != plan['file_bytes'] for path, plan in zip(temporary_paths, report['splits'])):
                    raise ValueError('Output size differs from planned split layout')
                report['verified_payloads'] = self.verify(temporary_paths)
                report['verified_ordinary'] = self.ordinary_hashes
                report['status'] = 'verified'
                for src, dst in zip(temporary_paths, paths):
                    os.link(src, dst)
                    published.append(dst)
                temporary_report = Path(directory) / report_path.name
                temporary_report.write_text(json.dumps(report, indent=2) + '\n')
                os.link(temporary_report, report_path)
                published.append(report_path)
            except BaseException:
                writer.close()
                for path in published:
                    path.unlink()
                raise
