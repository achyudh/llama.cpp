#!/usr/bin/env python3
"""Build a synthetic full-graph EXL3 fixture from test-llama-archs' Qwen4Exp model."""
import argparse
import ctypes as C
from pathlib import Path
import sys
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'gguf-py'))
import gguf


class Matrix(C.Structure):
    _fields_ = [('k', C.c_int64), ('n', C.c_int64), ('rate', C.c_int), ('packed', C.c_void_p),
                ('bytes', C.c_size_t), ('su', C.c_void_p), ('nsu', C.c_size_t), ('sv', C.c_void_p), ('nsv', C.c_size_t)]


def make(source, library, output):
    output.mkdir(exist_ok=False)
    reader = gguf.GGUFReader(source)
    values = {k: f.contents() for k, f in reader.fields.items() if not k.startswith('GGUF.')}
    types = {k: f.types for k, f in reader.fields.items() if not k.startswith('GGUF.')}
    data = {t.name: np.array(t.data) for t in reader.tensors if not t.name.endswith(('.scale', '.input_scale'))}
    rng = np.random.default_rng(2468)
    def random(shape):
        return rng.normal(0, .03, shape).astype(np.float32)
    # Match the published V/K head ratio; this permutation differs from its inverse.
    values['qwen4exp.ssm.inner_size'] = 768
    values['qwen4exp.ssm.time_step_rank'] = 6
    values['qwen4exp.expert_count'] = 4
    values['qwen4exp.expert_shared_feed_forward_length'] = 256
    values['qwen4exp.attention.indexer.head_count'] = 2
    for name in list(data):
        if name.startswith('blk.0.'):
            shapes = {'attn_qkv.weight': (1280, 256), 'attn_gate.weight': (768, 256),
                      'ssm_out.weight': (256, 768), 'ssm_conv1d.weight': (1280, 4),
                      'ssm_alpha.weight': (6, 256), 'ssm_beta.weight': (6, 256),
                      'ssm_dt.bias': (6,), 'ssm_a': (6,)}
            if name[6:] in shapes:
                data[name] = random(shapes[name[6:]])
                if name.endswith('ssm_a'): data[name] = -abs(data[name])
        if '.ffn_gate_inp.weight' in name: data[name] = random((4, 256))
        if '.ffn_down_exps.weight' in name: data[name] = random((4, 256, 384))
        if '.ffn_gate_up_exps.weight' in name:
            del data[name]
            data[name.replace('gate_up', 'gate')] = random((4, 384, 256))
            data[name.replace('gate_up', 'up')] = random((4, 384, 256))
        if '.ffn_gate_shexp.weight' in name or '.ffn_up_shexp.weight' in name: data[name] = random((256, 256))
        if '.ffn_down_shexp.weight' in name: data[name] = random((256, 256))
        if '.indexer.q_proj.weight' in name: data[name] = random((256, 256))
    # Append one full attention draft block, including its own HC head.
    for name, a in list(data.items()):
        if name.startswith('blk.1.'): data[name.replace('blk.1.', 'blk.2.', 1)] = a.copy()
    values['qwen4exp.block_count'] = 3
    values['qwen4exp.nextn_predict_layers'] = 1
    types['qwen4exp.nextn_predict_layers'] = [gguf.GGUFValueType.UINT32]
    values['qwen4exp.attention.compress_ratios'] = [4, 4, 4]
    values['qwen4exp.attention.recurrent_layers'] = [1, 0, 0]
    for key in ['qwen4exp.swiglu_clamp_exp', 'qwen4exp.swiglu_clamp_shexp']:
        values[key] = [0., 0., 0.]
    for name, shape in [('eh_proj.embedding', (256, 256)), ('eh_proj.hidden', (256, 256)),
                        ('enorm', (256,)), ('hnorm', (4, 256)), ('hc_head_norm', (4, 256)),
                        ('hc_head_down', (8, 1024)), ('hc_head_up', (1024, 8))]:
        data['blk.2.nextn.' + name + '.weight'] = random(shape)
    lib = C.CDLL(str(library.resolve()))
    ref = lib.ggml_exl3_mul_mat_ref
    ref.argtypes = [C.POINTER(Matrix), C.c_int64, C.c_void_p, C.c_size_t, C.c_void_p, C.c_size_t]
    ref.restype = C.c_bool
    compressed, reconstructed, rates = {}, {}, {}
    for name, a in data.items():
        # Keep non-matrix weights and HC/router/PLE parameters in their original format.
        select = (name == 'output.weight' or any(s in name for s in ['_exps.weight', '_shexp.weight',
                  'attn_qkv.weight', 'attn_gate.weight', 'ssm_out.weight', 'attn_q.weight', 'attn_k.weight',
                  'attn_v.weight', 'attn_output.weight', 'indexer.q_proj.weight', 'indexer.k_proj.weight', 'eh_proj.']))
        if not select or a.ndim not in (2, 3) or a.shape[-1] % 128 or a.shape[-2] % 128:
            compressed[name] = reconstructed[name] = a
            continue
        k, n, experts = a.shape[-1], a.shape[-2], a.shape[0] if a.ndim == 3 else 1
        rate = (7 if name.startswith('blk.0.') else 8) if '_exps.weight' in name else (10 if 'eh_proj.' in name else (12 if name == 'output.weight' else 14))
        p = rng.integers(-32768, 32768, (experts, 1, k*n*rate//32), dtype=np.int16)
        su = rng.uniform(.5, 1.5, (experts, 1, k)).astype(np.float16)
        sv = rng.uniform(.01, .03, (experts, 1, n)).astype(np.float16)
        weights = []
        x = np.eye(k, dtype=np.float32)
        for e in range(experts):
            y = np.zeros((k, n), np.float32)
            m = Matrix(k, n, rate, p[e].ctypes.data, p[e].nbytes, su[e].ctypes.data, k, sv[e].ctypes.data, n)
            if not ref(C.byref(m), k, x.ctypes.data, x.size, y.ctypes.data, y.size): raise ValueError(name)
            w = y.T.copy()
            # The ordinary GGUF path expects tiled V-head order.
            def tile(v): return v.reshape(2, 3, 128, -1).transpose(1, 0, 2, 3).reshape(v.shape)
            if name == 'blk.0.attn_qkv.weight': w[512:] = tile(w[512:])
            if name == 'blk.0.attn_gate.weight': w = tile(w)
            if name == 'blk.0.ssm_out.weight': w = tile(w.T).T.copy()
            weights.append(w)
        compressed[name], compressed[name+'.suh'], compressed[name+'.svh'] = p, su, sv
        reconstructed[name] = np.stack(weights) if experts > 1 else weights[0]
        rates[name] = rate
    for label, tensors in [('compressed', compressed), ('reconstructed', reconstructed)]:
        writer = gguf.GGUFWriter(output / (label+'.gguf'), 'qwen4exp')
        for key, val in values.items():
            if key == 'general.architecture': continue
            ty = types[key]
            if ty[0] == gguf.GGUFValueType.ARRAY and not val: continue
            if ty[0] == gguf.GGUFValueType.ARRAY: writer.add_key_value(key, val, ty[0], ty[1] if len(ty)>1 else gguf.GGUFValueType.STRING)
            else: writer.add_key_value(key, val, ty[0])
        if label == 'compressed':
            writer.add_uint32('exl3.version', 1); writer.add_string('exl3.codebook', 'mul1')
            for name, rate in rates.items(): writer.add_uint32('exl3.'+name+'.bits_x2', rate)
        for name, a in tensors.items(): writer.add_tensor(name, np.ascontiguousarray(a))
        writer.write_header_to_file(); writer.write_kv_data_to_file(); writer.write_tensors_to_file(); writer.close()
    print(f'{len(rates)} compressed banks; synthetic ordinary PLE rows; native GDN head order; separate MTP projections')


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source', type=Path, required=True); p.add_argument('--library', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args(); make(args.source, args.library, args.output)
