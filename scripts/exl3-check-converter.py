#!/usr/bin/env python3
"""Qualify EXL3 conversion against the frozen V3 tensors and graph outputs."""
import argparse
import hashlib
import json
from pathlib import Path
import re
import subprocess
import sys
import time
from typing import Any

import numpy as np
import torch
from safetensors.torch import save_file

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'gguf-py'))
import gguf
from conversion.qwen4exp import Qwen4ExpTextModel


ROUNDOFF_TENSORS = {
    'output_hc_norm.weight', 'blk.0.ssm_a',
    'blk.0.hc_attn_norm.weight', 'blk.0.hc_ffn_norm.weight',
    'blk.0.ple_norm_conv.weight', 'blk.0.ple_norm_key.weight', 'blk.0.ple_norm_query.weight',
    'blk.1.hc_attn_norm.weight', 'blk.1.hc_ffn_norm.weight',
    'blk.1.attn_q_norm.weight', 'blk.1.attn_k_norm.weight',
    'blk.1.indexer.q_norm.weight', 'blk.1.indexer.k_norm.weight',
    'blk.2.hc_attn_norm.weight', 'blk.2.hc_ffn_norm.weight',
    'blk.2.attn_q_norm.weight', 'blk.2.attn_k_norm.weight',
    'blk.2.indexer.q_norm.weight', 'blk.2.indexer.k_norm.weight',
    'blk.2.nextn.enorm.weight', 'blk.2.nextn.hnorm.weight', 'blk.2.nextn.hc_head_norm.weight',
}
SHARED_INPUT_SCALES = {
    'blk.1.indexer.k_proj.weight.suh': 'blk.1.indexer.q_proj.weight.suh',
    'blk.2.indexer.k_proj.weight.suh': 'blk.2.indexer.q_proj.weight.suh',
}


def compare_tensors(reference, paths):
    reader = gguf.GGUFReader(reference)
    expected = {t.name: (t.tensor_type, np.array(t.data)) for t in reader.tensors}
    observed = {}
    for path in paths:
        converted = gguf.GGUFReader(path)
        for tensor in converted.tensors:
            if tensor.name in observed:
                raise ValueError(f'Duplicate converted tensor: {tensor.name}')
            observed[tensor.name] = tensor.tensor_type, np.array(tensor.data)
    if expected.keys() != observed.keys():
        raise ValueError(f'Fixture tensor set changed: {sorted(expected.keys() ^ observed.keys())}')
    records = {}
    for name, (dtype, original) in expected.items():
        actual_type, actual = observed[name]
        if dtype != actual_type or original.shape != actual.shape:
            raise ValueError(f'Fixture tensor type/shape changed: {name}')
        if np.array_equal(original, actual):
            records[name] = dict(kind='identical')
        elif name in SHARED_INPUT_SCALES:
            if not np.array_equal(actual, expected[SHARED_INPUT_SCALES[name]][1]):
                raise ValueError(f'Fixture input scale does not equal shared Q/K source: {name}')
            records[name] = dict(kind='shared_qk_input_scale', source=SHARED_INPUT_SCALES[name])
        elif name in ROUNDOFF_TENSORS and dtype == gguf.GGMLQuantizationType.F32:
            error = float(np.max(np.abs(actual.astype(np.float64) - original.astype(np.float64))))
            if not np.isfinite(error) or error > 6e-8:
                raise ValueError(f'Fixture inverse-transform roundoff exceeded for {name}: {error}')
            records[name] = dict(kind='f32_inverse_roundoff', max_absolute_error=error, limit=6e-8)
        else:
            raise ValueError(f'Unexpected fixture tensor difference: {name}')
    return records

def prepare(actual, reference, root):
    source = root / 'source'
    source.mkdir()
    reader = gguf.GGUFReader(reference)
    values = {k: f.contents() for k, f in reader.fields.items() if not k.startswith('GGUF.')}
    data = {t.name: np.array(t.data) for t in reader.tensors}
    config = json.loads((actual / 'config.json').read_text())
    hp = config['text_config']
    hp.update(
        num_hidden_layers=2, hidden_size=256, num_attention_heads=2, num_key_value_heads=2, head_dim=128,
        intermediate_size=384, moe_intermediate_size=384, shared_expert_intermediate_size=256,
        num_experts=4, num_experts_per_tok=2, mtp_num_hidden_layers=1, vocab_size=128, hc_count=4, hc_lowrank=8,
        linear_num_key_heads=2, linear_num_value_heads=6, linear_key_head_dim=128, linear_value_head_dim=128,
        linear_conv_kernel_dim=4, layer_types=['linear_attention', 'full_attention'], full_attention_interval=2,
        ple_layer_ids=[1], ple_embed_dim=256, ngram_size=3, heads_per_ngram=2, split_ngram_parts=2,
        indexer_n_heads=2, indexer_head_dim=128, indexer_budget=16, indexer_compress_ratio=4,
        max_position_embeddings=256, eos_token_id=0, rms_norm_eps=1e-5,
    )
    hp['rope_parameters'] = {'rope_type': 'default', 'rope_theta': 10000.0, 'partial_rotary_factor': 1.0, 'mrope_section': [32, 32, 32]}
    config['architectures'] = ['Qwen4ExpForConditionalGeneration']
    config['image_token_id'] = None
    (source / 'config.json').write_text(json.dumps(config))
    Qwen4ExpTextModel._original_block_count = 2
    Qwen4ExpTextModel.no_mtp = False
    Qwen4ExpTextModel.mtp_only = False
    mapper = object.__new__(Qwen4ExpTextModel)
    mapper.hparams = hp
    mapper.block_count = 3
    mapper.tensor_map = gguf.get_tensor_name_map(gguf.MODEL_ARCH.QWEN4EXP, 3)
    keys = json.loads((actual / 'model.safetensors.index.json').read_text())['weight_map']
    selected = []
    for key in keys:
        match = re.search('model\\.language_model\\.layers\\.(\\d+)\\.', key)
        if match:
            old = int(match[1])
            if old == 0:
                selected.append(key)
            elif old == 3:
                selected.append(key.replace('.layers.3.', '.layers.1.'))
            elif old == 1 and '.ple.' in key:
                selected.append(key.replace('.layers.1.', '.layers.0.'))
        elif key.startswith('mtp.'):
            match = re.search('\\.experts\\.(\\d+)\\.', key)
            if not match or int(match[1]) < 4:
                selected.append(key)
        elif key.startswith(('model.language_model.', 'lm_head.')):
            selected.append(key)
    selected = [key for key in selected if (match := re.search(r'\.experts\.(\d+)\.', key)) is None or int(match[1]) < 4]
    tensors = {}
    skipped = []
    for raw in selected:
        if raw.endswith(('.suh', '.svh', '.mul1')):
            continue
        filtered = Qwen4ExpTextModel.filter_tensors((raw, lambda: None))
        if filtered is None:
            raise ValueError(f'Unexpected excluded fixture source: {raw}')
        alias = filtered[0]
        bid = int(alias.split('.')[2]) if alias.startswith('model.layers.') else None
        if raw.endswith('.trellis'):
            prefix = alias.removesuffix('.trellis')
            match = re.search('\\.experts\\.(\\d+)\\.(gate|up|down)_proj$', prefix)
            expert = int(match[1]) if match else 0
            if match:
                target = f'blk.{bid}.ffn_{match[2]}_exps.weight'
            elif '.indexer.index_qk_proj' in prefix:
                target = f'blk.{bid}.indexer.q_proj.weight'
            elif prefix.endswith(('.nextn_fc_embedding', '.nextn_fc_hidden')):
                target = f'blk.{bid}.nextn.eh_proj.' + ('embedding' if prefix.endswith('_embedding') else 'hidden') + '.weight'
            else:
                target = mapper.map_tensor_name(prefix + '.weight')
            su = data[target + '.suh'][expert].ravel()
            sv = data[target + '.svh'][expert].ravel()
            rate = values['exl3.' + target + '.bits_x2']
            k = len(su)
            n = len(sv)
            packed = data[target][expert].reshape(k // 16, n // 16, 8 * rate)
            if '.indexer.index_qk_proj' in prefix:
                key = f'blk.{bid}.indexer.k_proj.weight'
                sv = np.concatenate([sv, data[key + '.svh'][0].ravel()])
                packed = np.concatenate([packed, data[key][0].reshape(k // 16, 8, 8 * rate)], axis=1)
            prefix = raw.removesuffix('.trellis')
            for suffix, array in [('.trellis', packed), ('.suh', su), ('.svh', sv)]:
                tensors[prefix + suffix] = torch.from_numpy(array.copy())
            tensors[prefix + '.mul1'] = torch.tensor(-2082680531, dtype=torch.int32)
            continue
        if alias.endswith('.A_log'):
            target = f'blk.{bid}.ssm_a'
        elif alias.endswith('.dt_bias'):
            target = f'blk.{bid}.ssm_dt.bias'
        else:
            try:
                target = mapper.map_tensor_name(alias)
            except ValueError:
                skipped.append(raw)
                continue
        if target not in data:
            skipped.append(raw)
            continue
        array = data[target].copy()
        if '.linear_attn.' in alias:

            def grouped(a, dim):
                a = np.moveaxis(a, dim, 0)
                shape = a.shape
                a = a.reshape(3, 2, -1, *shape[1:]).transpose(1, 0, 2, *range(3, a.ndim + 2)).reshape(shape)
                return np.moveaxis(a, 0, dim)
            if alias.endswith(('.in_proj_a.weight', '.in_proj_b.weight', '.A_log', '.dt_bias')):
                array = grouped(array, 0)
            if '.conv1d.' in alias:
                array = np.concatenate([array[:512], grouped(array[512:], 0)])
        if alias.endswith('.A_log'):
            array = np.log(-array)
        elif alias.endswith('norm.weight') and (not alias.endswith('linear_attn.norm.weight')) or alias.endswith(('.ple.norm_key.weight', '.ple.norm_query.weight', '.ple.norm_conv.weight')):
            array = array - np.float32(1)
        tensors[raw] = torch.from_numpy(array.copy())
    if skipped:
        raise ValueError(f'Unmapped fixture sources: {skipped}')
    for bid in [1, 2]:
        for role in ['q', 'k', 'v']:
            raw = f'model.language_model.layers.1.self_attn.{role}_proj.bias' if bid == 1 else f'mtp.layers.0.self_attn.{role}_proj.bias'
            tensors[raw] = torch.from_numpy(data[f'blk.{bid}.attn_{role}.bias'].copy())
    save_file(tensors, source / 'model.safetensors')
    (source / 'model.safetensors.index.json').write_text(json.dumps({'weight_map': {k: 'model.safetensors' for k in tensors}}))
    ng = {}
    prefix = 'model.language_model.layers.0.ple.ple_embedding.ngram_embedding.'
    for suffix, key in [('layer_multipliers', 'layer_multipliers'), ('head_offsets', 'head_offsets'), ('head_vocab_sizes', 'head_vocab_sizes')]:
        ng[prefix + suffix] = torch.tensor(values['qwen4exp.ple.' + key], dtype=torch.int64)
    ng[prefix + 'head_bias'] = torch.from_numpy(data['per_layer_token_embd.bias'])
    table = data['per_layer_token_embd.weight']
    for i, part in enumerate(np.array_split(table, 2)):
        ng[prefix + f'shard_{i}.trellis'] = torch.from_numpy(part.copy())
    save_file(ng, source / 'ngram_embedding.safetensors', metadata=dict(format='exl3_ngram_trellis', version='1', K='8', codebook='mul1', rows='512', row_dim='64'))
    class FixtureModel(Qwen4ExpTextModel):
        model_arch = gguf.MODEL_ARCH.QWEN4EXP

        def set_vocab(self):
            # Freeze V3 runtime metadata to isolate tensor conversion in the graph comparison.
            for key, field in reader.fields.items():
                if key.startswith(('tokenizer.', 'qwen4exp.')):
                    value = field.contents()
                    ty = field.types
                    if ty[0] == gguf.GGUFValueType.ARRAY:
                        self.gguf_writer.add_key_value(key, value, ty[0], ty[1])
                    else:
                        self.gguf_writer.add_key_value(key, value, ty[0])

    model = FixtureModel(source, gguf.LlamaFileType.MOSTLY_F16, root / 'converted.gguf')
    model.write()
    return model.gguf_writer.format_shard_names(model.fname_out)


def check(model, reference, reference_outputs, build, output):
    output.mkdir(parents=True, exist_ok=False)
    start = time.monotonic()
    state: dict[str, Any] = dict(status='running', reference=str(reference.resolve()), stages=[])

    def save():
        (output / 'status.json').write_text(json.dumps(state, indent=2) + '\n')

    save()
    try:
        paths = prepare(model, reference, output)
        # The V3 synthetic architecture/tokenizer metadata is frozen in prepare().
        state['tensor_comparison'] = compare_tensors(reference, paths)
        save()
        previous = None
        for repeat in range(2):
            numeric = output / f'numeric-{repeat}'
            numeric.mkdir()
            command = [str(build.resolve() / 'bin/test-exl3-model'), '--ngram', str(paths[0]),
                       str(reference.resolve()), str(numeric)]
            with (output / f'graph-{repeat}.log').open('wb') as log:
                result = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT)
            state['stages'].append(dict(command=command, returncode=result.returncode))
            save()
            if result.returncode:
                raise RuntimeError(f'Converted-versus-V3 graph comparison failed: graph-{repeat}.log')
            hashes = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in numeric.glob('*.bin')}
            if len(hashes) != 32 or previous is not None and hashes != previous:
                raise ValueError('Graph outputs are missing or differ across fresh processes')
            # Comparisons 1..6 use the independent V3 model as the reference argument.
            for case in range(1, 7):
                name = f'comparison-{case}.reference.bin'
                if (numeric / name).read_bytes() != (reference_outputs / name).read_bytes():
                    raise ValueError(f'Reference graph output changed from the accepted V3 run: {name}')
            previous = hashes
        state.update(status='pass', elapsed_seconds=time.monotonic() - start, outputs_sha256=previous,
                     tensor_diff_allowlist='literal names in ROUNDOFF_TENSORS and SHARED_INPUT_SCALES',
                     reference_output_cases=6, fresh_processes=2)
        save()
    except BaseException as error:
        state.update(status='fail', error=repr(error))
        save()
        raise


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', type=Path, required=True)
    parser.add_argument('--reference', type=Path, required=True)
    parser.add_argument('--reference-outputs', type=Path, required=True)
    parser.add_argument('--build', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    check(args.model, args.reference, args.reference_outputs, args.build, args.output)
