import json
from pathlib import Path
import runpy
import sys

import numpy as np
import pytest
import torch
from safetensors.torch import save_file

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'gguf-py'))
import gguf
from conversion.qwen4exp import Qwen4ExpTextModel


def checkpoint(path):
    path.mkdir()
    config = dict(architectures=['Qwen4ExpForCausalLM'], model_type='qwen4_exp', num_hidden_layers=2,
                  hidden_size=128, num_attention_heads=2, num_key_value_heads=1, intermediate_size=256,
                  num_experts=12, num_experts_per_tok=2, moe_intermediate_size=256, mtp_num_hidden_layers=1,
                  linear_num_key_heads=2, linear_num_value_heads=6, linear_key_head_dim=128,
                  linear_value_head_dim=128, ple_layer_ids=[1], ple_embed_dim=32, ngram_size=3,
                  heads_per_ngram=1, split_ngram_parts=2, indexer_n_heads=2, indexer_head_dim=128,
                  quantization_config=dict(quant_method='exl3', codebook='mul1'))
    (path / 'config.json').write_text(json.dumps(config))
    tensors = {}

    def projection(prefix, k, n, rate, seed):
        rng = np.random.default_rng(seed)
        tensors[prefix + '.trellis'] = torch.from_numpy(rng.integers(-32768, 32768, (k // 16, n // 16, 8 * rate), dtype=np.int16))
        tensors[prefix + '.suh'] = torch.arange(k, dtype=torch.float16) / 128 + seed
        tensors[prefix + '.svh'] = torch.arange(n, dtype=torch.float16) / 128 + seed
        tensors[prefix + '.mul1'] = torch.tensor(-2082680531, dtype=torch.int32)

    for expert in reversed(range(12)):
        for i, role in enumerate(['gate', 'up', 'down']):
            projection(f'model.language_model.layers.0.mlp.experts.{expert}.{role}_proj',
                       256 if role == 'down' else 128, 128 if role == 'down' else 256, 7, 3 * expert + i)
    projection('model.language_model.layers.1.self_attn.indexer.index_qk_proj', 128, 384, 14, 40)
    projection('mtp.fc_embedding', 128, 128, 10, 41)
    projection('mtp.fc_hidden', 128, 128, 10, 42)
    projection('mtp.layers.0.self_attn.o_proj', 128, 128, 14, 43)
    projection('model.language_model.layers.0.linear_attn.in_proj_qkv', 128, 1280, 14, 44)
    projection('model.language_model.layers.0.linear_attn.in_proj_z', 128, 768, 14, 45)
    projection('model.language_model.layers.0.linear_attn.out_proj', 768, 128, 14, 46)
    tensors['model.language_model.embed_tokens.weight'] = torch.arange(128 * 32, dtype=torch.bfloat16).reshape(32, 128)
    tensors['model.language_model.layers.0.mlp.shared_expert_gate.weight'] = torch.linspace(-.2, .2, 128, dtype=torch.float16).reshape(1, 128)
    tensors['model.language_model.layers.0.linear_attn.conv1d.weight'] = torch.arange(1280 * 4, dtype=torch.float32).reshape(1280, 1, 4) / 128
    tensors['model.language_model.layers.0.linear_attn.in_proj_b.weight'] = torch.arange(6 * 128, dtype=torch.float16).reshape(6, 128) / 128
    values = torch.tensor([.0001, .0003, -.001, .1], dtype=torch.float16).repeat(128).reshape(4, 128)
    for name in ['model.language_model.layers.0.attn_hyper_connection.hc_norm.weight',
                 'model.language_model.layers.0.mlp_hyper_connection.hc_norm.weight',
                 'model.language_model.hyper_connection_mixer.hc_norm.weight',
                 'mtp.hyper_connection_mixer.hc_norm.weight', 'mtp.pre_fc_norm_hidden.weight',
                 'model.language_model.layers.0.ple.norm_conv.weight',
                 'model.language_model.layers.0.ple.norm_key.weight',
                 'model.language_model.layers.0.ple.norm_query.weight']:
        tensors[name] = values.clone()
    tensors['model.language_model.layers.0.linear_attn.A_log'] = torch.linspace(-.7, .3, 6, dtype=torch.float16)
    tensors['model.visual.dummy.weight'] = torch.ones(2, dtype=torch.float16)
    save_file(tensors, path / 'model.safetensors')
    (path / 'model.safetensors.index.json').write_text(json.dumps(dict(weight_map={n: 'model.safetensors' for n in tensors})))
    prefix = 'model.language_model.layers.0.ple.ple_embedding.ngram_embedding.'
    ngram = {
        prefix + 'head_bias': torch.arange(32, dtype=torch.float16).reshape(2, 16),
        prefix + 'layer_multipliers': torch.tensor([2**45 + 1, 2**45 + 3, 2**45 + 5], dtype=torch.int64),
        prefix + 'head_offsets': torch.tensor([0, 4], dtype=torch.int64),
        prefix + 'head_vocab_sizes': torch.tensor([3, 3], dtype=torch.int64),
    }
    for i in [1, 0]:
        ngram[prefix + f'shard_{i}.trellis'] = (torch.arange(36, dtype=torch.int16) + i * 36).reshape(4, 9)
    save_file(ngram, path / 'ngram_embedding.safetensors', metadata=dict(format='exl3_ngram_trellis', version='1',
              codebook='mul1', K='8', rows='8', row_dim='16'))
    return tensors, ngram


def converter(path, output, **kwargs):
    Qwen4ExpTextModel.no_mtp = False
    Qwen4ExpTextModel.mtp_only = False
    return Qwen4ExpTextModel(path, gguf.LlamaFileType.MOSTLY_F16, output, **kwargs)


def write(model):
    model.prepare_tensors()
    # Tokenizer and architecture metadata are exercised separately with a complete model fixture.
    model.set_gguf_parameters = lambda: None
    model._exl3.write()
    return json.loads(model.fname_out.with_suffix('.exl3.json').read_text())


def test_streamed_checkpoint(tmp_path, monkeypatch):
    source, ngram = checkpoint(tmp_path / 'source')
    monkeypatch.setattr('conversion.exl3.CHUNK_BYTES', 512)
    model = converter(tmp_path / 'source', tmp_path / 'model.gguf', split_max_tensors=8)
    report = write(model)
    assert report['status'] == 'verified'
    assert len(report['source_tensors']) == len(source) + len(ngram)
    assert report['source_tensors']['model.visual.dummy.weight']['reason'] == 'vision'
    tensors = {}
    for shard in model.gguf_writer.format_shard_names(model.fname_out):
        reader = gguf.GGUFReader(shard)
        assert reader.alignment == 4096
        for tensor in reader.tensors:
            tensors[tensor.name] = (np.array(tensor.data), tensor.tensor_type)
    for role in ['gate', 'up', 'down']:
        expected = np.stack([source[f'model.language_model.layers.0.mlp.experts.{e}.{role}_proj.trellis'].numpy().ravel() for e in range(12)])[:, None, :]
        np.testing.assert_array_equal(tensors[f'blk.0.ffn_{role}_exps.weight'][0], expected)
    native = 'model.language_model.layers.1.self_attn.indexer.index_qk_proj'
    for role, first, end in [('q', 0, 256), ('k', 256, 384)]:
        name = f'blk.1.indexer.{role}_proj.weight'
        expected = source[native + '.trellis'][:, first // 16:end // 16].numpy().ravel()[None, None, :]
        np.testing.assert_array_equal(tensors[name][0], expected)
        np.testing.assert_array_equal(tensors[name + '.suh'][0].ravel(), source[native + '.suh'].numpy())
        np.testing.assert_array_equal(tensors[name + '.svh'][0].ravel(), source[native + '.svh'][first:end].numpy())
    for role, prefix in [('embedding', 'mtp.fc_embedding'), ('hidden', 'mtp.fc_hidden')]:
        np.testing.assert_array_equal(tensors[f'blk.2.nextn.eh_proj.{role}.weight'][0].ravel(), source[prefix + '.trellis'].numpy().ravel())
    assert 'blk.2.nextn.eh_proj.weight' not in tensors
    assert tensors['token_embd.weight'][1] == gguf.GGMLQuantizationType.BF16
    assert tensors['token_embd.weight'][0].tobytes() == source['model.language_model.embed_tokens.weight'].view(torch.int16).numpy().tobytes()
    assert tensors['blk.0.ffn_gate_inp_shexp.weight'][1] == gguf.GGMLQuantizationType.F16
    for name, expected_name in [('model.language_model.layers.0.attn_hyper_connection.hc_norm.weight', 'blk.0.hc_attn_norm.weight'),
                                ('model.language_model.layers.0.mlp_hyper_connection.hc_norm.weight', 'blk.0.hc_ffn_norm.weight'),
                                ('model.language_model.hyper_connection_mixer.hc_norm.weight', 'output_hc_norm.weight'),
                                ('mtp.hyper_connection_mixer.hc_norm.weight', 'blk.2.nextn.hc_head_norm.weight'),
                                ('mtp.pre_fc_norm_hidden.weight', 'blk.2.nextn.hnorm.weight'),
                                ('model.language_model.layers.0.ple.norm_conv.weight', 'blk.0.ple_norm_conv.weight'),
                                ('model.language_model.layers.0.ple.norm_key.weight', 'blk.0.ple_norm_key.weight'),
                                ('model.language_model.layers.0.ple.norm_query.weight', 'blk.0.ple_norm_query.weight')]:
        expected = source[name].float().numpy() + np.float32(1)
        np.testing.assert_array_equal(tensors[expected_name][0], expected)
        assert np.any(expected != (source[name] + 1).float().numpy())
        assert report['verified_ordinary'][expected_name]['transform'] == 'f32_add_one'
    for native, target in [('in_proj_qkv', 'attn_qkv'), ('in_proj_z', 'attn_gate'), ('out_proj', 'ssm_out')]:
        expected = source[f'model.language_model.layers.0.linear_attn.{native}.trellis'].numpy().ravel()
        np.testing.assert_array_equal(tensors[f'blk.0.{target}.weight'][0].ravel(), expected)
    conv = source['model.language_model.layers.0.linear_attn.conv1d.weight'].numpy().squeeze()
    expected = np.concatenate([conv[:512], conv[512:].reshape(2, 3, 128, 4).transpose(1, 0, 2, 3).reshape(768, 4)])
    np.testing.assert_array_equal(tensors['blk.0.ssm_conv1d.weight'][0], expected)
    beta = source['model.language_model.layers.0.linear_attn.in_proj_b.weight'].numpy()
    np.testing.assert_array_equal(tensors['blk.0.ssm_beta.weight'][0], beta.reshape(2, 3, 128).transpose(1, 0, 2).reshape(6, 128))
    a = source['model.language_model.layers.0.linear_attn.A_log'].float().reshape(2, 3).T.flatten()
    np.testing.assert_array_equal(tensors['blk.0.ssm_a'][0], -torch.exp(a).numpy())
    prefix = 'model.language_model.layers.0.ple.ple_embedding.ngram_embedding.'
    expected = torch.cat([ngram[prefix + f'shard_{i}.trellis'] for i in range(2)]).numpy()
    np.testing.assert_array_equal(tensors['per_layer_token_embd.weight'][0], expected)
    assert model._read_hash_constants('ple_embedding.layer_multipliers') == [2**45 + 1, 2**45 + 3, 2**45 + 5]
    assert report['splits'][-1]['payload_bytes'] == expected.nbytes
    with pytest.raises(ValueError, match='already exists'):
        model._exl3.write()


@pytest.mark.parametrize('failure', ['marker', 'missing_aux', 'missing_expert', 'unknown', 'bad_shards', 'bad_ranges', 'unknown_filtered', 'bad_shard_separator'])
def test_reject_bad_checkpoint(tmp_path, failure):
    source, ngram = checkpoint(tmp_path / 'source')
    path = tmp_path / 'source'
    if failure == 'marker':
        source[next(n for n in source if n.endswith('.mul1'))] = torch.tensor(0, dtype=torch.int32)
    elif failure == 'missing_aux':
        del source[next(n for n in source if n.endswith('.suh'))]
    elif failure == 'missing_expert':
        source = {n: v for n, v in source.items() if '.experts.11.' not in n}
    elif failure == 'unknown':
        source['model.language_model.surprise.weight'] = torch.ones((2, 2))
    elif failure == 'unknown_filtered':
        source['model.audio.surprise.weight'] = torch.ones((2, 2))
    elif failure == 'bad_shards':
        del ngram[next(n for n in ngram if n.endswith('shard_1.trellis'))]
    elif failure == 'bad_ranges':
        ngram[next(n for n in ngram if n.endswith('.head_offsets'))] = torch.tensor([0, 1], dtype=torch.int64)
    elif failure == 'bad_shard_separator':
        name = next(n for n in ngram if n.endswith('shard_0.trellis'))
        ngram[name.replace('.ngram_embedding.shard_', '.ngram_embeddingXshard_')] = ngram.pop(name)
    save_file(source, path / 'model.safetensors')
    (path / 'model.safetensors.index.json').write_text(json.dumps(dict(weight_map={n: 'model.safetensors' for n in source})))
    save_file(ngram, path / 'ngram_embedding.safetensors', metadata=dict(format='exl3_ngram_trellis', version='1',
              codebook='mul1', K='8', rows='8', row_dim='16'))
    with pytest.raises((ValueError, KeyError)):
        converter(path, tmp_path / 'bad.gguf').prepare_tensors()
    assert not list(tmp_path.glob('*.gguf'))


def test_source_changed_and_corrupt_output(tmp_path):
    checkpoint(tmp_path / 'source')
    model = converter(tmp_path / 'source', tmp_path / 'model.gguf')
    write(model)
    paths = model.gguf_writer.format_shard_names(model.fname_out)
    reader = gguf.GGUFReader(paths[0])
    offset = next(t.data_offset for t in reader.tensors if t.name == 'blk.0.ffn_gate_exps.weight')
    del reader
    with paths[0].open('r+b') as f:
        f.seek(offset)
        byte = f.read(1)
        f.seek(offset)
        f.write(bytes([byte[0] ^ 1]))
    with pytest.raises(ValueError, match='differs from source'):
        model._exl3.verify(paths)
    source = model._exl3.payloads[0].parts[0][0]
    with source.file.open('ab') as f:
        f.write(b'changed')
    with pytest.raises(ValueError, match='Source changed'):
        model._exl3.read(source, 0, 2)


def test_dry_run_then_conversion(tmp_path):
    checkpoint(tmp_path / 'source')
    model = converter(tmp_path / 'source', tmp_path / 'model.gguf', dry_run=True)
    model.prepare_tensors()
    model._exl3.write()
    report = json.loads((tmp_path / 'model.exl3-plan.json').read_text())
    assert report['status'] == 'planned'
    assert not list(tmp_path.glob('*.gguf'))
    converted = write(converter(tmp_path / 'source', tmp_path / 'model.gguf'))
    assert [s['file_bytes'] for s in converted['splits']] == [s['file_bytes'] for s in report['splits']]


@pytest.mark.parametrize('dry_run', [False, True])
def test_output_without_extension(tmp_path, dry_run):
    checkpoint(tmp_path / 'source')
    model = converter(tmp_path / 'source', tmp_path / 'model-3.68', dry_run=dry_run)
    model.prepare_tensors()
    model._exl3.write()
    assert model.fname_out.name == 'model-3.68.gguf'
    suffix = '.exl3-plan.json' if dry_run else '.exl3.json'
    assert (tmp_path / ('model-3.68' + suffix)).is_file()
    assert not (tmp_path / ('model-3' + suffix)).exists()


def test_reject_half_rounded_norm(tmp_path):
    source, _ = checkpoint(tmp_path / 'source')
    model = converter(tmp_path / 'source', tmp_path / 'model.gguf')
    write(model)
    paths = model.gguf_writer.format_shard_names(model.fname_out)
    reader = gguf.GGUFReader(paths[0])
    offset = next(t.data_offset for t in reader.tensors if t.name == 'blk.0.hc_attn_norm.weight')
    del reader
    rounded = (source['model.language_model.layers.0.attn_hyper_connection.hc_norm.weight'] + 1).float().numpy()
    with paths[0].open('r+b') as f:
        f.seek(offset)
        f.write(rounded.tobytes())
    with pytest.raises(ValueError, match='Ordinary output differs from FP32 source transform'):
        model._exl3.verify(paths)


@pytest.mark.parametrize('change', ['ple_inverse', 'unlisted', 'missing_bias', 'wrong_shared_scale'])
def test_fixture_diff_gate(tmp_path, change):
    compare = runpy.run_path(str(Path(__file__).resolve().parents[1] / 'scripts/exl3-check-converter.py'))['compare_tensors']
    reference = {
        'blk.0.ple_norm_key.weight': np.zeros((4, 128), dtype=np.float32),
        'token_embd.weight': np.ones((4, 128), dtype=np.float32),
        'blk.1.attn_q.bias': np.zeros(128, dtype=np.float32),
        'blk.1.indexer.q_proj.weight.suh': np.ones((1, 1, 128), dtype=np.float16),
        'blk.1.indexer.k_proj.weight.suh': np.full((1, 1, 128), 2, dtype=np.float16),
    }
    converted = {name: array.copy() for name, array in reference.items()}
    if change == 'ple_inverse':
        converted['blk.0.ple_norm_key.weight'] += np.float32(1)
    elif change == 'unlisted':
        converted['token_embd.weight'][0, 0] = np.nextafter(np.float32(1), np.float32(2))
    elif change == 'missing_bias':
        del converted['blk.1.attn_q.bias']
    else:
        converted['blk.1.indexer.k_proj.weight.suh'][:] = np.float16(1.5)
    for name, arrays in [('reference', reference), ('converted', converted)]:
        writer = gguf.GGUFWriter(tmp_path / (name + '.gguf'), 'qwen4exp')
        for key, array in arrays.items():
            writer.add_tensor(key, array)
        writer.write_header_to_file()
        writer.write_kv_data_to_file()
        writer.write_tensors_to_file()
        writer.close()
    with pytest.raises(ValueError):
        compare(tmp_path / 'reference.gguf', [tmp_path / 'converted.gguf'])
