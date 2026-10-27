#!/usr/bin/env python3
"""Qualify the production EXL3 GPU expert cache using public weights and frozen synthetic graphs."""

import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import re
import subprocess
import time


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--model', type=Path, required=True)
parser.add_argument('--build', type=Path, required=True)
parser.add_argument('--fixture', type=Path, required=True, help='Frozen V3 synthetic model fixture directory')
parser.add_argument('--output', type=Path, required=True)
parser.add_argument('--banks', type=Path, help='Reuse previously prepared actual-weight banks')
parser.add_argument('--prepare-only', action='store_true')
parser.add_argument('--diagnostic', action='store_true', help='One actual-weight process only; not final qualification')
args = parser.parse_args()
args.output.mkdir(exist_ok=False)
args.build = args.build.resolve()
scripts = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location('exl3_import', scripts/'exl3-import.py')
importer = importlib.util.module_from_spec(spec)
spec.loader.exec_module(importer)
sources = [Path(__file__), scripts/'exl3-import.py'] + [scripts.parent/p for p in
    ['tests/test-exl3-model.cpp', 'src/llama-moe-cache.cpp', 'src/llama-model.cpp', 'src/llama-context.cpp',
     'ggml/src/ggml-backend.cpp', 'ggml/src/ggml-cuda/exl3.cu', 'ggml/src/ggml-cuda/exl3.cuh']]
identities = {str(p): sha(p) for p in sources}
for name in ['test-exl3-model', 'libllama.so', 'libggml-base.so', 'libggml-cpu.so', 'libggml-hip.so', 'libggml.so']:
    p = args.build/'bin'/name
    identities[str(p)] = sha(p)
state = dict(status='running', identities_sha256=identities, boot_id=Path('/proc/sys/kernel/random/boot_id').read_text().strip(),
    index_sha256=sha(args.model/'model.safetensors.index.json'), cmake_sha256=sha(args.build/'CMakeCache.txt'),
    fixture_sha256={name: sha(args.fixture/name) for name in ['compressed.gguf', 'reconstructed.gguf']},
    scope='Bounded production cache/scheduler qualification, not published-model serving or full-source registration fit',
    runs=[])


def save():
    (args.output/'status.json').write_text(json.dumps(state, indent=2)+'\n')


def fields(line):
    return {key: int(value) for key, value in re.findall(r'(\w+)=(-?\d+)', line)}


def execute(mode, folder, registered):
    folder.mkdir()
    env = os.environ.copy()
    for key in ['LD_LIBRARY_PATH', 'ROCBLAS_USE_HIPBLASLT', 'GGML_CUDA_DISABLE_GRAPHS']:
        env.pop(key, None)
    env.update(ROCR_VISIBLE_DEVICES='GPU-06370343B04276BF', LLAMA_MOE_PERF='1', GGML_CUDA_GRAPH_DEBUG='1',
               GGML_OP_OFFLOAD_MIN_BATCH='1', GPU_PINNED_MIN_XFER_SIZE='0',
               GGML_CUDA_REGISTER_HOST='1' if registered else '0', LLAMA_MOE_HOST_REGISTER_MIB='4096' if registered else '0')
    executable = args.build/'bin/test-exl3-model'
    if mode == 'synthetic':
        command = [executable, '--rocm', args.fixture/'compressed.gguf', args.fixture/'reconstructed.gguf', folder]
    else:
        command = [executable, '--rocm-banks', args.fixture/'compressed.gguf', bank_dir, folder]
    started = time.monotonic()
    run = subprocess.run([str(a) for a in command], stdout=subprocess.PIPE, stderr=subprocess.STDOUT, env=env, timeout=600)
    log = run.stdout.decode(errors='replace')
    (folder/'run.log').write_text(log)
    item = dict(mode=mode, registered=registered, command=[str(a) for a in command],
                environment={k: env[k] for k in ['ROCR_VISIBLE_DEVICES', 'LLAMA_MOE_PERF', 'GGML_CUDA_GRAPH_DEBUG',
                  'GGML_OP_OFFLOAD_MIN_BATCH', 'GPU_PINNED_MIN_XFER_SIZE', 'GGML_CUDA_REGISTER_HOST', 'LLAMA_MOE_HOST_REGISTER_MIB']},
                returncode=run.returncode, elapsed_seconds=time.monotonic()-started)
    state['runs'].append(item)
    save()
    assert run.returncode == 0, 'GPU test failed; see '+str(folder/'run.log')
    assert 'CUDA graph warmup complete' in log, 'No GGML graph capture observed'
    perf = [fields(line) for line in log.splitlines() if line.startswith('moe_perf:')]
    if mode == 'synthetic':
        comparisons = re.findall(r'full graph: exact=(\d) relative_rms=([\deE.+-]+) max_abs=([\deE.+-]+)', log)
        assert len(comparisons) == 22
        assert any(v['d2d_bytes'] > 0 and v['total'] == 1 for v in perf), 'Production bypass never reused cached data'
        assert any(v['bypass_bytes'] > 0 for v in perf), 'Production bypass was not exercised'
        item['comparisons'] = comparisons
    else:
        cases = [fields(line) for line in log.splitlines() if line.startswith('exl3_cache_case:')]
        copies = [fields(line) for line in log.splitlines() if line.startswith('exl3_cache_d2d:')]
        assert len(cases) == 108 and len(copies) == 3
        actual = {v['task']: v for v in perf if v['layer'] == -1 and v['total'] == 0}
        for case in cases:
            counter = actual[case['id']]
            for key in ['hits', 'misses', 'evictions']:
                assert case[key] == counter[key], (case, counter)
            assert counter['fill_bytes'] == case['packed_bytes']+case['aux_bytes']
            assert counter['fill_calls'] == case['packed_calls']+case['aux_calls']
            assert counter['id_calls'] == 1 and counter['id_bytes'] == case['rows']*10*4
            assert counter['padding_bytes'] == counter['bypass_bytes'] == counter['d2d_bytes'] == 0
        for case in copies:
            counter = actual[case['id']]
            assert case['bytes'] == counter['d2d_bytes'] and case['calls'] == counter['d2d_calls']
            assert counter['fill_bytes'] == counter['id_bytes'] == 0
        assert sum(c['evictions'] for c in cases) > 0
        assert any(c['layer'] == 0 and c['misses'] == 0 and c['hits'] > 0 for c in cases)
        assert any(c['misses'] and c['hits'] for c in cases)
        registrations = [fields(line) for line in log.splitlines() if line.startswith('moe_host_registration:')]
        assert len(registrations) == 1
        reg = registrations[0]
        if registered:
            assert reg['completed'] == reg['attempted'] == 27 and reg['failed'] == reg['skipped'] == 0
            assert reg['registered_bytes'] == sum(b['bytes'] for b in manifest)
        else:
            assert reg['registered_bytes'] == reg['completed'] == 0
        item.update(cases=cases, telemetry=actual, d2d=copies, registration=reg)
    item['output_sha256'] = {p.name: sha(p) for p in sorted(folder.glob('*.bin'))}
    assert len(item['output_sha256']) == (44 if mode == 'synthetic' else 432), 'Missing full numerical output files'
    save()
    return item


save()
try:
    if args.banks:
        bank_dir = args.banks.resolve()
        manifest = json.loads((bank_dir/'manifest.json').read_text())
        for b in manifest:
            assert b['file_sha256'] == sha(bank_dir/b['file']) and b['index_sha256'] == state['index_sha256']
    else:
        bank_dir = args.output/'banks'
        bank_dir.mkdir()
        manifest = []
        for layer, original in enumerate((0, 4, 5)):
            for p, projection in enumerate(('gate_proj', 'up_proj', 'down_proj')):
                name = f'bank-{layer}-{p}.gguf'
                result = importer.export(args.model, [f'model.language_model.layers.{original}.mlp.experts.{e}.{projection}' for e in range(512)], bank_dir/name)
                result.update(file=name, file_sha256=sha(bank_dir/name))
                result['bytes'] = result['experts']*(result['k']*result['n']*result['bits_x2']//16+2*(result['k']+result['n']))
                manifest.append(result)
                (bank_dir/'manifest.json').write_text(json.dumps(manifest, indent=2)+'\n')
    state['banks'] = manifest
    save()
    if args.prepare_only:
        state['status'] = 'prepared'
    else:
        if not args.diagnostic:
            first = execute('synthetic', args.output/'synthetic-0', False)
            second = execute('synthetic', args.output/'synthetic-1', False)
            assert first['output_sha256'] == second['output_sha256']
        baseline = None
        for number, registered in enumerate((False,) if args.diagnostic else (False, True, True, False)):
            result = execute('actual', args.output/f'actual-{number}', registered)
            if baseline is None:
                baseline = result['output_sha256']
            assert result['output_sha256'] == baseline, 'Fresh process/registration changed numerical outputs'
        for path, digest in identities.items():
            assert sha(Path(path)) == digest, 'Source/binary changed during qualification'
        state['status'] = 'diagnostic-pass' if args.diagnostic else 'pass'
except Exception as exc:
    state.update(status='failed', error=repr(exc))
    raise
finally:
    save()
