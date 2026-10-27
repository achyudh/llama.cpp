#!/usr/bin/env python3
"""Check lossless GGUF imports and CPU graph dispatch against qualified projection hashes."""

import argparse
import ctypes
import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess
import sys

import numpy as np
import torch

scripts = Path(__file__).resolve().parent
sys.path.insert(0, str(scripts.parent / 'gguf-py'))
from gguf import GGUFReader, GGUFWriter

spec = importlib.util.spec_from_file_location('exl3_import', scripts / 'exl3-import.py')
importer = importlib.util.module_from_spec(spec)
spec.loader.exec_module(importer)

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--model', type=Path, required=True)
parser.add_argument('--build', type=Path, required=True)
parser.add_argument('--qualification', type=Path, required=True)
parser.add_argument('--output', type=Path, required=True)
args = parser.parse_args()
args.output.mkdir(exist_ok=False)
torch.set_num_threads(4)
proof = json.loads(args.qualification.read_text())
assert proof['status'] == 'pass'


class Matrix(ctypes.Structure):
    _fields_ = [('ne0', ctypes.c_int64), ('ne1', ctypes.c_int64), ('bits_x2', ctypes.c_int),
                ('packed', ctypes.c_void_p), ('packed_bytes', ctypes.c_size_t),
                ('suh', ctypes.c_void_p), ('suh_elements', ctypes.c_size_t),
                ('svh', ctypes.c_void_p), ('svh_elements', ctypes.c_size_t)]


library = ctypes.CDLL(str((args.build / 'bin/libggml-base.so').resolve()))
reference = library.ggml_exl3_mul_mat_ref
reference.argtypes = [ctypes.POINTER(Matrix), ctypes.c_int64, ctypes.c_void_p, ctypes.c_size_t, ctypes.c_void_p, ctypes.c_size_t]
reference.restype = ctypes.c_bool
state = dict(status='running', banks=[], cases=[], qualification_sha256=hashlib.sha256(args.qualification.read_bytes()).hexdigest(),
    source_sha256={str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in [Path(__file__), scripts/'exl3-import.py']},
    binary_sha256={name: hashlib.sha256((args.build/'bin'/name).read_bytes()).hexdigest() for name in ['test-exl3','libggml-base.so']},
    scope='Lossless projection-bank import and CPU graph dispatch; not a full model or serving benchmark')


def save():
    (args.output/'status.json').write_text(json.dumps(state, indent=2)+'\n')


def write_inputs(path, x, ids=None):
    w = GGUFWriter(path, 'exl3_inputs')
    w.add_tensor('input', x)
    if ids is not None:
        w.add_tensor('ids', ids)
    w.write_header_to_file(); w.write_kv_data_to_file(); w.write_tensors_to_file(progress=False); w.close()


def execute(bundle, inputs, output, log):
    run = subprocess.run([str(args.build/'bin/test-exl3'), '--bundle', str(bundle), str(inputs), str(output)],
                         stdout=subprocess.PIPE, stderr=subprocess.STDOUT, timeout=120)
    log.write_bytes(run.stdout)
    return run


def check(prefixes, first=None, count=None):
    number = len(state['banks'])
    folder = args.output / f'bank-{number:02}'
    folder.mkdir()
    path = folder/'projection.gguf'
    item = importer.export(args.model, prefixes, path, first or 0, count)
    assert item['index_sha256'] == proof['index_sha256']
    state['banks'].append(item)
    save()
    reader = GGUFReader(path)
    tensors = {t.name: t.data for t in reader.tensors}
    p, su, sv = tensors['exl3.packed'], tensors['exl3.suh'], tensors['exl3.svh']
    e, k, n = item['experts'], item['k'], item['n']
    old = next(v for v in proof['projections'] if v['prefix'] == prefixes[0] and v['output_first'] == first)
    assert old['k'] == k and old['n'] == n and old['bits_x2'] == item['bits_x2']
    assert old['packed_sha256'] == hashlib.sha256(p[0].tobytes()).hexdigest()
    assert old['suh_sha256'] == hashlib.sha256(su[0].tobytes()).hexdigest()
    assert old['svh_sha256'] == hashlib.sha256(sv[0].tobytes()).hexdigest()
    base = torch.randn((5, k), generator=torch.Generator().manual_seed(1234)).half().float().numpy()
    for rows in (1, 4, 5):
        for slots in ((1, 4) if e > 1 else (0,)):
            case_dir = folder/f'rows-{rows}-slots-{slots}'
            case_dir.mkdir()
            ids = np.tile(np.array([0, 1, 0, 1], dtype=np.int32), (rows, 1)) if e > 1 else None
            x = ((base[:rows, None, :] + np.arange(slots, dtype=np.float32)[None, :, None] / 32)
                 .astype(np.float16).astype(np.float32)) if slots else base[:rows].copy()
            write_inputs(case_dir/'inputs.gguf', x, ids)
            completed = execute(path, case_dir/'inputs.gguf', case_dir/'output.bin', case_dir/'graph.log')
            assert completed.returncode == 0, completed.stdout.decode()
            used = 4 if ids is not None else 1
            actual = np.fromfile(case_dir/'output.bin', dtype='<f4').reshape(rows, used, n)
            expected = np.empty_like(actual)
            for expert in range(e):
                positions = [(r,s) for r in range(rows) for s in range(used) if ids is None or ids[r,s] == expert]
                gathered = np.stack([x[r, s % slots] if slots else x[r] for r,s in positions])
                out = np.empty((len(positions), n), dtype=np.float32)
                view = Matrix(k,n,item['bits_x2'],p[expert].ctypes.data,p[expert].nbytes,
                              su[expert].ctypes.data,k,sv[expert].ctypes.data,n)
                assert reference(ctypes.byref(view),len(positions),gathered.ctypes.data,gathered.size,out.ctypes.data,out.size)
                for pos, y in zip(positions, out): expected[pos] = y
            assert np.array_equal(actual, expected)
            prior = next(c for c in old['cases'] if c['rows']==rows and c['amplitude']==1.0)
            digest = hashlib.sha256(actual[:,0].copy().tobytes()).hexdigest()
            assert digest == prior['output_sha256']['reference'], 'Qualified projection hash changed'
            state['cases'].append(dict(bank=number,rows=rows,input_slots=slots,exact_reference=True,
                                       qualified_projection_hash_exact=True,output_sha256=hashlib.sha256(actual.tobytes()).hexdigest()))
            save()
    # Refuse a second import without changing the original file.
    before = hashlib.sha256(path.read_bytes()).hexdigest()
    try:
        importer.export(args.model,prefixes,path,first or 0,count)
    except ValueError:
        pass
    else:
        raise AssertionError('Existing output was not rejected')
    assert hashlib.sha256(path.read_bytes()).hexdigest()==before
    if e > 1:
        invalid = np.array([[-1, 0, 1, 0]],dtype=np.int32)
        write_inputs(folder/'invalid-inputs.gguf',base[:1,None,:].copy(),invalid)
        rejected = execute(path,folder/'invalid-inputs.gguf',folder/'invalid-output.bin',folder/'invalid.log')
        assert rejected.returncode==1 and not (folder/'invalid-output.bin').exists()
    print(number,prefixes[0],k,n,item['bits_x2']/2,'pass',flush=True)


save()
try:
    for layer in (0,4):
        for projection in ('gate_proj','up_proj','down_proj'):
            check([f'model.language_model.layers.{layer}.mlp.experts.{expert}.{projection}' for expert in (0,511)])
    check(['model.language_model.layers.0.mlp.shared_expert.gate_proj'])
    # The qualified attention slice is the final 256 output columns.
    q = next(p for p in proof['projections'] if p['prefix'].endswith('linear_attn.in_proj_qkv'))
    check([q['prefix']],q['output_first'],256)
    check(['mtp.fc_hidden'])
    for offset in (0,124160,248192): check(['lm_head'],offset,128)
    try:
        importer.export(args.model,['model.language_model.layers.0.mlp.experts.0.gate_proj',
                                    'model.language_model.layers.4.mlp.experts.0.gate_proj'],args.output/'mixed-rate.gguf')
    except ValueError:
        pass
    else:
        raise AssertionError('Mixed-rate bank was not rejected')
    assert not (args.output/'mixed-rate.gguf').exists()
    for name,digest in state['binary_sha256'].items():
        assert hashlib.sha256((args.build/'bin'/name).read_bytes()).hexdigest()==digest, 'Binary changed during qualification'
    for name,digest in state['source_sha256'].items():
        assert hashlib.sha256(Path(name).read_bytes()).hexdigest()==digest, 'Source changed during qualification'
    state['status']='pass'
except Exception as exc:
    state.update(status='failed',error=type(exc).__name__+': '+str(exc))
    raise
finally:
    save()
