#!/usr/bin/env python3
"""Reproduce the CPU-only ASM0 artifacts inside radiance-env."""
import difflib
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
BUILD = ROOT / 'build'
REF = ROOT / 'reference'
LLVM = Path('/nix/store/zbpn9mmalvssw588sbmy02zahnhc9bid-llvm-22.0.0-rocm/bin')
CLANG = Path(os.environ['HIP_CLANG_PATH']) / 'clang++'
LINK = CLANG.parent / 'ld.lld'
CLR = Path('/nix/store/1k1g3f4iickxf269fqv0ji6awf9v5vbp-clr-7.2.3/lib')
PYTHON = '/home/achyudh/projects/scratch/exl3-eval-2026-10-26/venv/bin/python'
commands = []


def run(argv, output=None, env=None):
    argv = list(map(str, argv))
    commands.append(argv)
    result = subprocess.run(argv, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, env=env)
    if result.stderr:
        print(result.stderr, file=sys.stderr, end='')
    result.check_returncode()
    if output:
        Path(output).write_text(result.stdout)
    return result.stdout


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


BUILD.mkdir(exist_ok=True)
run([CLANG, '-x', 'hip', '--cuda-device-only', '--offload-arch=gfx1201',
     '--rocm-device-lib-path=' + os.environ['HIP_DEVICE_LIB_PATH'], '-mcode-object-version=5',
     '-O2', '-S', REF / 'abi_probe.cpp', '-o', BUILD / 'abi_probe.s'])
run([sys.executable, ROOT / 'src/generate.py', '--abi', BUILD / 'abi_probe.s', '--out', BUILD])
original = (REF / 'gemm-wmma.original.cuh').read_text()
assert original.count('#include "common.cuh"') == 1
adapted = original.replace('#include "common.cuh"', '// common.cuh types and macros supplied by the standalone harness.')
(REF / 'gemm-wmma.standalone.cuh').write_text(adapted)
(REF / 'standalone.patch').write_text(''.join(difflib.unified_diff(original.splitlines(True), adapted.splitlines(True),
                                                             fromfile='gemm-wmma.original.cuh', tofile='gemm-wmma.standalone.cuh')))
resources = {}
for variant in ('strict', 'pipeline'):
    run([LLVM / 'llvm-mc', '-triple', 'amdgcn-amd-amdhsa', '-mcpu=gfx1201', '-filetype=obj',
         BUILD / f'{variant}.s', '-o', BUILD / f'{variant}.o'])
    run([LINK, '-shared', BUILD / f'{variant}.o', '-o', BUILD / f'{variant}.co'])
    notes = run([LLVM / 'llvm-readobj', '--notes', '--symbols', BUILD / f'{variant}.co'], BUILD / f'{variant}.metadata.txt')
    isa = run([LLVM / 'llvm-objdump', '-d', '--mcpu=gfx1201', BUILD / f'{variant}.co'], BUILD / f'{variant}.isa.txt')
    for field in ('.group_segment_fixed_size: 24576', '.private_segment_fixed_size: 0', '.vgpr_count:     240',
                  '.sgpr_count:     40', '.wavefront_size: 32', '.kernarg_segment_size: 40'):
        assert field in notes, (variant, field)
    assert '<unknown>' not in isa and 'scratch_' not in isa
    assert isa.count('v_wmma_f32_16x16x16_f16 ') == 64
    assert isa.count('s_barrier_signal ') == 3 and isa.count('s_barrier_wait ') == 3
    resources[variant] = {'vgpr_declared': 240, 'sgpr_declared': 40, 'lds_bytes': 24576, 'scratch_bytes': 0,
                          'static_wmma': 64, 'barriers_per_k32_step': 1, 'codeobject_bytes': (BUILD/f'{variant}.co').stat().st_size}
run([CLANG, '-x', 'hip', '--offload-arch=gfx1201', '--rocm-device-lib-path=' + os.environ['HIP_DEVICE_LIB_PATH'],
     '--no-offload-compress', '-O3', '-std=c++17', '-Wall', '-Wextra', ROOT / 'src/bench.cpp',
     '-L' + str(CLR), '-Wl,-rpath,' + str(CLR), '-lamdhip64', '-o', BUILD / 'bench'])
run([CLANG, '-x', 'hip', '--cuda-device-only', '--offload-arch=gfx1201',
     '--rocm-device-lib-path=' + os.environ['HIP_DEVICE_LIB_PATH'], '-O3', '-std=c++17', '-S',
     ROOT / 'src/bench.cpp', '-o', BUILD / 'baseline.s'])
baseline = (BUILD/'baseline.s').read_text()
resources['frozen_hip'] = {}
for field in ('group_segment_fixed_size', 'next_free_vgpr', 'next_free_sgpr', 'private_segment_fixed_size'):
    values = re.findall(r'\.amdhsa_' + field + r'\s+(\d+)', baseline)
    assert len(values) == 1, (field, values)
    resources['frozen_hip'][field] = int(values[0])
run([BUILD / 'bench', '--help'], BUILD / 'help.txt')
run([BUILD / 'bench', '--describe'], BUILD / 'describe.txt')
rejections = []
for shape in ([1, 1, 32], [65536, 65536, 128]):
    command = [str(BUILD/'bench'), '--check', '--codeobject', 'not-loaded', '--symbol', 'asm0_pipeline',
               '--shape', *map(str, shape)]
    attempt = subprocess.run(command, text=True, capture_output=True)
    assert attempt.returncode == 1 and ('expected positive' in attempt.stderr or 'offset overflow' in attempt.stderr)
    rejections.append({'command': command, 'exit_code': attempt.returncode, 'stderr': attempt.stderr})
(BUILD/'argument-rejections.json').write_text(json.dumps(rejections, indent=2)+'\n')
env = dict(os.environ)
env['LD_LIBRARY_PATH'] = '/nix/store/604gsr59rj7dzd0nrhp143rpvf7gyiaz-gcc-15.3.0-lib/lib:/nix/store/0zs3jf215fw4mmg3yw4arglkpxp7qrjy-zlib-1.3.2/lib'
env['OPENBLAS_NUM_THREADS'] = '1'
run([PYTHON, ROOT / 'src/check_layout.py', '--build', BUILD], BUILD / 'cpu-layout.log', env)
run([PYTHON, ROOT / 'src/check_mutations.py', '--build', BUILD], BUILD / 'cpu-mutations.log', env)
tools = {}
for tool in (CLANG, LINK, LLVM/'llvm-mc', LLVM/'llvm-objdump', LLVM/'llvm-readobj'):
    tools[str(tool)] = {'resolved': str(tool.resolve()), 'sha256': sha(tool), 'version': run([tool, '--version']).strip()}
manifest = {'status': 'cpu-prepared', 'device_qualified': False, 'gpu_executed': False,
            'build_python': {'executable': sys.executable, 'version': sys.version},
            'scope': 'standalone core; no inference integration or fresh timing',
            'tools': tools, 'resources': resources, 'commands': commands,
            'artifacts': {str(p.relative_to(ROOT)): sha(p) for sub in ('src', 'reference', 'build')
                          for p in sorted((ROOT/sub).rglob('*')) if p.is_file() and p.name != 'manifest.json' and '__pycache__' not in str(p)}}
(BUILD/'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
print('ASM0 CPU preparation passed; no GPU execution; see build/manifest.json')
