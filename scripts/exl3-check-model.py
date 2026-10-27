#!/usr/bin/env python3
"""Reproduce the synthetic Qwen4Exp EXL3 graph/cache qualification on CPU."""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import re
import subprocess
import time


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def check(build, output):
    build = build.resolve()
    output.mkdir(parents=True, exist_ok=False)
    source = Path(__file__).resolve().parent
    identities = [source / 'exl3-check-model.py', source / 'exl3-make-model-fixture.py']
    identities += [build / 'bin' / name for name in ['test-llama-archs', 'test-exl3-model',
        'libllama.so', 'libggml.so', 'libggml-base.so', 'libggml-cpu.so']]
    before = {str(p): sha(p) for p in identities}
    state = dict(status='running', scope='Synthetic CPU full-graph/cache qualification, not ROCm or published-model serving',
                 identities_sha256=before, stages=[])
    def save():
        (output / 'status.json').write_text(json.dumps(state, indent=2)+'\n')
    def run(args, log):
        start = time.monotonic()
        with (output / log).open('wb') as handle:
            result = subprocess.run([str(a) for a in args], stdout=handle, stderr=subprocess.STDOUT)
        state['stages'].append(dict(command=[str(a) for a in args], log=log, returncode=result.returncode,
                                    elapsed_seconds=time.monotonic()-start))
        save()
        if result.returncode: raise RuntimeError('Failed stage: '+log)
    save()
    try:
        base = output / 'base'; base.mkdir()
        run([build/'bin/test-llama-archs', '-a', '^qwen4exp$', '-o', base], 'base.log')
        spec = importlib.util.spec_from_file_location('fixture', source/'exl3-make-model-fixture.py')
        helper = importlib.util.module_from_spec(spec); spec.loader.exec_module(helper)
        helper.make(base/'qwen4exp-moe.gguf', build/'bin/libggml-base.so', output/'pair')
        command = [build/'bin/test-exl3-model', output/'pair/compressed.gguf', output/'pair/reconstructed.gguf']
        run(command, 'graph-cache.log')
        run(command, 'graph-cache-repeat.log')
        pattern = r'full graph: exact=(\d) relative_rms=([\deE.+-]+) max_abs=([\deE.+-]+)'
        first = re.findall(pattern, (output/'graph-cache.log').read_text())
        second = re.findall(pattern, (output/'graph-cache-repeat.log').read_text())
        if first != second or len(first) != 14: raise RuntimeError('Fresh-process numerical results differ or cases are missing')
        if any(sha(p) != before[str(p)] for p in identities): raise RuntimeError('Source/binary changed during qualification')
        state.update(status='pass', comparisons=[dict(exact=bool(int(e)), relative_rms=float(r), max_abs=float(a)) for e,r,a in first],
                     fresh_process_measurements_equal=True,
                     fixtures_sha256={str(p.relative_to(output)): sha(p) for p in (output/'pair').glob('*.gguf')})
        save()
    except BaseException as e:
        state.update(status='failed', error=repr(e)); save(); raise


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--build', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args(); check(args.build, args.output)
