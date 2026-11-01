#!/usr/bin/env python3
"""Explicit future GPU qualification stages. Never invoked by the CPU build."""
import argparse
import datetime
import errno
import fcntl
import hashlib
import json
from pathlib import Path
import subprocess

ROOT = Path(__file__).resolve().parents[1]
BUILD = ROOT / 'build'
CAMPAIGN = Path('/home/achyudh/projects/scratch/exl3-eval-2026-10-26/results/measure-20261028/campaign.lock')
ENV = Path('/home/achyudh/projects/scratch/radiance-eval-20261031/scripts/radiance-env')
p = argparse.ArgumentParser(description=__doc__)
p.add_argument('stage', choices=['probe', 'small', 'bench'])
args = p.parse_args()
manifest_path = BUILD / 'manifest.json'
manifest_hash = hashlib.sha256(manifest_path.read_bytes()).hexdigest()
manifest = json.loads(manifest_path.read_text())
for name, digest in manifest['artifacts'].items():
    assert hashlib.sha256((ROOT/name).read_bytes()).hexdigest() == digest, f'changed artifact: {name}'
required = {'small': 'probe', 'bench': 'small'}.get(args.stage)
if required:
    marker = json.loads((ROOT/'results'/f'{required}-passed.json').read_text())
    assert marker['manifest_sha256'] == manifest_hash, 'prerequisite belongs to another build'

with CAMPAIGN.open('a') as lock:
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    device = Path('/sys/class/drm/card0/device')
    try:
        busy = int((device/'gpu_busy_percent').read_text())
    except OSError as error:
        if error.errno != errno.EBUSY or (device/'power/runtime_status').read_text().strip() != 'suspended':
            raise
        busy = 0
    assert busy <= 10 and int((device/'mem_info_vram_used').read_text()) < 2**30, 'GPU is occupied'
    for proc in Path('/proc').iterdir():
        if not proc.name.isdecimal():
            continue
        try:
            assert (proc/'comm').read_text().strip() not in {
                'llama-server', 'llama-bench', 'test-exl3-model', 'radiance', 'radiance-legacy',
                'rad-kbench', 'r4d_selftest', 'moe-tp1', 'bench'}, f'GPU client at {proc}'
        except OSError:
            pass
    stamp = datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%SZ')
    result = ROOT/'results'/f'{args.stage}-{stamp}'
    result.mkdir(parents=True)
    environment = {'manifest_sha256': manifest_hash, 'stage': args.stage,
                   'note': 'No power, voltage or clock settings changed by runner.'}
    for name in ('gpu_busy_percent', 'mem_info_vram_used', 'pp_dpm_sclk', 'pp_dpm_mclk', 'pp_dpm_pcie'):
        try:
            environment[name] = (device/name).read_text()
        except OSError as error:
            environment[name] = str(error)
    (result/'environment.json').write_text(json.dumps(environment, indent=2)+'\n')
    # Strict first; any failure/timeout stops the stage and prevents further launches.
    for variant in ('strict', 'pipeline'):
        command = [str(ENV), 'timeout', '600' if args.stage == 'bench' else '90',
                   str(BUILD/'bench'), '--codeobject', str(BUILD/f'{variant}.co'), '--symbol', 'asm0_'+variant]
        if args.stage == 'probe':
            command += ['--check', '--shape', '16', '16', '128']
        elif args.stage == 'small':
            command += ['--check', '--graph']
        else:
            command += ['--bench', '--banks', '8', '--samples', str(result/f'{variant}.csv')]
        (result/f'{variant}-command.json').write_text(json.dumps(command, indent=2)+'\n')
        with (result/f'{variant}.log').open('w') as log:
            exit_code = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT).returncode
        (result/f'{variant}-exit.json').write_text(json.dumps({'exit_code': exit_code})+'\n')
        if exit_code:
            raise SystemExit(f'{variant} failed ({exit_code}); inspect {result}; no further GPU work launched')
    marker = {'manifest_sha256': manifest_hash, 'stage': args.stage, 'result': str(result), 'status': 'passed'}
    (ROOT/'results'/f'{args.stage}-passed.json').write_text(json.dumps(marker, indent=2)+'\n')
    print(json.dumps(marker, indent=2))
