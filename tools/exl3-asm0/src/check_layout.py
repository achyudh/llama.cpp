#!/usr/bin/env python3
"""CPU instruction/layout model. This is not hardware qualification."""
import argparse
import json
import re
from pathlib import Path
import numpy as np


def simulate(source, m, n, k, wg_x, wg_y, group_log2, x, w, out, written):
    text = source.split('    .p2alignl')[0]
    code = []
    labels = {}
    for line in text.splitlines():
        line = line.strip()
        if line.endswith(':'):
            labels[line[:-1]] = len(code)
        elif line and not line.startswith('.'):
            code.append(line)
    s = [0] * 40
    v = np.zeros((240, 128), dtype=np.uint32)
    v[0] = np.arange(128, dtype=np.uint32)
    active = np.ones(128, dtype=bool)
    lds = np.full(24576 // 4, 0xdeadbeef, dtype=np.uint32)
    args = [m, n, k, group_log2]
    memory = {4: x.view(np.uint32).reshape(-1), 8: w.view(np.uint32).reshape(-1), 12: out.view(np.uint32).reshape(-1)}
    pending = {'load': [], 'ds': []}
    unavailable = set()
    scc = False
    pc = 0
    count = 0

    def reg_range(token):
        nums = re.findall(r'\d+', token)
        return list(range(int(nums[0]), int(nums[-1]) + 1))

    def get(token):
        if token == 'exec_lo':
            return active.copy()
        if token == 'ttmp9':
            return wg_x
        if token == 'ttmp7':
            return wg_y
        if token.startswith('s'):
            return s[int(token[1:])]
        if token.startswith('v'):
            r = int(token[1:])
            assert r not in unavailable, (code[pc-1], 'read before memory wait', r)
            return v[r].copy()
        return int(token, 0)

    def put(token, value):
        nonlocal active
        if token == 'exec_lo':
            active = np.array(value, dtype=bool)
        elif token.startswith('s'):
            s[int(token[1:])] = value & 0xffffffff if isinstance(value, int) else value.copy()
        else:
            v[int(token[1:]), active] = np.asarray(value, dtype=np.uint32) if np.ndim(value) == 0 else np.asarray(value, dtype=np.uint32)[active]

    def wait(kind, left):
        q = pending[kind]
        while len(q) > left:
            for r in q.pop(0):
                unavailable.remove(r)

    while pc < len(code):
        instruction = code[pc]
        pc += 1
        count += 1
        assert count < 100000, 'nonterminating scalar control'
        op, _, operands = instruction.partition(' ')
        a = [z.strip() for z in operands.split(',')]
        if op == 's_endpgm':
            break
        if op.startswith('s_load'):
            if a[0] == 's[16:19]':
                s[16:20] = args
            continue
        if op in ('s_wait_kmcnt', 's_wait_alu', 's_wait_storecnt'):
            continue
        if op.startswith('s_wait_'):
            kind = 'load' if 'loadcnt' in op else 'ds'
            wait(kind, int(a[0], 0))
            continue
        if op.startswith('s_barrier'):
            assert not pending['ds'], 'barrier before LDS completion'
            continue
        if op.startswith('s_cbranch'):
            if scc == op.endswith('scc1'):
                pc = labels[a[0]]
            continue
        if op == 's_branch':
            pc = labels[a[0]]
            continue
        if op.startswith('s_cmp_'):
            lhs, rhs = map(get, a)
            how = op.split('_')[2]
            scc = {'ge': lhs >= rhs, 'lt': lhs < rhs, 'le': lhs <= rhs, 'eq': lhs == rhs}[how]
            continue
        if op == 's_cselect_b32':
            put(a[0], get(a[1]) if scc else get(a[2]))
            continue
        if op.startswith('v_cmpx_'):
            assert op == 'v_cmpx_gt_u32'
            active &= get(a[0]) > get(a[1])
            continue
        if op.startswith('buffer_'):
            match = re.fullmatch(r'(v\[\d+:\d+\]|v\d+), (v\d+), s\[(\d+):\d+\], (s\d+) offen offset:(\d+)', operands)
            assert match, instruction
            dst, addr, bank, offset, imm = match.groups()
            bank = int(bank)
            offsets = get(addr).astype(np.uint64) + get(offset) + int(imm)
            regs = reg_range(dst)
            assert np.all(offsets[active] % 4 == 0)
            mem = memory[bank]
            for j, r in enumerate(regs):
                pos = offsets // 4 + j
                valid = active & (pos < len(mem))
                if op.startswith('buffer_load'):
                    assert r not in unavailable, ('overlapping load destinations', r)
                    v[r, active] = 0
                    v[r, valid] = mem[pos[valid]]
                else:
                    assert r not in unavailable
                    assert valid.sum() == active.sum(), 'masked output outside allocation'
                    np.add.at(written, pos[valid], 1)
                    mem[pos[valid]] = v[r, valid]
            if op.startswith('buffer_load'):
                pending['load'].append(regs)
                unavailable.update(regs)
            continue
        if op.startswith('ds_'):
            match = re.fullmatch(r'(v\[\d+:\d+\]|v\d+), (v\[\d+:\d+\]|v\d+) offset:(\d+)', operands)
            assert match, instruction
            left, right, off = match.groups()
            load = op.startswith('ds_load')
            regs = reg_range(left if load else right)
            pos = (get(right if load else left).astype(np.uint64) + int(off)) // 4
            assert np.all((pos[active] + 3) < len(lds))
            for j, r in enumerate(regs):
                assert r not in unavailable, (instruction, r)
                if load:
                    v[r, active] = lds[pos[active] + j]
                else:
                    lds[pos[active] + j] = v[r, active]
            pending['ds'].append(regs if load else [])
            if load:
                unavailable.update(regs)
            continue
        if op.startswith('v_wmma'):
            dest, ar, br, cr = map(reg_range, a)
            assert dest == cr and len(dest) == 8 and len(ar) == len(br) == 4
            assert not unavailable.intersection(ar + br + cr), instruction
            assert active.all(), 'WMMA under partial exec'
            for wave in range(4):
                lanes = slice(wave*32, (wave+1)*32)
                aa = v[ar, lanes].T.copy().view(np.float16).reshape(32, 8)
                bb = v[br, lanes].T.copy().view(np.float16).reshape(32, 8)
                aa = np.concatenate([aa[:16], aa[16:]], axis=1).astype(np.float32)
                bb = np.concatenate([bb[:16], bb[16:]], axis=1).astype(np.float32)
                old = v[cr, lanes].copy().view(np.float32)
                cc = np.concatenate([old[:, :16], old[:, 16:]], axis=0)
                result = aa @ bb.T + cc
                packed = np.concatenate([result[:8], result[8:]], axis=1)
                v[dest, lanes] = packed.view(np.uint32)
            continue
        if op in ('s_mov_b32', 'v_mov_b32'):
            put(a[0], get(a[1]))
            continue
        x0, y0 = get(a[1]), get(a[2])
        if 'lshlrev' in op or 'lshrrev' in op:
            x0, y0 = y0, x0
        if 'lshl' in op:
            value = x0 << y0
        elif 'lshr' in op:
            value = x0 >> y0
        elif '_and_' in op:
            value = x0 & y0
        elif '_mul_' in op:
            value = x0 * y0
        elif '_add_' in op:
            value = x0 + y0
        elif '_sub_' in op:
            value = x0 - y0
        else:
            raise AssertionError(instruction)
        put(a[0], value)
    assert not unavailable, 'outstanding register loads on exit'
    return count


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--build', type=Path, required=True)
    args = p.parse_args()
    rng = np.random.default_rng(3101)
    cases = [(1, 1, 128), (15, 17, 128), (17, 15, 256), (127, 129, 128),
             (129, 127, 384), (129, 129, 128), (257, 17, 256), (1025, 1, 128)]
    reports = []
    for variant in ('strict', 'pipeline'):
        source = (args.build / f'{variant}.s').read_text()
        for m, n, k in cases:
            x = rng.integers(-8, 9, (m, k)).astype(np.float16)
            w = rng.integers(-8, 9, (n, k)).astype(np.float16)
            out = np.full((m, n), np.nan, dtype=np.float32)
            written = np.zeros(m*n, dtype=np.int32)
            gm = min(8, 1 << ((m+127)//128).bit_length()-1)
            total = 0
            for gy in range(((m+127)//128 + gm-1)//gm):
                for gx in range(((n+127)//128)*gm):
                    total += simulate(source, m, n, k, gx, gy, gm.bit_length()-1, x, w, out, written)
            expected = x.astype(np.float64) @ w.astype(np.float64).T
            assert np.array_equal(out, expected), (variant, (m, n, k), np.nanmax(abs(out-expected)))
            assert np.all(written == 1), (variant, 'missing or duplicate stores')
            reports.append({'variant': variant, 'shape': [m, n, k], 'instructions_modeled': total,
                            'exact_integer_reference': True, 'single_output_owner': True})
    result = {'status': 'passed', 'device_qualified': False, 'numpy_version': np.__version__,
              'scope': 'emitted instruction control/address/layout model with memory wait scoreboard; lockstep workgroup model, no ISA latency or GPU ABI proof',
              'cases': reports}
    (args.build / 'cpu-layout.json').write_text(json.dumps(result, indent=2) + '\n')
    print(f'CPU instruction/layout checks: {len(reports)} passed; device qualification pending')


if __name__ == '__main__':
    main()
