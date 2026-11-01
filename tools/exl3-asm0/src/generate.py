#!/usr/bin/env python3
"""Emit the isolated ASM0 gfx1201 GEMM candidates; never execute GPU code."""
import argparse
import json
from pathlib import Path

STAGE = 128 * 48 * 2
VARIANTS = ("strict", "pipeline")


def emit_kernel(variant, abi):
    name = "asm0_" + variant
    lines = [f'.amdgcn_target "amdgcn-amd-amdhsa--gfx1201"',
             '.amdhsa_code_object_version 5', '.text', f'.protected {name}',
             f'.globl {name}', '.p2align 8', f'.type {name},@function', f'{name}:']

    def e(s):
        lines.extend('    ' + x for x in s.splitlines())

    def label(s):
        lines.append(f'.L{name}_{s}:')

    def barrier():
        e('s_wait_dscnt 0\ns_barrier_signal -1\ns_barrier_wait -1')

    def aload(p, scalar):
        r = 200 + p * 4
        e(f'buffer_load_b128 v[{r}:{r+3}], v217, s[4:7], {scalar} offen offset:{p*16}')

    def bload(p, bank, scalar):
        kk, j = divmod(p, 4)
        r = bank + p * 4
        e(f'buffer_load_b128 v[{r}:{r+3}], v{220+j}, s[8:11], {scalar} offen offset:{kk*32}')

    def astore(p, stage):
        r = 200 + p * 4
        e(f'ds_store_b128 v218, v[{r}:{r+3}] offset:{stage*STAGE+p*16}')

    def aread(frag, bank, stage):
        kk, i = divmod(frag, 4)
        e(f'ds_load_b128 v[{bank}:{bank+3}], v219 offset:{stage*STAGE+i*16*96+kk*32}')

    # gfx12 workgroup IDs are in TTMP9 (x) and TTMP7 (y); captured clang ABI probe is the reference.
    e('s_load_b64 s[4:5], s[0:1], 0\ns_load_b64 s[8:9], s[0:1], 8\ns_load_b64 s[12:13], s[0:1], 16\ns_load_b128 s[16:19], s[0:1], 24')
    e('v_mov_b32 v216, v0\ns_wait_kmcnt 0')
    e('s_lshl_b32 s30, 1, s19\ns_sub_u32 s31, s30, 1\ns_and_b32 s22, ttmp9, s31\ns_lshl_b32 s30, ttmp7, s19\ns_add_u32 s22, s22, s30\ns_lshr_b32 s23, ttmp9, s19\ns_lshl_b32 s24, s22, 7\ns_lshl_b32 s25, s23, 7')
    e(f's_cmp_ge_u32 s24, s16\ns_cbranch_scc1 .L{name}_end\ns_cmp_ge_u32 s25, s17\ns_cbranch_scc1 .L{name}_end')
    e('s_lshl_b32 s20, s18, 1\ns_lshl_b32 s21, s17, 2\ns_mul_i32 s6, s16, s20\ns_mul_i32 s10, s17, s20\ns_mul_i32 s14, s16, s21')
    e('s_mov_b32 s7, 0x31004000\ns_mov_b32 s11, 0x31004000\ns_mov_b32 s15, 0x31004000')
    e('s_and_b32 s26, s22, 31\ns_lshl_b32 s26, s26, 8')
    label('stagger')
    e(f's_cmp_lt_u32 s26, s20\ns_cbranch_scc1 .L{name}_stagger_done\ns_sub_u32 s26, s26, s20\ns_branch .L{name}_stagger')
    label('stagger_done')
    e('s_lshr_b32 s28, s18, 5\ns_lshl_b32 s29, s20, 4')
    e('v_add_u32 v217, s24, v216\nv_mul_lo_u32 v217, v217, s20\nv_mul_lo_u32 v218, 96, v216')
    # One row of 32 halves per workitem stages A; B fragments bypass LDS.
    e('v_lshrrev_b32 v227, 6, v216\nv_lshlrev_b32 v227, 6, v227\nv_and_b32 v228, 15, v216\nv_add_u32 v219, v227, v228\nv_mul_lo_u32 v219, 96, v219\nv_and_b32 v229, 16, v216\nv_add_u32 v219, v219, v229')
    e('v_and_b32 v230, 32, v216\nv_lshlrev_b32 v230, 1, v230\nv_add_u32 v220, s25, v230\nv_add_u32 v220, v220, v228\nv_mul_lo_u32 v220, v220, s20\nv_add_u32 v220, v220, v229')
    for j in range(1, 4):
        e(f'v_add_u32 v{220+j}, s29, v{219+j}')
    e('v_lshrrev_b32 v229, 1, v229\nv_add_u32 v232, s24, v227\nv_add_u32 v232, v232, v229\nv_add_u32 v233, s25, v230\nv_add_u32 v233, v233, v228\nv_mul_lo_u32 v234, v232, s21\nv_lshlrev_b32 v230, 2, v233\nv_add_u32 v234, v234, v230')
    for r in range(128):
        e(f'v_mov_b32 v{r}, 0')
    for p in range(4):
        aload(p, 's26')
    for p in range(8):
        bload(p, 128, 's26')
    e('s_wait_loadcnt 0')
    for p in range(4):
        astore(p, 0)
    barrier()
    aread(0, 192, 0)

    def step(cur):
        cur_b, next_b = (128, 160) if cur == 0 else (160, 128)
        # Only VMEM consumes the changing scalar offset; immutable setup pairs stay unchanged.
        e('s_add_u32 s27, s26, 64\ns_cmp_ge_u32 s27, s20\ns_cselect_b32 s30, s20, 0\ns_sub_u32 s27, s27, s30')
        if variant == 'strict':
            for p in range(4):
                aload(p, 's27')
            for p in range(8):
                bload(p, next_b, 's27')
        for frag in range(8):
            kk, i = divmod(frag, 4)
            bank = 192 + (frag % 2) * 4
            e('s_wait_dscnt 0')
            if frag == 7:
                if variant == 'strict':
                    e('s_wait_loadcnt 0')
                    for p in range(4):
                        astore(p, cur ^ 1)
                barrier()
            for j in range(4):
                op = frag * 4 + j
                a = (i * 4 + j) * 8
                b = cur_b + (kk * 4 + j) * 4
                e(f'v_wmma_f32_16x16x16_f16 v[{a}:{a+7}], v[{bank}:{bank+3}], v[{b}:{b+3}], v[{a}:{a+7}]')
                if j == 0:
                    if frag < 7:
                        aread(frag + 1, 192 + ((frag + 1) % 2) * 4, cur)
                    else:
                        aread(0, 192, cur ^ 1)
                if variant == 'pipeline':
                    if op < 4:
                        aload(op, 's27')
                    elif op < 12:
                        bload(op - 4, next_b, 's27')
                    if op == 16:
                        e('s_wait_loadcnt 8')
                    if 16 <= op < 20:
                        astore(op - 16, cur ^ 1)
        e(f's_wait_loadcnt 0\ns_mov_b32 s26, s27\ns_sub_u32 s28, s28, 1\ns_wait_alu 0xfffe\ns_cmp_eq_u32 s28, 0\ns_cbranch_scc1 .L{name}_epilogue')

    label('loop')
    step(0)
    step(1)
    e(f's_branch .L{name}_loop')
    label('epilogue')
    e('s_wait_dscnt 0')
    e('s_add_u32 s30, s24, 128\ns_cmp_le_u32 s30, s16\ns_cselect_b32 s34, 1, 0\ns_add_u32 s30, s25, 128\ns_cmp_le_u32 s30, s17\ns_cselect_b32 s35, 1, 0\ns_and_b32 s34, s34, s35')
    e(f's_cmp_eq_u32 s34, 1\ns_cbranch_scc0 .L{name}_tail')
    for i in range(4):
        for t in range(8):
            e(f's_mul_i32 s32, s21, {i*16+t}\ns_wait_alu 0xfffe')
            for j in range(4):
                r = (i * 4 + j) * 8 + t
                e(f'buffer_store_b32 v{r}, v234, s[12:15], s32 offen offset:{j*64}')
    e(f's_branch .L{name}_finish')
    label('tail')
    e('s_mov_b32 s31, exec_lo')
    for i in range(4):
        for t in range(8):
            e(f'v_add_u32 v236, {i*16+t}, v232\nv_cmpx_gt_u32 s16, v236\ns_mov_b32 s33, exec_lo\ns_mul_i32 s32, s21, {i*16+t}\ns_wait_alu 0xfffe')
            for j in range(4):
                r = (i * 4 + j) * 8 + t
                e(f'v_add_u32 v237, {j*16}, v233\nv_cmpx_gt_u32 s17, v237\nbuffer_store_b32 v{r}, v234, s[12:15], s32 offen offset:{j*64}\ns_mov_b32 exec_lo, s33')
            e('s_mov_b32 exec_lo, s31')
    label('finish')
    e('s_wait_storecnt 0')
    label('end')
    e('s_endpgm')
    e(f'.size {name}, .-{name}')
    # Keep AMD's required 3 cache lines of padding, and use the compiler's captured ABI descriptor.
    e('.p2alignl 7, 3214868480\n.fill 96, 4, 3214868480')
    descriptor = abi[abi.index('\t.section\t.rodata'):abi.index('\t.text\n.Lfunc_end0')]
    descriptor = descriptor.replace('asm0_abi_probe', name)
    descriptor = descriptor.replace('fixed_size 0', f'fixed_size {2*STAGE}', 1)
    descriptor = descriptor.replace('next_free_vgpr 2', 'next_free_vgpr 240')
    descriptor = descriptor.replace('next_free_sgpr 16', 'next_free_sgpr 40')
    descriptor = descriptor.replace('reserve_vcc 0', 'reserve_vcc 1')
    lines.append(descriptor)
    meta = abi[abi.index('\t.amdgpu_metadata'):]
    meta = meta.replace('asm0_abi_probe', name).replace('fixed_size: 0', f'fixed_size: {2*STAGE}', 1)
    meta = meta.replace('workgroup_size: 1024', 'workgroup_size: 128')
    meta = meta.replace('sgpr_count:     16', 'sgpr_count:     40')
    meta = meta.replace('vgpr_count:     2', 'vgpr_count:     240')
    lines.append(meta)
    return '\n'.join(lines) + '\n'


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--abi', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    args = p.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    for variant in VARIANTS:
        (args.out / f'{variant}.s').write_text(emit_kernel(variant, args.abi.read_text()))
    (args.out / 'design.json').write_text(json.dumps({
        'target': 'gfx1201', 'device_qualified': False,
        'tile': [128, 128, 32], 'threads': 128, 'wave_size': 32,
        'vgpr': 240, 'sgpr': 40, 'lds_bytes': 2*STAGE, 'scratch_bytes': 0,
        'a_lds_stride_halves': 48, 'b_direct_vgpr': True,
        'variants': list(VARIANTS), 'precision': 'f16 x f16 -> f32',
        'k_order': '(row_block % 32)*128 % k, then ascending wrapped steps of 32',
        'abi': {'kernarg_bytes': 40, 'group_log2_offset': 36, 'workgroup_x': 'ttmp9', 'workgroup_y': 'ttmp7'},
    }, indent=2) + '\n')


if __name__ == '__main__':
    main()
