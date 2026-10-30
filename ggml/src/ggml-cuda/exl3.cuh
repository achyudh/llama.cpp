#pragma once

#include "common.cuh"
#include "ggml-exl3.h"

bool ggml_cuda_supports_exl3(const ggml_tensor * op, int cc);
void ggml_cuda_op_mul_mat_exl3(ggml_backend_cuda_context & ctx, ggml_tensor * dst);

#ifdef GGML_USE_HIP
void ggml_cuda_exl3_aux_scatter(ggml_backend_cuda_context & ctx, const ggml_tensor * scratch, const ggml_exl3_aux_batch & batch);

// GGML_EXL3_F32_ROWS decodes a dense bank once for all rows; routed banks use GGML_EXL3_F32_FRAGMENT.
enum ggml_cuda_exl3_mode { GGML_EXL3_F32, GGML_EXL3_F16_DOT, GGML_EXL3_F32_FRAGMENT, GGML_EXL3_F32_ROWS };
enum ggml_cuda_exl3_stage { GGML_EXL3_INPUT, GGML_EXL3_MATRIX, GGML_EXL3_OUTPUT, GGML_EXL3_ALL };

struct ggml_cuda_exl3_args {
    const uint32_t * packed;
    const half * suh;
    const half * svh;
    const float * x;
    const int32_t * ids;
    float * dst;
    void * xh;
    float * yh;
    int k, n, experts, assignments, used, input_slots, bits_x2;
    const int32_t * perm; // input transform only: row i of xh takes assignment perm[i]
    // cache slots: expert e reads its packed weights, suh and svh at the addresses slots[e*slot_stride + 0..2]
    const int64_t * slots = nullptr;
    int slot_stride = 0;
};

// Internal launch interface shared with the projection benchmark. Scratch is supplied by the caller.
GGML_BACKEND_API void ggml_cuda_exl3_launch(const ggml_cuda_exl3_args & args, ggml_cuda_exl3_mode mode,
                                           ggml_cuda_exl3_stage stage, cudaStream_t stream);

// 1024 + the byte sum of the hashed state is below 2048, so its FP16 bits are 0x6400 | sum; the FP32 FMA of the
// reference is exact for these operands, so one FP16 FMA rounds to the same value.
static __device__ __forceinline__ uint32_t ggml_cuda_exl3_mul1_bits(uint16_t state) {
    const uint32_t p = uint32_t(state) * 0x83dcd12du;
    return __builtin_amdgcn_udot4(p, 0x01010101u, 0x6400u, false);
}

static __device__ __forceinline__ half ggml_cuda_exl3_mul1(uint16_t state) {
    return __hfma(__ushort_as_half(uint16_t(ggml_cuda_exl3_mul1_bits(state))), __ushort_as_half(0x1eee), __ushort_as_half(0xc931));
}

static __device__ __forceinline__ half2 ggml_cuda_exl3_mul1x2(uint16_t s0, uint16_t s1) {
    const uint32_t bits = ggml_cuda_exl3_mul1_bits(s0) | (ggml_cuda_exl3_mul1_bits(s1) << 16);
    const half2 scale = __half2half2(__ushort_as_half(0x1eee)), bias = __half2half2(__ushort_as_half(0xc931));
    return __hfma2(*(const half2 *) &bits, scale, bias);
}

template<int bits_x2>
static __device__ __forceinline__ half ggml_cuda_exl3_weight(const uint32_t * tile, int i) {
    constexpr int words = 4 * bits_x2;
    constexpr int ring = words * 32;
    const int start = (((i + 1) * bits_x2 / 2) + ring - 16) % ring;
    const int word = start / 32;
    const uint64_t pair = (uint64_t(tile[word]) << 32) | tile[(word + 1) % words];
    return ggml_cuda_exl3_mul1(uint16_t(pair >> (48 - (start & 31))));
}

// Four adjacent K3.5/K4 state windows fit in 32 bits. Adapted from ExLlamaV3 dq8_half/dq8.
// Copyright (c) 2025 Turboderp. See licenses/LICENSE-exllamav3.
// The lane's eight weights as pairs w[j] = (2j, 2j + 1).
template<int bits_x2>
static __device__ __forceinline__ void ggml_cuda_exl3_fragment2(const uint32_t * tile, int lane, half2 w[4]) {
    if constexpr (bits_x2 == 7 || bits_x2 == 8) {
        constexpr int words = 4 * bits_x2;
        constexpr int span = bits_x2 == 7 ? 27 : 28;
#pragma unroll
        for (int group = 0; group < 2; ++group) {
            const int end = (lane * 8 + group * 4 + 4) * bits_x2 / 2;
            const int start = end - span;
            const int lo = (start < 0 ? start + words * 32 : start) / 32;
            const int hi = (end - 1) / 32;
            const int shift = (hi + 1) * 32 - end;
            const uint32_t window = uint32_t(((uint64_t(tile[lo]) << 32) | tile[hi]) >> shift);
#pragma unroll
            for (int j = 0; j < 4; j += 2) {
                const int d0 = (4 * bits_x2 / 2) - ((j + 1) * bits_x2 / 2), d1 = (4 * bits_x2 / 2) - ((j + 2) * bits_x2 / 2);
                w[group * 2 + j / 2] = ggml_cuda_exl3_mul1x2(uint16_t(window >> d0), uint16_t(window >> d1));
            }
        }
    } else {
#pragma unroll
        for (int j = 0; j < 8; j += 2) {
            w[j / 2] = __halves2half2(ggml_cuda_exl3_weight<bits_x2>(tile, lane * 8 + j), ggml_cuda_exl3_weight<bits_x2>(tile, lane * 8 + j + 1));
        }
    }
}

template<int bits_x2>
static __device__ __forceinline__ void ggml_cuda_exl3_fragment(const uint32_t * tile, int lane, half w[8]) {
    half2 w2[4];
    ggml_cuda_exl3_fragment2<bits_x2>(tile, lane, w2);
#pragma unroll
    for (int j = 0; j < 4; ++j) {
        w[2 * j] = __low2half(w2[j]);
        w[2 * j + 1] = __high2half(w2[j]);
    }
}
#endif
