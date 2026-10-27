#include "exl3.cuh"
#include "ggml-exl3.h"

bool ggml_cuda_supports_exl3(const ggml_tensor * op, int cc) {
#ifdef GGML_USE_HIP
    if (cc < GGML_CUDA_CC_RDNA4 || cc >= GGML_CUDA_CC_OFFSET_AMD + 0x1300 || !ggml_exl3_op_supported(op)) {
        return false;
    }
    const int bits = ggml_get_op_params_i32(op, 0);
    const int64_t rows = op->src[4] ? op->ne[2] : op->ne[1];
    const int64_t assignments = op->ne[1] * op->ne[2];
    // Bound the initial decode workspace and grid. Larger batches stay on the reference backend.
    return (bits == 7 || bits == 8 || bits == 10 || bits == 12 || bits == 14) && rows <= 8 &&
        op->src[2]->ne[0] <= 131072 && op->ne[0] <= 131072 && op->src[0]->ne[2] <= INT_MAX &&
        assignments <= 512 && assignments * (op->src[2]->ne[0] + op->ne[0]) <= 16 * 1024 * 1024;
#else
    GGML_UNUSED(op); GGML_UNUSED(cc);
    return false;
#endif
}

#ifdef GGML_USE_HIP
static __device__ __forceinline__ int exl3_expert(const ggml_cuda_exl3_args & a, int assignment) {
    return a.ids ? a.ids[assignment] : 0;
}

template<typename T, bool input>
static __global__ void exl3_transform(ggml_cuda_exl3_args a) {
    // The format uses groups of 32 lanes. Dispatch is restricted to RDNA4 wave32 devices.
    const int assignment = blockIdx.y;
    const int expert = exl3_expert(a, assignment);
    const bool valid = expert >= 0 && expert < a.experts;
    const int width = input ? a.k : a.n;
    const int lane = threadIdx.x;
    const int base = blockIdx.x * 128;
    const int xrow = a.ids ? (assignment / a.used) * a.input_slots + (assignment % a.used) % a.input_slots : assignment;
    float v[4];
#pragma unroll
    for (int j = 0; j < 4; ++j) {
        const int col = base + j * 32 + lane;
        v[j] = !valid ? NAN : input ? a.x[size_t(xrow) * width + col] * __half2float(a.suh[size_t(expert) * width + col])
                                  : a.yh[size_t(assignment) * width + col];
    }
#pragma unroll
    for (int h = 1; h < 32; h *= 2) {
#pragma unroll
        for (int j = 0; j < 4; ++j) {
            const float other = __shfl_xor(v[j], h, 32);
            v[j] = (lane & h) ? other - v[j] : v[j] + other;
        }
    }
#pragma unroll
    for (int h = 1; h < 4; h *= 2) {
#pragma unroll
        for (int j = 0; j < 4; j += 2 * h) {
#pragma unroll
            for (int z = 0; z < h; ++z) {
                const float x = v[j + z], y = v[j + z + h];
                v[j + z] = x + y;
                v[j + z + h] = x - y;
            }
        }
    }
#pragma unroll
    for (int j = 0; j < 4; ++j) {
        const int col = base + j * 32 + lane;
        const float value = v[j] * 0.08838834764831844f;
        if (input) {
            ((T *) a.xh)[size_t(assignment) * width + col] = (T) value;
        } else {
            a.dst[size_t(assignment) * width + col] = valid ? value * __half2float(a.svh[size_t(expert) * width + col]) : NAN;
        }
    }
}

template<int bits_x2>
static __global__ void exl3_matrix_f32(ggml_cuda_exl3_args a) {
    __shared__ float partial[8][16];
    const int col = threadIdx.x % 16, part = threadIdx.x / 16;
    const int assignment = blockIdx.y, expert = exl3_expert(a, assignment);
    float sum = 0;
    if (expert >= 0 && expert < a.experts) {
        const size_t bank_words = size_t(a.k / 16) * (a.n / 16) * (4 * bits_x2);
        for (int kt = part; kt < a.k / 16; kt += 8) {
            const uint32_t * tile = a.packed + expert * bank_words + (size_t(kt) * (a.n / 16) + blockIdx.x) * (4 * bits_x2);
#pragma unroll
            for (int r = 0; r < 16; ++r) {
                const int lane = (col & 7) * 4 + ((r & 7) / 2);
                const int j = (r & 1) | ((r >> 3) << 1) | ((col >> 3) << 2);
                const float w = __half2float(ggml_cuda_exl3_weight<bits_x2>(tile, lane * 8 + j));
                sum = fmaf(((const float *) a.xh)[size_t(assignment) * a.k + kt * 16 + r], w, sum);
            }
        }
    }
    partial[part][col] = sum;
    __syncthreads();
    if (part == 0) {
        for (int j = 1; j < 8; ++j) { sum += partial[j][col]; }
        a.yh[size_t(assignment) * a.n + blockIdx.x * 16 + col] = sum;
    }
}

// The fragment layout and quad reduction follow ExLlamaV3's MIT-licensed RDNA direct GEMV core.
// Copyright (c) 2025 Turboderp. See licenses/LICENSE-exllamav3.
template<int bits_x2, bool fp16>
static __global__ void exl3_matrix_fragment(ggml_cuda_exl3_args a) {
    __shared__ float partial[4][16];
    const int lane = threadIdx.x % 32, warp = threadIdx.x / 32;
    const int assignment = blockIdx.y, expert = exl3_expert(a, assignment);
    float lo = 0, hi = 0;
    if (expert >= 0 && expert < a.experts) {
        const size_t bank_words = size_t(a.k / 16) * (a.n / 16) * (4 * bits_x2);
        for (int kt = warp; kt < a.k / 16; kt += 4) {
            const uint32_t * tile = a.packed + expert * bank_words + (size_t(kt) * (a.n / 16) + blockIdx.x) * (4 * bits_x2);
            half w[8];
            ggml_cuda_exl3_fragment<bits_x2>(tile, lane, w);
#if defined(__HIP_DEVICE_COMPILE__) && defined(__gfx1201__)
            if constexpr (fp16) {
                const half2 * x = (const half2 *) ((const half *) a.xh + size_t(assignment) * a.k + kt * 16);
                const half2 x01 = x[lane & 3], x89 = x[(lane & 3) + 4];
                typedef _Float16 half_pair __attribute__((ext_vector_type(2)));
                lo = __builtin_amdgcn_fdot2(__builtin_bit_cast(half_pair, x01),
                                          __builtin_bit_cast(half_pair, __halves2half2(w[0], w[1])), lo, false);
                lo = __builtin_amdgcn_fdot2(__builtin_bit_cast(half_pair, x89),
                                          __builtin_bit_cast(half_pair, __halves2half2(w[2], w[3])), lo, false);
                hi = __builtin_amdgcn_fdot2(__builtin_bit_cast(half_pair, x01),
                                          __builtin_bit_cast(half_pair, __halves2half2(w[4], w[5])), hi, false);
                hi = __builtin_amdgcn_fdot2(__builtin_bit_cast(half_pair, x89),
                                          __builtin_bit_cast(half_pair, __halves2half2(w[6], w[7])), hi, false);
            } else
#endif
            {
#pragma unroll
                for (int j = 0; j < 4; ++j) {
                    const size_t offset = size_t(assignment) * a.k + kt * 16 + (lane & 3) * 2 + (j & 1) + (j / 2) * 8;
                    const float v = fp16 ? __half2float(((const half *) a.xh)[offset]) : ((const float *) a.xh)[offset];
                    lo = fmaf(v, __half2float(w[j]), lo);
                    hi = fmaf(v, __half2float(w[j + 4]), hi);
                }
            }
        }
    }
    lo += __shfl_xor(lo, 1, 32); lo += __shfl_xor(lo, 2, 32);
    hi += __shfl_xor(hi, 1, 32); hi += __shfl_xor(hi, 2, 32);
    const float vlo = __shfl(lo, (lane & 7) * 4, 32), vhi = __shfl(hi, (lane & 7) * 4, 32);
    if (lane < 16) { partial[warp][lane] = (lane & 8) ? vhi : vlo; }
    __syncthreads();
    if (warp == 0 && lane < 16) {
        float sum = partial[0][lane];
        for (int w = 1; w < 4; ++w) { sum += partial[w][lane]; }
        a.yh[size_t(assignment) * a.n + blockIdx.x * 16 + lane] = sum;
    }
}

template<int bits_x2>
static void exl3_matrix_launch(const ggml_cuda_exl3_args & a, ggml_cuda_exl3_mode mode, cudaStream_t stream) {
    const dim3 grid(a.n / 16, a.assignments);
    if (mode == GGML_EXL3_F16_DOT) { exl3_matrix_fragment<bits_x2, true><<<grid, 128, 0, stream>>>(a); }
    else if (mode == GGML_EXL3_F32_FRAGMENT) { exl3_matrix_fragment<bits_x2, false><<<grid, 128, 0, stream>>>(a); }
    else { exl3_matrix_f32<bits_x2><<<grid, 128, 0, stream>>>(a); }
}

void ggml_cuda_exl3_launch(const ggml_cuda_exl3_args & a, ggml_cuda_exl3_mode mode,
                          ggml_cuda_exl3_stage stage, cudaStream_t stream) {
    if (stage == GGML_EXL3_INPUT || stage == GGML_EXL3_ALL) {
        const dim3 grid(a.k / 128, a.assignments);
        if (mode == GGML_EXL3_F16_DOT) { exl3_transform<half, true><<<grid, 32, 0, stream>>>(a); }
        else { exl3_transform<float, true><<<grid, 32, 0, stream>>>(a); }
    }
    if (stage == GGML_EXL3_MATRIX || stage == GGML_EXL3_ALL) {
        switch (a.bits_x2) {
            case 7:  exl3_matrix_launch<7 >(a, mode, stream); break;
            case 8:  exl3_matrix_launch<8 >(a, mode, stream); break;
            case 10: exl3_matrix_launch<10>(a, mode, stream); break;
            case 12: exl3_matrix_launch<12>(a, mode, stream); break;
            case 14: exl3_matrix_launch<14>(a, mode, stream); break;
            default: GGML_ABORT("Unsupported EXL3 rate");
        }
    }
    if (stage == GGML_EXL3_OUTPUT || stage == GGML_EXL3_ALL) {
        exl3_transform<float, false><<<dim3(a.n / 128, a.assignments), 32, 0, stream>>>(a);
    }
    CUDA_CHECK(cudaGetLastError());
}
#endif

void ggml_cuda_op_mul_mat_exl3(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
#ifdef GGML_USE_HIP
    GGML_ASSERT(ggml_cuda_supports_exl3(dst, ggml_cuda_info().devices[ctx.device].cc));
    const int k = dst->src[2]->ne[0], n = dst->ne[0], assignments = dst->ne[1] * dst->ne[2];
    ggml_cuda_pool_alloc<float> xh(ctx.pool(), size_t(assignments) * k);
    ggml_cuda_pool_alloc<float> yh(ctx.pool(), size_t(assignments) * n);
    const ggml_cuda_exl3_args args = {
        (const uint32_t *) dst->src[0]->data, (const half *) dst->src[2]->data, (const half *) dst->src[3]->data,
        (const float *) dst->src[1]->data, dst->src[4] ? (const int32_t *) dst->src[4]->data : nullptr,
        (float *) dst->data, xh.get(), yh.get(), k, n, int(dst->src[0]->ne[2]), assignments,
        dst->src[4] ? int(dst->ne[1]) : 1, int(dst->src[1]->ne[1]), ggml_get_op_params_i32(dst, 0)};
    ggml_cuda_exl3_launch(args, GGML_EXL3_F32_FRAGMENT, GGML_EXL3_ALL, ctx.stream());
#else
    GGML_UNUSED(ctx); GGML_UNUSED(dst);
    GGML_ABORT("EXL3 projections require the ROCm backend");
#endif
}
