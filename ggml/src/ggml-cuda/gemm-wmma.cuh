#pragma once

// RDNA4 FP16 WMMA GEMM shared by the EXL3 projections (with their output rotation) and F16 matrix multiplication.

#include "common.cuh"

#ifdef GGML_USE_HIP
// 128-point Hadamard transform of v[j] = element j * 32 + lane, without normalization.
static __device__ __forceinline__ void ggml_cuda_hadamard128(float v[4], int lane) {
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
}

// Y = X W^T for FP16 X [m][k] and FP16 W [n_cols][k], k contiguous, with FP32 accumulation on RDNA4 WMMA. With
// rotate (EXL3: X holds rotated rows, W a decoded column chunk) the epilogue applies the output rotation and svh scale
// of each 128-column group: dst[row][n0 + col]; without it, it stores dst[row][col] for col < n. A block computes
// BM x 128 in 4 waves of BM / 2 x 64; both operands go through double-buffered LDS, the next K step loaded into
// registers during the current step's math, one barrier per step of 32.
// On gfx1201 this runs at about 100 TFLOPS for 4096 x 12288 x 2560, hipBLASLt's FP16 GEMM at about 145. Thread traces
// put the difference in instruction scheduling (LDS fragment and global load waits), not in tiling or occupancy.
// The routed EXL3 GEMM (exl3.cu) shares the tiling and epilogue but decodes its W tiles itself.
#if defined(__HIP_DEVICE_COMPILE__) && (defined(__gfx1200__) || defined(__gfx1201__))
#define GGML_CUDA_GEMM_WMMA 1
#endif
typedef _Float16 gemm_wmma_half8 __attribute__((ext_vector_type(8)));
typedef float gemm_wmma_float8 __attribute__((ext_vector_type(8)));
typedef float gemm_wmma_float4 __attribute__((ext_vector_type(4)));

#ifdef GGML_CUDA_GEMM_WMMA
// Raw buffer loads take a scalar base and a 32-bit per-lane offset, so a fragment address costs one VGPR.
static __device__ __forceinline__ __amdgpu_buffer_rsrc_t gemm_wmma_rsrc(const void * p, uint32_t bytes) {
    return __builtin_amdgcn_make_buffer_rsrc((void *) p, (short) 0, (int) bytes, 0x31004000);
}
// Orders LDS only: a full __syncthreads also invalidates the vector caches on every K step.
static __device__ __forceinline__ void gemm_wmma_lds_barrier() {
    __builtin_amdgcn_fence(__ATOMIC_RELEASE, "workgroup", "local");
    __builtin_amdgcn_s_barrier();
    __builtin_amdgcn_fence(__ATOMIC_ACQUIRE, "workgroup", "local");
}
static __device__ __forceinline__ gemm_wmma_half8 gemm_wmma_load16(__amdgpu_buffer_rsrc_t r, uint32_t voffset, uint32_t soffset) {
    return __builtin_bit_cast(gemm_wmma_half8, __builtin_amdgcn_raw_buffer_load_b128(r, (int) voffset, (int) soffset, 0));
}
static __device__ __forceinline__ gemm_wmma_float8 gemm_wmma_load32(__amdgpu_buffer_rsrc_t r, uint32_t voffset, uint32_t soffset) {
    const gemm_wmma_float4 lo = __builtin_bit_cast(gemm_wmma_float4, __builtin_amdgcn_raw_buffer_load_b128(r, (int) voffset, (int) soffset, 0));
    const gemm_wmma_float4 hi = __builtin_bit_cast(gemm_wmma_float4, __builtin_amdgcn_raw_buffer_load_b128(r, (int) voffset + 16, (int) soffset, 0));
    return __builtin_shufflevector(lo, hi, 0, 1, 2, 3, 4, 5, 6, 7);
}
#endif

namespace gemm_wmma {
constexpr int BN = 128, BK = 32, LDK = BK + 8, LDY = BN + 4, THREADS = 128;
constexpr int WN = 64, WAVES_N = BN / WN;
constexpr int B_PIECES = BN * BK / 8 / THREADS; // 8-half pieces of W per thread and K step
constexpr int YROWS = 32; // output rows rotated per epilogue phase
constexpr int GROUP_M = 8; // row blocks per launch-order group
static_assert(BN == 128, "the epilogue rotates one 128-column group per block");
}

#ifdef GGML_CUDA_GEMM_WMMA
// two X stages during the K loop, then one wave row of the output tile at a time for the rotation
template<int BM>
constexpr int gemm_wmma_smem() {
    using namespace gemm_wmma;
    return std::max(2 * (BM + BN) * LDK * 2, YROWS * LDY * 4);
}

// Stores the accumulators of a BM x 128 tile: rows row0.. (of m) of columns col0.. With rotate, applies the output
// rotation and svh scale of the 128-column group and stores dst[row][n0 + col]; routed rows go to dst row perm[row].
// With NT threads of 4 or 8 waves, a wave holds WM = BM / (NT / 32 / WAVES_N) rows.
template<int BM, bool routed, bool rotate, int NT = gemm_wmma::THREADS>
static __device__ __forceinline__ void gemm_wmma_epilogue(char * smem,
                                                          gemm_wmma_float8 (&acc)[BM / (NT / 32 / gemm_wmma::WAVES_N) / 16][gemm_wmma::WN / 16],
                                                          const half * __restrict__ svh, float * __restrict__ dst,
                                                          const int32_t * __restrict__ perm, int m, int n, int n0, int row0,
                                                          int col0) {
    using namespace gemm_wmma;
    constexpr int WM = BM / (NT / 32 / WAVES_N);
    float * ys = (float *) smem;
    const int tid = threadIdx.x, lane = tid % 32, warp = tid / 32;
    const int wm = warp / WAVES_N, wn = warp % WAVES_N;
    // a lane holds column wn * WN + j * 16 + lane % 16 of rows wm * WM + i * 16 + 8 * (lane / 16) + 0..7
    if constexpr (!rotate) {
#pragma unroll
        for (int i = 0; i < WM / 16; ++i) {
#pragma unroll
            for (int j = 0; j < WN / 16; ++j) {
                const int col = col0 + wn * WN + j * 16 + lane % 16;
#pragma unroll
                for (int v = 0; v < 8; ++v) {
                    const int row = row0 + wm * WM + i * 16 + 8 * (lane / 16) + v;
                    if (row < m && col < n) {
                        dst[size_t(routed ? perm[row] : row) * n + col] = acc[i][j][v];
                    }
                }
            }
        }
        return;
    }
    // each phase stages the 32 rows of fragments i = 2 * ih, 2 * ih + 1 of one wave row and rotates them; phases past
    // the last row are skipped
    for (int part = 0; part < BM / YROWS && row0 + part * YROWS < m; ++part) {
        const int pwm = part / (WM / YROWS), ih = part % (WM / YROWS);
        if (wm == pwm) {
#pragma unroll
            for (int i = 0; i < WM / 16; ++i) {
                if (i / (YROWS / 16) != ih) { continue; }
#pragma unroll
                for (int j = 0; j < WN / 16; ++j) {
#pragma unroll
                    for (int v = 0; v < 8; ++v) {
                        ys[((i % (YROWS / 16)) * 16 + 8 * (lane / 16) + v) * LDY + wn * WN + j * 16 + lane % 16] = acc[i][j][v];
                    }
                }
            }
        }
        gemm_wmma_lds_barrier();
        for (int r = warp; r < YROWS; r += NT / 32) {
            const int row = row0 + part * YROWS + r;
            float v[4];
#pragma unroll
            for (int j = 0; j < 4; ++j) { v[j] = ys[r * LDY + j * 32 + lane]; }
            ggml_cuda_hadamard128(v, lane);
            if (row < m) {
                const size_t out = routed ? perm[row] : row;
#pragma unroll
                for (int j = 0; j < 4; ++j) {
                    const int col = n0 + col0 + j * 32 + lane;
                    dst[out * n + col] = v[j] * 0.08838834764831844f * __half2float(svh[col]);
                }
            }
        }
        gemm_wmma_lds_barrier();
    }
}

// One output tile, rows row0.. (of m) and columns col0.. (of n_cb blocks); vb staggers its K order.
template<int BM, bool rotate, typename TX>
static __device__ __forceinline__ void gemm_f16_wmma_tile(char * smem, const TX * __restrict__ x, const half * __restrict__ w,
                                                          const half * __restrict__ svh, float * __restrict__ dst, int m,
                                                          int n, int k, int n0, int row0, int col0, int n_cb, int vb) {
    using namespace gemm_wmma;
    constexpr int WM = BM / (THREADS / 32 / WAVES_N);
    constexpr int A_PIECES = BM * BK / 8 / THREADS; // and of X
    constexpr int STAGE = (BM + BN) * LDK;          // halves per LDS stage
    static_assert(WM % YROWS == 0, "an epilogue phase covers part of one wave row");
    _Float16 * stages = (_Float16 *) smem;

    const int tid = threadIdx.x, lane = tid % 32, warp = tid / 32;
    const int wm = warp / WAVES_N, wn = warp % WAVES_N;

    gemm_wmma_float8 acc[WM / 16][WN / 16];
#pragma unroll
    for (int i = 0; i < WM / 16; ++i) {
#pragma unroll
        for (int j = 0; j < WN / 16; ++j) { acc[i][j] = gemm_wmma_float8{0, 0, 0, 0, 0, 0, 0, 0}; }
    }

    // byte offsets: per lane and piece fixed for the whole loop, the K position goes in the scalar offset
    // without rotate, W rows from n up to the last column block are out of bounds and read as zeros
    constexpr int XS = sizeof(TX); // bytes per X element
    const __amdgpu_buffer_rsrc_t xr = gemm_wmma_rsrc(x, uint32_t(m) * k * XS);
    const __amdgpu_buffer_rsrc_t wr = gemm_wmma_rsrc(w, uint32_t(rotate ? n_cb * BN : n) * k * 2);
    uint32_t xo[A_PIECES], wo[B_PIECES];
#pragma unroll
    for (int p = 0; p < A_PIECES; ++p) {
        const int idx = tid + p * THREADS, r = idx / (BK / 8), c = (idx % (BK / 8)) * 8;
        // rows past m read row m - 1: an output row depends only on its own input row and is not written
        xo[p] = (uint32_t(min(row0 + r, m - 1)) * k + c) * XS;
    }
#pragma unroll
    for (int p = 0; p < B_PIECES; ++p) {
        const int idx = tid + p * THREADS, r = idx / (BK / 8), c = (idx % (BK / 8)) * 8;
        wo[p] = (uint32_t(col0 + r) * k + c) * 2;
    }

    gemm_wmma_half8 ra[A_PIECES], rb[B_PIECES];
    gemm_wmma_float8 raf[A_PIECES]; // FP32 X, converted when stored
    const int kstart = (vb % 32) * 128 % k;

// staggered K: tile vb runs k from (vb % 32) * 128 and wraps, so blocks sharing operands do not read the same
// addresses (and memory channels) in lockstep; only the accumulation order changes
#define GW_KP(K0) ((K0) + kstart < k ? (K0) + kstart : (K0) + kstart - k)
#define GW_LOAD(K0)                                                                                      \
    {                                                                                                      \
        const int kp = GW_KP(K0);                                                                        \
        _Pragma("unroll") for (int p = 0; p < A_PIECES; ++p) {                                             \
            if constexpr (XS == 4) { raf[p] = gemm_wmma_load32(xr, xo[p], kp * 4); }                       \
            else { ra[p] = gemm_wmma_load16(xr, xo[p], kp * 2); }                                          \
        }                                                                                                  \
        _Pragma("unroll") for (int p = 0; p < B_PIECES; ++p) { rb[p] = gemm_wmma_load16(wr, wo[p], kp * 2); }   \
    }
#define GW_STORE(STAGE_PTR)                                                                              \
    _Pragma("unroll") for (int p = 0; p < A_PIECES; ++p) {                                                 \
        const int idx = tid + p * THREADS, r = idx / (BK / 8), c = (idx % (BK / 8)) * 8;                   \
        if constexpr (XS == 4) { ra[p] = __builtin_convertvector(raf[p], gemm_wmma_half8); }               \
        *(gemm_wmma_half8 *) ((STAGE_PTR) + r * LDK + c) = ra[p];                                               \
    }                                                                                                      \
    _Pragma("unroll") for (int p = 0; p < B_PIECES; ++p) {                                                 \
        const int idx = tid + p * THREADS, r = idx / (BK / 8), c = (idx % (BK / 8)) * 8;                   \
        *(gemm_wmma_half8 *) ((STAGE_PTR) + (BM + r) * LDK + c) = rb[p];                                        \
    }
// one K step on stage CUR, prefetching the next step into registers and then into the other stage, which was last
// read before the previous barrier. The prefetch is unconditional: past the end, buffer loads return zeros and the
// stage written is not read again.
#define GW_STEP(K0, CUR)                                                                                 \
    {                                                                                                      \
        GW_LOAD((K0) + BK);                                                                              \
        const _Float16 * as = stages + (CUR) * STAGE + (wm * WM + lane % 16) * LDK + 8 * (lane / 16);      \
        const _Float16 * bs = stages + (CUR) * STAGE + (BM + wn * WN + lane % 16) * LDK + 8 * (lane / 16); \
        _Pragma("unroll") for (int kk = 0; kk < BK / 16; ++kk) {                                           \
            gemm_wmma_half8 b[WN / 16];                                                                         \
            _Pragma("unroll") for (int j = 0; j < WN / 16; ++j) { b[j] = *(const gemm_wmma_half8 *) (bs + j * 16 * LDK + kk * 16); } \
            _Pragma("unroll") for (int i = 0; i < WM / 16; ++i) {                                          \
                const gemm_wmma_half8 a = *(const gemm_wmma_half8 *) (as + i * 16 * LDK + kk * 16);                  \
                _Pragma("unroll") for (int j = 0; j < WN / 16; ++j) {                                      \
                    acc[i][j] = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32_gfx12(a, b[j], acc[i][j]);      \
                }                                                                                          \
            }                                                                                              \
        }                                                                                                  \
        /* keep the math between the prefetch loads and their stores: the compiler would otherwise wait for */ \
        /* the loads at once and sink the register-only WMMAs past the barrier */                           \
        __builtin_amdgcn_sched_barrier(0);                                                                 \
        GW_STORE(stages + ((CUR) ^ 1) * STAGE);                                                          \
        gemm_wmma_lds_barrier();                                                                                \
    }

    GW_LOAD(0);
    GW_STORE(stages);
    gemm_wmma_lds_barrier();
    // k is a multiple of 128, so the steps come in pairs
    for (int k0 = 0; k0 < k; k0 += 2 * BK) {
        GW_STEP(k0, 0);
        GW_STEP(k0 + BK, 1);
    }
#undef GW_STEP
#undef GW_STORE
#undef GW_LOAD
#undef GW_KP

    gemm_wmma_epilogue<BM, false, rotate>(smem, acc, svh, dst, nullptr, m, n, n0, row0, col0);
}
#endif // GGML_CUDA_GEMM_WMMA

// X is FP16, or FP32 (TX = float) converted to FP16 on its way to LDS.
template<int BM, bool rotate, typename TX = half>
__launch_bounds__(gemm_wmma::THREADS)
static __global__ void gemm_f16_wmma(const TX * __restrict__ x, const half * __restrict__ w, const half * __restrict__ svh,
                                     float * __restrict__ dst, int m, int n, int k, int n0) {
#ifdef GGML_CUDA_GEMM_WMMA
    using namespace gemm_wmma;
    __shared__ __attribute__((aligned(16))) char smem[gemm_wmma_smem<BM>()];
    // grouped order: consecutive blocks walk GROUP_M row blocks down one column strip, so the blocks in flight share
    // a few W strips and X row blocks in L2 instead of sweeping every W strip for each row block
    const int n_rb = (m + BM - 1) / BM, n_cb = gridDim.x / n_rb;
    const int group = blockIdx.x / (GROUP_M * n_cb), first = group * GROUP_M, size = min(n_rb - first, GROUP_M);
    const int row_block = first + (blockIdx.x % (GROUP_M * n_cb)) % size, col_block = (blockIdx.x % (GROUP_M * n_cb)) / size;
    gemm_f16_wmma_tile<BM, rotate, TX>(smem, x, w, svh, dst, m, n, k, n0, row_block * BM, col_block * BN, n_cb, blockIdx.x);
#else
    GGML_UNUSED_VARS(x, w, svh, dst, m, n, k, n0);
    NO_DEVICE_CODE;
#endif
}
#endif // GGML_USE_HIP
