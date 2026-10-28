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
    const int64_t k = op->src[2]->ne[0], n = op->ne[0];
    // Widths cover the 248320-column vocabulary head with int column indices.
    if (!(bits == 7 || bits == 8 || bits == 10 || bits == 12 || bits == 14) || k > 262144 || n > 262144 ||
            op->src[0]->ne[2] > INT_MAX) {
        return false;
    }
    // Banks above 8 rows go through an FP16 GEMM whose input copy is bounded (a transform block row per 4 rows).
    if (rows > 8) {
        return assignments <= 4 * 65535 && assignments * k <= 256 * 1024 * 1024;
    }
    // Bound the decode workspace and grid.
    return rows <= 8 && assignments <= 512 && assignments * (k + n) <= 16 * 1024 * 1024;
#else
    GGML_UNUSED(op); GGML_UNUSED(cc);
    return false;
#endif
}

bool ggml_cuda_exl3_needs_sync(const ggml_tensor * op) {
    return op->op == GGML_OP_MUL_MAT_EXL3 && op->src[4] && op->ne[2] > 8;
}

#ifdef GGML_USE_HIP
static __device__ __forceinline__ int exl3_expert(const ggml_cuda_exl3_args & a, int assignment) {
    return a.ids ? a.ids[assignment] : 0;
}

// 128-point Hadamard transform of v[j] = element j * 32 + lane, without normalization.
static __device__ __forceinline__ void exl3_hadamard128(float v[4], int lane) {
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

#define EXL3_TRANSFORM_ROWS 4 // the supports check bounds the grid rows

template<typename T, bool input>
static __global__ void exl3_transform(ggml_cuda_exl3_args a) {
    // The format uses groups of 32 lanes. Dispatch is restricted to RDNA4 wave32 devices. A wave rotates one row.
    const int row = blockIdx.y * EXL3_TRANSFORM_ROWS + threadIdx.x / 32;
    if (row >= a.assignments) {
        return;
    }
    const int assignment = input && a.perm ? a.perm[row] : row;
    const int expert = exl3_expert(a, assignment);
    const bool valid = expert >= 0 && expert < a.experts;
    const int width = input ? a.k : a.n;
    const int lane = threadIdx.x % 32;
    const int base = blockIdx.x * 128;
    const int xrow = a.ids ? (assignment / a.used) * a.input_slots + (assignment % a.used) % a.input_slots : assignment;
    float v[4];
#pragma unroll
    for (int j = 0; j < 4; ++j) {
        const int col = base + j * 32 + lane;
        v[j] = !valid ? NAN : input ? a.x[size_t(xrow) * width + col] * __half2float(a.suh[size_t(expert) * width + col])
                                  : a.yh[size_t(assignment) * width + col];
    }
    exl3_hadamard128(v, lane);
#pragma unroll
    for (int j = 0; j < 4; ++j) {
        const int col = base + j * 32 + lane;
        const float value = v[j] * 0.08838834764831844f;
        if (input) {
            ((T *) a.xh)[size_t(row) * width + col] = (T) value;
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

// A dense bank is shared by every row, so decode each fragment once and apply it to all rows. Per-row accumulation
// and reduction follow exl3_matrix_fragment<bits_x2, false> exactly, so each row's result is bit-identical to it.
template<int bits_x2, int rows>
static __global__ void exl3_matrix_rows(ggml_cuda_exl3_args a) {
    __shared__ float partial[4][rows][16];
    const int lane = threadIdx.x % 32, warp = threadIdx.x / 32;
    const float * xh = (const float *) a.xh;
    float lo[rows], hi[rows];
#pragma unroll
    for (int r = 0; r < rows; ++r) { lo[r] = 0; hi[r] = 0; }
    for (int kt = warp; kt < a.k / 16; kt += 4) {
        const uint32_t * tile = a.packed + (size_t(kt) * (a.n / 16) + blockIdx.x) * (4 * bits_x2);
        half w[8];
        ggml_cuda_exl3_fragment<bits_x2>(tile, lane, w);
#pragma unroll
        for (int r = 0; r < rows; ++r) {
#pragma unroll
            for (int j = 0; j < 4; ++j) {
                const float v = xh[size_t(r) * a.k + kt * 16 + (lane & 3) * 2 + (j & 1) + (j / 2) * 8];
                lo[r] = fmaf(v, __half2float(w[j]), lo[r]);
                hi[r] = fmaf(v, __half2float(w[j + 4]), hi[r]);
            }
        }
    }
#pragma unroll
    for (int r = 0; r < rows; ++r) {
        lo[r] += __shfl_xor(lo[r], 1, 32); lo[r] += __shfl_xor(lo[r], 2, 32);
        hi[r] += __shfl_xor(hi[r], 1, 32); hi[r] += __shfl_xor(hi[r], 2, 32);
        const float vlo = __shfl(lo[r], (lane & 7) * 4, 32), vhi = __shfl(hi[r], (lane & 7) * 4, 32);
        if (lane < 16) { partial[warp][r][lane] = (lane & 8) ? vhi : vlo; }
    }
    __syncthreads();
    if (warp == 0 && lane < 16) {
#pragma unroll
        for (int r = 0; r < rows; ++r) {
            float sum = partial[0][r][lane];
            for (int w = 1; w < 4; ++w) { sum += partial[w][r][lane]; }
            a.yh[size_t(r) * a.n + blockIdx.x * 16 + lane] = sum;
        }
    }
}

// Decode columns [n0, n0 + gridDim.x * 16) of a bank into FP16 w[col - n0][k], k contiguous, for a GEMM: the dense
// bank, or the bank of expert experts[blockIdx.z] into slot blockIdx.z of w.
// A block decodes eight 16 x 16 tiles of one column strip and writes each column's 128 values contiguously.
template<int bits_x2>
static __global__ void exl3_decode_f16(const uint32_t * packed, const int32_t * experts, half * w, int k, int n, int n0) {
    __shared__ half strip[16][128 + 8];
    const int lane = threadIdx.x % 32, warp = threadIdx.x / 32;
    const int nt = n0 / 16 + blockIdx.x;
    if (experts) {
        packed += size_t(experts[blockIdx.z]) * (k / 16) * (n / 16) * (4 * bits_x2);
        w += size_t(blockIdx.z) * gridDim.x * 16 * k;
    }
#pragma unroll
    for (int t = 0; t < 2; ++t) {
        const int kt = blockIdx.y * 8 + warp * 2 + t;
        const uint32_t * tile = packed + (size_t(kt) * (n / 16) + nt) * (4 * bits_x2);
        half v[8];
        ggml_cuda_exl3_fragment<bits_x2>(tile, lane, v);
#pragma unroll
        for (int j = 0; j < 8; ++j) {
            const int r = (lane & 3) * 2 + (j & 1) + ((j >> 1) & 1) * 8;
            const int c = (lane >> 2) + 8 * (j >> 2);
            strip[c][(warp * 2 + t) * 16 + r] = v[j];
        }
    }
    __syncthreads();
    const int c = threadIdx.x / 8, part = threadIdx.x % 8;
    const uint4 * src = (const uint4 *) &strip[c][part * 16];
    uint4 * dst = (uint4 *) (w + size_t(blockIdx.x * 16 + c) * k + blockIdx.y * 128 + part * 16);
    dst[0] = src[0];
    dst[1] = src[1];
}

static void exl3_decode_f16_launch(const ggml_cuda_exl3_args & a, const int32_t * experts, int n_experts, half * w,
                                   int n0, int n_cols, cudaStream_t stream) {
    const dim3 grid(n_cols / 16, a.k / 128, n_experts);
    switch (a.bits_x2) {
        case 7:  exl3_decode_f16<7 ><<<grid, 128, 0, stream>>>(a.packed, experts, w, a.k, a.n, n0); break;
        case 8:  exl3_decode_f16<8 ><<<grid, 128, 0, stream>>>(a.packed, experts, w, a.k, a.n, n0); break;
        case 10: exl3_decode_f16<10><<<grid, 128, 0, stream>>>(a.packed, experts, w, a.k, a.n, n0); break;
        case 12: exl3_decode_f16<12><<<grid, 128, 0, stream>>>(a.packed, experts, w, a.k, a.n, n0); break;
        case 14: exl3_decode_f16<14><<<grid, 128, 0, stream>>>(a.packed, experts, w, a.k, a.n, n0); break;
        default: GGML_ABORT("Unsupported EXL3 rate");
    }
}

// Y = X W^T for FP16 X [m][k] (rotated rows) and FP16 W [n_cols][k] (a decoded column chunk), FP32 accumulation on
// RDNA4 WMMA, then the output rotation and svh scale of each 128-column group: dst[row][n0 + col]. A block computes
// BM x 128 in 4 waves of BM / 2 x 64; both operands go through double-buffered LDS, the next K step loaded into
// registers during the current step's math, one barrier per step of 32.
// On gfx1201 this runs at about 100 TFLOPS for 4096 x 12288 x 2560, hipBLASLt's FP16 GEMM at about 145. Thread traces
// put the difference in instruction scheduling (LDS fragment and global load waits), not in tiling or occupancy.
// Routed: x holds the rows sorted by expert and w the decoded experts of one chunk, all n columns each. Block b
// computes column block b % (n / BN) of tile b / (n / BN) = {first sorted row, rows (at most BM), w slot, expert},
// and writes sorted row i to dst row perm[i].
#if defined(__HIP_DEVICE_COMPILE__) && (defined(__gfx1200__) || defined(__gfx1201__))
#define EXL3_GEMM_WMMA 1
#endif
typedef _Float16 exl3_half8 __attribute__((ext_vector_type(8)));
typedef float exl3_float8 __attribute__((ext_vector_type(8)));

#ifdef EXL3_GEMM_WMMA
// Raw buffer loads take a scalar base and a 32-bit per-lane offset, so a fragment address costs one VGPR.
static __device__ __forceinline__ __amdgpu_buffer_rsrc_t exl3_rsrc(const void * p, uint32_t bytes) {
    return __builtin_amdgcn_make_buffer_rsrc((void *) p, (short) 0, (int) bytes, 0x31004000);
}
// Orders LDS only: a full __syncthreads also invalidates the vector caches on every K step.
static __device__ __forceinline__ void exl3_lds_barrier() {
    __builtin_amdgcn_fence(__ATOMIC_RELEASE, "workgroup", "local");
    __builtin_amdgcn_s_barrier();
    __builtin_amdgcn_fence(__ATOMIC_ACQUIRE, "workgroup", "local");
}
static __device__ __forceinline__ exl3_half8 exl3_load16(__amdgpu_buffer_rsrc_t r, uint32_t voffset, uint32_t soffset) {
    return __builtin_bit_cast(exl3_half8, __builtin_amdgcn_raw_buffer_load_b128(r, (int) voffset, (int) soffset, 0));
}
#endif

// BM is 128 rows, or 64 for routed tiles, which mostly hold fewer rows than that.
namespace exl3_gemm {
constexpr int BN = 128, BK = 32, LDK = BK + 8, LDY = BN + 4, THREADS = 128;
constexpr int WN = 64, WAVES_N = BN / WN;
constexpr int B_PIECES = BN * BK / 8 / THREADS; // 8-half pieces of W per thread and K step
constexpr int YROWS = 32; // output rows rotated per epilogue phase
constexpr int GROUP_M = 8; // row blocks per launch-order group
static_assert(BN == 128, "the epilogue rotates one 128-column group per block");
}

template<int BM, bool routed>
__launch_bounds__(exl3_gemm::THREADS)
static __global__ void exl3_gemm_f16(const half * __restrict__ x, const half * __restrict__ w, const half * __restrict__ svh,
                                     float * __restrict__ dst, int m, int n, int k, int n0,
                                     const int4 * __restrict__ tiles, const int32_t * __restrict__ perm) {
#ifdef EXL3_GEMM_WMMA
    using namespace exl3_gemm;
    constexpr int WM = BM / (THREADS / 32 / WAVES_N);
    constexpr int A_PIECES = BM * BK / 8 / THREADS; // and of X
    constexpr int STAGE = (BM + BN) * LDK;          // halves per LDS stage
    constexpr int SMEM = std::max(2 * STAGE * 2, YROWS * LDY * 4);
    static_assert(WM % YROWS == 0, "an epilogue phase covers part of one wave row");
    // two X stages during the K loop, then one wave row of the output tile at a time for the rotation
    __shared__ __attribute__((aligned(16))) char smem[SMEM];
    _Float16 * stages = (_Float16 *) smem;
    float * ys = (float *) smem;

    const int tid = threadIdx.x, lane = tid % 32, warp = tid / 32;
    const int wm = warp / WAVES_N, wn = warp % WAVES_N;
    // grouped order: consecutive blocks walk GROUP_M row blocks down one column strip, so the blocks in flight share
    // a few W strips and X row blocks in L2 instead of sweeping every W strip for each row block
    int n_cb, row0, col0;
    if constexpr (routed) {
        // the blocks in flight cover all column blocks of a few tiles, so a few experts' weights
        n_cb = n / BN;
        const int4 tile = tiles[blockIdx.x / n_cb];
        x += size_t(tile.x) * k;
        m = tile.y;
        w += size_t(tile.z) * n * k;
        svh += size_t(tile.w) * n;
        perm += tile.x;
        row0 = 0;
        col0 = (blockIdx.x % n_cb) * BN;
    } else {
        const int n_rb = (m + BM - 1) / BM;
        n_cb = gridDim.x / n_rb;
        const int group = blockIdx.x / (GROUP_M * n_cb), first = group * GROUP_M, size = min(n_rb - first, GROUP_M);
        const int row_block = first + (blockIdx.x % (GROUP_M * n_cb)) % size, col_block = (blockIdx.x % (GROUP_M * n_cb)) / size;
        row0 = row_block * BM;
        col0 = col_block * BN;
    }

    exl3_float8 acc[WM / 16][WN / 16];
#pragma unroll
    for (int i = 0; i < WM / 16; ++i) {
#pragma unroll
        for (int j = 0; j < WN / 16; ++j) { acc[i][j] = exl3_float8{0, 0, 0, 0, 0, 0, 0, 0}; }
    }

    // byte offsets: per lane and piece fixed for the whole loop, the K position goes in the scalar offset
    const __amdgpu_buffer_rsrc_t xr = exl3_rsrc(x, uint32_t(m) * k * 2), wr = exl3_rsrc(w, uint32_t(n_cb * BN) * k * 2);
    uint32_t xo[A_PIECES], wo[B_PIECES];
#pragma unroll
    for (int p = 0; p < A_PIECES; ++p) {
        const int idx = tid + p * THREADS, r = idx / (BK / 8), c = (idx % (BK / 8)) * 8;
        // rows past m read row m - 1: an output row depends only on its own input row and is not written
        xo[p] = (uint32_t(min(row0 + r, m - 1)) * k + c) * 2;
    }
#pragma unroll
    for (int p = 0; p < B_PIECES; ++p) {
        const int idx = tid + p * THREADS, r = idx / (BK / 8), c = (idx % (BK / 8)) * 8;
        wo[p] = (uint32_t(col0 + r) * k + c) * 2;
    }

    exl3_half8 ra[A_PIECES], rb[B_PIECES];
    const int kstart = (blockIdx.x % 32) * 128 % k;

// staggered K: block b runs k from (b % 32) * 128 and wraps, so blocks sharing operands do not read the same
// addresses (and memory channels) in lockstep; only the accumulation order changes
#define EXL3_KP(K0) ((K0) + kstart < k ? (K0) + kstart : (K0) + kstart - k)
#define EXL3_LOAD(K0)                                                                                      \
    {                                                                                                      \
        const int kp = EXL3_KP(K0);                                                                        \
        _Pragma("unroll") for (int p = 0; p < A_PIECES; ++p) { ra[p] = exl3_load16(xr, xo[p], kp * 2); }   \
        _Pragma("unroll") for (int p = 0; p < B_PIECES; ++p) { rb[p] = exl3_load16(wr, wo[p], kp * 2); }   \
    }
#define EXL3_STORE(STAGE_PTR)                                                                              \
    _Pragma("unroll") for (int p = 0; p < A_PIECES; ++p) {                                                 \
        const int idx = tid + p * THREADS, r = idx / (BK / 8), c = (idx % (BK / 8)) * 8;                   \
        *(exl3_half8 *) ((STAGE_PTR) + r * LDK + c) = ra[p];                                               \
    }                                                                                                      \
    _Pragma("unroll") for (int p = 0; p < B_PIECES; ++p) {                                                 \
        const int idx = tid + p * THREADS, r = idx / (BK / 8), c = (idx % (BK / 8)) * 8;                   \
        *(exl3_half8 *) ((STAGE_PTR) + (BM + r) * LDK + c) = rb[p];                                        \
    }
// one K step on stage CUR, prefetching the next step into registers and then into the other stage, which was last
// read before the previous barrier. The prefetch is unconditional: past the end, buffer loads return zeros and the
// stage written is not read again.
#define EXL3_STEP(K0, CUR)                                                                                 \
    {                                                                                                      \
        EXL3_LOAD((K0) + BK);                                                                              \
        const _Float16 * as = stages + (CUR) * STAGE + (wm * WM + lane % 16) * LDK + 8 * (lane / 16);      \
        const _Float16 * bs = stages + (CUR) * STAGE + (BM + wn * WN + lane % 16) * LDK + 8 * (lane / 16); \
        _Pragma("unroll") for (int kk = 0; kk < BK / 16; ++kk) {                                           \
            exl3_half8 b[WN / 16];                                                                         \
            _Pragma("unroll") for (int j = 0; j < WN / 16; ++j) { b[j] = *(const exl3_half8 *) (bs + j * 16 * LDK + kk * 16); } \
            _Pragma("unroll") for (int i = 0; i < WM / 16; ++i) {                                          \
                const exl3_half8 a = *(const exl3_half8 *) (as + i * 16 * LDK + kk * 16);                  \
                _Pragma("unroll") for (int j = 0; j < WN / 16; ++j) {                                      \
                    acc[i][j] = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32_gfx12(a, b[j], acc[i][j]);      \
                }                                                                                          \
            }                                                                                              \
        }                                                                                                  \
        /* keep the math between the prefetch loads and their stores: the compiler would otherwise wait for */ \
        /* the loads at once and sink the register-only WMMAs past the barrier */                           \
        __builtin_amdgcn_sched_barrier(0);                                                                 \
        EXL3_STORE(stages + ((CUR) ^ 1) * STAGE);                                                          \
        exl3_lds_barrier();                                                                                \
    }

    EXL3_LOAD(0);
    EXL3_STORE(stages);
    exl3_lds_barrier();
    // k is a multiple of 128, so the steps come in pairs
    for (int k0 = 0; k0 < k; k0 += 2 * BK) {
        EXL3_STEP(k0, 0);
        EXL3_STEP(k0 + BK, 1);
    }
#undef EXL3_STEP
#undef EXL3_STORE
#undef EXL3_LOAD
#undef EXL3_KP

    // a lane holds column wn * WN + j * 16 + lane % 16 of rows wm * WM + i * 16 + 8 * (lane / 16) + 0..7; each phase
    // stages the 32 rows of fragments i = 2 * ih, 2 * ih + 1 of one wave row and rotates them
    for (int part = 0; part < BM / YROWS; ++part) {
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
        exl3_lds_barrier();
        for (int r = warp; r < YROWS; r += THREADS / 32) {
            const int row = row0 + part * YROWS + r;
            float v[4];
#pragma unroll
            for (int j = 0; j < 4; ++j) { v[j] = ys[r * LDY + j * 32 + lane]; }
            exl3_hadamard128(v, lane);
            if (row < m) {
                const size_t out = routed ? perm[row] : row;
#pragma unroll
                for (int j = 0; j < 4; ++j) {
                    const int col = n0 + col0 + j * 32 + lane;
                    dst[out * n + col] = v[j] * 0.08838834764831844f * __half2float(svh[col]);
                }
            }
        }
        exl3_lds_barrier();
    }
#else
    GGML_UNUSED_VARS(x, w, svh, dst, m, n, k, n0, tiles, perm);
    NO_DEVICE_CODE;
#endif
}

template<int bits_x2>
static void exl3_matrix_launch(const ggml_cuda_exl3_args & a, ggml_cuda_exl3_mode mode, cudaStream_t stream) {
    if (mode == GGML_EXL3_F32_ROWS && !a.ids && a.experts >= 1 && a.assignments <= 8) {
        switch (a.assignments) {
            case 1: exl3_matrix_rows<bits_x2, 1><<<a.n / 16, 128, 0, stream>>>(a); return;
            case 2: exl3_matrix_rows<bits_x2, 2><<<a.n / 16, 128, 0, stream>>>(a); return;
            case 3: exl3_matrix_rows<bits_x2, 3><<<a.n / 16, 128, 0, stream>>>(a); return;
            case 4: exl3_matrix_rows<bits_x2, 4><<<a.n / 16, 128, 0, stream>>>(a); return;
            case 5: exl3_matrix_rows<bits_x2, 5><<<a.n / 16, 128, 0, stream>>>(a); return;
            case 6: exl3_matrix_rows<bits_x2, 6><<<a.n / 16, 128, 0, stream>>>(a); return;
            case 7: exl3_matrix_rows<bits_x2, 7><<<a.n / 16, 128, 0, stream>>>(a); return;
            case 8: exl3_matrix_rows<bits_x2, 8><<<a.n / 16, 128, 0, stream>>>(a); return;
            default: break;
        }
    }
    const dim3 grid(a.n / 16, a.assignments);
    if (mode == GGML_EXL3_F16_DOT) { exl3_matrix_fragment<bits_x2, true><<<grid, 128, 0, stream>>>(a); }
    else if (mode == GGML_EXL3_F32_FRAGMENT || mode == GGML_EXL3_F32_ROWS) { exl3_matrix_fragment<bits_x2, false><<<grid, 128, 0, stream>>>(a); }
    else { exl3_matrix_f32<bits_x2><<<grid, 128, 0, stream>>>(a); }
}

void ggml_cuda_exl3_launch(const ggml_cuda_exl3_args & a, ggml_cuda_exl3_mode mode,
                          ggml_cuda_exl3_stage stage, cudaStream_t stream) {
    if (stage == GGML_EXL3_INPUT || stage == GGML_EXL3_ALL) {
        const dim3 grid(a.k / 128, (a.assignments + EXL3_TRANSFORM_ROWS - 1) / EXL3_TRANSFORM_ROWS);
        if (mode == GGML_EXL3_F16_DOT) { exl3_transform<half, true><<<grid, 32 * EXL3_TRANSFORM_ROWS, 0, stream>>>(a); }
        else { exl3_transform<float, true><<<grid, 32 * EXL3_TRANSFORM_ROWS, 0, stream>>>(a); }
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
        const dim3 grid(a.n / 128, (a.assignments + EXL3_TRANSFORM_ROWS - 1) / EXL3_TRANSFORM_ROWS);
        exl3_transform<float, false><<<grid, 32 * EXL3_TRANSFORM_ROWS, 0, stream>>>(a);
    }
    CUDA_CHECK(cudaGetLastError());
}

// Routed banks above 8 rows: sort the assignments by expert on the host, rotate the rows to FP16 in that order, then
// per chunk of experts decode their banks to FP16 and run the GEMM over the chunk's tiles of up to BM rows each.
static void exl3_routed_gemm(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    using exl3_gemm::BN;
    // a used expert gets a median of 38 rows of a 4096-token batch (10 of 512 experts per token)
    constexpr int BM = 64;
    const int k = dst->src[2]->ne[0], n = dst->ne[0], used = dst->ne[1], assignments = dst->ne[1] * dst->ne[2];
    const int experts = dst->src[0]->ne[2];
    cudaStream_t stream = ctx.stream();

    std::vector<int32_t> ids(assignments);
    CUDA_CHECK(cudaMemcpyAsync(ids.data(), dst->src[4]->data, ids.size() * sizeof(int32_t), cudaMemcpyDeviceToHost, stream));
    CUDA_CHECK(cudaStreamSynchronize(stream));

    // counting sort by expert; rows of out-of-range experts are not computed and read NaN, like the other paths
    std::vector<int> start(experts + 1, 0);
    for (const int32_t e : ids) {
        if (e >= 0 && e < experts) { ++start[e + 1]; }
    }
    std::vector<int32_t> experts_used;
    for (int e = 0; e < experts; ++e) {
        if (start[e + 1]) { experts_used.push_back(e); }
        start[e + 1] += start[e];
    }
    const int valid = start[experts];
    // device copy: the sorted assignments, the experts used, then 16-byte aligned tiles grouped by expert chunk
    // decoded chunks of 32 MiB stay in the 64 MiB Infinity Cache until the GEMM reads them: at 4096 tokens 16 and
    // 64 MiB are 30% slower, 128 MiB and more 45-60%
    const int chunk_experts = std::max<int>(1, (32ll << 20) / (2ll * n * k));
    const size_t used_offset = valid, tiles_offset = (used_offset + experts_used.size() + 3) / 4 * 4;
    std::vector<int32_t> host(tiles_offset);
    std::vector<int> fill(start.begin(), start.end() - 1);
    for (int i = 0; i < assignments; ++i) {
        if (ids[i] >= 0 && ids[i] < experts) { host[fill[ids[i]]++] = i; }
    }
    std::copy(experts_used.begin(), experts_used.end(), host.begin() + used_offset);
    std::vector<int> chunk_tiles = {0};
    for (size_t s0 = 0; s0 < experts_used.size(); s0 += chunk_experts) {
        for (size_t s = s0; s < std::min(experts_used.size(), s0 + chunk_experts); ++s) {
            const int e = experts_used[s];
            for (int r0 = start[e]; r0 < start[e + 1]; r0 += BM) {
                host.insert(host.end(), {r0, std::min(BM, start[e + 1] - r0), int(s - s0), e});
            }
        }
        chunk_tiles.push_back((host.size() - tiles_offset) / 4);
    }
    for (int i = 0; i < assignments; ++i) {
        if (ids[i] < 0 || ids[i] >= experts) {
            CUDA_CHECK(cudaMemsetAsync((float *) dst->data + size_t(i) * n, 0xff, n * sizeof(float), stream));
        }
    }
    if (valid == 0) {
        return;
    }
    ggml_cuda_pool_alloc<int32_t> dev(ctx.pool(), host.size());
    CUDA_CHECK(cudaMemcpyAsync(dev.get(), host.data(), host.size() * sizeof(int32_t), cudaMemcpyHostToDevice, stream));
    // the host copy is released on return
    CUDA_CHECK(cudaStreamSynchronize(stream));

    ggml_cuda_pool_alloc<half> xh(ctx.pool(), size_t(valid) * k);
    ggml_cuda_pool_alloc<half> w(ctx.pool(), size_t(std::min<size_t>(chunk_experts, experts_used.size())) * n * k);
    const ggml_cuda_exl3_args args = {
        (const uint32_t *) dst->src[0]->data, (const half *) dst->src[2]->data, (const half *) dst->src[3]->data,
        (const float *) dst->src[1]->data, (const int32_t *) dst->src[4]->data, (float *) dst->data, xh.get(), nullptr,
        k, n, experts, valid, used, int(dst->src[1]->ne[1]), ggml_get_op_params_i32(dst, 0), dev.get()};
    ggml_cuda_exl3_launch(args, GGML_EXL3_F16_DOT, GGML_EXL3_INPUT, stream);
    const int4 * tiles = (const int4 *) (dev.get() + tiles_offset);
    for (size_t c = 0; c + 1 < chunk_tiles.size(); ++c) {
        const int s0 = c * chunk_experts, n_experts = std::min<int>(chunk_experts, experts_used.size() - s0);
        exl3_decode_f16_launch(args, dev.get() + used_offset + s0, n_experts, w.get(), 0, n, stream);
        const int blocks = n / BN * (chunk_tiles[c + 1] - chunk_tiles[c]);
        exl3_gemm_f16<BM, true><<<blocks, exl3_gemm::THREADS, 0, stream>>>(xh.get(), w.get(), args.svh, args.dst, 0, n, k, 0,
                                                                           tiles + chunk_tiles[c], dev.get());
    }
    CUDA_CHECK(cudaGetLastError());
}
#endif

void ggml_cuda_op_mul_mat_exl3(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
#ifdef GGML_USE_HIP
    GGML_ASSERT(ggml_cuda_supports_exl3(dst, ggml_cuda_info().devices[ctx.device].cc));
    if (ggml_is_empty(dst)) {
        return;
    }
    const int k = dst->src[2]->ne[0], n = dst->ne[0], assignments = dst->ne[1] * dst->ne[2];
    if (dst->src[4] && dst->ne[2] > 8) {
        exl3_routed_gemm(ctx, dst);
        return;
    }
    if (!dst->src[4] && assignments > 8) {
        // Rotate the rows to FP16, decode the bank to FP16 in column chunks and multiply with FP32 accumulation; the
        // GEMM applies the output rotation and scale.
        ggml_cuda_pool_alloc<half> xh(ctx.pool(), size_t(assignments) * k);
        const int chunk = std::min(n, std::max(128, int((64ll << 20) / (2ll * k)) / 128 * 128));
        ggml_cuda_pool_alloc<half> w(ctx.pool(), size_t(chunk) * k);
        const ggml_cuda_exl3_args args = {
            (const uint32_t *) dst->src[0]->data, (const half *) dst->src[2]->data, (const half *) dst->src[3]->data,
            (const float *) dst->src[1]->data, nullptr, (float *) dst->data, xh.get(), (float *) dst->data, k, n, 1,
            assignments, 1, int(dst->src[1]->ne[1]), ggml_get_op_params_i32(dst, 0)};
        cudaStream_t stream = ctx.stream();
        ggml_cuda_exl3_launch(args, GGML_EXL3_F16_DOT, GGML_EXL3_INPUT, stream);
        for (int n0 = 0; n0 < n; n0 += chunk) {
            const int n_cols = std::min(chunk, n - n0);
            exl3_decode_f16_launch(args, nullptr, 1, w.get(), n0, n_cols, stream);
            const int blocks = n_cols / exl3_gemm::BN * ((assignments + 127) / 128);
            exl3_gemm_f16<128, false><<<blocks, exl3_gemm::THREADS, 0, stream>>>(xh.get(), w.get(), args.svh, args.dst,
                                                                            assignments, n, k, n0, nullptr, nullptr);
        }
        CUDA_CHECK(cudaGetLastError());
        return;
    }
    ggml_cuda_pool_alloc<float> xh(ctx.pool(), size_t(assignments) * k);
    ggml_cuda_pool_alloc<float> yh(ctx.pool(), size_t(assignments) * n);
    const ggml_cuda_exl3_args args = {
        (const uint32_t *) dst->src[0]->data, (const half *) dst->src[2]->data, (const half *) dst->src[3]->data,
        (const float *) dst->src[1]->data, dst->src[4] ? (const int32_t *) dst->src[4]->data : nullptr,
        (float *) dst->data, xh.get(), yh.get(), k, n, int(dst->src[0]->ne[2]), assignments,
        dst->src[4] ? int(dst->ne[1]) : 1, int(dst->src[1]->ne[1]), ggml_get_op_params_i32(dst, 0)};
    ggml_cuda_exl3_launch(args, args.ids ? GGML_EXL3_F32_FRAGMENT : GGML_EXL3_F32_ROWS, GGML_EXL3_ALL, ctx.stream());
#else
    GGML_UNUSED(ctx); GGML_UNUSED(dst);
    GGML_ABORT("EXL3 projections require the ROCm backend");
#endif
}
