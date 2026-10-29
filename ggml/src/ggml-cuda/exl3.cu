#include "exl3.cuh"
#include "gemm-wmma.cuh"
#include "ggml-exl3.h"

#define EXL3_ROUTE_BLOCK       256 // assignments per block of the routing sort
#define EXL3_ROUTE_MAX_EXPERTS 16384 // the histogram's LDS: a routed op on a MoE cache bank has one per slot
#define EXL3_ROUTE_THREADS     1024

// a chunk of the routed GEMM: its rows in the expert-sorted order and its tiles in the tile list
struct ggml_cuda_routed_chunk {
    int32_t row0, rows, tile0, tiles;
};

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
    // Banks above 8 rows go through an FP16 GEMM whose input copy is bounded (a transform block row per 4 rows); a
    // routed one counts its experts in LDS.
    if (rows > 8) {
        return assignments <= 4 * 65535 && assignments * k <= 256 * 1024 * 1024 &&
            (!op->src[4] || op->src[0]->ne[2] <= EXL3_ROUTE_MAX_EXPERTS);
    }
    // Bound the decode workspace and grid.
    return rows <= 8 && assignments <= 512 && assignments * (k + n) <= 16 * 1024 * 1024;
#else
    GGML_UNUSED(op); GGML_UNUSED(cc);
    return false;
#endif
}

#ifdef GGML_USE_HIP
struct exl3_aux_scatter_args {
    const uint4 * src;
    uint4 * banks[GGML_EXL3_AUX_BANKS];
    size_t offsets[GGML_EXL3_AUX_BANKS];
    size_t widths[GGML_EXL3_AUX_BANKS];
    int32_t slots[GGML_EXL3_AUX_MAX_FILLS];
    int32_t count;
};

static __global__ void exl3_aux_scatter(exl3_aux_scatter_args a) {
    const int bank = blockIdx.z, fill = blockIdx.y;
    const size_t col = size_t(blockIdx.x)*blockDim.x + threadIdx.x;
    if (bank < GGML_EXL3_AUX_BANKS && fill < a.count && col < a.widths[bank]) {
        a.banks[bank][size_t(a.slots[fill])*a.widths[bank] + col] =
            a.src[a.offsets[bank] + size_t(fill)*a.widths[bank] + col];
    }
}

void ggml_cuda_exl3_aux_scatter(ggml_backend_cuda_context & ctx, const ggml_tensor * scratch, const ggml_exl3_aux_batch & batch) {
    exl3_aux_scatter_args a{};
    a.src = (const uint4 *) scratch->data;
    a.count = batch.count;
    size_t offset = 0, width = 0;
    for (int i = 0; i < GGML_EXL3_AUX_BANKS; ++i) {
        a.banks[i] = (uint4 *) batch.banks[i]->data;
        a.widths[i] = batch.banks[i]->nb[2]/sizeof(uint4);
        a.offsets[i] = offset;
        offset += a.widths[i]*batch.count;
        width = std::max(width, a.widths[i]);
    }
    std::copy(batch.slots, batch.slots + batch.count, a.slots);
    exl3_aux_scatter<<<dim3((width + 255)/256, batch.count, GGML_EXL3_AUX_BANKS), 256, 0, ctx.stream()>>>(a);
    CUDA_CHECK(cudaGetLastError());
}

static __device__ __forceinline__ int exl3_expert(const ggml_cuda_exl3_args & a, int assignment) {
    return a.ids ? a.ids[assignment] : 0;
}

#define EXL3_TRANSFORM_ROWS 4 // the supports check bounds the grid rows

// The format uses groups of 32 lanes. Dispatch is restricted to RDNA4 wave32 devices. A wave rotates one row: the
// input of assignment into row of xh, or row assignment of yh into dst.
template<typename T, bool input>
static __device__ __forceinline__ void exl3_transform_row(const ggml_cuda_exl3_args & a, int row, int assignment) {
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
    ggml_cuda_hadamard128(v, lane);
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

template<typename T, bool input>
static __global__ void exl3_transform(ggml_cuda_exl3_args a) {
    const int row = blockIdx.y * EXL3_TRANSFORM_ROWS + threadIdx.x / 32;
    if (row >= a.assignments) {
        return;
    }
    exl3_transform_row<T, input>(a, row, input && a.perm ? a.perm[row] : row);
}

// The input rows of one chunk of the routed GEMM: row i of xh takes assignment perm[chunk.row0 + i].
static __global__ void exl3_routed_input(ggml_cuda_exl3_args a, const ggml_cuda_routed_chunk * chunk) {
    const int rows = chunk->rows;
    const int32_t * perm = a.perm + chunk->row0;
    for (int row = blockIdx.y * EXL3_TRANSFORM_ROWS + threadIdx.x / 32; row < rows; row += gridDim.y * EXL3_TRANSFORM_ROWS) {
        exl3_transform_row<half, true>(a, row, perm[row]);
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

// Decode columns [n0, n0 + gridDim.x * 16) of a dense bank into FP16 w[col - n0][k], k contiguous, for a GEMM.
// A block decodes eight 16 x 16 tiles of one column strip and writes each column's 128 values contiguously.
template<int bits_x2>
static __global__ void exl3_decode_f16(const uint32_t * packed, half * w, int k, int n, int n0) {
    __shared__ half strip[16][128 + 8];
    const int lane = threadIdx.x % 32, warp = threadIdx.x / 32;
    const int nt = n0 / 16 + blockIdx.x;
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

static void exl3_decode_f16_launch(const ggml_cuda_exl3_args & a, half * w, int n0, int n_cols, cudaStream_t stream) {
    const dim3 grid(n_cols / 16, a.k / 128);
    switch (a.bits_x2) {
        case 7:  exl3_decode_f16<7 ><<<grid, 128, 0, stream>>>(a.packed, w, a.k, a.n, n0); break;
        case 8:  exl3_decode_f16<8 ><<<grid, 128, 0, stream>>>(a.packed, w, a.k, a.n, n0); break;
        case 10: exl3_decode_f16<10><<<grid, 128, 0, stream>>>(a.packed, w, a.k, a.n, n0); break;
        case 12: exl3_decode_f16<12><<<grid, 128, 0, stream>>>(a.packed, w, a.k, a.n, n0); break;
        case 14: exl3_decode_f16<14><<<grid, 128, 0, stream>>>(a.packed, w, a.k, a.n, n0); break;
        default: GGML_ABORT("Unsupported EXL3 rate");
    }
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

// Routing of the routed GEMM, on the device so that the op never waits for it: Radiance's stable counting sort in
// three passes, without an atomic deciding the order. hist counts each block of EXL3_ROUTE_BLOCK assignments per
// expert; plan scans those counts per expert (an expert's rows in earlier blocks), takes the offsets of the experts'
// runs and plans the tiles and chunks; place puts each assignment at its expert's offset plus the expert's rows in
// earlier blocks plus its rank in its block.

static __global__ void exl3_route_hist(const int32_t * ids, int32_t * block_count, uint32_t * dst, int assignments,
                                       int experts, int n) {
    extern __shared__ int32_t hist[];
    for (int e = threadIdx.x; e < experts; e += blockDim.x) {
        hist[e] = 0;
    }
    __syncthreads();
    const int i = blockIdx.x * EXL3_ROUTE_BLOCK + threadIdx.x;
    if (i < assignments) {
        const int e = ids[i];
        if (e >= 0 && e < experts) {
            atomicAdd(&hist[e], 1);
        } else {
            // rows of out-of-range experts are not computed and read NaN, like the other paths
            for (int c = 0; c < n; ++c) {
                dst[size_t(i) * n + c] = 0xffffffffu;
            }
        }
    }
    __syncthreads();
    for (int e = threadIdx.x; e < experts; e += blockDim.x) {
        block_count[size_t(blockIdx.x) * experts + e] = hist[e];
    }
}

// exclusive prefix sum over the block (a multiple of 32 threads, at most 1024) and the total
static __device__ int exl3_block_scan(int v, int * tmp, int & total) {
    const int lane = threadIdx.x % 32, wave = threadIdx.x / 32, n_waves = blockDim.x / 32;
    int x = v;
#pragma unroll
    for (int o = 1; o < 32; o *= 2) {
        const int y = __shfl_up(x, o, 32);
        x += lane >= o ? y : 0;
    }
    if (lane == 31) {
        tmp[wave] = x;
    }
    __syncthreads();
    if (wave == 0) {
        int t = lane < n_waves ? tmp[lane] : 0;
#pragma unroll
        for (int o = 1; o < 32; o *= 2) {
            const int y = __shfl_up(t, o, 32);
            t += lane >= o ? y : 0;
        }
        tmp[lane] = t;
    }
    __syncthreads();
    const int below = wave > 0 ? tmp[wave - 1] : 0;
    total = tmp[n_waves - 1];
    __syncthreads();
    return below + x - v;
}

// One block; each thread takes a contiguous range of the experts (a MoE cache bank has thousands of slots). The tiles
// are {first row in the expert-sorted order, rows (at most BM), 0, expert} for each expert's rows; chunk c takes the
// tiles whose first row is in [c * chunk_rows, (c + 1) * chunk_rows), so a chunk's rows span fewer than
// chunk_rows + BM. The chunks after the last are empty, so that the host can launch a bound of them.
template<int BM>
__launch_bounds__(EXL3_ROUTE_THREADS)
static __global__ void exl3_route_plan(int32_t * block_count, int n_blocks, int32_t * offset, int32_t * chunk_tile0,
                                       ggml_cuda_routed_chunk * chunks, int4 * tiles, int experts, int chunk_rows,
                                       int max_chunks) {
    __shared__ int tmp[32];
    const int per = (experts + blockDim.x - 1) / blockDim.x;
    const int e0 = min(experts, int(threadIdx.x) * per), e1 = min(experts, e0 + per);

    // the expert's rows in the blocks before each block, a batch of loads at a time; its total in offset for now
    int rows_sum = 0, tiles_sum = 0;
    for (int e = e0; e < e1; ++e) {
        constexpr int BATCH = 16;
        int rows = 0;
        for (int b0 = 0; b0 < n_blocks; b0 += BATCH) {
            int v[BATCH];
#pragma unroll
            for (int j = 0; j < BATCH; ++j) {
                v[j] = block_count[size_t(min(b0 + j, n_blocks - 1)) * experts + e];
            }
#pragma unroll
            for (int j = 0; j < BATCH; ++j) {
                if (b0 + j < n_blocks) {
                    block_count[size_t(b0 + j) * experts + e] = rows;
                    rows += v[j];
                }
            }
        }
        offset[e] = rows;
        rows_sum  += rows;
        tiles_sum += (rows + BM - 1) / BM;
    }
    int total_rows, total_tiles;
    const int row_base  = exl3_block_scan(rows_sum, tmp, total_rows);
    const int tile_base = exl3_block_scan(tiles_sum, tmp, total_tiles);
    for (int e = e0, row = row_base, tile = tile_base; e < e1; ++e) {
        const int rows = offset[e];
        offset[e] = row;
        for (int j = 0; j < (rows + BM - 1) / BM; ++j) {
            tiles[tile++] = make_int4(row + j * BM, min(BM, rows - j * BM), 0, e);
        }
        row += rows;
    }
    if (threadIdx.x == 0) {
        offset[experts] = total_rows;
    }
    __syncthreads();

    // a tile starts the chunks from the one after its predecessor's to its own (consecutive tiles start fewer than
    // BM <= chunk_rows rows apart, so at most one)
    const int n_chunks = total_tiles > 0 ? tiles[total_tiles - 1].x / chunk_rows + 1 : 0;
    if (n_chunks > max_chunks) {
        __builtin_trap();
    }
    for (int t = threadIdx.x; t < total_tiles; t += blockDim.x) {
        const int c = tiles[t].x / chunk_rows, prev = t > 0 ? tiles[t - 1].x / chunk_rows : -1;
        for (int cc = prev + 1; cc <= c; ++cc) {
            chunk_tile0[cc] = t;
        }
    }
    if (threadIdx.x == 0) {
        chunk_tile0[n_chunks] = total_tiles;
    }
    __syncthreads();
    for (int c = threadIdx.x; c < max_chunks; c += blockDim.x) {
        if (c < n_chunks) {
            const int t0 = chunk_tile0[c], t1 = chunk_tile0[c + 1];
            const int4 last = tiles[t1 - 1];
            const int row0 = tiles[t0].x;
            chunks[c] = { row0, last.x + last.y - row0, t0, t1 - t0 };
        } else {
            chunks[c] = {};
        }
    }
}

static __global__ void exl3_route_place(const int32_t * ids, const int32_t * block_count, const int32_t * offset,
                                        int32_t * perm, int assignments, int experts) {
    __shared__ int32_t s_e[EXL3_ROUTE_BLOCK];
    const int i = blockIdx.x * EXL3_ROUTE_BLOCK + threadIdx.x;
    const int e = i < assignments ? ids[i] : -1;
    const bool valid = e >= 0 && e < experts;
    s_e[threadIdx.x] = valid ? e : -1;
    __syncthreads();
    if (!valid) {
        return;
    }
    // the rank within the block: its earlier assignments to the same expert
    int rank = 0;
    for (int j = 0; j < int(threadIdx.x); ++j) {
        rank += s_e[j] == e;
    }
    perm[offset[e] + block_count[size_t(blockIdx.x) * experts + e] + rank] = i;
}

#ifdef GGML_CUDA_GEMM_WMMA
template<int bits_x2, int BM>
constexpr int exl3_gemm_smem() {
    using namespace gemm_wmma;
    return std::max(2 * (BM + BN) * LDK * 2 + 2 * 2 * (BN / 16) * (4 * bits_x2) * 4, YROWS * LDY * 4);
}

// One routed tile of the GEMM that decodes the bank in its W load, as gemm_f16_wmma_tile otherwise: rows x (m of them,
// rotated) times the columns col0.. of one expert's packed bank, in NT threads (a wave computes 32 x 64). Each K step
// of 32 is two K by eight N trellis tiles of the column block; their packed words go through LDS two steps ahead (a
// lane's fragment reads words that other lanes loaded), and the waves decode them into the W stage one step ahead.
// Waves whose rows are all past m skip their X loads and math.
template<int bits_x2, int BM, int NT>
static __device__ __forceinline__ void exl3_gemm_tile(char * smem, const half * __restrict__ x, const uint32_t * __restrict__ packed,
                                                      const half * __restrict__ svh, float * __restrict__ dst,
                                                      const int32_t * __restrict__ perm, int m, int n, int k, int col0, int vb) {
    using namespace gemm_wmma;
    constexpr int WM = BM / (NT / 32 / WAVES_N);
    constexpr int A_PIECES = BM * BK / 8 / NT;
    constexpr int STAGE = (BM + BN) * LDK;
    constexpr int TW = 4 * bits_x2;   // words per 16 x 16 trellis tile
    constexpr int SEG = BN / 16 * TW; // words of the column block's tiles in one K tile row
    constexpr int PW = 2 * SEG;       // and in one K step
    constexpr int P_PIECES = (PW + NT - 1) / NT;
    constexpr int TPW = 2 * (BN / 16) / (NT / 32); // trellis tiles decoded per wave and step
    static_assert(WM == 32 && A_PIECES >= 1, "a wave computes 32 rows");
    _Float16 * stages = (_Float16 *) smem;
    uint32_t * pk = (uint32_t *) (smem + 2 * STAGE * 2);

    const int tid = threadIdx.x, lane = tid % 32, warp = tid / 32;
    const int wm = warp / WAVES_N, wn = warp % WAVES_N;

    gemm_wmma_float8 acc[WM / 16][WN / 16];
#pragma unroll
    for (int i = 0; i < WM / 16; ++i) {
#pragma unroll
        for (int j = 0; j < WN / 16; ++j) { acc[i][j] = gemm_wmma_float8{0, 0, 0, 0, 0, 0, 0, 0}; }
    }

    const int n_tiles = n / 16;
    const __amdgpu_buffer_rsrc_t xr = gemm_wmma_rsrc(x, uint32_t(m) * k * 2);
    const __amdgpu_buffer_rsrc_t pr = gemm_wmma_rsrc(packed, uint32_t(k / 16) * n_tiles * TW * 4);
    uint32_t xo[A_PIECES], po[P_PIECES];
    bool xlive[A_PIECES];
#pragma unroll
    for (int p = 0; p < A_PIECES; ++p) {
        const int idx = tid + p * NT, r = idx / (BK / 8), c = (idx % (BK / 8)) * 8;
        // rows past m are not loaded: an output row depends only on its own input row and is not written
        xo[p] = (uint32_t(min(r, m - 1)) * k + c) * 2;
        xlive[p] = r < m;
    }
    const bool live = wm * WM < m;
#pragma unroll
    for (int p = 0; p < P_PIECES; ++p) {
        const int idx = min(tid + p * NT, PW - 1);
        po[p] = ((idx / SEG) * n_tiles * TW + col0 / 16 * TW + idx % SEG) * 4;
    }
    gemm_wmma_half8 ra[A_PIECES] = {};
    uint32_t rp[P_PIECES];
    const int kstart = (vb % 32) * 128 % k;

#define EG_KP(K0) ((K0) + kstart < k ? (K0) + kstart : (K0) + kstart - k)
#define EG_LOAD_X(K0)                                                                                    \
    {                                                                                                      \
        const int kp = EG_KP(K0);                                                                        \
        _Pragma("unroll") for (int p = 0; p < A_PIECES; ++p) { if (xlive[p]) { ra[p] = gemm_wmma_load16(xr, xo[p], kp * 2); } } \
    }
#define EG_LOAD_P(K0)                                                                                    \
    {                                                                                                      \
        const int kp = EG_KP(K0);                                                                        \
        _Pragma("unroll") for (int p = 0; p < P_PIECES; ++p) {                                             \
            rp[p] = __builtin_amdgcn_raw_buffer_load_b32(pr, (int) po[p], (int) (kp / 16 * n_tiles * TW * 4), 0); \
        }                                                                                                  \
    }
#define EG_STORE_X(STAGE_PTR)                                                                            \
    _Pragma("unroll") for (int p = 0; p < A_PIECES; ++p) {                                                 \
        const int idx = tid + p * NT, r = idx / (BK / 8), c = (idx % (BK / 8)) * 8;                       \
        *(gemm_wmma_half8 *) ((STAGE_PTR) + r * LDK + c) = ra[p];                                          \
    }
#define EG_STORE_P(PK)                                                                                   \
    _Pragma("unroll") for (int p = 0; p < P_PIECES; ++p) {                                                 \
        if (PW % NT == 0 || tid + p * NT < PW) { (PK)[tid + p * NT] = rp[p]; }                             \
    }
// lane holds k = (lane & 3) * 2 + (j & 1) + 8 * ((j >> 1) & 1) and n = lane / 4 + 8 * (j >> 2) of its tile
#define EG_DECODE(PK, STAGE_PTR)                                                                         \
    _Pragma("unroll") for (int h = 0; h < TPW; ++h) {                                                      \
        const int kt = (warp * TPW + h) / (BN / 16), nt = (warp * TPW + h) % (BN / 16);                  \
        half2 v[4];                                                                                        \
        ggml_cuda_exl3_fragment2<bits_x2>((PK) + kt * SEG + nt * TW, lane, v);                             \
        _Pragma("unroll") for (int j = 0; j < 4; ++j) {                                                    \
            const int r = (lane & 3) * 2 + 8 * (j & 1), c = lane / 4 + 8 * (j >> 1);                        \
            *(half2 *) ((STAGE_PTR) + (BM + nt * 16 + c) * LDK + kt * 16 + r) = v[j];                      \
        }                                                                                                  \
    }
// step K0 multiplies stage CUR; it decodes the next step's W from the packed words of buffer CUR ^ 1 into the other
// stage (both last read before the previous barrier), loads the next step's X and the packed words two steps ahead,
// and stores them into the other stage and packed buffer CUR (decoded in the previous step).
#define EG_STEP(K0, CUR)                                                                                 \
    {                                                                                                      \
        EG_LOAD_X((K0) + BK);                                                                            \
        EG_LOAD_P((K0) + 2 * BK);                                                                        \
        EG_DECODE(pk + ((CUR) ^ 1) * PW, stages + ((CUR) ^ 1) * STAGE);                                  \
        const _Float16 * as = stages + (CUR) * STAGE + (wm * WM + lane % 16) * LDK + 8 * (lane / 16);      \
        const _Float16 * bs = stages + (CUR) * STAGE + (BM + wn * WN + lane % 16) * LDK + 8 * (lane / 16); \
        if (live) _Pragma("unroll") for (int kk = 0; kk < BK / 16; ++kk) {                                 \
            gemm_wmma_half8 b[WN / 16];                                                                    \
            _Pragma("unroll") for (int j = 0; j < WN / 16; ++j) { b[j] = *(const gemm_wmma_half8 *) (bs + j * 16 * LDK + kk * 16); } \
            _Pragma("unroll") for (int i = 0; i < WM / 16; ++i) {                                          \
                const gemm_wmma_half8 a = *(const gemm_wmma_half8 *) (as + i * 16 * LDK + kk * 16);       \
                _Pragma("unroll") for (int j = 0; j < WN / 16; ++j) {                                      \
                    acc[i][j] = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32_gfx12(a, b[j], acc[i][j]);      \
                }                                                                                          \
            }                                                                                              \
        }                                                                                                  \
        __builtin_amdgcn_sched_barrier(0);                                                                 \
        EG_STORE_X(stages + ((CUR) ^ 1) * STAGE);                                                        \
        EG_STORE_P(pk + (CUR) * PW);                                                                     \
        gemm_wmma_lds_barrier();                                                                           \
    }

    EG_LOAD_P(0);
    EG_STORE_P(pk);
    EG_LOAD_P(BK);
    EG_STORE_P(pk + PW);
    EG_LOAD_X(0);
    EG_STORE_X(stages);
    gemm_wmma_lds_barrier();
    EG_DECODE(pk, stages);
    gemm_wmma_lds_barrier();
    for (int k0 = 0; k0 < k; k0 += 2 * BK) {
        EG_STEP(k0, 0);
        EG_STEP(k0 + BK, 1);
    }
#undef EG_STEP
#undef EG_DECODE
#undef EG_STORE_P
#undef EG_STORE_X
#undef EG_LOAD_P
#undef EG_LOAD_X
#undef EG_KP

    gemm_wmma_epilogue<BM, true, true, NT>(smem, acc, svh, dst, perm, m, n, 0, 0, col0);
}
#endif // GGML_CUDA_GEMM_WMMA

// The routed GEMM over one chunk of rows: block b computes column block b % (n / BN) of the chunk's tiles
// b / (n / BN), b / (n / BN) + gridDim.x / (n / BN), ...; x holds the chunk's rotated rows.
template<int bits_x2, int BM, int NT>
__launch_bounds__(NT)
static __global__ void exl3_routed_gemm_decode(const half * __restrict__ xh, const uint32_t * __restrict__ packed,
                                               const half * __restrict__ svh, float * __restrict__ dst,
                                               const int4 * __restrict__ tiles, const int32_t * __restrict__ perm,
                                               const ggml_cuda_routed_chunk * __restrict__ chunk, int n, int k, bool cache_order) {
#ifdef GGML_CUDA_GEMM_WMMA
    using namespace gemm_wmma;
    __shared__ __attribute__((aligned(16))) char smem[exl3_gemm_smem<bits_x2, BM>()];
    const int n_cb = n / BN, cb = blockIdx.x % n_cb;
    const int tile0 = chunk->tile0, n_tiles = chunk->tiles, row0 = chunk->row0;
    const size_t bank_words = size_t(k / 16) * (n / 16) * (4 * bits_x2);
    for (int t = blockIdx.x / n_cb; t < n_tiles; t += gridDim.x / n_cb) {
        const int4 tile = tiles[tile0 + t];
        // the K start staggers by the tile's position in the expert-sorted list, which for cached banks depends on the
        // slots the experts happen to occupy: those stagger by column block only, so moving an expert does not change
        // its accumulation order
        exl3_gemm_tile<bits_x2, BM, NT>(smem, xh + size_t(tile.x - row0) * k, packed + size_t(tile.w) * bank_words,
                                        svh + size_t(tile.w) * n, dst, perm + tile.x, tile.y, n, k, cb * BN,
                                        cache_order ? cb : (tile0 + t) * n_cb + cb);
        gemm_wmma_lds_barrier();
    }
#else
    GGML_UNUSED_VARS(xh, packed, svh, dst, tiles, perm, chunk, n, k, cache_order);
    NO_DEVICE_CODE;
#endif
}

// Routed banks above 8 rows, decoding the banks in the GEMM: sort the assignments by expert on the device, then per
// chunk of rows rotate them to FP16 and run the GEMM over the chunk's tiles of up to BM rows of one expert each. The
// host launches the chunks of all assignments; with invalid experts the last ones can be empty.
static void exl3_routed_gemm(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    using gemm_wmma::BN;
    constexpr int BM = 128, NT = 256;
    const int k = dst->src[2]->ne[0], n = dst->ne[0], used = dst->ne[1], assignments = dst->ne[1] * dst->ne[2];
    const int experts = dst->src[0]->ne[2];
    cudaStream_t stream = ctx.stream();

    // a chunk's rotated rows take at most 32 MiB, so they stay in the Infinity Cache for its column blocks
    const int chunk_rows = std::max<int>(BM, (32ll << 20) / (2ll * k));
    const int max_chunks = (assignments + chunk_rows - 1) / chunk_rows;
    const int max_tiles  = std::min(experts, assignments) + (assignments + BM - 1) / BM;
    const int n_blocks   = (assignments + EXL3_ROUTE_BLOCK - 1) / EXL3_ROUTE_BLOCK;

    // workspace: block counts, offsets, sorted assignments, chunk starts, then 16-byte aligned chunks and tiles
    const size_t o_offset = size_t(n_blocks) * experts, o_perm = o_offset + experts + 1;
    const size_t o_tile0 = o_perm + assignments, o_chunks = (o_tile0 + max_chunks + 1 + 3) / 4 * 4;
    const size_t o_tiles = o_chunks + size_t(max_chunks) * sizeof(ggml_cuda_routed_chunk) / sizeof(int32_t);
    ggml_cuda_pool_alloc<int32_t> ws(ctx.pool(), o_tiles + size_t(max_tiles) * 4);
    const int32_t * ids = (const int32_t *) dst->src[4]->data;
    ggml_cuda_routed_chunk * chunks = (ggml_cuda_routed_chunk *) (ws.get() + o_chunks);
    int4 * tiles = (int4 *) (ws.get() + o_tiles);

    exl3_route_hist<<<n_blocks, EXL3_ROUTE_BLOCK, experts * sizeof(int32_t), stream>>>(ids, ws.get(), (uint32_t *) dst->data,
                                                                                         assignments, experts, n);
    exl3_route_plan<BM><<<1, EXL3_ROUTE_THREADS, 0, stream>>>(ws.get(), n_blocks, ws.get() + o_offset, ws.get() + o_tile0,
                                                              chunks, tiles, experts, chunk_rows, max_chunks);
    exl3_route_place<<<n_blocks, EXL3_ROUTE_BLOCK, 0, stream>>>(ids, ws.get(), ws.get() + o_offset, ws.get() + o_perm,
                                                                 assignments, experts);

    const int xh_rows = std::min(chunk_rows + BM, assignments);
    ggml_cuda_pool_alloc<half> xh(ctx.pool(), size_t(xh_rows) * k);
    const ggml_cuda_exl3_args args = {
        (const uint32_t *) dst->src[0]->data, (const half *) dst->src[2]->data, (const half *) dst->src[3]->data,
        (const float *) dst->src[1]->data, ids, (float *) dst->data, xh.get(), nullptr,
        k, n, experts, assignments, used, int(dst->src[1]->ne[1]), ggml_get_op_params_i32(dst, 0), ws.get() + o_perm};
    const int input_blocks = (xh_rows + EXL3_TRANSFORM_ROWS - 1) / EXL3_TRANSFORM_ROWS;
    // the GEMM blocks loop over the chunk's tiles, about 16 blocks per WGP (nsm counts WGPs; 4 were 2-7% slower, 32
    // no faster)
    const int n_cb = n / BN;
    const int blocks = n_cb * std::max(1, std::min(max_tiles, 16 * ggml_cuda_info().devices[ctx.device].nsm / n_cb));
    const bool cache_order = ggml_get_op_params_i32(dst, 1) != 0;
    for (int c = 0; c < max_chunks; ++c) {
        exl3_routed_input<<<dim3(k / 128, input_blocks), 32 * EXL3_TRANSFORM_ROWS, 0, stream>>>(args, chunks + c);
        switch (args.bits_x2) {
            case 7:  exl3_routed_gemm_decode<7,  BM, NT><<<blocks, NT, 0, stream>>>(xh.get(), args.packed, args.svh, args.dst, tiles, args.perm, chunks + c, n, k, cache_order); break;
            case 8:  exl3_routed_gemm_decode<8,  BM, NT><<<blocks, NT, 0, stream>>>(xh.get(), args.packed, args.svh, args.dst, tiles, args.perm, chunks + c, n, k, cache_order); break;
            case 10: exl3_routed_gemm_decode<10, BM, NT><<<blocks, NT, 0, stream>>>(xh.get(), args.packed, args.svh, args.dst, tiles, args.perm, chunks + c, n, k, cache_order); break;
            case 12: exl3_routed_gemm_decode<12, BM, NT><<<blocks, NT, 0, stream>>>(xh.get(), args.packed, args.svh, args.dst, tiles, args.perm, chunks + c, n, k, cache_order); break;
            case 14: exl3_routed_gemm_decode<14, BM, NT><<<blocks, NT, 0, stream>>>(xh.get(), args.packed, args.svh, args.dst, tiles, args.perm, chunks + c, n, k, cache_order); break;
            default: GGML_ABORT("Unsupported EXL3 rate");
        }
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
            exl3_decode_f16_launch(args, w.get(), n0, n_cols, stream);
            const int blocks = n_cols / gemm_wmma::BN * ((assignments + 127) / 128);
            gemm_f16_wmma<128, true><<<blocks, gemm_wmma::THREADS, 0, stream>>>(xh.get(), w.get(), args.svh, args.dst,
                                                                      assignments, n, k, n0);
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
