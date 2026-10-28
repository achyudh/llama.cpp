#include "exl3.cuh"
#include "gemm-wmma.cuh"
#include "ggml-exl3.h"

#define EXL3_ROUTE_BLOCK       256 // assignments per block of the routing sort
#define EXL3_ROUTE_MAX_EXPERTS 1024

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
    // routed one plans its chunks in one block with a thread per expert.
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

// Decode columns [n0, n0 + gridDim.x * 16) of a bank into FP16 w[col - n0][k], k contiguous, for a GEMM: the dense
// bank, or the bank of expert experts[blockIdx.z] into slot blockIdx.z of w. With a chunk of the routed GEMM, the
// experts are the chunk's, from experts[chunk.expert0].
// A block decodes eight 16 x 16 tiles of one column strip and writes each column's 128 values contiguously.
template<int bits_x2>
static __global__ void exl3_decode_f16(const uint32_t * packed, const int32_t * experts, const ggml_cuda_routed_chunk * chunk,
                                       half * w, int k, int n, int n0) {
    __shared__ half strip[16][128 + 8];
    const int lane = threadIdx.x % 32, warp = threadIdx.x / 32;
    if (chunk) {
        if (int(blockIdx.z) >= chunk->experts) {
            return;
        }
        experts += chunk->expert0;
    }
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

static void exl3_decode_f16_launch(const ggml_cuda_exl3_args & a, const int32_t * experts, const ggml_cuda_routed_chunk * chunk,
                                   int n_experts, half * w, int n0, int n_cols, cudaStream_t stream) {
    const dim3 grid(n_cols / 16, a.k / 128, n_experts);
    switch (a.bits_x2) {
        case 7:  exl3_decode_f16<7 ><<<grid, 128, 0, stream>>>(a.packed, experts, chunk, w, a.k, a.n, n0); break;
        case 8:  exl3_decode_f16<8 ><<<grid, 128, 0, stream>>>(a.packed, experts, chunk, w, a.k, a.n, n0); break;
        case 10: exl3_decode_f16<10><<<grid, 128, 0, stream>>>(a.packed, experts, chunk, w, a.k, a.n, n0); break;
        case 12: exl3_decode_f16<12><<<grid, 128, 0, stream>>>(a.packed, experts, chunk, w, a.k, a.n, n0); break;
        case 14: exl3_decode_f16<14><<<grid, 128, 0, stream>>>(a.packed, experts, chunk, w, a.k, a.n, n0); break;
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

// One block, a thread per expert. The chunks are groups of max_experts used experts, as the decoded weights of one
// chunk have to fit; a group with more than chunk_tiles tiles is split into chunks of chunk_tiles tiles, so that the
// rotated rows of a chunk fit (its first and last experts can then be in two chunks). The chunks after the last are
// empty, so that the host can launch a bound of them.
template<int BM>
__launch_bounds__(EXL3_ROUTE_MAX_EXPERTS)
static __global__ void exl3_route_plan(int32_t * block_count, int n_blocks, int32_t * offset, int32_t * used_experts,
                                       ggml_cuda_routed_chunk * chunks, int4 * tiles, int experts, int max_experts,
                                       int chunk_tiles, int max_chunks) {
    __shared__ int tmp[32];
    __shared__ int s_offset[EXL3_ROUTE_MAX_EXPERTS], s_tile[EXL3_ROUTE_MAX_EXPERTS], s_used[EXL3_ROUTE_MAX_EXPERTS];
    __shared__ int s_chunk[EXL3_ROUTE_MAX_EXPERTS];
    const int e = threadIdx.x;

    // the expert's rows in the blocks before each block, a batch of loads at a time
    int rows = 0;
    if (e < experts) {
        constexpr int BATCH = 16;
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
    }
    const int n_tiles = (rows + BM - 1) / BM;
    int total_rows, total_tiles, n_used;
    const int row0  = exl3_block_scan(rows, tmp, total_rows);
    const int tile0 = exl3_block_scan(n_tiles, tmp, total_tiles);
    const int used  = exl3_block_scan(rows > 0, tmp, n_used);
    if (e < experts) {
        offset[e]   = row0;
        s_offset[e] = row0;
        s_tile[e]   = tile0;
        if (rows > 0) {
            used_experts[used] = e;
            s_used[used] = e;
        }
    }
    if (e == 0) {
        offset[experts] = total_rows;
    }
    __syncthreads();

    // chunks: group g of used experts, split by tiles
    const int n_groups = (n_used + max_experts - 1) / max_experts;
    const int g = e;
    int g_tile0 = 0, g_tiles = 0;
    if (g < n_groups) {
        g_tile0 = s_tile[s_used[g * max_experts]];
        g_tiles = ((g + 1) * max_experts < n_used ? s_tile[s_used[(g + 1) * max_experts]] : total_tiles) - g_tile0;
    }
    const int n_split = (g_tiles + chunk_tiles - 1) / chunk_tiles;
    int n_chunks;
    const int g_chunk0 = exl3_block_scan(n_split, tmp, n_chunks);
    if (n_chunks > max_chunks) {
        __builtin_trap();
    }
    if (g < n_groups) {
        s_chunk[g] = g_chunk0;
        // the used experts of the group, in order, own its tiles; tile t belongs to the last expert starting at or before it
        const int u0 = g * max_experts, u1 = min(n_used, u0 + max_experts);
        for (int j = 0; j < n_split; ++j) {
            const int t0 = g_tile0 + j * chunk_tiles, t1 = min(g_tile0 + g_tiles, t0 + chunk_tiles);
            int uf = u0, ul = u0;
            for (int u = u0; u < u1; ++u) {
                const int tu = s_tile[s_used[u]];
                uf = tu <= t0     ? u : uf;
                ul = tu <= t1 - 1 ? u : ul;
            }
            const int ef = s_used[uf], el = s_used[ul];
            const int r0 = s_offset[ef] + (t0 - s_tile[ef]) * BM;
            const int r1 = min(s_offset[el] + (t1 - s_tile[el]) * BM, ul + 1 < n_used ? s_offset[s_used[ul + 1]] : total_rows);
            chunks[g_chunk0 + j] = { r0, r1 - r0, t0, t1 - t0, uf, ul - uf + 1 };
        }
    }
    for (int c = n_chunks + e; c < max_chunks; c += blockDim.x) {
        chunks[c] = {};
    }
    __syncthreads();

    // the tiles of each used expert, relative to their chunk
    if (e < experts && rows > 0) {
        const int grp = used / max_experts;
        const int g0  = s_tile[s_used[grp * max_experts]];
        for (int j = 0; j < n_tiles; ++j) {
            const int t = tile0 + j;
            const ggml_cuda_routed_chunk & c = chunks[s_chunk[grp] + (t - g0) / chunk_tiles];
            tiles[t] = make_int4(row0 + j * BM - c.row0, min(BM, rows - j * BM), used - c.expert0, e);
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

// Routed banks above 8 rows: sort the assignments by expert on the device, then per chunk of experts rotate their
// rows to FP16, decode their banks to FP16 and run the GEMM over the chunk's tiles of up to BM rows each. The host
// launches a bound of chunks for the shapes; the chunks past the last are empty.
static void exl3_routed_gemm(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    using gemm_wmma::BN;
    // a used expert gets a median of 38 rows of a 4096-token batch (10 of 512 experts per token)
    constexpr int BM = 64;
    const int k = dst->src[2]->ne[0], n = dst->ne[0], used = dst->ne[1], assignments = dst->ne[1] * dst->ne[2];
    const int experts = dst->src[0]->ne[2];
    cudaStream_t stream = ctx.stream();

    // decoded chunks of 32 MiB stay in the 64 MiB Infinity Cache until the GEMM reads them: at 4096 tokens 16 and
    // 64 MiB are 30% slower, 128 MiB and more 45-60%
    const int chunk_experts = std::max<int>(1, (32ll << 20) / (2ll * n * k));
    // and a chunk's rotated rows take at most 32 MiB
    const int chunk_tiles = std::max<int>(2, (32ll << 20) / (2ll * k) / BM);
    const int chunk_rows  = chunk_tiles * BM;
    const int max_used    = std::min(experts, assignments);
    const int max_tiles   = max_used + (assignments + BM - 1) / BM;
    // a group of chunk_experts used experts is one chunk per chunk_tiles of its tiles, rounded up
    const int max_chunks  = (max_used + chunk_experts - 1) / chunk_experts + max_tiles / chunk_tiles;
    const int n_blocks    = (assignments + EXL3_ROUTE_BLOCK - 1) / EXL3_ROUTE_BLOCK;

    // workspace: block counts, offsets, sorted assignments, used experts, then 16-byte aligned chunks and tiles
    const size_t o_offset = size_t(n_blocks) * experts, o_perm = o_offset + experts + 1, o_used = o_perm + assignments;
    const size_t o_chunks = (o_used + experts + 3) / 4 * 4;
    const size_t o_tiles = o_chunks + size_t(max_chunks) * sizeof(ggml_cuda_routed_chunk) / sizeof(int32_t);
    ggml_cuda_pool_alloc<int32_t> ws(ctx.pool(), o_tiles + size_t(max_tiles) * 4);
    const int32_t * ids = (const int32_t *) dst->src[4]->data;
    ggml_cuda_routed_chunk * chunks = (ggml_cuda_routed_chunk *) (ws.get() + o_chunks);
    int4 * tiles = (int4 *) (ws.get() + o_tiles);

    exl3_route_hist<<<n_blocks, EXL3_ROUTE_BLOCK, experts * sizeof(int32_t), stream>>>(ids, ws.get(), (uint32_t *) dst->data,
                                                                                         assignments, experts, n);
    exl3_route_plan<BM><<<1, (experts + 31) / 32 * 32, 0, stream>>>(ws.get(), n_blocks, ws.get() + o_offset, ws.get() + o_used,
                                                                    chunks, tiles, experts, chunk_experts, chunk_tiles, max_chunks);
    exl3_route_place<<<n_blocks, EXL3_ROUTE_BLOCK, 0, stream>>>(ids, ws.get(), ws.get() + o_offset, ws.get() + o_perm,
                                                                 assignments, experts);

    // launch sizes for the expected chunk, the rows of chunk_experts experts with the average rows; the kernels loop
    const int n_slots = std::min(chunk_experts, max_used);
    const int typical_rows = std::min<int>(chunk_rows, (int64_t(assignments) * n_slots + max_used - 1) / max_used);
    const int input_blocks = (typical_rows + typical_rows / 2 + EXL3_TRANSFORM_ROWS - 1) / EXL3_TRANSFORM_ROWS;
    const int tile_blocks = n_slots + (typical_rows + typical_rows / 2 + BM - 1) / BM;

    ggml_cuda_pool_alloc<half> xh(ctx.pool(), size_t(std::min(chunk_rows, assignments)) * k);
    ggml_cuda_pool_alloc<half> w(ctx.pool(), size_t(n_slots) * n * k);
    const ggml_cuda_exl3_args args = {
        (const uint32_t *) dst->src[0]->data, (const half *) dst->src[2]->data, (const half *) dst->src[3]->data,
        (const float *) dst->src[1]->data, ids, (float *) dst->data, xh.get(), nullptr,
        k, n, experts, assignments, used, int(dst->src[1]->ne[1]), ggml_get_op_params_i32(dst, 0), ws.get() + o_perm};
    for (int c = 0; c < max_chunks; ++c) {
        exl3_routed_input<<<dim3(k / 128, input_blocks), 32 * EXL3_TRANSFORM_ROWS, 0, stream>>>(args, chunks + c);
        exl3_decode_f16_launch(args, ws.get() + o_used, chunks + c, n_slots, w.get(), 0, n, stream);
        gemm_f16_wmma<BM, true, true><<<n / BN * tile_blocks, gemm_wmma::THREADS, 0, stream>>>(xh.get(), w.get(), args.svh,
                                                                                    args.dst, 0, n, k, 0, tiles, args.perm, chunks + c);
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
            exl3_decode_f16_launch(args, nullptr, nullptr, 1, w.get(), n0, n_cols, stream);
            const int blocks = n_cols / gemm_wmma::BN * ((assignments + 127) / 128);
            gemm_f16_wmma<128, false, true><<<blocks, gemm_wmma::THREADS, 0, stream>>>(xh.get(), w.get(), args.svh, args.dst,
                                                                            assignments, n, k, n0, nullptr, nullptr, nullptr);
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
