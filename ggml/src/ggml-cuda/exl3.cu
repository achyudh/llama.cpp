#include "exl3.cuh"
#include "gemm-wmma.cuh"
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
    using gemm_wmma::BN;
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
        gemm_f16_wmma<BM, true, true><<<blocks, gemm_wmma::THREADS, 0, stream>>>(xh.get(), w.get(), args.svh, args.dst, 0, n, k, 0,
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
            const int blocks = n_cols / gemm_wmma::BN * ((assignments + 127) / 128);
            gemm_f16_wmma<128, false, true><<<blocks, gemm_wmma::THREADS, 0, stream>>>(xh.get(), w.get(), args.svh, args.dst,
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
