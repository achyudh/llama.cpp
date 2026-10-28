#include "top-k-mask.cuh"
#include "convert.cuh"

#define CUDA_TOP_K_MASK_BLOCK_SIZE 512

// one block per row: fill the row with -inf, then copy the mask cells of the live picks and the extra cells
template <typename T>
static __global__ void top_k_mask_kernel(
        const T * mask, const int32_t * top_k, const float * score, const int32_t * blocks, const int32_t * extra, T * dst,
        const int64_t n_kv, const int64_t n_mask, const int64_t ne1, const int64_t ne2,
        const int64_t m1, const int64_t m2, const int64_t m3,
        const int64_t n_top, const int64_t n_pick, const int64_t n_blk, const int64_t n_extra,
        const int64_t sm1, const int64_t sm2, const int64_t sm3,
        const int64_t sd1, const int64_t sd2, const int64_t sd3,
        const int64_t sk0, const int64_t sk1, const int64_t sk2, const int64_t sk3,
        const int64_t ss0, const int64_t ss1, const int64_t ss2, const int64_t ss3,
        const int64_t sb0, const int64_t sb1,
        const int64_t se0, const int64_t se1, const int64_t se2, const int64_t se3,
        const bool vec) {
    ggml_cuda_pdl_lc();

    const int64_t ir = blockIdx.x;
    const int64_t i1 = ir % ne1;
    const int64_t i2 = (ir / ne1) % ne2;
    const int64_t i3 = ir / (ne1*ne2);

    const T       * mask_row  = mask  + (i1 % m1)*sm1 + (i2 % m2)*sm2 + (i3 % m3)*sm3;
    T             * dst_row   = dst   + i1*sd1 + i2*sd2 + i3*sd3;
    const int32_t * top_k_row = top_k + i1*sk1 + i2*sk2 + i3*sk3;

    ggml_cuda_pdl_sync();

    const T ninf = ggml_cuda_cast<T>(-INFINITY);
    const int64_t col0 = n_kv - n_mask;
    if (vec) {
        constexpr int nv = sizeof(int4)/sizeof(T);
        T fill[nv];
#pragma unroll
        for (int j = 0; j < nv; ++j) {
            fill[j] = ninf;
        }
        const int4 fill4 = *(const int4 *) fill;
        for (int64_t i = threadIdx.x; i < n_kv/nv; i += blockDim.x) {
            ((int4 *) dst_row)[i] = fill4;
        }
    } else {
        for (int64_t i = threadIdx.x; i < n_kv; i += blockDim.x) {
            dst_row[i] = ninf;
        }
    }

    __syncthreads();

    // one thread per pick or extra cell; cells kept twice get the same value
    for (int64_t i = threadIdx.x; i < n_top + n_extra; i += blockDim.x) {
        if (i >= n_top) {
            const int32_t c = extra[(i - n_top)*se0 + i1*se1 + i2*se2 + i3*se3];
            if (c >= 0 && c < n_kv) {
                dst_row[c] = c < col0 ? ggml_cuda_cast<T>(0.0f) : mask_row[c - col0];
            }
            continue;
        }

        const int32_t p = top_k_row[i*sk0];
        if (p < 0 || p >= n_pick) {
            continue;
        }
        if (score && !(score[p*ss0 + i1*ss1 + i2*ss2 + i3*ss3] > -INFINITY)) {
            continue;
        }
        if (!blocks) {
            if (p < n_kv) {
                dst_row[p] = p < col0 ? ggml_cuda_cast<T>(0.0f) : mask_row[p - col0];
            }
            continue;
        }
        for (int64_t k = 0; k < n_blk; ++k) {
            const int32_t c = blocks[k*sb0 + p*sb1];
            if (c >= 0 && c < n_kv) {
                dst_row[c] = c < col0 ? ggml_cuda_cast<T>(0.0f) : mask_row[c - col0];
            }
        }
    }
}

template <typename T>
static void top_k_mask_cuda(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * mask   = dst->src[0];
    const ggml_tensor * top_k  = dst->src[1];
    const ggml_tensor * score  = dst->src[2];
    const ggml_tensor * blocks = dst->src[3];
    const ggml_tensor * extra  = dst->src[4];

    const int64_t n_kv = dst->ne[0];

    GGML_ASSERT(dst->nb[0] == sizeof(T) && mask->nb[0] == sizeof(T));
    GGML_ASSERT(dst->nb[1] % sizeof(T) == 0 && mask->nb[1] % sizeof(T) == 0);

    // 16-byte stores need every dst row aligned
    const bool vec = (n_kv*sizeof(T)) % sizeof(int4) == 0 &&
        dst->nb[1] % sizeof(int4) == 0 && dst->nb[2] % sizeof(int4) == 0 && dst->nb[3] % sizeof(int4) == 0 &&
        (uintptr_t) dst->data % sizeof(int4) == 0;

    const int64_t n_pick = blocks ? blocks->ne[1] : score ? score->ne[0] : n_kv;

    const size_t si = sizeof(int32_t);
    const size_t sf = sizeof(float);

    const dim3 block_dims(CUDA_TOP_K_MASK_BLOCK_SIZE, 1, 1);
    const dim3 grid_dims(ggml_nrows(dst), 1, 1);
    const ggml_cuda_kernel_launch_params launch_params = ggml_cuda_kernel_launch_params(grid_dims, block_dims, 0, ctx.stream());

    ggml_cuda_kernel_launch(top_k_mask_kernel<T>, launch_params,
            (const T *) mask->data, (const int32_t *) top_k->data,
            score  ? (const float   *) score->data  : nullptr,
            blocks ? (const int32_t *) blocks->data : nullptr,
            extra  ? (const int32_t *) extra->data  : nullptr,
            (T *) dst->data,
            n_kv, mask->ne[0], dst->ne[1], dst->ne[2],
            mask->ne[1], mask->ne[2], mask->ne[3],
            top_k->ne[0], n_pick, blocks ? blocks->ne[0] : 1, extra ? extra->ne[0] : 0,
            mask->nb[1]/sizeof(T), mask->nb[2]/sizeof(T), mask->nb[3]/sizeof(T),
            dst->nb[1]/sizeof(T),  dst->nb[2]/sizeof(T),  dst->nb[3]/sizeof(T),
            top_k->nb[0]/si, top_k->nb[1]/si, top_k->nb[2]/si, top_k->nb[3]/si,
            score  ? score->nb[0]/sf  : 0, score ? score->nb[1]/sf : 0, score ? score->nb[2]/sf : 0, score ? score->nb[3]/sf : 0,
            blocks ? blocks->nb[0]/si : 0, blocks ? blocks->nb[1]/si : 0,
            extra  ? extra->nb[0]/si  : 0, extra ? extra->nb[1]/si : 0, extra ? extra->nb[2]/si : 0, extra ? extra->nb[3]/si : 0,
            vec);
}

// ggml_top_k_cells: one block per row; per chunk of TOP_K_CELLS_CHUNK cells, the kept cells with a finite mask value set
// their bits in an LDS bitmap (an OR, so the result does not depend on the order), then a scan of the words' counts
// writes them in ascending order after those of the earlier chunks; the rest of the row is -1
#define TOP_K_CELLS_CHUNK (1 << 18)
#define TOP_K_CELLS_WORDS (TOP_K_CELLS_CHUNK/32)
#define TOP_K_CELLS_PER_THREAD (TOP_K_CELLS_WORDS/CUDA_TOP_K_MASK_BLOCK_SIZE)

template <typename T>
static __global__ void top_k_cells_kernel(
        const T * mask, const int32_t * top_k, const float * score, const int32_t * blocks, const int32_t * extra,
        int32_t * dst, const int64_t n_kv, const int64_t n_mask, const int64_t n_list, const int64_t ne1, const int64_t ne2,
        const int64_t m1, const int64_t m2, const int64_t m3,
        const int64_t n_top, const int64_t n_pick, const int64_t n_blk, const int64_t n_extra,
        const int64_t sm1, const int64_t sm2, const int64_t sm3,
        const int64_t sd1, const int64_t sd2, const int64_t sd3,
        const int64_t sk0, const int64_t sk1, const int64_t sk2, const int64_t sk3,
        const int64_t ss0, const int64_t ss1, const int64_t ss2, const int64_t ss3,
        const int64_t sb0, const int64_t sb1,
        const int64_t se0, const int64_t se1, const int64_t se2, const int64_t se3) {
    static_assert(TOP_K_CELLS_WORDS % CUDA_TOP_K_MASK_BLOCK_SIZE == 0, "whole words per thread");
    constexpr int nw = CUDA_TOP_K_MASK_BLOCK_SIZE/WARP_SIZE;

    ggml_cuda_pdl_lc();

    const int64_t ir = blockIdx.x;
    const int64_t i1 = ir % ne1;
    const int64_t i2 = (ir / ne1) % ne2;
    const int64_t i3 = ir / (ne1*ne2);

    const T       * mask_row  = mask  + (i1 % m1)*sm1 + (i2 % m2)*sm2 + (i3 % m3)*sm3;
    int32_t       * dst_row   = dst   + i1*sd1 + i2*sd2 + i3*sd3;
    const int32_t * top_k_row = top_k + i1*sk1 + i2*sk2 + i3*sk3;

    __shared__ uint32_t bits[TOP_K_CELLS_WORDS];
    __shared__ int      warp_sum[nw];

    ggml_cuda_pdl_sync();

    const int     tid  = threadIdx.x;
    const int     lane = tid % WARP_SIZE;
    const int     warp = tid / WARP_SIZE;
    const int64_t col0 = n_kv - n_mask;

    int64_t base = 0; // cells written by the earlier chunks
    for (int64_t c0 = 0; c0 < n_kv; c0 += TOP_K_CELLS_CHUNK) {
        for (int i = tid; i < TOP_K_CELLS_WORDS; i += CUDA_TOP_K_MASK_BLOCK_SIZE) {
            bits[i] = 0;
        }
        __syncthreads();

        auto keep = [&](const int32_t c) {
            if (c >= c0 && c < c0 + TOP_K_CELLS_CHUNK && c < n_kv &&
                    (c < col0 || isfinite(ggml_cuda_cast<float>(mask_row[c - col0])))) {
                atomicOr(&bits[(c - c0)/32], 1u << ((c - c0) % 32));
            }
        };

        for (int64_t i = tid; i < n_top + n_extra; i += CUDA_TOP_K_MASK_BLOCK_SIZE) {
            if (i >= n_top) {
                keep(extra[(i - n_top)*se0 + i1*se1 + i2*se2 + i3*se3]);
                continue;
            }
            const int32_t p = top_k_row[i*sk0];
            if (p < 0 || p >= n_pick) {
                continue;
            }
            if (score && !(score[p*ss0 + i1*ss1 + i2*ss2 + i3*ss3] > -INFINITY)) {
                continue;
            }
            if (!blocks) {
                keep(p);
                continue;
            }
            for (int64_t k = 0; k < n_blk; ++k) {
                keep(blocks[k*sb0 + p*sb1]);
            }
        }
        __syncthreads();

        // the thread's words are consecutive, so the scan of the threads' counts orders the cells
        int cnt = 0;
#pragma unroll
        for (int w = 0; w < TOP_K_CELLS_PER_THREAD; ++w) {
            cnt += __popc(bits[tid*TOP_K_CELLS_PER_THREAD + w]);
        }
        int incl = cnt;
#pragma unroll
        for (int o = 1; o < WARP_SIZE; o *= 2) {
            const int t = __shfl_up_sync(0xffffffff, incl, o, WARP_SIZE);
            if (lane >= o) {
                incl += t;
            }
        }
        if (lane == WARP_SIZE - 1) {
            warp_sum[warp] = incl;
        }
        __syncthreads();
        int before = 0;
        int total  = 0;
        for (int w = 0; w < nw; ++w) {
            before += w < warp ? warp_sum[w] : 0;
            total  += warp_sum[w];
        }
        int64_t pos = base + before + incl - cnt;
        for (int w = 0; w < TOP_K_CELLS_PER_THREAD; ++w) {
            uint32_t b = bits[tid*TOP_K_CELLS_PER_THREAD + w];
            while (b) {
                const int j = __ffs(b) - 1;
                b &= b - 1;
                if (pos < n_list) {
                    dst_row[pos] = (int32_t) (c0 + (int64_t) (tid*TOP_K_CELLS_PER_THREAD + w)*32 + j);
                }
                ++pos;
            }
        }
        base += total;
        __syncthreads();
    }

    for (int64_t i = base + tid; i < n_list; i += CUDA_TOP_K_MASK_BLOCK_SIZE) {
        dst_row[i] = -1;
    }
}

template <typename T>
static void top_k_cells_cuda(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * mask   = dst->src[0];
    const ggml_tensor * top_k  = dst->src[1];
    const ggml_tensor * score  = dst->src[2];
    const ggml_tensor * blocks = dst->src[3];
    const ggml_tensor * extra  = dst->src[4];

    const int64_t n_kv   = ggml_get_op_params_i32(dst, 0);
    const int64_t n_pick = blocks ? blocks->ne[1] : score ? score->ne[0] : n_kv;

    GGML_ASSERT(dst->nb[0] == sizeof(int32_t) && mask->nb[0] == sizeof(T));

    const size_t si = sizeof(int32_t);
    const size_t sf = sizeof(float);
    const size_t st = sizeof(T);

    const dim3 block_dims(CUDA_TOP_K_MASK_BLOCK_SIZE, 1, 1);
    const dim3 grid_dims(ggml_nrows(dst), 1, 1);
    const ggml_cuda_kernel_launch_params launch_params = ggml_cuda_kernel_launch_params(grid_dims, block_dims, 0, ctx.stream());

    ggml_cuda_kernel_launch(top_k_cells_kernel<T>, launch_params,
            (const T *) mask->data, (const int32_t *) top_k->data,
            score  ? (const float   *) score->data  : nullptr,
            blocks ? (const int32_t *) blocks->data : nullptr,
            extra  ? (const int32_t *) extra->data  : nullptr,
            (int32_t *) dst->data,
            n_kv, mask->ne[0], dst->ne[0], dst->ne[1], dst->ne[2],
            mask->ne[1], mask->ne[2], mask->ne[3],
            top_k->ne[0], n_pick, blocks ? blocks->ne[0] : 1, extra ? extra->ne[0] : 0,
            mask->nb[1]/st, mask->nb[2]/st, mask->nb[3]/st,
            dst->nb[1]/si,  dst->nb[2]/si,  dst->nb[3]/si,
            top_k->nb[0]/si, top_k->nb[1]/si, top_k->nb[2]/si, top_k->nb[3]/si,
            score  ? score->nb[0]/sf  : 0, score ? score->nb[1]/sf : 0, score ? score->nb[2]/sf : 0, score ? score->nb[3]/sf : 0,
            blocks ? blocks->nb[0]/si : 0, blocks ? blocks->nb[1]/si : 0,
            extra  ? extra->nb[0]/si  : 0, extra ? extra->nb[1]/si : 0, extra ? extra->nb[2]/si : 0, extra ? extra->nb[3]/si : 0);
}

void ggml_cuda_op_top_k_mask(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    if (dst->type == GGML_TYPE_I32) {
        if (dst->src[0]->type == GGML_TYPE_F16) {
            top_k_cells_cuda<half>(ctx, dst);
        } else {
            top_k_cells_cuda<float>(ctx, dst);
        }
        return;
    }
    switch (dst->type) {
        case GGML_TYPE_F16:
            top_k_mask_cuda<half>(ctx, dst);
            break;
        case GGML_TYPE_F32:
            top_k_mask_cuda<float>(ctx, dst);
            break;
        default:
            GGML_ABORT("unsupported type");
    }
}
