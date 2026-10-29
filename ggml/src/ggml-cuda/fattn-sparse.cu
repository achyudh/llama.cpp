#include "fattn-sparse.cuh"
#include "fattn-common.cuh"
#include "fattn-paged-decode.cuh"

// Two kernels and the stock split combine:
//   1. fs_mask_count / fs_mask_write: per query row, the ascending list of the mask columns that are finite (at most
//      n_kv_max) and its length. A block per 2048 columns of a row counts them, then writes them after the counts of
//      the chunks before it (a ballot per 32 columns), so the list is in column order.
//   2. fs_attend: grid (split, KV head, query row), one wave per query head of the KV head (GQA G). The block stages
//      16 rows of its split at a time, K and V dequantized to f32 in LDS, so each selected row is read and converted
//      once for the G heads. A wave scores the staged rows with a lane per row (f32 dot against Q in LDS, the mask value
//      added), does one online-softmax step over them, then accumulates P V with a lane per D/32 output dims.
//      A prefill (more than FS_MAX_Q rows) has one split per row, so that a row's output depends only on its own list,
//      and takes fs_attend_heads when G is a multiple of FS_HPW (the waves split the heads, not the rows; see there).
//   3. flash_attn_combine_results merges the splits (skipped with one split).

#define FS_MAX_Q  8   // query rows per sequence of a decode step; more rows are a prefill (see FS_PREFILL_MIN_KV)

static __device__ __forceinline__ uint32_t fs_ballot(const bool p) {
#ifdef GGML_USE_HIP
    return (uint32_t) __ballot(p);
#else
    return __ballot_sync(0xffffffff, p);
#endif // GGML_USE_HIP
}

// mask columns per block of the compaction: 256 threads x 8
#define FS_CHUNK 2048

static __device__ __forceinline__ bool fs_live(const half * mask, const int i, const int n_kv) {
    return i < n_kv && isfinite(__half2float(mask[i]));
}

// pass 1: the number of live columns in each chunk of each row
__launch_bounds__(256, 1)
static __global__ void fs_mask_count(
        const half * mask_ptr, int32_t * chunk_cnt, const int n_kv, const int n_q, const int mask_ne3,
        const int64_t s31, const int64_t s33) {
    const int chunk = blockIdx.x;
    const int row   = blockIdx.y;
    const int seq   = blockIdx.z;
    const half * mask = mask_ptr + (seq % mask_ne3)*s33 + row*s31;

    int c = 0;
#pragma unroll
    for (int item = 0; item < FS_CHUNK/256; ++item) {
        c += fs_live(mask, chunk*FS_CHUNK + item*256 + threadIdx.x, n_kv);
    }
    c = warp_reduce_sum<WARP_SIZE>(c);

    __shared__ int partial[256/WARP_SIZE];
    if (threadIdx.x % WARP_SIZE == 0) {
        partial[threadIdx.x / WARP_SIZE] = c;
    }
    __syncthreads();
    if (threadIdx.x == 0) {
        int t = 0;
        for (int w = 0; w < 256/WARP_SIZE; ++w) {
            t += partial[w];
        }
        chunk_cnt[((int64_t) seq*n_q + row)*gridDim.x + chunk] = t;
    }
}

// pass 2: each chunk writes its live columns, in order, after those of the chunks before it (at most n_kv_max per
// row); the last chunk also writes the row's count
__launch_bounds__(256, 1)
static __global__ void fs_mask_write(
        const half * mask_ptr, const int32_t * chunk_cnt, int32_t * idx_ptr, int32_t * cnt_ptr, const int n_kv,
        const int n_q, const int mask_ne3, const int64_t s31, const int64_t s33, const int n_kv_max) {
    constexpr int per_lane = FS_CHUNK/256;
    const int tid    = threadIdx.x;
    const int warp   = tid / WARP_SIZE;
    const int lane   = tid % WARP_SIZE;
    const int chunk  = blockIdx.x;
    const int row    = blockIdx.y;
    const int seq    = blockIdx.z;
    const int64_t r  = (int64_t) seq*n_q + row;

    const half    * mask = mask_ptr + (seq % mask_ne3)*s33 + row*s31;
    const int32_t * cc   = chunk_cnt + r*gridDim.x;
    int32_t       * idx  = idx_ptr + r*n_kv_max;

    __shared__ int warp_offsets[256/WARP_SIZE];
    __shared__ int base;

    // the live columns before this chunk
    int before = 0;
    for (int c = tid; c < chunk; c += 256) {
        before += cc[c];
    }
    before = warp_reduce_sum<WARP_SIZE>(before);
    if (lane == 0) {
        warp_offsets[warp] = before;
    }
    __syncthreads();
    if (tid == 0) {
        int t = 0;
        for (int w = 0; w < 256/WARP_SIZE; ++w) {
            t += warp_offsets[w];
        }
        base = t;
        if (chunk == (int) gridDim.x - 1) {
            cnt_ptr[r] = min(t + cc[chunk], n_kv_max);
        }
    }
    __syncthreads();
    if (base >= n_kv_max) {
        return;
    }

    const int i0 = chunk*FS_CHUNK;
    uint32_t sel[per_lane];
    int warp_count = 0;
#pragma unroll
    for (int item = 0; item < per_lane; ++item) {
        sel[item] = fs_ballot(fs_live(mask, i0 + (warp*per_lane + item)*WARP_SIZE + lane, n_kv));
        warp_count += __popc(sel[item]);
    }
    __syncthreads();
    if (lane == 0) {
        warp_offsets[warp] = warp_count;
    }
    __syncthreads();
    if (tid == 0) {
        int offset = base;
        for (int w = 0; w < 256/WARP_SIZE; ++w) {
            const int c = warp_offsets[w];
            warp_offsets[w] = offset;
            offset += c;
        }
    }
    __syncthreads();

    const uint32_t below = (1u << lane) - 1u;
    int item_offset = warp_offsets[warp];
#pragma unroll
    for (int item = 0; item < per_lane; ++item) {
        const int dst = item_offset + __popc(sel[item] & below);
        if ((sel[item] >> lane) & 1u && dst < n_kv_max) {
            idx[dst] = i0 + (warp*per_lane + item)*WARP_SIZE + lane;
        }
        item_offset += __popc(sel[item]);
    }
}

// the length of each cell list (ggml_flash_attn_ext_set_cells): its entries up to the first -1
__launch_bounds__(256, 1)
static __global__ void fs_list_count(const int32_t * lists, int32_t * cnt_ptr, const int n_list) {
    const int32_t * list = lists + (int64_t) blockIdx.x*n_list;

    int c = 0;
    for (int i = threadIdx.x; i < n_list; i += 256) {
        c += list[i] >= 0;
    }
    c = warp_reduce_sum<WARP_SIZE>(c);

    __shared__ int partial[256/WARP_SIZE];
    if (threadIdx.x % WARP_SIZE == 0) {
        partial[threadIdx.x / WARP_SIZE] = c;
    }
    __syncthreads();
    if (threadIdx.x == 0) {
        int t = 0;
        for (int w = 0; w < 256/WARP_SIZE; ++w) {
            t += partial[w];
        }
        cnt_ptr[blockIdx.x] = t;
    }
}

// gfx12: K/V rows through raw buffer loads (a scalar resource and row offset, the lane's offset in a vector register)
#if defined(__HIP_DEVICE_COMPILE__) && (defined(__gfx1200__) || defined(__gfx1201__))
#define FS_BUFFER_LOADS 1
static __device__ __forceinline__ __amdgpu_buffer_rsrc_t fs_rsrc(const char * p, const uint32_t bytes) {
    return __builtin_amdgcn_make_buffer_rsrc((void *) p, (short) 0, (int) bytes, 0x31004000);
}
#endif

// a lane's EPL consecutive elements from element e0 of a row (EPL divides 32, so they lie in one quant block), raw,
// dequantized later. Q8_0 / Q4_0 blocks are only 2-byte aligned: the compiler merges the 2-byte loads into wider
// unaligned ones, as does the buffer form
template <ggml_type type, int EPL>
struct fs_frag {
    uint16_t u[type == GGML_TYPE_F16 ? EPL : EPL/2]; // Q4_0: one nibble of each of EPL bytes
    uint16_t d;
    int      shift; // Q4_0: 0 for the low nibbles, 4 for the high ones

    // the offsets of the scale and of the lane's elements within the row
    __device__ __forceinline__ uint32_t offsets(const uint32_t e0, uint32_t & off_d) {
        if constexpr (type == GGML_TYPE_F16) {
            off_d = 0;
            return 2*e0;
        } else if constexpr (type == GGML_TYPE_Q8_0) {
            off_d = (e0/QK8_0)*(uint32_t) sizeof(block_q8_0);
            return off_d + (uint32_t) sizeof(ggml_half) + e0 % QK8_0;
        } else {
            const uint32_t j = e0 % QK4_0;
            off_d = (e0/QK4_0)*(uint32_t) sizeof(block_q4_0);
            shift = j < QK4_0/2 ? 0 : 4;
            return off_d + (uint32_t) sizeof(ggml_half) + j % (QK4_0/2);
        }
    }

    __device__ __forceinline__ void load(const char * row, const int e0) {
        constexpr int n = sizeof(u)/sizeof(u[0]);
        uint32_t off_d;
        const uint32_t off = offsets(e0, off_d);
        if constexpr (type != GGML_TYPE_F16) {
            d = *(const uint16_t *) (row + off_d);
        }
        const uint16_t * p = (const uint16_t *) (row + off);
#pragma unroll
        for (int i = 0; i < n; ++i) {
            u[i] = p[i];
        }
    }

#ifdef FS_BUFFER_LOADS
    // the same from a buffer resource: the row at byte soff (uniform), the lane's offsets in vector registers
    __device__ __forceinline__ void load(const __amdgpu_buffer_rsrc_t rsrc, const uint32_t soff, const int e0) {
        uint32_t off_d;
        const uint32_t off = offsets(e0, off_d);
        if constexpr (type != GGML_TYPE_F16) {
            d = __builtin_amdgcn_raw_buffer_load_b16(rsrc, (int) off_d, (int) soff, 0);
        }
        static_assert(sizeof(u) == 2 || sizeof(u) == 4 || sizeof(u) == 8 || sizeof(u) == 16, "one load");
        if constexpr (sizeof(u) == 2) {
            u[0] = __builtin_amdgcn_raw_buffer_load_b16(rsrc, (int) off, (int) soff, 0);
        } else if constexpr (sizeof(u) == 4) {
            const uint32_t x = __builtin_amdgcn_raw_buffer_load_b32(rsrc, (int) off, (int) soff, 0);
            memcpy(u, &x, sizeof(u));
        } else if constexpr (sizeof(u) == 8) {
            const auto x = __builtin_amdgcn_raw_buffer_load_b64(rsrc, (int) off, (int) soff, 0);
            memcpy(u, &x, sizeof(u));
        } else {
            const auto x = __builtin_amdgcn_raw_buffer_load_b128(rsrc, (int) off, (int) soff, 0);
            memcpy(u, &x, sizeof(u));
        }
    }
#endif // FS_BUFFER_LOADS

    __device__ __forceinline__ void dequant(float * out) const {
        if constexpr (type == GGML_TYPE_F16) {
#pragma unroll
            for (int i = 0; i < EPL; ++i) {
                out[i] = __half2float(__ushort_as_half(u[i]));
            }
        } else if constexpr (type == GGML_TYPE_Q8_0) {
            // u = q + 128 as an unsigned byte, h = 1024 + u as an f16 (a byte permute makes two), then
            // d*h - 1152*d: both terms (22 and 15 significant bits) and the result d*q are exact in f32, so this equals
            // d*q
            static_assert(EPL % 4 == 0, "whole 32-bit words");
            const float df  = __half2float(__ushort_as_half(d));
            const float off = -1152.0f*df;
#pragma unroll
            for (int i = 0; i < EPL/4; ++i) {
                const uint32_t w = ((uint32_t) u[2*i] | ((uint32_t) u[2*i + 1] << 16)) ^ 0x80808080u;
#pragma unroll
                for (int j = 0; j < 2; ++j) {
#ifdef GGML_USE_HIP
                    // bytes 2j, 0x64, 2j + 1, 0x64
                    const uint32_t h2 = __builtin_amdgcn_perm(0x64646464u, w, 0x04000400u | (2*j + 1) << 16 | 2*j);
#else
                    const uint32_t h2 = 0x64006400u | ((w >> 16*j) & 0xFF) | ((w >> 16*j) & 0xFF00) << 8;
#endif // GGML_USE_HIP
                    const half2 h = *(const half2 *) &h2;
                    out[4*i + 2*j + 0] = fmaf(df, __low2float(h),  off);
                    out[4*i + 2*j + 1] = fmaf(df, __high2float(h), off);
                }
            }
        } else {
            const float df = __half2float(__ushort_as_half(d));
#pragma unroll
            for (int i = 0; i < EPL/2; ++i) {
                const uint32_t b0 = (u[i] & 0xFF) >> shift;
                const uint32_t b1 = (u[i] >> 8)   >> shift;
                out[2*i + 0] = df*((int) (b0 & 0xF) - 8);
                out[2*i + 1] = df*((int) (b1 & 0xF) - 8);
            }
        }
    }
};

// rows staged per step
#define FS_ROWS 16

static __device__ __forceinline__ float fs_dot2(const float2 a, const float2 b, const float acc) {
    return fmaf(a.y, b.y, fmaf(a.x, b.x, acc));
}

template <int D, int G, ggml_type type_K, ggml_type type_V>
__launch_bounds__(G*WARP_SIZE, 1)
static __global__ void fs_attend(
        const char * Q, const char * K, const char * V, const half * mask, const int32_t * idx_ptr,
        const int32_t * cnt_ptr, float * parts, float2 * meta, float * dst, const float scale,
        const int n_q, const int n_head, const int n_kv_max, const int mask_ne3,
        const int64_t nbq1, const int64_t nbq2, const int64_t nbq3,
        const int64_t nbk1, const int64_t nbk2, const int64_t nbk3,
        const int64_t nbv1, const int64_t nbv2, const int64_t nbv3,
        const int64_t s31, const int64_t s33) {
    constexpr int EPL      = D/WARP_SIZE; // output dims per lane
    constexpr int NT       = G*WARP_SIZE;
    constexpr int K_STRIDE = D/2 + 1;     // float2 per staged K row
    constexpr int GROUPS   = D/8;         // 8-element groups per row

    const int split   = blockIdx.x;
    const int n_split = gridDim.x;
    const int h_kv    = blockIdx.y;
    const int qrow    = blockIdx.z % n_q;
    const int seq     = blockIdx.z / n_q;
    const int lane    = threadIdx.x % WARP_SIZE;
    const int g       = threadIdx.x / WARP_SIZE; // the wave's query head within the KV head's group

    __shared__ float2 sq[G][D/2];
    __shared__ float2 sk[FS_ROWS][K_STRIDE];
    __shared__ float sv[FS_ROWS][D];
    __shared__ float smask[FS_ROWS];

    for (int i = threadIdx.x; i < G*D/2; i += NT) {
        const int gg = i / (D/2);
        const float2 * qg = (const float2 *) (Q + qrow*nbq1 + (h_kv*G + gg)*nbq2 + seq*nbq3);
        const float2 q2 = qg[i % (D/2)];
        sq[gg][i % (D/2)] = q2;
    }

    const half    * mrow  = mask ? mask + (seq % mask_ne3)*s33 + qrow*s31 : nullptr; // none with cell lists
    const int32_t * list  = idx_ptr + ((int64_t) seq*n_q + qrow)*n_kv_max;
    const int       count = cnt_ptr[(int64_t) seq*n_q + qrow];
    const int       per   = (count + n_split - 1)/n_split;
    const int       r0    = split*per;
    const int       r1    = min(r0 + per, count);

    const char * Kh = K + h_kv*nbk2 + seq*nbk3;
    const char * Vh = V + h_kv*nbv2 + seq*nbv3;

    float m = -INFINITY;
    float l = 0.0f;
    float acc[EPL];
#pragma unroll
    for (int i = 0; i < EPL; ++i) {
        acc[i] = 0.0f;
    }

    for (int rb = r0; rb < r1; rb += FS_ROWS) {
        const int nr = min(FS_ROWS, r1 - rb);

        // Stage dequantized rows in FP32, one 8-element group per item.
        __syncthreads();
        for (int it = threadIdx.x; it < nr*GROUPS; it += NT) {
            const int r  = it / GROUPS;
            const int e0 = (it % GROUPS)*8;
            const int c  = list[rb + r];
            float x[8];
            fs_frag<type_K, 8> fk;
            fk.load(Kh + c*nbk1, e0);
            fk.dequant(x);
#pragma unroll
            for (int i = 0; i < 4; ++i) {
                sk[r][e0/2 + i] = make_float2(x[2*i], x[2*i + 1]);
            }
            fs_frag<type_V, 8> fv;
            fv.load(Vh + c*nbv1, e0);
            fv.dequant(x);
#pragma unroll
            for (int i = 0; i < 8; ++i) {
                sv[r][e0 + i] = x[i];
            }
            if (e0 == 0) {
                smask[r] = mrow ? __half2float(mrow[c]) : 0.0f;
            }
        }
        __syncthreads();

        // scores, a lane per row
        float s = -INFINITY;
        if (lane < nr) {
            float t = 0.0f;
#pragma unroll 8
            for (int d2 = 0; d2 < D/2; ++d2) {
                t = fs_dot2(sq[g][d2], sk[lane][d2], t);
            }
            s = t*scale + smask[lane];
        }

        const float mx = warp_reduce_max<WARP_SIZE>(s);
        const float mn = fmaxf(m, mx);
        if (mn == -INFINITY) {
            continue;
        }
        const float a = expf(m - mn);
        const float p = expf(s - mn);
        m = mn;
        l = fmaf(l, a, warp_reduce_sum<WARP_SIZE>(p));
#pragma unroll
        for (int i = 0; i < EPL; ++i) {
            acc[i] *= a;
        }

        // P V: a lane per EPL output dims
        for (int r = 0; r < nr; ++r) {
            const float pr = __shfl_sync(0xffffffff, p, r, WARP_SIZE);
#pragma unroll
            for (int i = 0; i < EPL; ++i) {
                acc[i] = fmaf(pr, sv[r][lane*EPL + i], acc[i]);
            }
        }
    }

    const int j = (seq*n_q + qrow)*n_head + h_kv*G + g; // dst row: [D, n_head, n_q, n_seq]
    if (n_split == 1) {
#pragma unroll
        for (int i = 0; i < EPL; ++i) {
            dst[(int64_t) j*D + lane*EPL + i] = l > 0.0f ? acc[i]/l : 0.0f;
        }
    } else {
#pragma unroll
        for (int i = 0; i < EPL; ++i) {
            parts[((int64_t) j*n_split + split)*D + lane*EPL + i] = acc[i];
        }
        if (lane == 0) {
            meta[(int64_t) j*n_split + split] = make_float2(m, l);
        }
    }
}

// broadcast of lane src (uniform) to the wave
static __device__ __forceinline__ float fs_bcast(const float x, const int src) {
#ifdef GGML_USE_HIP
    return __int_as_float(__builtin_amdgcn_readlane(__float_as_int(x), src));
#else
    return __shfl_sync(0xffffffff, x, src, WARP_SIZE);
#endif // GGML_USE_HIP
}

static __device__ __forceinline__ float fs_shfl(const float x, const int src) {
    return __shfl_sync(0xffffffff, x, src, WARP_SIZE);
}

// lane ^ w's x (w a power of 2 below 32): DPP within a row of 16 lanes, permlanex16 across the two rows
template <int w>
static __device__ __forceinline__ float fs_xor(const float x) {
#ifdef GGML_USE_HIP
    static_assert(w >= 1 && w <= 16 && (w & (w - 1)) == 0, "a lane bit");
    const int xi = __float_as_int(x);
    if constexpr (w == 16) {
        return __int_as_float(__builtin_amdgcn_permlanex16(xi, xi, 0x76543210, 0xfedcba98, false, false));
    } else {
        return __int_as_float(__builtin_amdgcn_update_dpp(0, xi, 0x160 + w, 0xf, 0xf, true)); // row_xmask:w
    }
#else
    return __shfl_xor_sync(0xffffffff, x, w, WARP_SIZE);
#endif // GGML_USE_HIP
}

static constexpr __device__ int fs_log2(const int x) {
    return x <= 1 ? 0 : 1 + fs_log2(x/2);
}

// f(std::integral_constant<int, b>) for b in [0, n)
template <int n, int b = 0, typename F>
static __device__ __forceinline__ void fs_static_for(F && f) {
    if constexpr (b < n) {
        f(std::integral_constant<int, b>{});
        fs_static_for<n, b + 1>(f);
    }
}

// query heads per wave of fs_attend_heads
#define FS_HPW 4

// The prefill form for G a multiple of FS_HPW: a wave owns FS_HPW query heads of the KV head (Q in registers, D/32
// dims per lane) and reads K/V of its rows straight into registers, 32/FS_HPW rows per step; the FS_HPW x rows
// partial dots are reduce-scattered to one score per lane (lane = head*rows + row), and each head's online softmax
// runs over its lanes. The G/FS_HPW waves of a block read the same rows (from L2) and share nothing else.
// The rows' cells are wave-uniform, so their addresses are scalar.
template <int D, int G, ggml_type type_K, ggml_type type_V>
__launch_bounds__((G/FS_HPW)*WARP_SIZE, 1)
static __global__ void fs_attend_heads(
        const char * __restrict__ Q, const char * __restrict__ K, const char * __restrict__ V,
        const half * __restrict__ mask, const int32_t * __restrict__ idx_ptr, const int32_t * __restrict__ cnt_ptr,
        float * __restrict__ dst, const float scale, const int n_q, const int n_head, const int n_kv_max,
        const int mask_ne3,
        const int64_t nbq1, const int64_t nbq2, const int64_t nbq3,
        const int64_t nbk1, const int64_t nbk2, const int64_t nbk3,
        const int64_t nbv1, const int64_t nbv2, const int64_t nbv3,
        const int64_t s31, const int64_t s33, const int n_kv) {
    constexpr int EPL = D/WARP_SIZE;     // dims per lane
    constexpr int HPW = FS_HPW;
    constexpr int R   = WARP_SIZE/HPW;   // rows per step
    static_assert(G % HPW == 0 && HPW*R == WARP_SIZE && (HPW & (HPW - 1)) == 0, "one partial score per lane");

    const int h_kv = blockIdx.y;
    const int qrow = blockIdx.z % n_q;
    const int seq  = blockIdx.z / n_q;
    const int lane = threadIdx.x % WARP_SIZE;
    const int h0   = h_kv*G + (threadIdx.x / WARP_SIZE)*HPW; // the wave's first query head
    const int my_h = lane / R;                                 // after the reduce-scatter: head lane / R, row lane % R
    const int my_r = lane % R;

    // q[a] is head a ^ my_h: the head steps of the reduce-scatter then keep and send fixed halves of the partials
    float q[HPW][EPL];
#pragma unroll
    for (int a = 0; a < HPW; ++a) {
        const float * qh = (const float *) (Q + qrow*nbq1 + (h0 + (a ^ my_h))*nbq2 + seq*nbq3);
#pragma unroll
        for (int i = 0; i < EPL; ++i) {
            q[a][i] = qh[lane*EPL + i];
        }
    }

    const half    * mrow  = mask ? mask + (seq % mask_ne3)*s33 + qrow*s31 : nullptr; // none with cell lists
    const int32_t * list  = idx_ptr + ((int64_t) seq*n_q + qrow)*n_kv_max;
    const int       count = cnt_ptr[(int64_t) seq*n_q + qrow];

    const char * Kh = K + h_kv*nbk2 + seq*nbk3;
    const char * Vh = V + h_kv*nbv2 + seq*nbv3;
#ifdef FS_BUFFER_LOADS
    // ggml_cuda_flash_attn_ext_sparse checked that n_kv rows fit in 32-bit offsets
    const __amdgpu_buffer_rsrc_t Kr = fs_rsrc(Kh, (uint32_t) (n_kv*nbk1));
    const __amdgpu_buffer_rsrc_t Vr = fs_rsrc(Vh, (uint32_t) (n_kv*nbv1));
#endif // FS_BUFFER_LOADS

    float m = -INFINITY; // of head my_h
    float l = 0.0f;
    float acc[HPW][EPL]; // of head hh
#pragma unroll
    for (int hh = 0; hh < HPW; ++hh) {
#pragma unroll
        for (int i = 0; i < EPL; ++i) {
            acc[hh][i] = 0.0f;
        }
    }

    for (int rb = 0; rb < count; rb += R) {
        const int nr = min(R, count - rb);

        // the rows' cells (scalar loads); missing rows repeat the last cell and are masked below
        // with a mask, lane r also fetches the mask value of row rb + r
        float mk = 0.0f;
        if (mrow) {
            mk = lane < nr ? __half2float(mrow[list[rb + lane]]) : -INFINITY;
        }

        fs_frag<type_K, EPL> fk[R];
        fs_frag<type_V, EPL> fv[R];
#pragma unroll
        for (int r = 0; r < R; ++r) {
            const int c = list[min(rb + r, count - 1)];
#ifdef FS_BUFFER_LOADS
            fk[r].load(Kr, (uint32_t) c*(uint32_t) nbk1, lane*EPL);
            fv[r].load(Vr, (uint32_t) c*(uint32_t) nbv1, lane*EPL);
#else
            fk[r].load(Kh + c*nbk1, lane*EPL);
            fv[r].load(Vh + c*nbv1, lane*EPL);
#endif // FS_BUFFER_LOADS
        }

        float v[HPW*R]; // v[a*R + r]: head a ^ my_h, row r
#pragma unroll
        for (int r = 0; r < R; ++r) {
            float k[EPL];
            fk[r].dequant(k);
#pragma unroll
            for (int a = 0; a < HPW; ++a) {
                float t = q[a][0]*k[0];
#pragma unroll
                for (int i = 1; i < EPL; ++i) {
                    t = fmaf(q[a][i], k[i], t);
                }
                v[a*R + r] = t;
            }
        }

        // reduce-scatter over the 5 lane bits, each halving the values a lane keeps. Head bits: a lane keeps the
        // heads whose bit matches its own, which with q permuted by my_h are v[0, w). Row bits: the swap is a bit
        // select (a ternary on v makes the compiler index v at run time).
        fs_static_for<fs_log2(HPW)>([&](auto b) {
            constexpr int w = WARP_SIZE/2 >> b;
#pragma unroll
            for (int i = 0; i < w; ++i) {
                v[i] += fs_xor<w>(v[i + w]);
            }
        });
        fs_static_for<fs_log2(R)>([&](auto b) {
            constexpr int w = R/2 >> b;
            const int hi = -((lane >> fs_log2(w)) & 1);
#pragma unroll
            for (int i = 0; i < w; ++i) {
                const int lo_bits = __float_as_int(v[i]);
                const int hi_bits = __float_as_int(v[i + w]);
                const int diff    = (lo_bits ^ hi_bits) & hi;
                const float send = __int_as_float(hi_bits ^ diff);
                const float keep = __int_as_float(lo_bits ^ diff);
                v[i] = keep + fs_xor<w>(send);
            }
        });

        const float mask_r = mrow ? fs_shfl(mk, my_r) : 0.0f;
        const float s      = my_r < nr ? v[0]*scale + mask_r : -INFINITY;

        // the head's online softmax step over its R lanes
        float mx = s;
        fs_static_for<fs_log2(R)>([&](auto b) {
            mx = fmaxf(mx, fs_xor<(1 << b)>(mx));
        });
        const float mn = fmaxf(m, mx);
        const float a  = mn == -INFINITY ? 1.0f : expf(m - mn);
        const float p  = mn == -INFINITY ? 0.0f : expf(s - mn);
        float ps = p;
        fs_static_for<fs_log2(R)>([&](auto b) {
            ps += fs_xor<(1 << b)>(ps);
        });
        m = mn;
        l = fmaf(l, a, ps);

        // a is 1 unless the head's maximum rose
        if (fs_ballot(a != 1.0f)) {
#pragma unroll
            for (int hh = 0; hh < HPW; ++hh) {
                const float ah = fs_bcast(a, hh*R);
#pragma unroll
                for (int i = 0; i < EPL; ++i) {
                    acc[hh][i] *= ah;
                }
            }
        }
#pragma unroll
        for (int r = 0; r < R; ++r) {
            if (r < nr) {
                float x[EPL];
                fv[r].dequant(x);
#pragma unroll
                for (int hh = 0; hh < HPW; ++hh) {
                    const float pr = fs_bcast(p, hh*R + r);
#pragma unroll
                    for (int i = 0; i < EPL; ++i) {
                        acc[hh][i] = fmaf(pr, x[i], acc[hh][i]);
                    }
                }
            }
        }
    }

#pragma unroll
    for (int hh = 0; hh < HPW; ++hh) {
        const float   lh = fs_bcast(l, hh*R);
        const int64_t j  = ((int64_t) seq*n_q + qrow)*n_head + h0 + hh; // dst row: [D, n_head, n_q, n_seq]
#pragma unroll
        for (int i = 0; i < EPL; ++i) {
            dst[j*D + lane*EPL + i] = lh > 0.0f ? acc[hh][i]/lh : 0.0f;
        }
    }
}

// the GQA ratios and type pairs that are instantiated
static bool fs_gqa_ok(const int D, const int G) {
    switch (G) {
        case 1: case 2: case 4: case 6: case 8: case 12:
            return true;
        case 16:
            return D <= 128;
        default:
            return false;
    }
}

static bool fs_types_ok(const ggml_type tk, const ggml_type tv) {
    return (tk == GGML_TYPE_F16  && tv == GGML_TYPE_F16)  ||
           (tk == GGML_TYPE_Q8_0 && tv == GGML_TYPE_Q8_0) ||
           (tk == GGML_TYPE_Q8_0 && tv == GGML_TYPE_Q4_0);
}

// A prefill (more than FS_MAX_Q rows) takes the sparse kernel from this K/V length on: dense attention costs about
// n_kv per query, the sparse kernel about n_kv_max. GGML_CUDA_FATTN_SPARSE_MIN_KV overrides it.
#define FS_PREFILL_MIN_KV 12288

static int64_t fs_prefill_min_kv() {
    static const int64_t min_kv = [] {
        const char * env = getenv("GGML_CUDA_FATTN_SPARSE_MIN_KV");
        return env ? (int64_t) atoll(env) : (int64_t) FS_PREFILL_MIN_KV;
    }();
    return min_kv;
}

bool ggml_cuda_fattn_sparse_supported(int device, const ggml_tensor * dst) {
#ifndef GGML_USE_HIP
    GGML_UNUSED_VARS(device, dst);
    return false;
#else
    const ggml_tensor * Q    = dst->src[0];
    const ggml_tensor * K    = dst->src[1];
    const ggml_tensor * V    = dst->src[2];
    const ggml_tensor * mask  = dst->src[3];
    const ggml_tensor * cells = dst->src[GGML_FA_SRC_CELLS];

    // GGML_CUDA_FATTN_SPARSE=0 selects dense attention for comparison.
    static const bool enabled = [] {
        const char * env = getenv("GGML_CUDA_FATTN_SPARSE");
        return env == nullptr || atoi(env) != 0;
    }();
    const int cc = ggml_cuda_info().devices[device].cc;
    if (!enabled || !GGML_CUDA_CC_IS_RDNA4(cc) || ggml_cuda_info().devices[device].warp_size != 32) {
        return false;
    }
    if (ggml_get_op_params_i32(dst, 4) <= 0 || !(mask || cells) || dst->src[4] || dst->src[GGML_FA_SRC_PAGES] ||
            ggml_flash_attn_ext_mask_col0(dst) != 0) {
        return false;
    }
    float params[3];
    memcpy(params, dst->op_params, sizeof(params)); // scale, max_bias, logit_softcap
    if (params[1] != 0.0f || params[2] != 0.0f) {
        return false;
    }
    if (Q->type != GGML_TYPE_F32 || dst->type != GGML_TYPE_F32 || !ggml_is_contiguous(dst) || Q->nb[0] != sizeof(float)) {
        return false;
    }
    if (mask && (mask->type != GGML_TYPE_F16 || !ggml_is_contiguous(mask) || mask->ne[2] != 1 || mask->ne[3] <= 0 ||
            Q->ne[3] % mask->ne[3] != 0 || mask->ne[0] != K->ne[1] || mask->ne[1] < Q->ne[1])) {
        return false;
    }
    // ggml_flash_attn_ext_set_cells checked the shape: a list per query, n_kv_max long
    if (cells && (mask || ggml_get_op_params_i32(dst, 4) != cells->ne[0])) {
        return false;
    }
    if (K->ne[1] <= 0 || K->ne[1] > INT32_MAX - FS_CHUNK || Q->ne[1] <= 0 || Q->ne[2] <= 0 || Q->ne[3] <= 0 ||
            Q->ne[2] > INT32_MAX / Q->ne[3] / Q->ne[1]) {
        return false;
    }
    const int64_t D = Q->ne[0];
    if ((D != 128 && D != 256) || K->ne[0] != D || V->ne[0] != D) {
        return false;
    }
    // a row per block in the grid's z; the K/V rows of a head at 32-bit offsets (fs_attend_heads' buffer loads)
    if (Q->ne[1] > FS_MAX_Q && (K->ne[1] < fs_prefill_min_kv() || Q->ne[1]*Q->ne[3] > 65535 ||
            K->ne[1]*K->nb[1] > UINT32_MAX || V->ne[1]*V->nb[1] > UINT32_MAX)) {
        return false;
    }
    if (K->ne[2] == 0 || Q->ne[2] % K->ne[2] != 0 || V->ne[1] != K->ne[1] || V->ne[2] != K->ne[2] || K->ne[3] != Q->ne[3] || V->ne[3] != Q->ne[3]) {
        return false;
    }
    if (!fs_gqa_ok(D, Q->ne[2]/K->ne[2]) || !fs_types_ok(K->type, V->type)) {
        return false;
    }
    return K->nb[0] == ggml_type_size(K->type) && V->nb[0] == ggml_type_size(V->type);
#endif // GGML_USE_HIP
}

template <int D, int G, ggml_type type_K, ggml_type type_V>
static void fs_launch(ggml_backend_cuda_context & ctx, ggml_tensor * dst, const int32_t * idx, const int32_t * cnt,
                      const int n_kv_max, const int n_split, float * parts, float2 * meta) {
    const ggml_tensor * Q    = dst->src[0];
    const ggml_tensor * K    = dst->src[1];
    const ggml_tensor * V    = dst->src[2];
    const ggml_tensor * mask = dst->src[3];

    float scale;
    memcpy(&scale, dst->op_params, sizeof(float));

    // no mask with cell lists
    const half *  mask_d   = mask ? (const half *) mask->data : nullptr;
    const int     mask_ne3 = mask ? mask->ne[3] : 1;
    const int64_t s31      = mask ? mask->nb[1]/sizeof(half) : 0;
    const int64_t s33      = mask ? mask->nb[3]/sizeof(half) : 0;

    const dim3 grid(n_split, K->ne[2], Q->ne[1]*Q->ne[3]);
    if constexpr (G % FS_HPW == 0) {
        if (Q->ne[1] > FS_MAX_Q) {
            GGML_ASSERT(n_split == 1);
            fs_attend_heads<D, G, type_K, type_V><<<grid, (G/FS_HPW)*WARP_SIZE, 0, ctx.stream()>>>(
                (const char *) Q->data, (const char *) K->data, (const char *) V->data, mask_d, idx, cnt,
                (float *) dst->data, scale, Q->ne[1], Q->ne[2], n_kv_max, mask_ne3,
                Q->nb[1], Q->nb[2], Q->nb[3], K->nb[1], K->nb[2], K->nb[3], V->nb[1], V->nb[2], V->nb[3], s31, s33,
                K->ne[1]);
            return;
        }
    }
    fs_attend<D, G, type_K, type_V><<<grid, G*WARP_SIZE, 0, ctx.stream()>>>(
        (const char *) Q->data, (const char *) K->data, (const char *) V->data, mask_d, idx, cnt,
        parts, meta, (float *) dst->data, scale, Q->ne[1], Q->ne[2], n_kv_max, mask_ne3,
        Q->nb[1], Q->nb[2], Q->nb[3], K->nb[1], K->nb[2], K->nb[3], V->nb[1], V->nb[2], V->nb[3], s31, s33);
}

template <int D, ggml_type type_K, ggml_type type_V>
static void fs_launch_G(ggml_backend_cuda_context & ctx, ggml_tensor * dst, const int G, const int32_t * idx,
                        const int32_t * cnt, const int n_kv_max, const int n_split, float * parts, float2 * meta) {
    switch (G) {
        case  1: fs_launch<D,  1, type_K, type_V>(ctx, dst, idx, cnt, n_kv_max, n_split, parts, meta); break;
        case  2: fs_launch<D,  2, type_K, type_V>(ctx, dst, idx, cnt, n_kv_max, n_split, parts, meta); break;
        case  4: fs_launch<D,  4, type_K, type_V>(ctx, dst, idx, cnt, n_kv_max, n_split, parts, meta); break;
        case  6: fs_launch<D,  6, type_K, type_V>(ctx, dst, idx, cnt, n_kv_max, n_split, parts, meta); break;
        case  8: fs_launch<D,  8, type_K, type_V>(ctx, dst, idx, cnt, n_kv_max, n_split, parts, meta); break;
        case 12: fs_launch<D, 12, type_K, type_V>(ctx, dst, idx, cnt, n_kv_max, n_split, parts, meta); break;
        case 16:
            if constexpr (D <= 128) {
                fs_launch<D, 16, type_K, type_V>(ctx, dst, idx, cnt, n_kv_max, n_split, parts, meta);
                break;
            }
            GGML_ABORT("fatal error");
        default: GGML_ABORT("fatal error");
    }
}

template <int D>
static void fs_launch_D(ggml_backend_cuda_context & ctx, ggml_tensor * dst, const int G, const int32_t * idx,
                        const int32_t * cnt, const int n_kv_max, const int n_split, float * parts, float2 * meta) {
    const ggml_type tk = dst->src[1]->type;
    const ggml_type tv = dst->src[2]->type;
    if (tk == GGML_TYPE_F16) {
        fs_launch_G<D, GGML_TYPE_F16, GGML_TYPE_F16>(ctx, dst, G, idx, cnt, n_kv_max, n_split, parts, meta);
    } else if (tv == GGML_TYPE_Q8_0) {
        fs_launch_G<D, GGML_TYPE_Q8_0, GGML_TYPE_Q8_0>(ctx, dst, G, idx, cnt, n_kv_max, n_split, parts, meta);
    } else {
        fs_launch_G<D, GGML_TYPE_Q8_0, GGML_TYPE_Q4_0>(ctx, dst, G, idx, cnt, n_kv_max, n_split, parts, meta);
    }
}

void ggml_cuda_flash_attn_ext_sparse(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * Q     = dst->src[0];
    const ggml_tensor * K     = dst->src[1];
    const ggml_tensor * mask  = dst->src[3];
    const ggml_tensor * cells = dst->src[GGML_FA_SRC_CELLS];

    const int D        = Q->ne[0];
    const int n_q      = Q->ne[1];
    const int n_head   = Q->ne[2];
    const int n_seq    = Q->ne[3];
    const int n_rows   = n_q*n_seq;
    // the lists' row stride: the compacted mask rows are at most K/V long
    const int n_kv_max = cells ? cells->ne[0] : std::min<int64_t>(ggml_get_op_params_i32(dst, 4), K->ne[1]);
    const int G        = n_head/K->ne[2];

    cudaStream_t stream = ctx.stream();

    ggml_cuda_pool_alloc<int32_t> idx_alloc(ctx.pool());
    ggml_cuda_pool_alloc<int32_t> cnt(ctx.pool(), n_rows);
    ggml_cuda_pool_alloc<int32_t> chunk_cnt(ctx.pool());
    const int32_t * idx_ptr;
    if (cells) {
        idx_ptr = (const int32_t *) cells->data;
        fs_list_count<<<n_rows, 256, 0, stream>>>(idx_ptr, cnt.ptr, n_kv_max);
    } else {
        const int n_chunks = (mask->ne[0] + FS_CHUNK - 1)/FS_CHUNK;
        idx_alloc.alloc((size_t) n_rows*n_kv_max);
        chunk_cnt.alloc((size_t) n_rows*n_chunks);
        idx_ptr = idx_alloc.ptr;

        int32_t * idx = idx_alloc.ptr;
        const dim3 grid(n_chunks, n_q, n_seq);
        const int64_t s31 = mask->nb[1]/sizeof(half);
        const int64_t s33 = mask->nb[3]/sizeof(half);
        fs_mask_count<<<grid, 256, 0, stream>>>((const half *) mask->data, chunk_cnt.ptr, mask->ne[0], n_q, mask->ne[3], s31, s33);
        fs_mask_write<<<grid, 256, 0, stream>>>((const half *) mask->data, chunk_cnt.ptr, idx, cnt.ptr, mask->ne[0], n_q,
            mask->ne[3], s31, s33, n_kv_max);
    }

    // about 2 blocks per CU, at least two staging steps per split
    const int nsm     = ggml_cuda_info().devices[ctx.device].nsm;
    const int blocks  = K->ne[2]*n_rows;
    const int n_kv_split = std::min<int64_t>(n_kv_max, K->ne[1]);
    // a prefill has a row per block: one split, so that each row's sum order is fixed
    const int n_split = n_q > FS_MAX_Q ? 1 :
        std::max(1, std::min((2*nsm + blocks - 1)/blocks, (n_kv_split + 2*FS_ROWS - 1)/(2*FS_ROWS)));

    ggml_cuda_pool_alloc<float>  parts(ctx.pool());
    ggml_cuda_pool_alloc<float2> meta(ctx.pool());
    if (n_split > 1) {
        parts.alloc((size_t) n_rows*n_head*n_split*D);
        meta.alloc((size_t) n_rows*n_head*n_split);
    }

    switch (D) {
        case 128: fs_launch_D<128>(ctx, dst, G, idx_ptr, cnt.ptr, n_kv_max, n_split, parts.ptr, meta.ptr); break;
        case 256: fs_launch_D<256>(ctx, dst, G, idx_ptr, cnt.ptr, n_kv_max, n_split, parts.ptr, meta.ptr); break;
        default: GGML_ABORT("fatal error");
    }

    if (n_split > 1) {
        const dim3 grid(n_q, n_head, n_seq);
        const size_t shared = 2*n_split*sizeof(float);
        if (D == 128) {
            flash_attn_combine_results<128, true><<<grid, 128, shared, stream>>>(parts.ptr, meta.ptr, (float *) dst->data, n_split);
        } else {
            flash_attn_combine_results<256, true><<<grid, 256, shared, stream>>>(parts.ptr, meta.ptr, (float *) dst->data, n_split);
        }
    }
    CUDA_CHECK(cudaGetLastError());
}
