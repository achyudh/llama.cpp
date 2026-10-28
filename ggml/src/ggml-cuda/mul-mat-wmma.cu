#include "mul-mat-wmma.cuh"
#include "gemm-wmma.cuh"

bool ggml_cuda_should_use_mul_mat_wmma(const ggml_tensor * src0, const ggml_tensor * src1, const ggml_tensor * dst, int cc) {
#ifdef GGML_USE_HIP
    static const bool disabled = getenv("GGML_CUDA_DISABLE_MUL_MAT_WMMA") != nullptr;
    if (disabled || !GGML_CUDA_CC_IS_RDNA4(cc) || cc >= GGML_CUDA_CC_OFFSET_AMD + 0x1300 || src0->type != GGML_TYPE_F16 ||
            src1->type != GGML_TYPE_F32 || dst->type != GGML_TYPE_F32) {
        return false;
    }
    const int64_t k = src0->ne[0], n = src0->ne[1], m = src1->ne[1];
    // one matrix each, rows contiguous; K in pairs of 32-wide steps and 16-byte rows; 32-bit buffer offsets
    return src0->ne[2] == 1 && src0->ne[3] == 1 && src1->ne[2] == 1 && src1->ne[3] == 1 &&
        ggml_is_contiguous(src0) && ggml_is_contiguous(src1) && ggml_is_contiguous(dst) &&
        m >= GGML_CUDA_MUL_MAT_WMMA_MIN_ROWS && k % 64 == 0 &&
        n * k * 2 < INT32_MAX && m * k * 4 < INT32_MAX;
#else
    GGML_UNUSED_VARS(src0, src1, dst, cc);
    return false;
#endif
}

void ggml_cuda_mul_mat_wmma(ggml_backend_cuda_context & ctx, const ggml_tensor * src0, const ggml_tensor * src1, ggml_tensor * dst) {
#ifdef GGML_USE_HIP
    const int k = src0->ne[0], n = src0->ne[1], m = src1->ne[1];
    cudaStream_t stream = ctx.stream();
    // the GEMM reads the FP32 activations and converts them on their way to LDS
    const int blocks = ((n + gemm_wmma::BN - 1) / gemm_wmma::BN) * ((m + 127) / 128);
    gemm_f16_wmma<128, false, false, float><<<blocks, gemm_wmma::THREADS, 0, stream>>>(
        (const float *) src1->data, (const half *) src0->data, nullptr, (float *) dst->data, m, n, k, 0, nullptr, nullptr);
    CUDA_CHECK(cudaGetLastError());
#else
    GGML_UNUSED_VARS(ctx, src0, src1, dst);
    GGML_ABORT("the WMMA matrix multiplication requires the ROCm backend");
#endif
}
