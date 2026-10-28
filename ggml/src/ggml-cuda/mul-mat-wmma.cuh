#pragma once

#include "common.cuh"

// fewer rows go through mmf or BLAS
#define GGML_CUDA_MUL_MAT_WMMA_MIN_ROWS 64

// F16 x F32 matrix multiplication on RDNA4 through the FP16 WMMA GEMM of the EXL3 projections, for batches where it
// beats the BLAS path (rocBLAS FP16 runs at about 12 TFLOPS on gfx1201). GGML_CUDA_DISABLE_MUL_MAT_WMMA=1 turns it off.
bool ggml_cuda_should_use_mul_mat_wmma(const ggml_tensor * src0, const ggml_tensor * src1, const ggml_tensor * dst, int cc);
void ggml_cuda_mul_mat_wmma(ggml_backend_cuda_context & ctx, const ggml_tensor * src0, const ggml_tensor * src1, ggml_tensor * dst);
