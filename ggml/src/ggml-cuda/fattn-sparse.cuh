#pragma once

#include "common.cuh"

// RDNA4 sparse attention for up to 8 query rows. Attention reads at most n_kv_max K/V rows per query;
// mask compaction still scans the full K/V length.
bool ggml_cuda_fattn_sparse_decode_supported(int device, const ggml_tensor * dst);

void ggml_cuda_flash_attn_ext_sparse_decode(ggml_backend_cuda_context & ctx, ggml_tensor * dst);
