#pragma once

#include "ggml.h"

#ifdef __cplusplus
extern "C" {
#endif

// Internal EXL3 mul1 reference codec. Rates are expressed as twice the bits per weight.
GGML_API size_t ggml_exl3_tile_size(int bits_x2);
GGML_API ggml_fp16_t ggml_exl3_mul1(uint16_t state);
// Output is one row-major 16 x 16 tile in the rotated basis, without suh/svh scales.
GGML_API bool ggml_exl3_decode_tile(const void * packed, size_t packed_bytes, int bits_x2, ggml_fp16_t dst[256]);

// LSB-first embedding ring, FP16 scale then packed symbols; no Hadamard or projection scales.
GGML_API size_t ggml_exl3_ngram_row_size(int bits, int dim);
GGML_API bool ggml_exl3_decode_ngram_row(const void * packed, size_t packed_bytes, int bits, int dim,
                                       const ggml_fp16_t * bias, size_t bias_elements,
                                       ggml_fp16_t * dst, size_t dst_elements);

struct ggml_exl3_matrix {
    int64_t ne0;
    int64_t ne1;
    int bits_x2;
    const void * packed;
    size_t packed_bytes;
    const ggml_fp16_t * suh;
    size_t suh_elements;
    const ggml_fp16_t * svh;
    size_t svh_elements;
};

// Reference only: contiguous FP32 [rows, ne0] -> [rows, ne1], with FP64 transforms/accumulation and no bias.
GGML_API bool ggml_exl3_mul_mat_ref(const struct ggml_exl3_matrix * matrix, int64_t rows,
                                  const float * x, size_t x_elements, float * dst, size_t dst_elements);

struct gguf_context;
struct ggml_exl3_bundle {
    struct ggml_tensor * packed;
    struct ggml_tensor * suh;
    struct ggml_tensor * svh;
    int bits_x2;
};

// The caller owns both contexts and their tensor storage. No allocation or data conversion.
GGML_API bool ggml_exl3_load_bundle(const struct gguf_context * metadata, struct ggml_context * tensors,
                                   struct ggml_exl3_bundle * bundle);
GGML_API bool ggml_exl3_op_supported(const struct ggml_tensor * op);
GGML_API bool ggml_exl3_compute_forward(struct ggml_tensor * op);
// Thread ith of nth computes its share of the output; results do not depend on nth.
GGML_API bool ggml_exl3_compute_forward_part(struct ggml_tensor * op, int ith, int nth);

#define GGML_EXL3_AUX_BANKS 6
#define GGML_EXL3_AUX_MAX_FILLS 80
#define GGML_EXL3_AUX_SCRATCH_BYTES (2 * 1024 * 1024)

struct ggml_backend;
struct ggml_exl3_aux_batch {
    struct ggml_tensor * banks[GGML_EXL3_AUX_BANKS];
    int32_t slots[GGML_EXL3_AUX_MAX_FILLS];
    int32_t count;
    int32_t n_slots;
};

// Internal backend extension. Payload is bank-major, then fill-major; the caller keeps it alive until completion.
typedef bool (*ggml_exl3_aux_upload_t)(struct ggml_backend * backend, struct ggml_tensor * scratch,
                                      const void * payload, size_t bytes, const struct ggml_exl3_aux_batch * batch);

#ifdef __cplusplus
}
#endif
