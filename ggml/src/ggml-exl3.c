// EXL3 packing, permutation and mul1 codebook adapted from ExLlamaV3.
// Copyright (c) 2025 Turboderp
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in all
// copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.

#include "ggml-exl3.h"
#include "ggml-impl.h"
#include "gguf.h"

#include <math.h>
#include <stdlib.h>
#include <string.h>

size_t ggml_exl3_tile_size(int bits_x2) {
    if (bits_x2 < 2 || bits_x2 > 16 || ((bits_x2 & 1) && bits_x2 > 7)) {
        return 0;
    }
    return 16 * bits_x2;
}

ggml_fp16_t ggml_exl3_mul1(uint16_t state) {
    const uint32_t product = (uint32_t) state * UINT32_C(0x83dcd12d);
    const int sum = (product & 255) + ((product >> 8) & 255) + ((product >> 16) & 255) + (product >> 24);
    const float scale = ggml_fp16_to_fp32(0x1eee);
    const float bias  = ggml_fp16_to_fp32(0xc931);
    return ggml_fp32_to_fp16(fmaf(1024.0f + sum, scale, bias));
}

static uint32_t ggml_exl3_load_u32(const uint8_t * p) {
    return (uint32_t) p[0] | ((uint32_t) p[1] << 8) | ((uint32_t) p[2] << 16) | ((uint32_t) p[3] << 24);
}

bool ggml_exl3_decode_tile(const void * packed, size_t packed_bytes, int bits_x2, ggml_fp16_t dst[256]) {
    const size_t size = ggml_exl3_tile_size(bits_x2);
    if (!packed || !dst || !size || packed_bytes != size) {
        return false;
    }

    const uint8_t * bytes = packed;
    const int ring_bits = (int) size * 8;
    const int words = (int) size / 4;
    int end = 0;
    for (int i = 0; i < 256; ++i) {
        end += bits_x2 / 2 + ((bits_x2 & 1) && (i & 1));
        const int start = (end + ring_bits - 16) % ring_bits;
        const int word = start / 32;
        const uint32_t a = ggml_exl3_load_u32(bytes + 4 * word);
        const uint32_t b = ggml_exl3_load_u32(bytes + 4 * ((word + 1) % words));
        const uint16_t state = (uint16_t) ((((uint64_t) a << 32) | b) >> (48 - (start & 31)));

        const int lane = i / 8;
        const int j = i & 7;
        const int row = (lane & 3) * 2 + (j & 1) + ((j & 2) ? 8 : 0);
        const int col = lane / 4 + ((j & 4) ? 8 : 0);
        dst[row * 16 + col] = ggml_exl3_mul1(state);
    }
    return true;
}

static bool ggml_exl3_mul_size(size_t a, size_t b, size_t * result) {
    if (a && b > SIZE_MAX / a) {
        return false;
    }
    *result = a * b;
    return true;
}

static void ggml_exl3_had_128(double * x) {
    for (int step = 1; step < 128; step *= 2) {
        for (int base = 0; base < 128; base += 2 * step) {
            for (int j = 0; j < step; ++j) {
                const double a = x[base + j];
                const double b = x[base + j + step];
                x[base + j] = a + b;
                x[base + j + step] = a - b;
            }
        }
    }
    for (int j = 0; j < 128; ++j) {
        x[j] *= 0.08838834764831844055;
    }
}

bool ggml_exl3_mul_mat_ref(const struct ggml_exl3_matrix * matrix, int64_t rows,
                          const float * x, size_t x_elements, float * dst, size_t dst_elements) {
    if (!matrix || !x || !dst || !matrix->packed || !matrix->suh || !matrix->svh ||
        matrix->ne0 <= 0 || matrix->ne1 <= 0 || rows <= 0 || matrix->ne0 % 128 || matrix->ne1 % 128 ||
        (uint64_t) matrix->ne0 > SIZE_MAX || (uint64_t) matrix->ne1 > SIZE_MAX || (uint64_t) rows > SIZE_MAX) {
        return false;
    }
    const size_t k = (size_t) matrix->ne0;
    const size_t n = (size_t) matrix->ne1;
    const size_t m = (size_t) rows;
    const size_t tile_size = ggml_exl3_tile_size(matrix->bits_x2);
    size_t tiles, packed_bytes, nx, ny, x_bytes, y_bytes;
    if (!tile_size || !ggml_exl3_mul_size(k / 16, n / 16, &tiles) ||
        !ggml_exl3_mul_size(tiles, tile_size, &packed_bytes) || matrix->packed_bytes != packed_bytes ||
        matrix->suh_elements != k || matrix->svh_elements != n ||
        !ggml_exl3_mul_size(m, k, &nx) || !ggml_exl3_mul_size(m, n, &ny) ||
        x_elements != nx || dst_elements != ny ||
        !ggml_exl3_mul_size(nx, sizeof(double), &x_bytes) || !ggml_exl3_mul_size(ny, sizeof(double), &y_bytes)) {
        return false;
    }
    double * xh = malloc(x_bytes);
    double * yh = calloc(1, y_bytes);
    if (!xh || !yh) {
        free(xh);
        free(yh);
        return false;
    }
    for (size_t r = 0; r < m; ++r) {
        for (size_t i = 0; i < k; ++i) {
            xh[r * k + i] = (double) x[r * k + i] * (double) ggml_fp16_to_fp32(matrix->suh[i]);
        }
        for (size_t i = 0; i < k; i += 128) {
            ggml_exl3_had_128(xh + r * k + i);
        }
    }
    ggml_fp16_t tile[256];
    for (size_t ki = 0; ki < k; ki += 16) {
        for (size_t ni = 0; ni < n; ni += 16) {
            const size_t offset = (ki / 16 * (n / 16) + ni / 16) * tile_size;
            if (!ggml_exl3_decode_tile((const uint8_t *) matrix->packed + offset, tile_size, matrix->bits_x2, tile)) {
                free(xh);
                free(yh);
                return false;
            }
            for (size_t i = 0; i < 16; ++i) {
                for (size_t j = 0; j < 16; ++j) {
                    const double w = ggml_fp16_to_fp32(tile[i * 16 + j]);
                    for (size_t r = 0; r < m; ++r) {
                        yh[r * n + ni + j] += xh[r * k + ki + i] * w;
                    }
                }
            }
        }
    }
    for (size_t r = 0; r < m; ++r) {
        for (size_t j = 0; j < n; j += 128) {
            ggml_exl3_had_128(yh + r * n + j);
        }
        for (size_t j = 0; j < n; ++j) {
            dst[r * n + j] = (float) (yh[r * n + j] * (double) ggml_fp16_to_fp32(matrix->svh[j]));
        }
    }
    free(xh);
    free(yh);
    return true;
}

static bool ggml_exl3_bank_valid(const struct ggml_tensor * packed, const struct ggml_tensor * suh,
                                 const struct ggml_tensor * svh, int bits_x2) {
    const uint16_t little_endian = 1;
    // GGUF I16 storage is native-endian in memory; the reference codec consumes little-endian bytes.
    if (*(const uint8_t *) &little_endian != 1) { return false; }
    if (!packed || !suh || !svh || packed->type != GGML_TYPE_I16 ||
        suh->type != GGML_TYPE_F16 || svh->type != GGML_TYPE_F16 ||
        !ggml_is_contiguous(packed) || !ggml_is_contiguous(suh) || !ggml_is_contiguous(svh) ||
        packed->ne[1] != 1 || packed->ne[3] != 1 || packed->ne[2] <= 0 ||
        suh->ne[3] != 1 || svh->ne[3] != 1 ||
        suh->ne[0] <= 0 || svh->ne[0] <= 0 || suh->ne[0] % 128 || svh->ne[0] % 128 ||
        !((suh->ne[1] == packed->ne[2] && suh->ne[2] == 1) || (suh->ne[1] == 1 && suh->ne[2] == packed->ne[2])) ||
        !((svh->ne[1] == packed->ne[2] && svh->ne[2] == 1) || (svh->ne[1] == 1 && svh->ne[2] == packed->ne[2]))) {
        return false;
    }
    const size_t tile_size = ggml_exl3_tile_size(bits_x2);
    size_t count, bytes, all_bytes;
    return tile_size && ggml_exl3_mul_size(suh->ne[0] / 16, svh->ne[0] / 16, &count) &&
        ggml_exl3_mul_size(count, tile_size, &bytes) && bytes / 2 == (uint64_t) packed->ne[0] &&
        ggml_exl3_mul_size(bytes, packed->ne[2], &all_bytes) && all_bytes <= INT64_MAX;
}

static bool ggml_exl3_inputs_valid(const struct ggml_tensor * packed, const struct ggml_tensor * suh,
                                   const struct ggml_tensor * svh, const struct ggml_tensor * x,
                                   const struct ggml_tensor * ids, int bits_x2) {
    if (!ggml_exl3_bank_valid(packed, suh, svh, bits_x2) || !x || x->type != GGML_TYPE_F32 ||
        !ggml_is_contiguous(x) || x->ne[0] != suh->ne[0] || x->ne[3] != 1) {
        return false;
    }
    size_t rows, elements, bytes, input_rows;
    if (x->ne[1] <= 0 || x->ne[2] <= 0 ||
        !ggml_exl3_mul_size(x->ne[1], x->ne[2], &input_rows) ||
        !ggml_exl3_mul_size(input_rows, x->ne[0], &elements) ||
        !ggml_exl3_mul_size(elements, sizeof(float), &bytes) || bytes > INT64_MAX) {
        return false;
    }
    if (ids) {
        if (ids->type != GGML_TYPE_I32 || !ggml_is_contiguous(ids) || ids->ne[0] <= 0 || ids->ne[1] <= 0 || ids->ne[2] != 1 || ids->ne[3] != 1 ||
            ids->ne[1] != x->ne[2] || x->ne[1] <= 0 || ids->ne[0] % x->ne[1] ||
            !ggml_exl3_mul_size(ids->ne[0], ids->ne[1], &rows)) {
            return false;
        }
    } else {
        if (packed->ne[2] != 1 || x->ne[2] != 1) {
            return false;
        }
        rows = x->ne[1];
    }
    return rows > 0 && ggml_exl3_mul_size(rows, svh->ne[0], &elements) &&
        ggml_exl3_mul_size(elements, sizeof(float), &bytes) && bytes <= INT64_MAX;
}

struct ggml_tensor * ggml_mul_mat_exl3(struct ggml_context * ctx, struct ggml_tensor * packed,
                                      struct ggml_tensor * suh, struct ggml_tensor * svh,
                                      struct ggml_tensor * x, struct ggml_tensor * ids, int bits_x2) {
    if (!ctx || !ggml_exl3_inputs_valid(packed, suh, svh, x, ids, bits_x2)) {
        return NULL;
    }
    const int64_t ne[3] = {svh->ne[0], ids ? ids->ne[0] : x->ne[1], ids ? ids->ne[1] : 1};
    struct ggml_tensor * out = ggml_new_tensor(ctx, GGML_TYPE_F32, 3, ne);
    out->op = GGML_OP_MUL_MAT_EXL3;
    out->src[0] = packed;
    out->src[1] = x;
    out->src[2] = suh;
    out->src[3] = svh;
    out->src[4] = ids;
    ggml_set_op_params_i32(out, 0, bits_x2);
    return out;
}

bool ggml_exl3_op_supported(const struct ggml_tensor * op) {
    return op && op->op == GGML_OP_MUL_MAT_EXL3 && op->type == GGML_TYPE_F32 && ggml_is_contiguous(op) &&
        ggml_exl3_inputs_valid(op->src[0], op->src[2], op->src[3], op->src[1], op->src[4], ggml_get_op_params_i32(op, 0)) &&
        op->ne[0] == op->src[3]->ne[0] && op->ne[1] == (op->src[4] ? op->src[4]->ne[0] : op->src[1]->ne[1]) &&
        op->ne[2] == (op->src[4] ? op->src[4]->ne[1] : 1) && op->ne[3] == 1;
}

bool ggml_exl3_compute_forward(struct ggml_tensor * op) {
    if (!ggml_exl3_op_supported(op) || !op->data) {
        return false;
    }
    for (int i = 0; i < 5; ++i) {
        if (op->src[i] && !op->src[i]->data) {
            return false;
        }
    }
    const struct ggml_tensor * p = op->src[0], * x = op->src[1], * su = op->src[2], * sv = op->src[3], * ids = op->src[4];
    struct ggml_exl3_matrix matrix = {su->ne[0], sv->ne[0], ggml_get_op_params_i32(op, 0),
        p->data, p->nb[2], su->data, su->ne[0], sv->data, sv->ne[0]};
    if (!ids) {
        return ggml_exl3_mul_mat_ref(&matrix, x->ne[1], x->data, ggml_nelements(x), op->data, ggml_nelements(op));
    }
    const int32_t * selected = ids->data;
    const size_t slots = ids->ne[0], total = (size_t) ids->ne[0] * ids->ne[1];
    for (size_t i = 0; i < total; ++i) {
        if (selected[i] < 0 || selected[i] >= p->ne[2]) {
            return false;
        }
    }
    for (int64_t expert = 0; expert < p->ne[2]; ++expert) {
        size_t count = 0, nx, ny, bx, by;
        for (size_t i = 0; i < total; ++i) { count += selected[i] == expert; }
        if (!count) { continue; }
        if (!ggml_exl3_mul_size(count, matrix.ne0, &nx) || !ggml_exl3_mul_size(count, matrix.ne1, &ny) ||
            !ggml_exl3_mul_size(nx, sizeof(float), &bx) || !ggml_exl3_mul_size(ny, sizeof(float), &by)) {
            return false;
        }
        float * gathered = malloc(bx), * result = malloc(by);
        if (!gathered || !result) { free(gathered); free(result); return false; }
        size_t row = 0;
        for (size_t i = 0; i < total; ++i) {
            if (selected[i] != expert) { continue; }
            const size_t input = i / slots * x->ne[1] + i % slots % x->ne[1];
            memcpy(gathered + row++ * matrix.ne0, (const float *) x->data + input * matrix.ne0, matrix.ne0 * sizeof(float));
        }
        matrix.packed = (const uint8_t *) p->data + expert * p->nb[2];
        matrix.suh = (const ggml_fp16_t *) su->data + expert * matrix.ne0;
        matrix.svh = (const ggml_fp16_t *) sv->data + expert * matrix.ne1;
        const bool ok = ggml_exl3_mul_mat_ref(&matrix, count, gathered, nx, result, ny);
        if (ok) {
            row = 0;
            for (size_t i = 0; i < total; ++i) {
                if (selected[i] == expert) {
                    memcpy((float *) op->data + i * matrix.ne1, result + row++ * matrix.ne1, matrix.ne1 * sizeof(float));
                }
            }
        }
        free(gathered);
        free(result);
        if (!ok) { return false; }
    }
    return true;
}

static int64_t ggml_exl3_key(const struct gguf_context * ctx, const char * key, enum gguf_type type) {
    const int64_t id = gguf_find_key(ctx, key);
    return id >= 0 && gguf_get_kv_type(ctx, id) == type ? id : -1;
}

bool ggml_exl3_load_bundle(const struct gguf_context * metadata, struct ggml_context * tensors,
                          struct ggml_exl3_bundle * bundle) {
    if (!metadata || !tensors || !bundle || gguf_get_n_tensors(metadata) != 3) { return false; }
    const int64_t arch = ggml_exl3_key(metadata, "general.architecture", GGUF_TYPE_STRING);
    const int64_t version = ggml_exl3_key(metadata, "exl3.version", GGUF_TYPE_UINT32);
    const int64_t codebook = ggml_exl3_key(metadata, "exl3.codebook", GGUF_TYPE_STRING);
    const int64_t rate = ggml_exl3_key(metadata, "exl3.bits_x2", GGUF_TYPE_UINT32);
    const int64_t k = ggml_exl3_key(metadata, "exl3.input_features", GGUF_TYPE_UINT64);
    const int64_t n = ggml_exl3_key(metadata, "exl3.output_features", GGUF_TYPE_UINT64);
    const int64_t e = ggml_exl3_key(metadata, "exl3.expert_count", GGUF_TYPE_UINT32);
    if (arch < 0 || version < 0 || codebook < 0 || rate < 0 || k < 0 || n < 0 || e < 0 ||
        strcmp(gguf_get_val_str(metadata, arch), "exl3_projection") || gguf_get_val_u32(metadata, version) != 1 ||
        strcmp(gguf_get_val_str(metadata, codebook), "mul1") || gguf_get_val_u32(metadata, rate) > 16) {
        return false;
    }
    struct ggml_exl3_bundle found = {ggml_get_tensor(tensors, "exl3.packed"),
        ggml_get_tensor(tensors, "exl3.suh"), ggml_get_tensor(tensors, "exl3.svh"), (int) gguf_get_val_u32(metadata, rate)};
    if (!ggml_exl3_bank_valid(found.packed, found.suh, found.svh, found.bits_x2) ||
        gguf_get_val_u64(metadata, k) != (uint64_t) found.suh->ne[0] ||
        gguf_get_val_u64(metadata, n) != (uint64_t) found.svh->ne[0] ||
        gguf_get_val_u32(metadata, e) != (uint64_t) found.packed->ne[2]) {
        return false;
    }
    struct ggml_tensor * values[] = {found.packed, found.suh, found.svh};
    for (int i = 0; i < 3; ++i) {
        const int64_t id = gguf_find_tensor(metadata, ggml_get_name(values[i]));
        if (id < 0 || gguf_get_tensor_type(metadata, id) != values[i]->type ||
            memcmp(gguf_get_tensor_ne(metadata, id), values[i]->ne, sizeof(values[i]->ne))) {
            return false;
        }
    }
    *bundle = found;
    return true;
}
