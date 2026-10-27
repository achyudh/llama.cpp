#include "ggml-exl3.h"
#include "test-exl3-fixtures.h"
#include "ggml-backend.h"
#include "ggml-cpp.h"
#include "gguf.h"

#undef NDEBUG
#include <cassert>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <limits>
#include <vector>

static void test_bounds() {
    uint8_t packed[256] = {};
    ggml_fp16_t out[256];
    std::memset(out, 0xa5, sizeof(out));
    for (int rate : {-1, 0, 1, 9, 11, 13, 15, 17, 1000000}) {
        assert(ggml_exl3_tile_size(rate) == 0);
        assert(!ggml_exl3_decode_tile(packed, sizeof(packed), rate, out));
    }
    assert(!ggml_exl3_decode_tile(nullptr, 112, 7, out));
    assert(!ggml_exl3_decode_tile(packed, 112, 7, nullptr));
    assert(!ggml_exl3_decode_tile(packed, 111, 7, out));
    assert(!ggml_exl3_decode_tile(packed, 113, 7, out));
    for (auto value : out) {
        assert(value == 0xa5a5);
    }
}

static void test_tiles() {
    for (int rate : {2, 3, 4, 5, 6, 7, 8, 10, 12, 14, 16}) {
        std::vector<uint8_t> packed(ggml_exl3_tile_size(rate) + 1, 0);
        ggml_fp16_t out[258];
        out[0] = out[257] = 0x5a5a;
        assert(ggml_exl3_decode_tile(packed.data() + 1, packed.size() - 1, rate, out + 1));
        for (int i = 1; i <= 256; ++i) {
            assert(out[i] == ggml_exl3_mul1(0));
        }
        assert(out[0] == 0x5a5a && out[257] == 0x5a5a);
    }
}

static void test_golden() {
    ggml_fp16_t actual[256];
    assert(ggml_exl3_decode_tile(exl3_packed_7, sizeof(exl3_packed_7), 7, actual));
    assert(std::memcmp(actual, exl3_expected_7, sizeof(actual)) == 0);
    assert(ggml_exl3_decode_tile(exl3_packed_8, sizeof(exl3_packed_8), 8, actual));
    assert(std::memcmp(actual, exl3_expected_8, sizeof(actual)) == 0);
}

static void test_ngram() {
    ggml_fp16_t bias[160], out[162];
    for (int i = 0; i < 160; ++i) { bias[i] = ggml_fp32_to_fp16((i % 7 - 3)*0.0625f); }
    for (int bits = 1; bits <= 8; ++bits) {
        std::vector<uint8_t> row(ggml_exl3_ngram_row_size(bits, 160) + 1, 0);
        row[1] = 0x00; row[2] = 0x38; // FP16 scale 0.5, unaligned source
        out[0] = out[161] = 0x5a5a;
        assert(ggml_exl3_decode_ngram_row(row.data()+1, row.size()-1, bits, 160, bias, 160, out+1, 160));
        for (int i = 0; i < 160; ++i) {
            assert(out[i+1] == ggml_fp32_to_fp16(fmaf(ggml_fp16_to_fp32(ggml_exl3_mul1(0)), 0.5f, ggml_fp16_to_fp32(bias[i]))));
        }
        assert(out[0] == 0x5a5a && out[161] == 0x5a5a);
    }
    uint8_t row[162] = {};
    for (auto & v : out) { v = 0x5a5a; }
    for (int bits : {-1, 0, 9, 100}) { assert(!ggml_exl3_decode_ngram_row(row, 162, bits, 160, bias, 160, out+1, 160)); }
    for (int dim : {-1, 0, 15, 159, 257, 1000000000}) { assert(!ggml_exl3_ngram_row_size(8, dim)); }
    assert(!ggml_exl3_decode_ngram_row(nullptr, 162, 8, 160, bias, 160, out+1, 160));
    assert(!ggml_exl3_decode_ngram_row(row, 162, 8, 160, nullptr, 160, out+1, 160));
    assert(!ggml_exl3_decode_ngram_row(row, 162, 8, 160, bias, 159, out+1, 160));
    assert(!ggml_exl3_decode_ngram_row(row, 161, 8, 160, bias, 160, out+1, 160));
    assert(!ggml_exl3_decode_ngram_row(row, 163, 8, 160, bias, 160, out+1, 160));
    assert(!ggml_exl3_decode_ngram_row(row, 162, 8, 160, bias, 160, out+1, 159));
    assert(!ggml_exl3_decode_ngram_row(row, 162, 8, 160, bias, 160, nullptr, 160));
    for (auto v : out) { assert(v == 0x5a5a); }
}

static void test_ngram_native(const char * path, const char * output) {
    ggml_context * raw = nullptr;
    gguf_context_ptr meta(gguf_init_from_file(path, {false, &raw}));
    ggml_context_ptr ctx(raw);
    assert(meta && ctx);
    const auto n = gguf_get_val_u32(meta.get(), gguf_find_key(meta.get(), "cases"));
    std::ofstream file(output, std::ios::binary);
    for (uint32_t c = 0; c < n; ++c) {
        const std::string key = "case." + std::to_string(c);
        const int bits = gguf_get_val_u32(meta.get(), gguf_find_key(meta.get(), (key+".bits").c_str()));
        auto * packed = ggml_get_tensor(ctx.get(), (key+".packed").c_str());
        auto * bias = ggml_get_tensor(ctx.get(), (key+".bias").c_str());
        auto * reference = ggml_get_tensor(ctx.get(), (key+".reference").c_str());
        assert(packed && bias && reference && packed->type == GGML_TYPE_I16 && bias->type == GGML_TYPE_F16 && reference->type == GGML_TYPE_F16);
        std::vector<ggml_fp16_t> result(ggml_nelements(reference));
        assert(ggml_exl3_decode_ngram_row(packed->data, ggml_nbytes(packed), bits, result.size(),
            (const ggml_fp16_t *) bias->data, ggml_nelements(bias), result.data(), result.size()));
        if (memcmp(result.data(), reference->data, result.size()*2)) {
            fprintf(stderr, "EXL3 n-gram mismatch: case=%u bits=%d dim=%zu\n", c, bits, result.size());
            assert(false);
        }
        assert(file.write((const char *) result.data(), result.size()*2));
    }
    printf("EXL3 native n-gram rows: %u exact cases\n", n);
}

static void test_projection() {
    const auto one = ggml_fp32_to_fp16(1.0f);
    const double w = ggml_fp16_to_fp32(ggml_exl3_mul1(0));
    for (int rate : {7, 8, 10, 12, 14}) {
        for (int k : {128, 256}) {
            const int n = 384 - k;
            std::vector<uint8_t> packed(k / 16 * (n / 16) * ggml_exl3_tile_size(rate));
            std::vector<ggml_fp16_t> suh(k, one), svh(n, one);
            suh[0] = ggml_fp32_to_fp16(-2.0f);
            svh[0] = ggml_fp32_to_fp16(0.5f);
            if (k > 128) { suh[128] = ggml_fp32_to_fp16(3.0f); }
            if (n > 128) { svh[128] = ggml_fp32_to_fp16(-1.5f); }
            ggml_exl3_matrix matrix = {k, n, rate, packed.data(), packed.size(), suh.data(), suh.size(), svh.data(), svh.size()};
            for (int rows : {1, 4, 5}) {
                std::vector<float> x(rows * k), out(rows * n + 2, 123.0f);
                for (int i = 0; i < rows * k; ++i) { x[i] = float(i % 17 - 8) / 16; }
                assert(ggml_exl3_mul_mat_ref(&matrix, rows, x.data(), x.size(), out.data() + 1, rows * n));
                for (int r = 0; r < rows; ++r) {
                    double dc = 0;
                    for (int i = 0; i < k; i += 128) { dc += x[r * k + i] * ggml_fp16_to_fp32(suh[i]); }
                    for (int j = 0; j < n; ++j) {
                        const float expected = j % 128 ? 0.0f : float(128 * w * dc * ggml_fp16_to_fp32(svh[j]));
                        assert(std::abs(out[1 + r * n + j] - expected) < 1e-5f);
                    }
                }
                assert(out.front() == 123.0f && out.back() == 123.0f);
            }
        }
    }
}

static void test_projection_bounds() {
    uint8_t packed[7168] = {};
    ggml_fp16_t scales[128] = {};
    float x[128] = {}, out[128];
    for (float & v : out) { v = 123.0f; }
    const ggml_exl3_matrix valid = {128, 128, 7, packed, sizeof(packed), scales, 128, scales, 128};
    assert(!ggml_exl3_mul_mat_ref(nullptr, 1, x, 128, out, 128));
    assert(!ggml_exl3_mul_mat_ref(&valid, 0, x, 128, out, 128));
    assert(!ggml_exl3_mul_mat_ref(&valid, -1, x, 128, out, 128));
    assert(!ggml_exl3_mul_mat_ref(&valid, 1, nullptr, 128, out, 128));
    assert(!ggml_exl3_mul_mat_ref(&valid, 1, x, 128, nullptr, 128));
    assert(!ggml_exl3_mul_mat_ref(&valid, 1, x, 127, out, 128));
    assert(!ggml_exl3_mul_mat_ref(&valid, 1, x, 128, out, 127));
    assert(!ggml_exl3_mul_mat_ref(&valid, std::numeric_limits<int64_t>::max(), x, 128, out, 128));
    for (int variant = 0; variant < 12; ++variant) {
        auto bad = valid;
        switch (variant) {
            case 0: bad.ne0 = 0; break;
            case 1: bad.ne0 = 64; break;
            case 2: bad.ne1 = -128; break;
            case 3: bad.ne1 = 129; break;
            case 4: bad.ne0 = bad.ne1 = std::numeric_limits<int64_t>::max() / 128 * 128; break;
            case 5: bad.bits_x2 = 9; break;
            case 6: bad.packed = nullptr; break;
            case 7: bad.packed_bytes--; break;
            case 8: bad.suh = nullptr; break;
            case 9: bad.svh = nullptr; break;
            case 10: bad.suh_elements--; break;
            case 11: bad.svh_elements++; break;
        }
        assert(!ggml_exl3_mul_mat_ref(&bad, 1, x, 128, out, 128));
    }
    for (float v : out) { assert(v == 123.0f); }
}

static ggml_backend_t cpu_backend() {
    auto device = ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_CPU);
    if (!device) {
        ggml_backend_load_all();
        device = ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_CPU);
    }
    return device ? ggml_backend_dev_init(device, nullptr) : nullptr;
}

static void compute(ggml_context * ctx, ggml_tensor * out, ggml_backend_t backend) {
    assert(out && ggml_backend_dev_supports_op(ggml_backend_get_device(backend), out));
    auto * graph = ggml_new_graph(ctx);
    ggml_build_forward_expand(graph, out);
    assert(ggml_backend_graph_compute(backend, graph) == GGML_STATUS_SUCCESS);
}

static ggml_exl3_bundle bank(ggml_context * ctx, int experts, int rate) {
    const int k = 128, n = 256;
    auto * p = ggml_new_tensor_3d(ctx, GGML_TYPE_I16, k / 16 * (n / 16) * ggml_exl3_tile_size(rate) / 2, 1, experts);
    auto * su = ggml_new_tensor_2d(ctx, GGML_TYPE_F16, k, experts);
    auto * sv = ggml_new_tensor_2d(ctx, GGML_TYPE_F16, n, experts);
    ggml_set_name(p, "exl3.packed"); ggml_set_name(su, "exl3.suh"); ggml_set_name(sv, "exl3.svh");
    for (size_t i = 0; i < ggml_nbytes(p); ++i) { ((uint8_t *) p->data)[i] = uint8_t(i * 23 + i / 317); }
    for (int i = 0; i < k * experts; ++i) { ((ggml_fp16_t *) su->data)[i] = ggml_fp32_to_fp16(float(i % 7 - 3) / 8); }
    for (int i = 0; i < n * experts; ++i) { ((ggml_fp16_t *) sv->data)[i] = ggml_fp32_to_fp16(float(i % 11 - 5) / 8); }
    return {p, su, sv, rate};
}

static void test_dispatch(ggml_backend_t backend) {
    for (int rate : {7, 8, 14}) {
        for (int slots : {0, 1, 2, 4}) {
            for (int rows : {1, 4, 5}) {
                auto * ctx = ggml_init({4 * 1024 * 1024, nullptr, false});
                const bool routed = slots > 0;
                auto b = bank(ctx, routed ? 4 : 1, rate);
                auto * x = routed ? ggml_new_tensor_3d(ctx, GGML_TYPE_F32, 128, slots, rows) : ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 128, rows);
                auto * ids = routed ? ggml_new_tensor_2d(ctx, GGML_TYPE_I32, 4, rows) : nullptr;
                for (int64_t i = 0; i < ggml_nelements(x); ++i) { ((float *) x->data)[i] = float(i % 19 - 9) / 16; }
                if (ids) {
                    const int32_t selected[] = {3, 0, 3, 2};
                    for (int i = 0; i < 4 * rows; ++i) { ((int32_t *) ids->data)[i] = selected[(i + i / 4) % 4]; }
                }
                auto * out = ggml_mul_mat_exl3(ctx, b.packed, b.suh, b.svh, x, ids, rate);
                compute(ctx, out, backend);
                std::vector<float> expected(ggml_nelements(out));
                for (int t = 0; t < rows; ++t) {
                    for (int s = 0; s < (routed ? 4 : 1); ++s) {
                        const int pos = t * (routed ? 4 : 1) + s;
                        const int e = ids ? ((int32_t *) ids->data)[pos] : 0;
                        ggml_exl3_matrix matrix = {128, 256, rate, (uint8_t *) b.packed->data + e * b.packed->nb[2], b.packed->nb[2],
                            (ggml_fp16_t *) b.suh->data + 128 * e, 128, (ggml_fp16_t *) b.svh->data + 256 * e, 256};
                        const float * input = (float *) x->data + 128 * (routed ? t * slots + s % slots : t);
                        assert(ggml_exl3_mul_mat_ref(&matrix, 1, input, 128, expected.data() + pos * 256, 256));
                    }
                }
                assert(std::memcmp(out->data, expected.data(), ggml_nbytes(out)) == 0);
                if (ids) {
                    std::memset(out->data, 0x5a, ggml_nbytes(out));
                    ((int32_t *) ids->data)[0] = 4;
                    assert(!ggml_exl3_compute_forward(out));
                    for (size_t i = 0; i < ggml_nbytes(out); ++i) { assert(((uint8_t *) out->data)[i] == 0x5a); }
                }
                assert(!ggml_mul_mat_exl3(ctx, b.packed, b.suh, b.svh, x, ids, 9));
                auto wrong = *b.packed; wrong.type = GGML_TYPE_F32;
                assert(!ggml_mul_mat_exl3(ctx, &wrong, b.suh, b.svh, x, ids, rate));
                ggml_free(ctx);
            }
        }
    }
}

static gguf_context * metadata(const ggml_exl3_bundle & b) {
    auto * meta = gguf_init_empty();
    gguf_set_val_str(meta, "general.architecture", "exl3_projection");
    gguf_set_val_u32(meta, "exl3.version", 1);
    gguf_set_val_str(meta, "exl3.codebook", "mul1");
    gguf_set_val_u32(meta, "exl3.bits_x2", b.bits_x2);
    gguf_set_val_u64(meta, "exl3.input_features", b.suh->ne[0]);
    gguf_set_val_u64(meta, "exl3.output_features", b.svh->ne[0]);
    gguf_set_val_u32(meta, "exl3.expert_count", b.packed->ne[2]);
    gguf_add_tensor(meta, b.packed); gguf_add_tensor(meta, b.suh); gguf_add_tensor(meta, b.svh);
    return meta;
}

static void test_bundle() {
    auto * ctx = ggml_init({4 * 1024 * 1024, nullptr, false});
    auto b = bank(ctx, 2, 7);
    auto * meta = metadata(b);
    FILE * file = tmpfile();
    assert(file && gguf_write_to_file_ptr(meta, file, false));
    rewind(file);
    ggml_context * loaded = nullptr;
    auto * read = gguf_init_from_file_ptr(file, {false, &loaded});
    assert(read && loaded);
    ggml_exl3_bundle result = {};
    assert(ggml_exl3_load_bundle(read, loaded, &result));
    for (auto * original : {b.packed, b.suh, b.svh}) {
        auto * copy = ggml_get_tensor(loaded, ggml_get_name(original));
        assert(copy && ggml_nbytes(copy) == ggml_nbytes(original));
        assert(std::memcmp(copy->data, original->data, ggml_nbytes(copy)) == 0);
    }
    fclose(file); gguf_free(read); ggml_free(loaded); gguf_free(meta);
    for (int variant = 0; variant < 8; ++variant) {
        auto * bad = metadata(b);
        switch (variant) {
            case 0: gguf_set_val_f32(bad, "exl3.version", 1); break;
            case 1: gguf_set_val_u32(bad, "exl3.version", 2); break;
            case 2: gguf_set_val_str(bad, "exl3.codebook", "unknown"); break;
            case 3: gguf_set_val_u32(bad, "exl3.bits_x2", UINT32_MAX); break;
            case 4: gguf_set_val_u64(bad, "exl3.input_features", UINT64_MAX); break;
            case 5: gguf_set_val_u32(bad, "exl3.expert_count", 1); break;
            case 6: gguf_set_val_str(bad, "general.architecture", "qwen4exp"); break;
            case 7: gguf_set_val_u64(bad, "exl3.bits_x2", 7); break;
        }
        assert(!ggml_exl3_load_bundle(bad, ctx, &result));
        gguf_free(bad);
    }
    ggml_free(ctx);
}

static int run_bundle(const char * bundle_path, const char * input_path, const char * output_path, ggml_backend_t backend) {
    ggml_context * weights = nullptr, * inputs = nullptr;
    auto * meta = gguf_init_from_file(bundle_path, {false, &weights});
    auto * data = gguf_init_from_file(input_path, {false, &inputs});
    ggml_exl3_bundle b = {};
    if (!meta || !data || !ggml_exl3_load_bundle(meta, weights, &b)) {
        std::fprintf(stderr, "Invalid EXL3 bundle or inputs\n");
        if (meta) { gguf_free(meta); } if (data) { gguf_free(data); }
        if (weights) { ggml_free(weights); } if (inputs) { ggml_free(inputs); }
        return 1;
    }
    auto * ctx = ggml_init({4 * 1024 * 1024, nullptr, true});
    auto * out = ggml_mul_mat_exl3(ctx, b.packed, b.suh, b.svh, ggml_get_tensor(inputs, "input"), ggml_get_tensor(inputs, "ids"), b.bits_x2);
    auto * ids = ggml_get_tensor(inputs, "ids");
    if (out && ids) {
        for (int64_t i = 0; i < ggml_nelements(ids); ++i) {
            int32_t value = ((int32_t *) ids->data)[i];
            if (value < 0 || value >= b.packed->ne[2]) { out = nullptr; break; }
        }
    }
    int status = 1;
    ggml_backend_buffer_t buffer = out ? ggml_backend_alloc_ctx_tensors(ctx, backend) : nullptr;
    if (out && buffer) {
        compute(ctx, out, backend);
        std::ofstream output(output_path, std::ios::binary);
        output.write((const char *) out->data, ggml_nbytes(out));
        status = output ? 0 : 1;
    }
    ggml_backend_buffer_free(buffer);
    ggml_free(ctx); gguf_free(meta); gguf_free(data); ggml_free(weights); ggml_free(inputs);
    return status;
}

// Native fixtures: all 65536 codebook half words, then rate/length, packed tile and 256 decoded half words.
static void test_native(const char * path) {
    std::ifstream in(path, std::ios::binary);
    assert(in);
    std::vector<uint16_t> codebook(65536);
    assert(in.read(reinterpret_cast<char *>(codebook.data()), codebook.size() * sizeof(uint16_t)));
    for (unsigned i = 0; i < codebook.size(); ++i) {
        assert(ggml_exl3_mul1(i) == codebook[i]);
    }
    int cases = 0;
    uint32_t rate;
    while (in.read(reinterpret_cast<char *>(&rate), sizeof(rate))) {
        uint32_t length = 0;
        assert(in.read(reinterpret_cast<char *>(&length), sizeof(length)));
        assert(length == ggml_exl3_tile_size(rate) && length > 0 && length <= 256);
        std::vector<uint8_t> packed(length + 1);
        ggml_fp16_t expected[256], actual[256];
        assert(in.read(reinterpret_cast<char *>(packed.data() + 1), length));
        assert(in.read(reinterpret_cast<char *>(expected), sizeof(expected)));
        assert(ggml_exl3_decode_tile(packed.data() + 1, length, rate, actual));
        if (std::memcmp(expected, actual, sizeof(actual)) != 0) {
            std::fprintf(stderr, "EXL3 mismatch in fixture %d, rate %u/2\n", cases, rate);
            assert(false);
        }
        ++cases;
    }
    assert(in.eof() && in.gcount() == 0 && cases > 0);
    std::printf("Native oracle: all 65536 codebook entries and %d tiles match exactly\n", cases);
}

int main(int argc, char ** argv) {
    auto backend = cpu_backend();
    if (argc == 4 && !strcmp(argv[1], "--ngram")) {
        test_ngram_native(argv[2], argv[3]);
        ggml_backend_free(backend);
        return 0;
    }
    if (argc == 5 && std::strcmp(argv[1], "--bundle") == 0) {
        if (!backend) { return 1; }
        int status = run_bundle(argv[2], argv[3], argv[4], backend);
        ggml_backend_free(backend);
        return status;
    }
    test_bounds();
    test_tiles();
    test_golden();
    test_ngram();
    test_projection();
    test_projection_bounds();
    test_bundle();
    if (backend) {
        test_dispatch(backend);
        ggml_backend_free(backend);
    }
    if (argc == 2) {
        test_native(argv[1]);
    } else {
        assert(argc == 1);
    }
    std::puts("EXL3 reference codec tests passed");
    return 0;
}
