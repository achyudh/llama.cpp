#undef NDEBUG
#include "llama.h"
#include "../src/llama-model.h"
#include "../src/llama-moe-cache.h"
#include "../src/llama-model-saver.h"
#include "ggml-backend-impl.h"
#include "ggml-cpu.h"
#include "ggml-exl3.h"
#include "gguf.h"

#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <stdexcept>
#include <vector>

// CPU storage and arithmetic with a separate device/buffer identity, to exercise real scheduler copies.
static ggml_backend_device test_device;
static ggml_backend_buffer_type test_buft;
static ggml_backend_dev_t cpu_device;
static bool allow_exl3 = true;
static const char * device_name(ggml_backend_dev_t) { return "EXL3-test-device"; }
static const char * buffer_name(ggml_backend_buffer_type_t) { return "EXL3-test-buffer"; }
static enum ggml_backend_dev_type device_type(ggml_backend_dev_t) { return GGML_BACKEND_DEVICE_TYPE_GPU; }
static void device_memory(ggml_backend_dev_t, size_t * free, size_t * total) { *free = *total = 1024*1024*1024; }
static void device_props(ggml_backend_dev_t, ggml_backend_dev_props * p) {
    ggml_backend_dev_get_props(cpu_device, p);
    p->name = "EXL3-test-device";
    p->description = "CPU cache integration test";
    p->type = GGML_BACKEND_DEVICE_TYPE_GPU;
    p->caps.buffer_from_host_ptr = false;
    p->caps.events = false;
}
static ggml_backend_buffer_type_t device_buft(ggml_backend_dev_t) { return &test_buft; }
static ggml_backend_buffer_type_t device_host_buft(ggml_backend_dev_t) { return ggml_backend_cpu_buffer_type(); }
static bool buffer_host(ggml_backend_buffer_type_t) { return false; }
static ggml_backend_buffer_t buffer_alloc(ggml_backend_buffer_type_t, size_t size) {
    auto * b = ggml_backend_buft_alloc_buffer(ggml_backend_cpu_buffer_type(), size);
    if (b) { b->buft = &test_buft; }
    return b;
}
static bool device_supports_buft(ggml_backend_dev_t, ggml_backend_buffer_type_t b) { return b == &test_buft; }
static bool device_supports_op(ggml_backend_dev_t, const ggml_tensor * op) {
    return (allow_exl3 || op->op != GGML_OP_MUL_MAT_EXL3) && ggml_backend_dev_supports_op(cpu_device, op);
}
static bool device_offload_op(ggml_backend_dev_t, const ggml_tensor * op) {
    return op->op == GGML_OP_MUL_MAT_EXL3;
}
static ggml_backend_t device_init(ggml_backend_dev_t, const char *) {
    auto * b = ggml_backend_cpu_init();
    b->device = &test_device;
    ggml_backend_cpu_set_n_threads(b, 2);
    return b;
}
static void init_device() {
    cpu_device = ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_CPU);
    assert(cpu_device);
    test_device = *cpu_device;
    test_device.iface.get_name = device_name;
    test_device.iface.get_type = device_type;
    test_device.iface.get_memory = device_memory;
    test_device.iface.get_props = device_props;
    test_device.iface.get_buffer_type = device_buft;
    test_device.iface.get_host_buffer_type = device_host_buft;
    test_device.iface.init_backend = device_init;
    test_device.iface.supports_op = device_supports_op;
    test_device.iface.supports_buft = device_supports_buft;
    test_device.iface.offload_op = device_offload_op;
    test_buft = *ggml_backend_cpu_buffer_type();
    test_buft.device = &test_device;
    test_buft.iface.get_name = buffer_name;
    test_buft.iface.alloc_buffer = buffer_alloc;
    test_buft.iface.is_host = buffer_host;
}

static llama_model * load(const char * file, bool device) {
    auto p = llama_model_default_params();
    ggml_backend_dev_t devices[] = { device ? &test_device : nullptr, nullptr };
    p.devices = devices;
    p.n_gpu_layers = device ? 99 : 0;
    p.load_mtp = true;
    p.load_mode = LLAMA_LOAD_MODE_MMAP;
    p.check_tensors = true;
    auto * m = llama_model_load_from_file(file, p);
    if (!m) { throw std::runtime_error("model load failed"); }
    return m;
}

static std::vector<float> run(llama_model * model, int rows, bool mtp, size_t cache) {
    auto p = llama_context_default_params();
    p.n_ctx = 128; p.n_batch = 32; p.n_ubatch = rows;
    p.n_threads = p.n_threads_batch = 2;
    p.ctx_type = mtp ? LLAMA_CONTEXT_TYPE_MTP : LLAMA_CONTEXT_TYPE_DEFAULT;
    p.flash_attn_type = LLAMA_FLASH_ATTN_TYPE_DISABLED;
    p.moe_cache_size = cache;
    auto * ctx = llama_init_from_model(model, p);
    if (!ctx) { throw std::runtime_error("context init failed"); }
    auto * batch = llama_batch_ext_init(ctx);
    std::vector<float> h(llama_model_n_embd_out(model));
    for (size_t j = 0; j < h.size(); ++j) { h[j] = std::sin(float(j)*.03f)*.1f; }
    std::vector<float> result;
    for (int start = 0; start < 10; start += rows) {
        llama_batch_ext_clear(batch);
        const int count = std::min(rows, 10-start);
        for (int j = 0; j < count; ++j) {
            const int id = llama_batch_ext_add_token(batch, 0, (start+j)*7 % 128);
            const llama_pos pos[] = { start+j, start+j, start+j, 0 };
            assert(id >= 0 && llama_batch_ext_set_pos(batch, id, pos));
            if (mtp) { assert(llama_batch_ext_set_embd_state(batch, id, {h.data(), 1, h.size()})); }
            assert(llama_batch_ext_set_output_logits(batch, id, true));
        }
        if (llama_process(ctx, LLAMA_PROCESS_TYPE_DECODE, batch)) { throw std::runtime_error("decode failed"); }
        for (int j = 0; j < count; ++j) {
            const float * y = llama_get_logits_ith(ctx, j);
            result.insert(result.end(), y, y+128);
        }
    }
    llama_batch_ext_free(batch);
    llama_free(ctx);
    return result;
}

static void compare(const std::vector<float> & ref, const std::vector<float> & got, bool exact) {
    assert(ref.size() == got.size());
    double sq = 0, error = 0, peak = 0;
    for (size_t i = 0; i < ref.size(); ++i) {
        assert(std::isfinite(ref[i]) && std::isfinite(got[i]));
        const double delta = double(ref[i])-got[i];
        error += delta*delta; sq += double(ref[i])*ref[i]; peak = std::max(peak, std::abs(delta));
    }
    const double rel = std::sqrt(error/std::max(sq, 1e-30));
    printf("full graph: exact=%d relative_rms=%.9g max_abs=%.9g\n", exact, rel, peak);
    if (exact) { assert(std::memcmp(ref.data(), got.data(), ref.size()*sizeof(float)) == 0); }
    else { assert(rel < 1e-4 && peak < 1e-4); }
}

static void compare_tensors(const ggml_tensor * a, const ggml_tensor * b) {
    assert(ggml_nbytes(a) == ggml_nbytes(b));
    std::vector<uint8_t> bytes(ggml_nbytes(b));
    ggml_backend_tensor_get(b, bytes.data(), 0, bytes.size());
    assert(!memcmp(a->data, bytes.data(), bytes.size()));
}

// Force reuse and replacement with distinct per-expert auxiliaries through the production fill callback.
static void test_eviction(llama_model & model) {
    auto * backend = device_init(&test_device, nullptr);
    {
        llama_moe_cache cache(model, {backend}, {&test_buft}, 1280*1024);
        auto * w = model.layers[0].ffn_gate_exps;
        const auto b = model.exl3.at(w);
        auto * cp = cache.get_experts(w), * su = cache.get_experts(b.suh), * sv = cache.get_experts(b.svh);
        assert(cp && su && sv && cp->ne[2] == su->ne[2] && cp->ne[2] == sv->ne[2]);
        assert(cp->ne[2] < 4 && cp->ne[2] >= 2);
        auto * map = cache.get_slot_map(0, 1, 2);
        assert(map);
        ggml_context_ptr ctx(ggml_init({1024*1024, nullptr, true}));
        auto * ids = ggml_new_tensor_1d(ctx.get(), GGML_TYPE_I32, 2);
        ggml_backend_buffer_ptr ids_buf(ggml_backend_buft_alloc_buffer(ggml_backend_cpu_buffer_type(), ggml_nbytes(ids)));
        assert(ggml_backend_tensor_alloc(ids_buf.get(), ids, ggml_backend_buffer_get_base(ids_buf.get())) == GGML_STATUS_SUCCESS);
        auto * dst_map = ggml_dup_tensor(ctx.get(), map);
        ggml_backend_buffer_ptr dst_buf(ggml_backend_buft_alloc_buffer(ggml_backend_cpu_buffer_type(), ggml_nbytes(dst_map)));
        assert(ggml_backend_tensor_alloc(dst_buf.get(), dst_map, ggml_backend_buffer_get_base(dst_buf.get())) == GGML_STATUS_SUCCESS);
        auto * lookup = ggml_get_rows(ctx.get(), dst_map, ids);
        auto * graph = ggml_new_graph(ctx.get());
        ggml_build_forward_expand(graph, lookup);
        const int32_t histories[][2] = {{0,1},{0,1},{2,3},{0,1},{3,2},{1,0}};
        for (const auto & history : histories) {
            std::memcpy(ids->data, history, sizeof(history));
            assert(cache.copy(backend, map, dst_map, graph));
            for (int e : history) {
                const int slot = ((int32_t *) dst_map->data)[e];
                assert(slot >= 0 && slot < cp->ne[2]);
                for (const auto pair : {std::make_pair(w, cp), std::make_pair(b.suh, su), std::make_pair(b.svh, sv)}) {
                    std::vector<uint8_t> bytes(pair.first->nb[2]);
                    ggml_backend_tensor_get(pair.second, bytes.data(), slot*pair.second->nb[2], bytes.size());
                    assert(std::memcmp(bytes.data(), (char *) pair.first->data+e*pair.first->nb[2], bytes.size()) == 0);
                }
            }
        }
        for (auto * source : {w, b.suh, b.svh}) {
            auto * dst = ggml_dup_tensor(ctx.get(), source);
            ggml_backend_buffer_ptr buffer(ggml_backend_buft_alloc_buffer(&test_buft, ggml_nbytes(dst)));
            assert(ggml_backend_tensor_alloc(buffer.get(), dst, ggml_backend_buffer_get_base(buffer.get())) == GGML_STATUS_SUCCESS);
            ggml_backend_buffer_clear(buffer.get(), 0x5a);
            assert(cache.copy_experts(backend, source, dst, 0, 0) == 1);
            assert(cache.copy_experts(backend, source, dst, 1, 1) == 1);
            assert(cache.copy_experts(backend, source, dst, 2, 3) == 0);
            std::vector<uint8_t> bytes(2*source->nb[2]);
            ggml_backend_tensor_get(dst, bytes.data(), 0, bytes.size());
            assert(!memcmp(source->data, bytes.data(), bytes.size()));
        }
        assert(!cache.get_slot_map(0, 33, 2));
        printf("EXL3 cache: coherent packed/suh/svh hits, evictions and refills pass\n");
    }
    ggml_backend_free(backend);
}

static int test_main(int argc, char ** argv) {
    if (argc != 3) { fprintf(stderr, "usage: %s compressed.gguf reconstructed.gguf\n", argv[0]); return 1; }
    llama_backend_init();
    llama_log_set([](enum ggml_log_level level, const char * text, void *) {
        if (level >= GGML_LOG_LEVEL_WARN) { fputs(text, stderr); }
    }, nullptr);
    init_device();
    auto * quant = load(argv[1], false);
    auto * ref = load(argv[2], false);
    auto load_params = llama_model_default_params();
    load_params.n_gpu_layers = 0; load_params.load_mtp = true; load_params.no_alloc = true;
    load_params.load_mode = LLAMA_LOAD_MODE_NONE;
    auto * descriptor = llama_model_load_from_file(argv[1], load_params);
    assert(descriptor && descriptor->exl3.size() == quant->exl3.size());
    llama_model_free(descriptor);
    load_params.no_alloc = false; load_params.load_mtp = false;
    auto * trunk = llama_model_load_from_file(argv[1], load_params);
    assert(trunk && trunk->exl3.size() < quant->exl3.size() && !trunk->layers.back().nextn.fc_hidden);
    compare(run(quant, 4, false, 0), run(trunk, 4, false, 0), true);
    llama_model_free(trunk);
    gguf_context_ptr bad(gguf_init_from_file(argv[1], {true, nullptr}));
    assert(bad);
    load_params.no_alloc = true; load_params.load_mtp = true;
    gguf_set_val_u32(bad.get(), "exl3.version", 2);
    assert(!llama_model_init_from_user(bad.get(), nullptr, nullptr, load_params));
    gguf_set_val_u32(bad.get(), "exl3.version", 1);
    gguf_set_val_str(bad.get(), "exl3.codebook", "unknown");
    assert(!llama_model_init_from_user(bad.get(), nullptr, nullptr, load_params));
    gguf_set_val_str(bad.get(), "exl3.codebook", "mul1");
    gguf_set_val_str(bad.get(), "exl3.output.weight.bits_x2", "12");
    assert(!llama_model_init_from_user(bad.get(), nullptr, nullptr, load_params));
    for (bool mtp : {false, true}) {
        for (int rows : {1,4,5}) { compare(run(ref, rows, mtp, 0), run(quant, rows, mtp, 0), false); }
    }
    FILE * file = tmpfile();
    assert(file);
    {
        llama_model_saver saver(quant);
        saver.add_kv_from_model(); saver.add_tensors_from_model(); saver.save(file);
    }
    rewind(file);
    auto params = llama_model_default_params();
    params.n_gpu_layers = 0; params.load_mtp = true; params.load_mode = LLAMA_LOAD_MODE_NONE;
    auto * roundtrip = llama_model_load_from_file_ptr(file, params);
    assert(roundtrip && roundtrip->exl3.size() == quant->exl3.size());
    for (const auto & entry : quant->exl3) {
        const auto found = std::find_if(roundtrip->exl3.begin(), roundtrip->exl3.end(), [&](const auto & other) {
            return !strcmp(entry.first->name, other.first->name);
        });
        assert(found != roundtrip->exl3.end() && found->second.bits_x2 == entry.second.bits_x2);
        compare_tensors(entry.first, found->first);
        compare_tensors(entry.second.suh, found->second.suh);
        compare_tensors(entry.second.svh, found->second.svh);
    }
    compare(run(quant, 1, true, 0), run(roundtrip, 1, true, 0), true);
    llama_model_free(roundtrip);
    fclose(file);
    auto * cached = load(argv[1], true);
    test_eviction(*cached);
    for (bool mtp : {false, true}) {
        for (int rows : {1,4,5}) { compare(run(quant, rows, mtp, 0), run(cached, rows, mtp, 1280*1024), true); }
    }
    allow_exl3 = false;
    auto * backend = device_init(&test_device, nullptr);
    { llama_moe_cache cache(*cached, {backend}, {&test_buft}, 1280*1024); assert(!cache.get_experts(cached->layers[0].ffn_gate_exps)); }
    ggml_backend_free(backend);
    llama_model_free(cached); llama_model_free(ref); llama_model_free(quant);
    llama_backend_free();
    puts("EXL3 full-model graph/cache integration passed (CPU test device, not ROCm qualification)");
    return 0;
}

int main(int argc, char ** argv) {
    try { return test_main(argc, argv); }
    catch (const std::exception & e) { fprintf(stderr, "%s\n", e.what()); return 1; }
}
