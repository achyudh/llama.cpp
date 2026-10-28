#undef NDEBUG
#include "llama.h"
#include "../src/llama-model.h"
#include "../src/llama-ext.h"
#include "../src/llama-moe-cache.h"
#include "../src/llama-model-saver.h"
#include "ggml-backend-impl.h"
#include "ggml-cpu.h"
#include "ggml-exl3.h"
#include "gguf.h"

#include <algorithm>
#include <array>
#include <cassert>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <cstdlib>
#include <fstream>
#include <string>
#include <stdexcept>
#include <set>
#include <vector>

// CPU storage and arithmetic with a separate device/buffer identity, to exercise real scheduler copies.
static ggml_backend_device test_device;
static ggml_backend_buffer_type test_buft;
static ggml_backend_dev_t cpu_device;
static bool allow_exl3 = true;
static ggml_backend_dev_t cache_device;
static bool rocm = false;
static bool ngram_inputs = false;
static std::string output_directory;
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
    ggml_backend_dev_t devices[] = { device ? cache_device : nullptr, nullptr };
    p.devices = devices;
    p.n_gpu_layers = device ? 99 : 0;
    p.load_mtp = true;
    p.load_mode = LLAMA_LOAD_MODE_MMAP;
    p.check_tensors = true;
    if (ngram_inputs) { p.lazy_mode = LLAMA_LAZY_MODE_ON; }
    auto * m = llama_model_load_from_file(file, p);
    if (!m) { throw std::runtime_error("model load failed"); }
    return m;
}

static std::vector<float> run(llama_model * model, int rows, bool mtp, size_t cache, const std::vector<int> & widths = {}) {
    auto p = llama_context_default_params();
    p.n_ctx = 128; p.n_batch = 32; p.n_ubatch = rows;
    p.n_threads = p.n_threads_batch = 2;
    p.ctx_type = mtp ? LLAMA_CONTEXT_TYPE_MTP : LLAMA_CONTEXT_TYPE_DEFAULT;
    p.flash_attn_type = LLAMA_FLASH_ATTN_TYPE_DISABLED;
    p.moe_cache_size = cache;
    // the host experts to pin, as --moe-pin takes it (all or MiB)
    if (const char * pin = getenv("LLAMA_ARG_MOE_PIN")) {
        p.moe_pin_size = std::string(pin) == "all" ? SIZE_MAX : size_t(std::stoull(pin)) * 1024 * 1024;
    }
    auto * ctx = llama_init_from_model(model, p);
    if (!ctx) { throw std::runtime_error("context init failed"); }
    static int task = 0;
    llama_perf_moe_scope(ctx, ++task, 0, 1);
    auto * batch = llama_batch_ext_init(ctx);
    std::vector<float> h(llama_model_n_embd_out(model));
    for (size_t j = 0; j < h.size(); ++j) { h[j] = std::sin(float(j)*.03f)*.1f; }
    std::vector<float> result;
    for (int start = 0, step = 0; start < 10; ++step) {
        llama_batch_ext_clear(batch);
        const int count = std::min(widths.empty() ? rows : widths[step % widths.size()], 10-start);
        for (int j = 0; j < count; ++j) {
            const int32_t tokens[] = {0,7,0,14,0,0,21,28,0,35};
            const int id = llama_batch_ext_add_token(batch, 0, ngram_inputs ? tokens[start+j] : (start+j)*7 % 128);
            const llama_pos pos[] = { start+j, start+j, start+j, 0 };
            assert(id >= 0 && llama_batch_ext_set_pos(batch, id, pos));
            if (mtp) { assert(llama_batch_ext_set_embd_state(batch, id, {h.data(), 1, h.size()})); }
            assert(llama_batch_ext_set_output_logits(batch, id, true));
        }
        start += count;
        if (llama_process(ctx, LLAMA_PROCESS_TYPE_DECODE, batch)) { throw std::runtime_error("decode failed"); }
        for (int j = 0; j < count; ++j) {
            const float * y = llama_get_logits_ith(ctx, j);
            result.insert(result.end(), y, y+128);
        }
    }
    llama_batch_ext_free(batch);
    llama_perf_moe_flush(ctx, true);
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
    if (!output_directory.empty()) {
        static int number = 0;
        const std::string prefix = output_directory + "/comparison-" + std::to_string(number++);
        auto save = [&](const char * name, const std::vector<float> & values) {
            std::ofstream f(prefix + "." + name + ".bin", std::ios::binary);
            assert(f.write((const char *) values.data(), values.size()*sizeof(float)));
        };
        save("reference", ref); save("actual", got);
    }
    printf("full graph: exact=%d relative_rms=%.9g max_abs=%.9g\n", exact, rel, peak);
    fflush(stdout);
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
    auto * backend = ggml_backend_dev_init(cache_device, nullptr);
    auto * buft = ggml_backend_dev_buffer_type(cache_device);
    {
        llama_moe_cache cache(model, {backend}, {buft}, 1280*1024);
        auto * w = model.layers[0].ffn_gate_exps;
        const auto b = model.exl3.at(w);
        auto * cp = cache.get_experts(w), * su = cache.get_experts(b.suh), * sv = cache.get_experts(b.svh);
        assert(cp && su && sv && cp->ne[2] == su->ne[2] && cp->ne[2] == sv->ne[2]);
        assert(cp->ne[2] < 4 && cp->ne[2] >= 2);
        auto * map = cache.get_slot_map(0, 1, 2);
        assert(map);
        ggml_context_ptr ctx(ggml_init({1024*1024, nullptr, true}));
        auto * ids = ggml_new_tensor_1d(ctx.get(), GGML_TYPE_I32, 2);
        ggml_backend_buffer_ptr ids_buf(ggml_backend_buft_alloc_buffer(buft, ggml_nbytes(ids)));
        assert(ggml_backend_tensor_alloc(ids_buf.get(), ids, ggml_backend_buffer_get_base(ids_buf.get())) == GGML_STATUS_SUCCESS);
        auto * dst_map = ggml_dup_tensor(ctx.get(), map);
        ggml_backend_buffer_ptr dst_buf(ggml_backend_buft_alloc_buffer(buft, ggml_nbytes(dst_map)));
        assert(ggml_backend_tensor_alloc(dst_buf.get(), dst_map, ggml_backend_buffer_get_base(dst_buf.get())) == GGML_STATUS_SUCCESS);
        auto * lookup = ggml_get_rows(ctx.get(), dst_map, ids);
        auto * graph = ggml_new_graph(ctx.get());
        ggml_build_forward_expand(graph, lookup);
        const int32_t histories[][2] = {{0,1},{0,1},{2,3},{0,1},{3,2},{1,0}};
        for (const auto & history : histories) {
            ggml_backend_tensor_set(ids, history, 0, sizeof(history));
            assert(cache.copy(backend, map, dst_map, graph));
            ggml_backend_synchronize(backend);
            std::vector<int32_t> selected(ggml_nelements(dst_map));
            ggml_backend_tensor_get(dst_map, selected.data(), 0, ggml_nbytes(dst_map));
            for (int e : history) {
                const int slot = selected[e];
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
            ggml_backend_buffer_ptr buffer(ggml_backend_buft_alloc_buffer(buft, ggml_nbytes(dst)));
            assert(ggml_backend_tensor_alloc(buffer.get(), dst, ggml_backend_buffer_get_base(buffer.get())) == GGML_STATUS_SUCCESS);
            ggml_backend_buffer_clear(buffer.get(), 0x5a);
            assert(cache.copy_experts(backend, source, dst, 0, 0) == 1);
            assert(cache.copy_experts(backend, source, dst, 1, 1) == 1);
            assert(cache.copy_experts(backend, source, dst, 2, 3) == 0);
            ggml_backend_synchronize(backend);
            std::vector<uint8_t> bytes(2*source->nb[2]);
            ggml_backend_tensor_get(dst, bytes.data(), 0, bytes.size());
            assert(!memcmp(source->data, bytes.data(), bytes.size()));
        }
        assert(!cache.get_slot_map(0, 33, 2));
        if (rocm) {
            assert(cache.get_slot_map(1, 8, 2));
            assert(!cache.get_slot_map(1, 9, 2));
        }
        printf("EXL3 cache: coherent packed/suh/svh hits, evictions and refills pass\n");
    }
    ggml_backend_free(backend);
}

#if defined(GGML_USE_HIP) && !defined(_WIN32)
struct cache_test_bank {
    std::unique_ptr<void, decltype(&std::free)> storage{nullptr, &std::free};
    ggml_context_ptr host_ctx, device_ctx;
    ggml_backend_buffer_ptr host_buffer, device_buffer;
    std::array<ggml_tensor *, 3> source, resident;
    int bits;

    cache_test_bank(const std::string & file, ggml_backend_t backend, int layer, int projection) {
        ggml_context * raw = nullptr;
        gguf_context_ptr meta(gguf_init_from_file(file.c_str(), {false, &raw}));
        ggml_context_ptr original(raw);
        ggml_exl3_bundle b{};
        assert(meta && ggml_exl3_load_bundle(meta.get(), raw, &b) && b.packed->ne[2] == 512);
        bits = b.bits_x2;
        host_ctx.reset(ggml_init({8*ggml_tensor_overhead(), nullptr, true}));
        device_ctx.reset(ggml_init({8*ggml_tensor_overhead(), nullptr, true}));
        const std::array<ggml_tensor *, 3> tensors = {b.packed, b.suh, b.svh};
        size_t bytes = 0;
        for (auto * t : tensors) { assert(ggml_nbytes(t) % 4096 == 0); bytes += ggml_nbytes(t); }
        storage.reset(std::aligned_alloc(4096, bytes));
        assert(storage);
        host_buffer.reset(ggml_backend_cpu_buffer_from_ptr(storage.get(), bytes));
        size_t offset = 0;
        for (int i = 0; i < 3; ++i) {
            auto * t = tensors[i];
            source[i] = ggml_new_tensor_3d(host_ctx.get(), t->type, t->ne[0], 1, 512);
            ggml_format_name(source[i], "cache_test.%d.%d.%d", layer, projection, i);
            assert(ggml_backend_tensor_alloc(host_buffer.get(), source[i], (char *) storage.get()+offset) == GGML_STATUS_SUCCESS);
            std::memcpy(source[i]->data, t->data, ggml_nbytes(t));
            offset += ggml_nbytes(t);
            resident[i] = ggml_dup_tensor(device_ctx.get(), source[i]);
        }
        device_buffer.reset(ggml_backend_alloc_ctx_tensors(device_ctx.get(), backend));
        assert(device_buffer);
        for (int i = 0; i < 3; ++i) { ggml_backend_tensor_set(resident[i], source[i]->data, 0, ggml_nbytes(source[i])); }
        ggml_backend_buffer_set_usage(host_buffer.get(), GGML_BACKEND_BUFFER_USAGE_WEIGHTS);
        ggml_backend_buffer_set_usage(device_buffer.get(), GGML_BACKEND_BUFFER_USAGE_WEIGHTS);
    }
};

struct cache_test_graph {
    ggml_context_ptr input_ctx, ctx;
    ggml_backend_buffer_ptr input_buffer;
    ggml_backend_sched_ptr sched;
    ggml_tensor * scores, * x, * xd, * ids, * slots = nullptr;
    std::array<ggml_tensor *, 3> outputs;
    ggml_cgraph * graph;
    int rows;

    cache_test_graph(llama_moe_cache * cache, int layer, int rows, const std::vector<std::unique_ptr<cache_test_bank>> & banks,
                     ggml_backend_t gpu, ggml_backend_t cpu) : rows(rows) {
        ggml_backend_t backends[] = {gpu, cpu};
        sched.reset(ggml_backend_sched_new(backends, nullptr, 2, 1024, false, true));
        if (cache) {
            ggml_backend_sched_set_copy_callback(sched.get(), [](ggml_backend_t backend, const ggml_tensor * src, ggml_tensor * dst, ggml_cgraph * g, void * data) {
                return ((llama_moe_cache *) data)->copy(backend, src, dst, g);
            }, cache);
        }
        input_ctx.reset(ggml_init({8*ggml_tensor_overhead(), nullptr, true}));
        scores = ggml_new_tensor_2d(input_ctx.get(), GGML_TYPE_F32, 512, rows);
        x = ggml_new_tensor_3d(input_ctx.get(), GGML_TYPE_F32, 2560, 1, rows);
        xd = ggml_new_tensor_3d(input_ctx.get(), GGML_TYPE_F32, 640, 10, rows);
        input_buffer.reset(ggml_backend_alloc_ctx_tensors(input_ctx.get(), gpu));
        assert(input_buffer);
        for (auto * t : {x, xd}) {
            std::vector<float> values(ggml_nelements(t));
            for (size_t j = 0; j < values.size(); ++j) { values[j] = .1f*std::sin(j*.031f+layer); }
            ggml_backend_tensor_set(t, values.data(), 0, ggml_nbytes(t));
        }
        ctx.reset(ggml_init({2*1024*1024, nullptr, true}));
        graph = ggml_new_graph_custom(ctx.get(), 1024, false);
        ids = ggml_cont(ctx.get(), ggml_top_k(ctx.get(), scores, 10));
        ggml_set_output(ids);
        auto * selected = ids;
        if (cache) {
            auto * map = cache->get_slot_map(layer, rows, 10);
            assert(map);
            auto * lookup = ggml_get_rows(ctx.get(), map, ggml_reshape_1d(ctx.get(), ids, 10*rows));
            ggml_backend_sched_set_tensor_backend(sched.get(), lookup, gpu);
            ggml_build_forward_expand(graph, lookup);
            slots = ggml_reshape_2d(ctx.get(), lookup, 10, rows);
            ggml_set_output(slots);
            selected = slots;
        }
        for (int p = 0; p < 3; ++p) {
            auto & bank = *banks[layer*3+p];
            std::array<ggml_tensor *, 3> weights = bank.resident;
            if (cache) {
                for (int j = 0; j < 3; ++j) { weights[j] = cache->get_experts(bank.source[j]); assert(weights[j]); }
            }
            outputs[p] = ggml_mul_mat_exl3(ctx.get(), weights[0], weights[1], weights[2], p == 2 ? xd : x, selected, bank.bits);
            assert(outputs[p] && ggml_backend_supports_op(gpu, outputs[p]));
            ggml_backend_sched_set_tensor_backend(sched.get(), outputs[p], gpu);
            ggml_set_output(outputs[p]);
            ggml_build_forward_expand(graph, outputs[p]);
        }
        assert(ggml_backend_sched_alloc_graph(sched.get(), graph));
        assert(ggml_backend_sched_get_n_splits(sched.get()) == (cache ? 2 : 1));
    }

    void select(const std::vector<int32_t> & selected) {
        std::vector<float> values(512*rows);
        for (int r = 0; r < rows; ++r) {
            for (int e = 0; e < 512; ++e) { values[r*512+e] = -1000.0f-e; }
            for (int j = 0; j < 10; ++j) { values[r*512+selected[r*10+j]] = 1000.0f-j; }
        }
        ggml_backend_tensor_set(scores, values.data(), 0, ggml_nbytes(scores));
    }

    int64_t compute() {
        const int64_t start = ggml_time_us();
        assert(ggml_backend_sched_graph_compute_async(sched.get(), graph) == GGML_STATUS_SUCCESS);
        ggml_backend_sched_synchronize(sched.get());
        return ggml_time_us()-start;
    }
};

static void save_bytes(const std::string & path, const void * data, size_t bytes) {
    std::ofstream file(path, std::ios::binary);
    assert(file.write((const char *) data, bytes));
}

static int test_actual_cache(const char * fixture, const char * directory, const char * output) {
    rocm = true;
    llama_backend_init();
    cache_device = ggml_backend_dev_by_name("ROCm0");
    assert(cache_device);
    ggml_backend_ptr gpu(ggml_backend_dev_init(cache_device, nullptr)), cpu(ggml_backend_cpu_init());
    auto * buft = ggml_backend_dev_buffer_type(cache_device);
    std::vector<std::unique_ptr<cache_test_bank>> banks;
    for (int l = 0; l < 3; ++l) {
        for (int p = 0; p < 3; ++p) {
            banks.emplace_back(new cache_test_bank(std::string(directory)+"/bank-"+std::to_string(l)+"-"+std::to_string(p)+".gguf", gpu.get(), l, p));
        }
    }
    // Only use the frozen model's loader/device ownership; these graphs run actual expert modules, not that model.
    std::unique_ptr<llama_model> model(load(fixture, true));
    assert(model->layers.size() == 3);
    model->exl3.clear();
    model->hparams.n_expert = 512;
    model->hparams.n_expert_used_arr.fill(10);
    size_t bytes_per_layer[3] = {};
    for (int l = 0; l < 3; ++l) {
        auto & layer = model->layers[l];
        layer.ffn_gate_up_exps = nullptr;
        layer.ffn_gate_exps = banks[l*3]->source[0];
        layer.ffn_up_exps = banks[l*3+1]->source[0];
        layer.ffn_down_exps = banks[l*3+2]->source[0];
        for (int p = 0; p < 3; ++p) {
            auto & b = *banks[l*3+p];
            const llm_tensor roles[] = {LLM_TENSOR_FFN_GATE_EXPS, LLM_TENSOR_FFN_UP_EXPS, LLM_TENSOR_FFN_DOWN_EXPS};
            model->exl3[b.source[0]] = {b.source[1], b.source[2], b.bits, roles[p], false};
            for (auto * t : b.source) { bytes_per_layer[l] += t->nb[2]; }
        }
    }
    const size_t budget = 52*(bytes_per_layer[0]+bytes_per_layer[1]+bytes_per_layer[2])+4096;
    llama_moe_cache cache(*model, {gpu.get()}, {buft}, budget);
    for (int l = 0; l < 3; ++l) {
        auto * w = cache.get_experts(banks[l*3]->source[0]);
        assert(w && w->ne[2] >= 50 && w->ne[2] < (l == 0 ? 512 : 1024));
        printf("exl3_cache_bank: layer=%d slots=%lld expert_bytes=%zu\n", l, (long long) w->ne[2], bytes_per_layer[l]);
        assert(!cache.get_slot_map(l, 33, 10));
    }
    assert(cache.get_experts(banks[3]->source[0])->data == cache.get_experts(banks[6]->source[0])->data);
    int number = 0;
    const int offsets[] = {0,0,64,128,192,256,320,384,448,448,0,0};
    for (int rows : {1,4,5}) {
        std::vector<std::unique_ptr<cache_test_graph>> cached, resident;
        for (int l = 0; l < 3; ++l) {
            cached.emplace_back(new cache_test_graph(&cache, l, rows, banks, gpu.get(), cpu.get()));
            resident.emplace_back(new cache_test_graph(nullptr, l, rows, banks, gpu.get(), cpu.get()));
        }
        for (int step = 0; step < 12; ++step) {
            for (int l = 0; l < 3; ++l) {
                const std::string prefix = std::string(output)+"/case-"+std::to_string(number);
                std::vector<int32_t> selected(10*rows);
                for (int r = 0; r < rows; ++r) {
                    for (int j = 0; j < 10; ++j) {
                        selected[r*10+j] = step < 2 || step > 9 ? (j == 9 ? 511 : j) :
                            (offsets[step]+r*(step % 2 ? 10 : 71)+j*(step % 2 ? 1 : 7))%512;
                    }
                }
                std::set<int32_t> unique(selected.begin(), selected.end());
                std::array<std::vector<int32_t>, 3> before;
                for (int k = 0; k < 3; ++k) {
                    auto * map = (const int32_t *) cache.get_slot_map(k, 1, 10)->data;
                    before[k].assign(map, map+512);
                }
                cached[l]->select(selected); resident[l]->select(selected);
                resident[l]->compute();
                const int64_t elapsed = cached[l]->compute();
                auto * map = (const int32_t *) cache.get_slot_map(l, 1, 10)->data;
                std::vector<int32_t> actual_ids(selected.size()), slots(selected.size());
                ggml_backend_tensor_get(cached[l]->ids, actual_ids.data(), 0, actual_ids.size()*4);
                ggml_backend_tensor_get(cached[l]->slots, slots.data(), 0, slots.size()*4);
                assert(actual_ids == selected);
                for (size_t j = 0; j < slots.size(); ++j) { assert(slots[j] == map[selected[j]]); }
                save_bytes(prefix+".ids.bin", selected.data(), selected.size()*4);
                save_bytes(prefix+".slots.bin", slots.data(), slots.size()*4);
                std::vector<float> got, ref;
                for (int p = 0; p < 3; ++p) {
                    auto * tensor = cached[l]->outputs[p];
                    const size_t count = ggml_nelements(tensor), start = got.size();
                    got.resize(start+count); ref.resize(start+count);
                    ggml_backend_tensor_get(tensor, got.data()+start, 0, count*4);
                    ggml_backend_tensor_get(resident[l]->outputs[p], ref.data()+start, 0, count*4);
                    assert(std::memcmp(got.data()+start, ref.data()+start, count*4) == 0);
                    auto & bank = *banks[l*3+p];
                    for (int j = 0; j < 3; ++j) {
                        auto * src = bank.source[j], * dst = cache.get_experts(src);
                        std::vector<uint8_t> bytes(src->nb[2]);
                        for (int e : unique) {
                            assert(map[e] >= 0 && map[e] < dst->ne[2]);
                            ggml_backend_tensor_get(dst, bytes.data(), map[e]*dst->nb[2], bytes.size());
                            assert(!std::memcmp(bytes.data(), (char *) src->data+e*src->nb[2], bytes.size()));
                        }
                    }
                }
                save_bytes(prefix+".cached.bin", got.data(), got.size()*4);
                save_bytes(prefix+".resident.bin", ref.data(), ref.size()*4);
                int misses = 0, evicted = 0, runs = 0, previous_e = -2, previous_slot = -2;
                for (int e : unique) {
                    if (before[l][e] < 0) {
                        ++misses;
                        if (e != previous_e+1 || map[e] != previous_slot+1) { ++runs; }
                        previous_e = e; previous_slot = map[e];
                    }
                }
                for (int k = 0; k < 3; ++k) {
                    auto * after = (const int32_t *) cache.get_slot_map(k, 1, 10)->data;
                    for (int e = 0; e < 512; ++e) { evicted += before[k][e] >= 0 && after[e] < 0; }
                }
                size_t packed = 0, aux = 0;
                for (int p = 0; p < 3; ++p) {
                    auto & b = *banks[l*3+p];
                    packed += b.source[0]->nb[2]*misses;
                    aux += (b.source[1]->nb[2]+b.source[2]->nb[2])*misses;
                }
                printf("exl3_cache_case: id=%d layer=%d rows=%d unique=%zu hits=%zu misses=%d evictions=%d packed_bytes=%zu aux_bytes=%zu packed_calls=%d aux_calls=%d wall_us=%lld\n",
                    number, l, rows, unique.size(), unique.size()-misses, misses, evicted, packed, aux, 3*runs, 6*runs, (long long) elapsed);
                cache.perf_flush(number++, 0, 1, 0, false);
                fflush(stdout);
            }
        }
    }
    // Reuse the production device-copy path and verify untouched experts retain the poison pattern.
    for (int l = 0; l < 3; ++l) {
        size_t copied_bytes = 0, copied_calls = 0;
        auto * map = (const int32_t *) cache.get_slot_map(l, 1, 10)->data;
        for (int p = 0; p < 3; ++p) {
            auto & bank = *banks[l*3+p];
            for (auto * src : bank.source) {
                ggml_context_ptr ctx(ggml_init({2*ggml_tensor_overhead(), nullptr, true}));
                auto * dst = ggml_dup_tensor(ctx.get(), src);
                ggml_backend_buffer_ptr buffer(ggml_backend_alloc_ctx_tensors(ctx.get(), gpu.get()));
                ggml_backend_buffer_clear(buffer.get(), 0x5a);
                for (int e = 0; e < 512;) {
                    const int count = cache.copy_experts(gpu.get(), src, dst, e, 511);
                    if (count) { copied_bytes += count*src->nb[2]; ++copied_calls; e += count; }
                    else { ++e; }
                }
                ggml_backend_synchronize(gpu.get());
                std::vector<uint8_t> bytes(ggml_nbytes(dst));
                ggml_backend_tensor_get(dst, bytes.data(), 0, bytes.size());
                for (int e = 0; e < 512; ++e) {
                    auto first = bytes.begin()+e*src->nb[2], last = first+src->nb[2];
                    if (map[e] >= 0) { assert(!std::memcmp(&*first, (char *) src->data+e*src->nb[2], src->nb[2])); }
                    else { assert(std::all_of(first, last, [](uint8_t b) { return b == 0x5a; })); }
                }
            }
        }
        printf("exl3_cache_d2d: id=%d layer=%d bytes=%zu calls=%zu\n", 10000+l, l, copied_bytes, copied_calls);
        cache.perf_flush(10000+l, 0, 1, 0, false);
    }
    puts("EXL3 actual-weight GPU cache qualification passed");
    return 0;
}
#endif

static int test_main(int argc, char ** argv) {
#if defined(GGML_USE_HIP) && !defined(_WIN32)
    if (argc == 5 && !strcmp(argv[1], "--rocm-banks")) { return test_actual_cache(argv[2], argv[3], argv[4]); }
#endif
    rocm = argc == 5 && !strcmp(argv[1], "--rocm");
    ngram_inputs = argc == 5 && !strcmp(argv[1], "--ngram");
    const int first = rocm || ngram_inputs ? 2 : 1;
    if (argc != (rocm || ngram_inputs ? 5 : 3)) { fprintf(stderr, "usage: %s [--rocm|--ngram] compressed.gguf reconstructed.gguf [output-directory]\n", argv[0]); return 1; }
    if (rocm || ngram_inputs) { output_directory = argv[4]; }
    llama_backend_init();
    llama_log_set([](enum ggml_log_level level, const char * text, void *) {
        if (rocm || level >= GGML_LOG_LEVEL_WARN) { fputs(text, stderr); }
    }, nullptr);
    init_device();
    cache_device = rocm ? ggml_backend_dev_by_name("ROCm0") : &test_device;
    assert(cache_device);
    auto * quant = load(argv[first], false);
    auto * ref = load(argv[first+1], false);
    ngram_inputs = ngram_inputs || quant->exl3_ngram_bits != 0;
    auto load_params = llama_model_default_params();
    ggml_backend_dev_t cpu_devices[] = {nullptr};
    load_params.devices = cpu_devices;
    load_params.n_gpu_layers = 0; load_params.load_mtp = true; load_params.no_alloc = true;
    load_params.load_mode = LLAMA_LOAD_MODE_NONE;
    auto * descriptor = llama_model_load_from_file(argv[first], load_params);
    assert(descriptor && descriptor->exl3.size() == quant->exl3.size());
    llama_model_free(descriptor);
    load_params.no_alloc = false; load_params.load_mtp = false;
    auto * trunk = llama_model_load_from_file(argv[first], load_params);
    assert(trunk && trunk->exl3.size() < quant->exl3.size() && !trunk->layers.back().nextn.fc_hidden);
    compare(run(quant, 4, false, 0), run(trunk, 4, false, 0), true);
    llama_model_free(trunk);
    gguf_context_ptr bad(gguf_init_from_file(argv[first], {true, nullptr}));
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
    params.devices = cpu_devices;
    params.n_gpu_layers = 0; params.load_mtp = true; params.load_mode = LLAMA_LOAD_MODE_NONE;
    auto * roundtrip = llama_model_load_from_file_ptr(file, params);
    assert(roundtrip && roundtrip->exl3.size() == quant->exl3.size());
    assert(roundtrip->exl3_ngram_bits == quant->exl3_ngram_bits);
    if (quant->exl3_ngram_bits) {
        compare_tensors(quant->per_layer_tok_embd, roundtrip->per_layer_tok_embd);
        compare_tensors(quant->per_layer_tok_embd_bias, roundtrip->per_layer_tok_embd_bias);
    }
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
    auto * cached = load(argv[first], true);
    test_eviction(*cached);
    for (bool mtp : {false, true}) {
        for (int rows : {1,4,5}) {
            auto values = run(cached, rows, mtp, 1280*1024);
            compare(run(quant, rows, mtp, 0), values, !rocm);
            if (rocm) { compare(run(cached, rows, mtp, 0), values, true); }
        }
    }
    if (rocm || ngram_inputs) {
        for (bool mtp : {false, true}) {
            compare(run(cached, 5, mtp, 0, {1,1,4,1,3}), run(cached, 5, mtp, 1280*1024, {1,1,4,1,3}), true);
        }
    }
    if (!rocm) {
        allow_exl3 = false;
        auto * backend = device_init(&test_device, nullptr);
        { llama_moe_cache cache(*cached, {backend}, {&test_buft}, 1280*1024); assert(!cache.get_experts(cached->layers[0].ffn_gate_exps)); }
        ggml_backend_free(backend);
    }
    llama_model_free(cached); llama_model_free(ref); llama_model_free(quant);
    llama_backend_free();
    puts(rocm ? "EXL3 synthetic graph/cache integration passed on ROCm" : "EXL3 full-model graph/cache integration passed (CPU test device, not ROCm qualification)");
    return 0;
}

int main(int argc, char ** argv) {
    try { return test_main(argc, argv); }
    catch (const std::exception & e) { fprintf(stderr, "%s\n", e.what()); return 1; }
}
