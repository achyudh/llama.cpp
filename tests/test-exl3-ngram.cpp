#undef NDEBUG
#include "llama.h"
#include "../src/llama-impl.h"
#include "../src/llama-mmap.h"
#include "../src/llama-model.h"
#include "ggml-cpp.h"
#include "ggml-cpu.h"
#include "gguf.h"

#include <algorithm>
#include <cassert>
#include <cstring>
#include <fstream>
#include <set>
#include <stdexcept>
#include <vector>
#if defined(__linux__)
#include <sys/resource.h>
#endif

static int actual(const char * fixture, const char * source, const char * output) {
    ggml_context * raw = nullptr;
    gguf_context_ptr meta(gguf_init_from_file(fixture, {false, &raw}));
    ggml_context_ptr ctx(raw);
    assert(meta && ctx);
    auto u32 = [&](const std::string & key) { return gguf_get_val_u32(meta.get(), gguf_find_key(meta.get(), key.c_str())); };
    auto u64 = [&](const char * key) { return gguf_get_val_u64(meta.get(), gguf_find_key(meta.get(), key)); };
    auto tensor = [&](const std::string & key) { auto * t = ggml_get_tensor(ctx.get(), key.c_str()); assert(t); return t; };
    for (uint32_t h = 0; h < u32("hashes"); ++h) {
        const std::string key = "hash." + std::to_string(h);
        auto * tokens = tensor(key+".tokens");
        auto * mult = tensor(key+".multipliers");
        assert(tokens->type == GGML_TYPE_I32 && mult->type == GGML_TYPE_I64);
        std::vector<uint64_t> multipliers(ggml_nelements(mult));
        for (size_t j = 0; j < multipliers.size(); ++j) { multipliers[j] = ((int64_t *) mult->data)[j]; }
        assert(multipliers.size() == size_t(ggml_nelements(tokens)));
        assert(llama_ngram_hash((int32_t *) tokens->data, multipliers.data(), multipliers.size(),
            u32(key+".eos"), u32(key+".size"), u32(key+".offset")) == int32_t(u32(key+".expected")));
    }
    llama_file file(source, "rb");
    llama_mmap mapping(&file, 0);
    const size_t offset = u64("table.offset"), rows = u64("table.rows");
    assert(offset <= mapping.size() && rows <= (mapping.size()-offset)/162);
    ggml_context_ptr descriptors(ggml_init({4*ggml_tensor_overhead(), nullptr, true}));
    auto * table = ggml_new_tensor_2d(descriptors.get(), GGML_TYPE_I16, 81, rows);
    auto * bias = ggml_new_tensor_2d(descriptors.get(), GGML_TYPE_F16, 160, 16);
    auto * address = (uint8_t *) mapping.addr()+offset;
    auto * bias_data = tensor("bias");
    assert(bias_data->type == GGML_TYPE_F16 && ggml_nbytes(bias_data) == 16*160*2);
    ggml_backend_buffer_ptr table_buffer(ggml_backend_cpu_buffer_from_ptr(mapping.addr(), mapping.size()));
    ggml_backend_buffer_ptr bias_buffer(ggml_backend_buft_alloc_buffer(ggml_backend_cpu_buffer_type(), 16*160*2));
    assert(ggml_backend_tensor_alloc(table_buffer.get(), table, address) == GGML_STATUS_SUCCESS);
    assert(ggml_backend_tensor_alloc(bias_buffer.get(), bias, ggml_backend_buffer_get_base(bias_buffer.get())) == GGML_STATUS_SUCCESS);
    ggml_backend_tensor_set(bias, bias_data->data, 0, 16*160*2);
    std::unique_ptr<llama_model> model(llama_model_create(LLM_ARCH_QWEN4EXP, llama_model_default_params()));
    model->per_layer_tok_embd = table;
    model->per_layer_tok_embd_bias = bias;
    model->exl3_ngram_bits = 8;
    model->hparams.ple_head_dim = 160;
    model->hparams.ple_n_heads = 16;
    model->can_prefetch.insert(table);
    for (int h = 0; h < 16; ++h) {
        model->hparams.ple_head_offsets[h] = ((int64_t *) tensor("offsets")->data)[h];
        model->hparams.ple_head_vocab_sizes[h] = ((int64_t *) tensor("sizes")->data)[h];
    }
    assert(model->get_ngram_rows(nullptr, 0, nullptr, 0) == 0);
    std::ofstream out(output, std::ios::binary);
    for (uint32_t c = 0; c < u32("cases"); ++c) {
        const std::string key = "case." + std::to_string(c);
        auto * ids = tensor(key+".ids"), * ref = tensor(key+".reference");
        assert(ids->type == GGML_TYPE_I32 && ref->type == GGML_TYPE_F16);
        const size_t n = ggml_nelements(ids);
        assert(ggml_nelements(ref) == int64_t(n*160));
        std::vector<ggml_fp16_t> result(n*160+2, 0x5a5a);
        const int64_t start = ggml_time_us();
        const size_t unique = model->get_ngram_rows((int32_t *) ids->data, n, result.data()+1, n*160);
        const int64_t elapsed = ggml_time_us()-start;
        assert(unique == std::set<int32_t>((int32_t *) ids->data, (int32_t *) ids->data+n).size());
        assert(!memcmp(result.data()+1, ref->data, n*160*2));
        assert(result.front() == 0x5a5a && result.back() == 0x5a5a);
        assert(out.write((const char *) (result.data()+1), n*160*2));
        printf("ngram_gather: case=%u rows=%zu unique=%zu packed_bytes=%zu output_bytes=%zu us=%lld\n", c, n, unique, unique*162, n*320, (long long) elapsed);
    }
    for (int variant = 0; variant < 6; ++variant) {
        int32_t ids[] = {0, 1};
        size_t elements = 320;
        switch (variant) {
            case 0: ids[1] = -1; break;
            case 1: ids[1] = rows; break;
            case 2: ids[1] = model->hparams.ple_head_offsets[15]+model->hparams.ple_head_vocab_sizes[15]; break;
            case 3: elements--; break;
            case 4: model->exl3_ngram_bits = 9; break;
            case 5: bias->type = GGML_TYPE_I16; break;
        }
        std::vector<ggml_fp16_t> result(320, 0x5a5a);
        bool rejected = false;
        try { model->get_ngram_rows(ids, 2, result.data(), elements); }
        catch (const std::runtime_error &) { rejected = true; }
        assert(rejected && std::all_of(result.begin(), result.end(), [](ggml_fp16_t x) { return x == 0x5a5a; }));
        model->exl3_ngram_bits = 8; bias->type = GGML_TYPE_F16;
    }
    printf("EXL3 mapped n-gram gather passed: cases=%u hashes=%u table_bytes=%zu\n", u32("cases"), u32("hashes"), rows*162);
#if defined(__linux__)
    struct rusage usage;
    assert(getrusage(RUSAGE_SELF, &usage) == 0);
    std::ifstream status("/proc/self/status");
    std::string line;
    size_t hwm = 0;
    while (std::getline(status, line)) {
        if (line.compare(0, 6, "VmHWM:") == 0) { assert(sscanf(line.c_str()+6, "%zu", &hwm) == 1); }
    }
    printf("ngram_memory: hwm_kib=%zu inherited_maxrss_kib=%ld major_faults=%ld minor_faults=%ld\n", hwm, usage.ru_maxrss, usage.ru_majflt, usage.ru_minflt);
    assert(hwm > 0 && hwm < 512*1024);
#endif
    return 0;
}

int main(int argc, char ** argv) {
    llama_backend_init();
    if (argc == 3 && !strcmp(argv[1], "--reject")) {
        auto p = llama_model_default_params();
        ggml_backend_dev_t devices[] = {nullptr};
        p.devices = devices; p.n_gpu_layers = 0; p.load_mtp = true;
        p.no_alloc = true; p.load_mode = LLAMA_LOAD_MODE_NONE;
        auto * model = llama_model_load_from_file(argv[2], p);
        if (model) { llama_model_free(model); return 1; }
        puts("Malformed n-gram model rejected");
        return 0;
    }
    if (argc != 5 || strcmp(argv[1], "--actual")) { return 1; }
    try { return actual(argv[2], argv[3], argv[4]); }
    catch (const std::exception & e) { fprintf(stderr, "%s\n", e.what()); return 1; }
}
