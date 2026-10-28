#include "../src/llama-model-loader.h"

#include "ggml-backend.h"
#include "ggml-cpu.h"
#include "gguf.h"

#include <cstdio>
#include <cstring>
#include <vector>

static void test_shared_mapping(int mapped_context, int n_contexts) {
    constexpr int n = 8192;
    const ggml_init_params params = {4 * ggml_tensor_overhead() + n * sizeof(float), nullptr, false};
    ggml_context_ptr source(ggml_init(params));
    auto * embedding = ggml_new_tensor_1d(source.get(), GGML_TYPE_F32, n);
    ggml_set_name(embedding, "token_embd.weight");
    for (int i = 0; i < n; ++i) {
        ((float *) embedding->data)[i] = i + 0.5f;
    }
    gguf_context_ptr meta(gguf_init_empty());
    gguf_set_val_str(meta.get(), "general.architecture", "llama");
    gguf_add_tensor(meta.get(), embedding);
    FILE * file = tmpfile();
    GGML_ASSERT(file);
    GGML_ASSERT(gguf_write_to_file_ptr(meta.get(), file, false));
    rewind(file);
    {
        std::vector<std::string> splits;
        llama_model_loader loader(nullptr, nullptr, nullptr, "", splits, file, LLAMA_LOAD_MODE_MMAP,
                                  false, false, false, nullptr, nullptr);
        std::vector<ggml_context *> contexts;
        std::vector<ggml_tensor *> tensors;
        for (int i = 0; i < n_contexts; ++i) {
            ggml_context_ptr ctx(ggml_init({4 * ggml_tensor_overhead(), nullptr, true}));
            auto * tensor = ggml_dup_tensor(ctx.get(), embedding);
            ggml_set_name(tensor, embedding->name);
            contexts.push_back(ctx.get());
            tensors.push_back(tensor);
            loader.ctx_map.emplace(llama_model_loader::ctx_key{ggml_backend_cpu_buffer_type(), i != 0}, std::move(ctx));
        }
        loader.init_mappings(false);
        const auto * weight = loader.get_weight(embedding->name);
        auto * data = (char *) loader.mappings[0]->addr() + weight->offs;
        std::vector<ggml_backend_buffer_ptr> buffers;
        for (int i = 0; i < n_contexts; ++i) {
            buffers.emplace_back(i == mapped_context
                ? ggml_backend_cpu_buffer_from_ptr(data, ggml_nbytes(embedding))
                : ggml_backend_alloc_ctx_tensors_from_buft(contexts[i], ggml_backend_cpu_buffer_type()));
        }
        for (int i = 0; i < n_contexts; ++i) {
            std::vector<ggml_backend_buffer_t> bufs{buffers[i].get()};
            GGML_ASSERT(loader.load_all_data(contexts[i], bufs, nullptr, nullptr, nullptr));
            GGML_ASSERT(!memcmp(tensors[i]->data, embedding->data, ggml_nbytes(embedding)));
        }
        for (auto * tensor : tensors) {
            GGML_ASSERT(!memcmp(tensor->data, embedding->data, ggml_nbytes(embedding)));
        }
        GGML_ASSERT(loader.size_released == (mapped_context < 0 ? ggml_nbytes(embedding) : 0));
    }
    fclose(file);
}

int main() {
    test_shared_mapping(0, 2);
    test_shared_mapping(1, 2);
    test_shared_mapping(-1, 2);
    test_shared_mapping(-1, 1);
    return 0;
}
