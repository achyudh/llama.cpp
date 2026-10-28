#include "llama-expert-stream.h"

#include "llama-impl.h"
#include "llama-model.h"
#include "llama-moe-cache.h"

#include <algorithm>
#include <cinttypes>
#include <stdexcept>

static bool llama_expert_stream_is_host_weight(const ggml_tensor * t) {
    return t->buffer != nullptr &&
        ggml_backend_buffer_get_usage(t->buffer) == GGML_BACKEND_BUFFER_USAGE_WEIGHTS &&
        ggml_backend_buffer_is_host(t->buffer);
}

llama_expert_stream::llama_expert_stream(const llama_model & model, ggml_backend_t backend, int64_t min_tokens)
    : backend(backend), min_tokens(min_tokens) {
    ggml_backend_dev_t dev = ggml_backend_get_device(backend);
    ggml_backend_buffer_type_t buft = ggml_backend_get_default_buffer_type(backend);
    const size_t align = ggml_backend_buft_get_alignment(buft);

    // each layer is laid out from the start of the staging buffer
    std::vector<std::vector<size_t>> offsets(model.layers.size());
    layers.resize(model.layers.size());
    size_t n_tensors = 0;
    for (size_t il = 0; il < model.layers.size(); ++il) {
        auto experts = llama_moe_cache_layer_experts(model, model.layers[il]);
        if (experts.empty() || !std::all_of(experts.begin(), experts.end(), llama_expert_stream_is_host_weight)) {
            continue;
        }
        size_t offset = 0;
        for (const ggml_tensor * w : experts) {
            index[w] = { (int) il, (int) layers[il].weights.size() };
            layers[il].weights.push_back(w);
            offsets[il].push_back(offset);
            offset += GGML_PAD(ggml_backend_buft_get_alloc_size(buft, w), align);
        }
        staging_bytes = std::max(staging_bytes, offset);
        n_tensors += experts.size();
    }
    if (n_tensors == 0) {
        throw std::runtime_error("expert streaming: the model has no host-resident experts");
    }

    ggml_init_params params = { n_tensors*ggml_tensor_overhead(), nullptr, true };
    ctx = ggml_init(params);
    staging = ggml_backend_buft_alloc_buffer(buft, staging_bytes);
    copy_backend = ggml_backend_dev_init(dev, nullptr);
    staged_event = ggml_backend_event_new(dev);
    consumed_event = ggml_backend_event_new(dev);
    if (!ctx || !staging || !copy_backend || !staged_event || !consumed_event) {
        release();
        throw std::runtime_error("expert streaming: failed to allocate the staging buffer, copy backend or events");
    }
    char * base = (char *) ggml_backend_buffer_get_base(staging);
    for (size_t il = 0; il < layers.size(); ++il) {
        for (size_t ip = 0; ip < layers[il].weights.size(); ++ip) {
            ggml_tensor * t = ggml_dup_tensor(ctx, layers[il].weights[ip]);
            ggml_format_name(t, "stream.%s", layers[il].weights[ip]->name);
            if (ggml_backend_tensor_alloc(staging, t, base + offsets[il][ip]) != GGML_STATUS_SUCCESS) {
                release();
                throw std::runtime_error("expert streaming: failed to place a staged tensor");
            }
            layers[il].staged.push_back(t);
        }
    }

    thread = std::thread([this] { run(); });

    LLAMA_LOG_INFO("%s: %8.2f MiB staging buffer on %s for batches of at least %" PRId64 " tokens\n", __func__,
        staging_bytes/1024.0/1024.0, ggml_backend_dev_name(dev), min_tokens);
}

llama_expert_stream::~llama_expert_stream() {
    if (thread.joinable()) {
        {
            std::lock_guard<std::mutex> lock(mutex);
            stop = true;
        }
        cv.notify_all();
        thread.join();
    }
    release();
}

void llama_expert_stream::release() {
    if (copy_backend) {
        ggml_backend_synchronize(copy_backend);
    }
    ggml_backend_synchronize(backend);
    if (staged_event)   { ggml_backend_event_free(staged_event);     staged_event = nullptr; }
    if (consumed_event) { ggml_backend_event_free(consumed_event);   consumed_event = nullptr; }
    if (copy_backend)   { ggml_backend_free(copy_backend);           copy_backend = nullptr; }
    if (staging)        { ggml_backend_buffer_free(staging);         staging = nullptr; }
    if (ctx)            { ggml_free(ctx);                            ctx = nullptr; }
}

bool llama_expert_stream::copy(ggml_backend_t backend, const ggml_tensor * w, ggml_tensor * dst, int64_t n_tokens) {
    if (backend != this->backend || n_tokens < min_tokens) {
        return false;
    }
    const auto it = index.find(w);
    if (it == index.end()) {
        return false;
    }
    const auto [il, ip] = it->second;
    acquire(il);
    ggml_backend_tensor_copy_async(backend, backend, layers[il].staged[ip], dst);
    if (++n_copied == layers[il].weights.size()) {
        // the staging buffer is free once these copies are done: start on the next layer with experts
        int next = il;
        do {
            next = (next + 1) % (int) layers.size();
        } while (layers[next].weights.empty());
        std::lock_guard<std::mutex> lock(mutex);
        request(next);
        // a later copy of this layer has to upload it again
        current = -1;
    }
    return true;
}

void llama_expert_stream::acquire(int il) {
    if (il == current) {
        return;
    }
    std::unique_lock<std::mutex> lock(mutex);
    if (staged != il && requested != il) {
        // not prefetched: let a running upload finish, then upload this layer
        cv.wait(lock, [&] { return requested < 0; });
        if (staged != il) {
            request(il);
        }
    }
    cv.wait(lock, [&] { return staged == il && requested != il; });
    lock.unlock();
    ggml_backend_event_wait(backend, staged_event);
    current = il;
    n_copied = 0;
}

void llama_expert_stream::request(int il) {
    // the copies out of staging so far are on the backend's stream: the thread waits for them before overwriting
    ggml_backend_event_record(consumed_event, backend);
    requested = il;
    staged = -1;
    cv.notify_all();
}

void llama_expert_stream::run() {
    while (true) {
        int il;
        {
            std::unique_lock<std::mutex> lock(mutex);
            cv.wait(lock, [&] { return stop || requested >= 0; });
            if (stop) {
                return;
            }
            il = requested;
        }
        ggml_backend_event_synchronize(consumed_event);
        for (size_t ip = 0; ip < layers[il].weights.size(); ++ip) {
            const ggml_tensor * w = layers[il].weights[ip];
            // from pageable memory this returns once the data is staged, so the thread, not the graph, waits for it
            ggml_backend_tensor_set_async(copy_backend, layers[il].staged[ip], w->data, 0, ggml_nbytes(w));
        }
        ggml_backend_event_record(staged_event, copy_backend);
        {
            std::lock_guard<std::mutex> lock(mutex);
            staged = il;
            requested = -1;
        }
        cv.notify_all();
    }
}
