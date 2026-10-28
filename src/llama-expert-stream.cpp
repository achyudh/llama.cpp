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
    const size_t n_layer = model.layers.size();

    // groups: an EXL3 bank with its two scale vectors, or a single expert tensor
    weights.resize(n_layer);
    staged.resize(n_layer);
    group_of.resize(n_layer);
    position.assign(n_layer, -1);
    std::vector<std::vector<size_t>> offsets(n_layer); // of each tensor in its group's slot
    std::vector<size_t> group_bytes;
    size_t n_tensors = 0;
    for (size_t il = 0; il < n_layer; ++il) {
        auto experts = llama_moe_cache_layer_experts(model, model.layers[il]);
        if (experts.empty() || !std::all_of(experts.begin(), experts.end(), llama_expert_stream_is_host_weight)) {
            continue;
        }
        int group = -1;
        size_t offset = 0;
        const ggml_tensor * bank = nullptr;
        for (size_t ip = 0; ip < experts.size(); ++ip) {
            ggml_tensor * t = experts[ip];
            const bool scale = bank && (model.exl3.at(bank).suh == t || model.exl3.at(bank).svh == t);
            if (!scale) {
                ++group;
                offset = 0;
                bank = model.exl3.count(t) ? t : nullptr;
            }
            index[t] = { (int) il, group, (int) ip };
            weights[il].push_back(t);
            group_of[il].push_back(group);
            offsets[il].push_back(offset);
            offset += GGML_PAD(ggml_backend_buft_get_alloc_size(buft, t), align);
            if ((size_t) group >= group_bytes.size()) {
                group_bytes.resize(group + 1, 0);
            }
            group_bytes[group] = std::max(group_bytes[group], offset);
        }
        if (!order.empty() && group != group_of[order[0]].back()) {
            throw std::runtime_error("expert streaming: the layers have different expert groups");
        }
        position[il] = (int) order.size();
        order.push_back((int) il);
        n_tensors += experts.size();
    }
    if (order.empty()) {
        throw std::runtime_error("expert streaming: the model has no host-resident experts");
    }

    // the last group of a layer is used again two groups after its slot is released: it gets a second slot
    const int n_groups = (int) group_bytes.size();
    std::vector<size_t> slot_offset;
    for (int g = 0; g < n_groups; ++g) {
        group_slots.push_back(g == n_groups - 1 && order.size() > 1 ? 2 : 1);
        group_first_slot.push_back((int) slot_offset.size());
        for (int k = 0; k < group_slots[g]; ++k) {
            slot_offset.push_back(staging_bytes);
            staging_bytes += group_bytes[g];
        }
    }

    ggml_init_params params = { n_tensors*ggml_tensor_overhead(), nullptr, true };
    ctx = ggml_init(params);
    staging = ggml_backend_buft_alloc_buffer(buft, staging_bytes);
    copy_backend = ggml_backend_dev_init(dev, nullptr);
    slots.resize(slot_offset.size());
    bool events = true;
    for (auto & s : slots) {
        s.staged = ggml_backend_event_new(dev);
        s.consumed = ggml_backend_event_new(dev);
        events = events && s.staged && s.consumed;
    }
    if (!ctx || !staging || !copy_backend || !events) {
        release();
        throw std::runtime_error("expert streaming: failed to allocate the staging buffer, copy backend or events");
    }
    char * base = (char *) ggml_backend_buffer_get_base(staging);
    for (int il : order) {
        for (size_t ip = 0; ip < weights[il].size(); ++ip) {
            ggml_tensor * t = ggml_dup_tensor(ctx, weights[il][ip]);
            ggml_format_name(t, "stream.%s", weights[il][ip]->name);
            char * addr = base + slot_offset[slot_of(il, group_of[il][ip])] + offsets[il][ip];
            if (ggml_backend_tensor_alloc(staging, t, addr) != GGML_STATUS_SUCCESS) {
                release();
                throw std::runtime_error("expert streaming: failed to place a staged tensor");
            }
            staged[il].push_back(t);
        }
    }

    thread = std::thread([this] { run(); });

    LLAMA_LOG_INFO("%s: %8.2f MiB staging buffer on %s (%zu slots for %d groups) for batches of at least %" PRId64 " tokens\n",
        __func__, staging_bytes/1024.0/1024.0, ggml_backend_dev_name(dev), slots.size(), n_groups, min_tokens);
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
    for (auto & s : slots) {
        if (s.staged)   { ggml_backend_event_free(s.staged);   s.staged = nullptr; }
        if (s.consumed) { ggml_backend_event_free(s.consumed); s.consumed = nullptr; }
    }
    if (copy_backend) { ggml_backend_free(copy_backend);   copy_backend = nullptr; }
    if (staging)      { ggml_backend_buffer_free(staging); staging = nullptr; }
    if (ctx)          { ggml_free(ctx);                    ctx = nullptr; }
}

int llama_expert_stream::slot_of(int il, int group) const {
    return group_first_slot[group] + position[il] % group_slots[group];
}

int llama_expert_stream::next_layer(int il, int steps) const {
    return order[(position[il] + steps) % order.size()];
}

ggml_tensor * llama_expert_stream::alias(ggml_backend_t backend, const ggml_tensor * w, int64_t n_tokens) const {
    if (backend != this->backend || n_tokens < min_tokens) {
        return nullptr;
    }
    const auto it = index.find(w);
    return it == index.end() ? nullptr : staged[it->second.layer][it->second.index];
}

bool llama_expert_stream::copy(ggml_backend_t backend, const ggml_tensor * w, ggml_tensor * dst, int64_t n_tokens) {
    if (backend != this->backend || n_tokens < min_tokens) {
        return false;
    }
    const auto it = index.find(w);
    if (it == index.end()) {
        return false;
    }
    const entry & e = it->second;
    acquire(e.layer, e.group);
    if (dst->data != staged[e.layer][e.index]->data) {
        ggml_backend_tensor_copy_async(backend, backend, staged[e.layer][e.index], dst);
    }
    return true;
}

void llama_expert_stream::acquire(int il, int group) {
    if (il == current_layer && group == current_group) {
        return;
    }
    std::unique_lock<std::mutex> lock(mutex);
    if (current_layer >= 0) {
        // everything that reads the previous group has been queued: its slot takes its next layer
        const int s = slot_of(current_layer, current_group);
        if (!slots[s].queued) {
            request(s, next_layer(current_layer, group_slots[current_group]));
        }
    }
    const int s = slot_of(il, group);
    if (slots[s].layer != il) {
        // not prefetched (the first batch, or a skipped layer): let a pending upload into the slot finish first
        cv.wait(lock, [&] { return !slots[s].queued; });
        if (slots[s].layer != il) {
            request(s, il);
        }
    }
    // the other slots of the group start on the following layers if they are idle (the first layers of a batch)
    for (int k = 1; k < group_slots[group]; ++k) {
        const int next = next_layer(il, k), s2 = slot_of(next, group);
        if (s2 != s && slots[s2].layer != next && !slots[s2].queued) {
            request(s2, next);
        }
    }
    cv.wait(lock, [&] { return slots[s].layer == il && slots[s].uploaded; });
    lock.unlock();
    ggml_backend_event_wait(backend, slots[s].staged);
    current_layer = il;
    current_group = group;
}

void llama_expert_stream::request(int s, int il) {
    // the compute stream has queued everything that reads the slot: the upload waits for it
    ggml_backend_event_record(slots[s].consumed, backend);
    slots[s].layer = il;
    slots[s].queued = true;
    slots[s].uploaded = false;
    queue.push_back(s);
    cv.notify_all();
}

void llama_expert_stream::run() {
    while (true) {
        int s, il;
        {
            std::unique_lock<std::mutex> lock(mutex);
            cv.wait(lock, [&] { return stop || !queue.empty(); });
            if (stop) {
                return;
            }
            s = queue.front();
            il = slots[s].layer;
        }
        ggml_backend_event_synchronize(slots[s].consumed);
        for (size_t ip = 0; ip < weights[il].size(); ++ip) {
            if (slot_of(il, group_of[il][ip]) != s) {
                continue;
            }
            // from pageable memory this returns once the data is staged, so the thread, not the graph, waits for it
            ggml_backend_tensor_set_async(copy_backend, staged[il][ip], weights[il][ip]->data, 0, ggml_nbytes(weights[il][ip]));
        }
        ggml_backend_event_record(slots[s].staged, copy_backend);
        {
            std::lock_guard<std::mutex> lock(mutex);
            queue.pop_front();
            slots[s].queued = false;
            slots[s].uploaded = true;
        }
        cv.notify_all();
    }
}
