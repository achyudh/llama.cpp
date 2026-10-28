#pragma once

#include "ggml-backend.h"

#include <array>
#include <cstdint>
#include <map>
#include <memory>
#include <vector>

struct llama_model;
struct llama_layer;

bool llama_moe_perf_enabled();

// the expert tensors of a layer and their EXL3 input/output scales: gate, up, down or gate_up, down
std::vector<ggml_tensor *> llama_moe_cache_layer_experts(const llama_model & model, const llama_layer & layer);

struct llama_moe_perf_data {
    enum metric {
        hits, misses, evictions, hit_bytes, fill_bytes, fill_calls, bypass_bytes, bypass_calls,
        bypass_large, bypass_other, padding_bytes, d2d_bytes, d2d_calls, id_bytes, id_calls,
        id_wait_us, plan_us, submit_us, copy_le64k, copy_le1m, copy_le4m, copy_gt4m, count
    };
    std::array<uint64_t, count> values = {};

    void upload(size_t bytes, bool fill);
    void print(int64_t task, int slot, int phase, int role, int layer, int type, bool total) const;
    llama_moe_perf_data & operator+=(const llama_moe_perf_data & other);
    llama_moe_perf_data operator-(const llama_moe_perf_data & other) const;
};

// keeps recently and repeatedly used experts of host-resident MoE layers in a device buffer
// each layer has a slot map in host memory: when the scheduler copies it to the device, the copy callback uploads the missing experts
class llama_moe_cache {
public:
    // backends are all the backends of the context, each GPU gets its own cache of the given size for the layers assigned to it
    // pin: bytes of the host experts to pin for the GPU (whole expert tensors, in layer order), SIZE_MAX = all
    llama_moe_cache(const llama_model & model, const std::vector<ggml_backend_t> & backends, const std::vector<ggml_backend_buffer_type_t> & bufts, size_t size, size_t pin = 0);
    ~llama_moe_cache();

    // the device that caches layer il
    ggml_backend_t backend(int32_t il) const;

    // the slot map of layer il, if its experts can be read from the cache for n_tokens tokens, nullptr otherwise
    ggml_tensor * get_slot_map(int32_t il, int64_t n_tokens, int64_t n_expert_used) const;

    // the experts of w in the cache, nullptr if w is not cached
    ggml_tensor * get_experts(const ggml_tensor * w) const;

    // ggml_backend_sched copy callback, returns false if src is not a slot map
    bool copy(ggml_backend_t backend, const ggml_tensor * src, ggml_tensor * dst, ggml_cgraph * graph);

    // for large batches: copy the experts of w that are in the cache, starting at expert e and up to expert last, to the copy dst of w
    // returns the number of experts copied, 0 if expert e is not in the cache
    int64_t copy_experts(ggml_backend_t backend, const ggml_tensor * w, ggml_tensor * dst, int64_t e, int64_t last);

    void perf_flush(int64_t task, int slot, int phase, int role, bool final);

    std::map<ggml_backend_buffer_type_t, size_t> memory_breakdown() const;

private:
    struct impl;
    std::unique_ptr<impl> pimpl;
};

using llama_moe_cache_ptr = std::unique_ptr<llama_moe_cache>;
