#pragma once

#include "ggml-backend.h"

#include <array>
#include <cstdint>
#include <map>
#include <memory>
#include <vector>

struct llama_model;
struct llama_layer;
class llama_kv_cache;

bool llama_moe_perf_enabled();

// the expert tensors of a layer and their EXL3 input/output scales: gate, up, down or gate_up, down
std::vector<ggml_tensor *> llama_moe_cache_layer_experts(const llama_model & model, const llama_layer & layer);

struct llama_moe_perf_data {
    enum metric {
        hits, misses, evictions, hit_bytes, fill_bytes, fill_calls, bypass_bytes, bypass_calls,
        bypass_large, bypass_other, padding_bytes, d2d_bytes, d2d_calls, id_bytes, id_calls,
        id_wait_us, plan_us, submit_us, copy_le64k, copy_le1m, copy_le4m, copy_gt4m,
        extra_hits, extra_misses, extra_fill_bytes, extra_invalidated, count
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
    llama_moe_cache(const llama_model & model, const std::vector<ggml_backend_t> & backends, const std::vector<ggml_backend_buffer_type_t> & bufts, size_t size, size_t pin = 0, bool aux_allowed = false, bool mtp = false);
    ~llama_moe_cache();

    // the device that caches layer il
    ggml_backend_t backend(int32_t il) const;

    // the slot map of layer il, if its experts can be read from the cache for n_tokens tokens, nullptr otherwise
    ggml_tensor * get_slot_map(int32_t il, int64_t n_tokens, int64_t n_expert_used) const;

    // whether the graph should queue the shared expert between the routing and the slot lookup (LLAMA_MOE_SHARED_EARLY)
    bool shared_early() const;

    // the route trigger of layer il, a host weight whose GET_ROWS by the selected experts starts the split of the
    // shared expert: its copy reads the selected experts back; nullptr if get_slot_map is nullptr or shared_early is false
    ggml_tensor * get_route_trigger(int32_t il, int64_t n_tokens, int64_t n_expert_used) const;

    // the experts of w in the cache, nullptr if w is not cached
    ggml_tensor * get_experts(const ggml_tensor * w) const;
    // lend the regions borrowed up to a level (0: base slots only, 1: staging and the top of the compute buffer, 2: also
    // its middle); lowering the level invalidates the experts in the regions it gives back
    void set_lend_level(int level);
    // lend the scheduler's compute buffer above its decode use: [b_start, size) at level 1, [a_start, b_start) at level 2
    void attach_compute(ggml_backend_buffer_t buffer, size_t a_start, size_t b_start);
    bool lends_compute() const;
    void retire_compute(ggml_backend_buffer_t buffer);
    void attach_kv(llama_kv_cache & kv);
    void set_kv_active(bool active);
    // before a graph runs: give back the regions its compute buffer use reaches; follow a reallocated compute buffer
    void fit_compute(ggml_backend_buffer_t buffer, size_t planned_size);

    // EXL3: the slot table of the packed weights w (I64 [3, n_slots], see ggml_exl3_set_slots), nullptr if w is not cached
    ggml_tensor * get_slot_table(const ggml_tensor * w) const;

    bool wants_staging() const;
    void attach_staging(ggml_backend_t backend, ggml_backend_buffer_t buffer);
    void set_staging_active(bool active);

    // ggml_backend_sched copy callback, returns false if src is not a slot map or route trigger
    bool copy(ggml_backend_t backend, const ggml_tensor * src, ggml_tensor * dst, ggml_cgraph * graph);

    // ggml_backend_sched ordered copy callback: route triggers, and the slot maps whose selected experts their route
    // trigger read back (they do not need the scheduler to wait or the other inputs of their split)
    bool copy_ordered(ggml_backend_t backend, const ggml_tensor * src, ggml_tensor * dst, ggml_cgraph * graph);

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
