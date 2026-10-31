#include "llama-moe-cache.h"
#include "llama-moe-cache-slru.h"
#include "llama-memory-lending.h"

#include "llama-impl.h"
#include "llama-model.h"

#include "ggml-cpp.h"
#include "ggml-exl3.h"

#include <algorithm>
#include <cerrno>
#include <cstdlib>
#include <cstring>
#include <cinttypes>
#include <cstdio>
#include <sstream>
#include <stdexcept>
#include <unordered_map>
#include <vector>

bool llama_moe_perf_enabled() {
    static const bool enabled = [] {
        const char * value = std::getenv("LLAMA_MOE_PERF");
        return value && std::strcmp(value, "1") == 0;
    }();
    return enabled;
}

void llama_moe_perf_data::upload(size_t bytes, bool fill) {
    values[fill ? fill_bytes : bypass_bytes] += bytes;
    values[fill ? fill_calls : bypass_calls]++;
    values[bytes <= 65536 ? copy_le64k : bytes <= 1048576 ? copy_le1m : bytes <= 4194304 ? copy_le4m : copy_gt4m]++;
}

llama_moe_perf_data & llama_moe_perf_data::operator+=(const llama_moe_perf_data & other) {
    for (size_t i = 0; i < count; ++i) {
        values[i] += other.values[i];
    }
    return *this;
}

llama_moe_perf_data llama_moe_perf_data::operator-(const llama_moe_perf_data & other) const {
    llama_moe_perf_data result;
    for (size_t i = 0; i < count; ++i) {
        result.values[i] = values[i] - other.values[i];
    }
    return result;
}

void llama_moe_perf_data::print(int64_t task, int slot, int phase, int role, int layer, int type, bool total) const {
    if (std::all_of(values.begin(), values.end(), [](uint64_t n) { return n == 0; })) {
        return;
    }
    static const char * names[count] = {
        "hits", "misses", "evictions", "hit_bytes", "fill_bytes", "fill_calls", "bypass_bytes", "bypass_calls",
        "bypass_large", "bypass_other", "padding_bytes", "d2d_bytes", "d2d_calls", "id_bytes", "id_calls",
        "id_wait_us", "plan_us", "submit_us", "copy_le64k", "copy_le1m", "copy_le4m", "copy_gt4m",
        "extra_hits", "extra_misses", "extra_fill_bytes", "extra_invalidated"
    };
    std::ostringstream fields;
    for (size_t i = 0; i < count; ++i) {
        fields << ' ' << names[i] << '=' << values[i];
    }
    LLAMA_LOG_INFO("moe_perf: schema=1 us=%" PRId64 " task=%" PRId64 " slot=%d phase=%d role=%d layer=%d type=%d total=%d%s\n",
        ggml_time_us(), task, slot, phase, role, layer, type, total, fields.str().c_str());
}

// gate, up, down or gate_up, down
std::vector<ggml_tensor *> llama_moe_cache_layer_experts(const llama_model & model, const llama_layer & layer) {
    std::vector<ggml_tensor *> res;
    for (ggml_tensor * t : { layer.ffn_gate_up_exps, layer.ffn_gate_exps, layer.ffn_up_exps, layer.ffn_down_exps }) {
        if (t != nullptr) {
            res.push_back(t);
            const auto it = model.exl3.find(t);
            if (it != model.exl3.end()) {
                res.push_back(it->second.suh);
                res.push_back(it->second.svh);
            }
        }
    }
    return res;
}

namespace {


static bool llama_moe_cache_same_layout(const std::vector<ggml_tensor *> & a, const std::vector<ggml_tensor *> & b) {
    if (a.size() != b.size()) {
        return false;
    }
    for (size_t i = 0; i < a.size(); ++i) {
        if (a[i]->type != b[i]->type || !ggml_are_same_shape(a[i], b[i]) || a[i]->nb[2] != b[i]->nb[2]) {
            return false;
        }
    }
    return true;
}

static bool llama_moe_cache_is_host_weight(const ggml_tensor * t) {
    return t->buffer != nullptr &&
        ggml_backend_buffer_get_usage(t->buffer) == GGML_BACKEND_BUFFER_USAGE_WEIGHTS &&
        ggml_backend_buffer_is_host(t->buffer);
}

}

struct llama_moe_cache::impl {
    static constexpr int64_t max_batch = 32;
    // a GPU with its own budget and banks, it caches the layers assigned to it
    struct device {
        ggml_backend_t backend;
        ggml_backend_buffer_type_t buft;
        size_t host_bytes = 0; // host experts of the layers it caches
        double split = 0.0;    // share of the budget

        // banks and their views
        ggml_context_ptr ctx;
        ggml_backend_buffer_ptr buf;
        size_t buf_size = 0;
        ggml_exl3_aux_upload_t aux_upload = nullptr;
        ggml_tensor * aux_scratch = nullptr;
        ggml_backend_buffer_ptr aux_host;
        ggml_backend_buffer_type_t aux_host_buft = nullptr;
        size_t aux_bytes = 0;
        bool aux_enabled = false;
        // pinned destination of the selected experts read back when a route trigger is copied, and its completion
        ggml_backend_buffer_ptr ids_host;
        ggml_backend_event_t ids_ready = nullptr;
    };

    // layers of the same device with the same expert tensor layout share the banks and the LRU of a group
    struct group {
        int32_t id;                       // device
        std::vector<ggml_tensor *> ref;   // expert tensors of the first layer
        std::vector<int32_t> layers;
        std::vector<ggml_tensor *> banks; // device storage of all slots, one per expert tensor
        // borrowed slot ranges after the base slots, in slot order and by level: level 1 (the staging buffer, the top of the
        // compute buffer) is reclaimed for streamed batches, level 2 (the middle of the compute buffer) also for bypass ones
        struct region {
            std::vector<ggml_tensor *> banks; // one per expert tensor, count + 1 slots (the last is the zero padding slot)
            int32_t first = 0;
            int32_t count = 0;
            int level = 1;
            bool compute = false;
            uint64_t lease = 0;
        };
        std::vector<region> regions;
        int32_t base_slots = 0;
        size_t host_bytes = 0;
        int32_t n_slots = 0;
        int exl3_bits_x2 = 0;
        int64_t max_tokens = max_batch;
        llama_moe_cache_slru slru;
        // EXL3 groups: I64 [ref.size(), table_rows], each slot's address of every expert tensor's storage; the ops read
        // their slots through it (ggml_exl3_set_slots), so slots can live in any buffer
        ggml_tensor * table = nullptr;
        int32_t table_rows = 0;
        std::vector<int64_t> table_host;
        std::vector<ggml_tensor *> table_views; // I64 [3, n_slots] per expert tensor, then [ref.size(), n_slots] for the aux scatter
    };

    bool has_table(const group & g) const { return g.exl3_bits_x2 && !g.ref.empty() && g.ref.size() % 3 == 0; }

    struct layer {
        int32_t ig = -1;                    // -1 if the layer is not cached
        int32_t n_expert_used = 0;
        ggml_tensor * slot_map = nullptr;   // I32 [1, n_expert] in host memory
        ggml_tensor * trigger = nullptr;    // I32 [1, n_expert] in host memory, its lookup starts the split after the routing
        std::vector<ggml_tensor *> experts; // host expert tensors, in the order of the banks
        llama_moe_perf_data perf;
        llama_moe_perf_data perf_last;
    };

    struct binding {
        int32_t il;
        int32_t ip;           // index of the bank
        ggml_tensor * cached; // view of the bank used in place of the host experts
    };

    struct stats {
        size_t hits   = 0;
        size_t misses = 0;
        size_t bytes  = 0;
    };

    int32_t n_expert_used;

    stats stats_small; // up to 8 tokens per ubatch
    stats stats_large;
    stats stats_copy;  // experts copied from the cache for large batches

    std::vector<device> devices;
    std::vector<group> groups;
    std::vector<layer> layers;
    std::unordered_map<const ggml_tensor *, binding> bindings; // host experts -> cached experts
    std::unordered_map<const ggml_tensor *, int32_t> layer_of; // slot map -> layer
    std::unordered_map<const ggml_tensor *, int32_t> trigger_of; // route trigger -> layer

    // LLAMA_MOE_SHARED_EARLY: the graph runs the shared expert between the routing and the slot lookup;
    // early reads the selected experts back before it, serial after it (a control)
    enum shared_mode { SHARED_OFF, SHARED_EARLY, SHARED_SERIAL } shared = SHARED_OFF;
    // the layer whose selected experts were read back by its route trigger and are not yet consumed
    int32_t pending_il = -1;
    const ggml_tensor * pending_sel = nullptr;

    std::vector<int32_t> ids;
    std::vector<llama_moe_cache_slru::fill> fills;

    // LLAMA_MOE_ROUTE_TRACE=path: the selected experts of every cached layer, for replaying the cache policy offline
    // (expert ids only; little-endian records, see open_trace)
    FILE * trace = nullptr;
    FILE * lease_trace = nullptr;
    uint64_t trace_routes = 0, trace_sequence = 0;
    llama_memory_lending lending;
    llama_memory_lending::owner staging_owner, compute_owner;
    int64_t trace_t0 = 0;
    std::vector<uint16_t> trace_ids;

    // slot maps in host memory
    ggml_context_ptr ctx_host;
    ggml_backend_buffer_ptr buf_host;
    size_t buf_host_size = 0;

    // views used by copy_experts
    ggml_context_ptr ctx_views;
    // the views of each table build; earlier builds stay alive, graphs built before a rebuild may still use them
    std::vector<ggml_context_ptr> ctx_tables;
    ggml_context_ptr ctx_extra;
    enum staging_mode { STAGING_OFF, STAGING_ON, STAGING_RESERVE } staging = STAGING_OFF;
    bool staging_active = false;
    int lend_level = 0;              // regions lent up to this level are in use (0: base slots only)
    ggml_context_ptr ctx_compute;    // banks lent from the scheduler's compute buffer
    size_t staging_used = 0;

    impl(const llama_model & model, const std::vector<ggml_backend_t> & backends, const std::vector<ggml_backend_buffer_type_t> & bufts, size_t size, size_t pin, bool aux_allowed, bool mtp) :
            n_expert_used(model.hparams.n_expert_used_max()), layers(model.layers.size()) {
        for (size_t i = 0; i < backends.size(); ++i) {
            const auto dev_type = ggml_backend_dev_type(ggml_backend_get_device(backends[i]));
            if (dev_type == GGML_BACKEND_DEVICE_TYPE_GPU || dev_type == GGML_BACKEND_DEVICE_TYPE_IGPU) {
                auto & d = devices.emplace_back();
                d.backend = backends[i];
                d.buft    = bufts[i];
            }
        }
        if (devices.empty()) {
            throw std::runtime_error("MoE cache requires a GPU backend");
        }
        if (model.split_mode() == LLAMA_SPLIT_MODE_TENSOR) {
            throw std::runtime_error("MoE cache does not support tensor parallelism");
        }
        if (model.hparams.n_expert == 0 || n_expert_used == 0) {
            throw std::runtime_error("MoE cache requires a MoE model");
        }

        // only cache layers that keep all of their experts in host memory, on the device the layer is assigned to
        const size_t begin = mtp ? model.hparams.n_layer() : 0;
        const size_t end = mtp ? model.layers.size() : model.hparams.n_layer();
        for (size_t il = begin; il < end; ++il) {
            auto experts = llama_moe_cache_layer_experts(model, model.layers[il]);
            if (experts.empty() || !std::all_of(experts.begin(), experts.end(), llama_moe_cache_is_host_weight)) {
                continue;
            }
            const auto it_dev = std::find_if(devices.begin(), devices.end(), [&](const device & d) { return ggml_backend_get_device(d.backend) == model.dev_layer(il); });
            if (it_dev == devices.end()) {
                continue;
            }
            int64_t supported_tokens = 0;
            ggml_context_ptr probe_ctx(ggml_init({8*ggml_tensor_overhead(), nullptr, true}));
            // Cache eligibility must also respect the device kernel's batch limit.
            for (int64_t tokens = 1; tokens <= max_batch; ++tokens) {
                bool supported = true;
                for (auto * w : experts) {
                    const auto b = model.exl3.find(w);
                    if (b == model.exl3.end()) {
                        continue;
                    }
                    auto * x = ggml_new_tensor_3d(probe_ctx.get(), GGML_TYPE_F32, b->second.suh->ne[0], 1, tokens);
                    auto * ids = ggml_new_tensor_2d(probe_ctx.get(), GGML_TYPE_I32, n_expert_used, tokens);
                    auto * op = ggml_mul_mat_exl3(probe_ctx.get(), w, b->second.suh, b->second.svh, x, ids, b->second.bits_x2);
                    supported = supported && op && ggml_backend_supports_op(it_dev->backend, op);
                    ggml_reset(probe_ctx.get());
                }
                if (!supported) { break; }
                supported_tokens = tokens;
            }
            if (!supported_tokens) {
                LLAMA_LOG_WARN("%s: layer %zu EXL3 cache disabled: device has no compressed execution kernel\n", __func__, il);
                continue;
            }
            const int32_t id = (int32_t) (it_dev - devices.begin());
            auto it = std::find_if(groups.begin(), groups.end(), [&](const group & g) { return g.id == id && llama_moe_cache_same_layout(g.ref, experts); });
            if (it == groups.end()) {
                groups.emplace_back();
                it = groups.end() - 1;
                it->id  = id;
                it->ref = experts;
                const auto b = model.exl3.find(experts.front());
                if (b != model.exl3.end()) {
                    it->exl3_bits_x2 = b->second.bits_x2;
                }
            }
            it->layers.push_back(il);
            layers[il].n_expert_used = model.hparams.n_expert_used(il);
            it->max_tokens = std::min(it->max_tokens, supported_tokens);
            for (const ggml_tensor * t : experts) {
                it->host_bytes         += ggml_nbytes(t);
                devices[id].host_bytes += ggml_nbytes(t);
            }
        }
        if (groups.empty()) {
            LLAMA_LOG_WARN("%s: no supported layer has all of its experts in host memory, MoE cache is disabled\n", __func__);
            return;
        }

        const char * aux_mode = aux_allowed && devices.size() == 1 ? std::getenv("LLAMA_MOE_AUX_BATCH") : nullptr;
        if (aux_mode && std::strcmp(aux_mode, "0") != 0 && std::strcmp(aux_mode, "1") != 0 && std::strcmp(aux_mode, "reserve") != 0) {
            throw std::invalid_argument("LLAMA_MOE_AUX_BATCH must be 0, 1 or reserve");
        }
        const char * staging_mode = aux_allowed ? std::getenv("LLAMA_MOE_CACHE_STAGING") : nullptr;
        if (staging_mode && std::strcmp(staging_mode, "0") != 0) {
            if (!aux_allowed || devices.size() != 1 || !aux_mode || std::strcmp(aux_mode, "1") != 0 ||
                    (std::strcmp(staging_mode, "1") != 0 && std::strcmp(staging_mode, "reserve") != 0)) {
                throw std::invalid_argument("LLAMA_MOE_CACHE_STAGING requires 1 or reserve, one target GPU and LLAMA_MOE_AUX_BATCH=1");
            }
            staging = std::strcmp(staging_mode, "1") == 0 ? STAGING_ON : STAGING_RESERVE;
        }
        const char * shared_mode = aux_allowed && devices.size() == 1 ? std::getenv("LLAMA_MOE_SHARED_EARLY") : nullptr;
        if (shared_mode && std::strcmp(shared_mode, "0") != 0 && std::strcmp(shared_mode, "1") != 0 && std::strcmp(shared_mode, "serial") != 0) {
            throw std::invalid_argument("LLAMA_MOE_SHARED_EARLY must be 0, 1 or serial");
        }
        if (shared_mode && std::strcmp(shared_mode, "0") != 0) {
            shared = std::strcmp(shared_mode, "1") == 0 ? SHARED_EARLY : SHARED_SERIAL;
            LLAMA_LOG_INFO("%s: MoE shared expert between the routing and the slot lookup, read back %s\n", __func__,
                shared == SHARED_EARLY ? "before it" : "after it");
        }
        if (aux_mode && std::strcmp(aux_mode, "0") != 0) {
            for (const group & g : groups) {
                if (!g.exl3_bits_x2 || g.ref.size() != 9) { continue; }
                device & d = devices[g.id];
                auto * reg = ggml_backend_dev_backend_reg(ggml_backend_get_device(d.backend));
                d.aux_upload = reg ? (ggml_exl3_aux_upload_t) ggml_backend_reg_get_proc_address(reg, "ggml_backend_exl3_aux_upload") : nullptr;
                d.aux_enabled = std::strcmp(aux_mode, "1") == 0;
            }
        }

        // one extra slot at the end, CUDA MMQ can read past the last expert
        auto alloc_size = [&](const group & g, int32_t n_slots) {
            const size_t alignment = ggml_backend_buft_get_alignment(devices[g.id].buft);
            size_t res = 0;
            for (const ggml_tensor * t : g.ref) {
                res += GGML_PAD(t->nb[2]*(n_slots + 1), alignment);
            }
            return res;
        };

        // the budget is split among the devices with host experts like the layers, by the tensor split or by default by free memory
        const float * tensor_split = model.tensor_split();
        const bool split_by_free = tensor_split == nullptr ||
            std::all_of(tensor_split, tensor_split + model.n_devices(), [](float x) { return x == 0.0f; });
        double split_sum = 0.0;
        for (device & d : devices) {
            if (d.host_bytes == 0) {
                continue;
            }
            ggml_backend_dev_t dev = ggml_backend_get_device(d.backend);
            if (split_by_free) {
                size_t free;
                size_t total;
                ggml_backend_dev_memory(dev, &free, &total);
                d.split = (double) free;
            } else {
                const auto it = std::find_if(model.devices.begin(), model.devices.end(), [&](const llama_device & ld) { return ld.dev == dev; });
                GGML_ASSERT(it != model.devices.end());
                d.split = (double) tensor_split[it - model.devices.begin()];
            }
            split_sum += d.split;
        }
        if (split_sum == 0.0) {
            // the devices do not report their free memory
            for (device & d : devices) {
                d.split    = d.host_bytes > 0 ? 1.0 : 0.0;
                split_sum += d.split;
            }
        }
        for (device & d : devices) {
            if (d.aux_upload && (double) size*d.split/split_sum > GGML_EXL3_AUX_SCRATCH_BYTES) {
                d.aux_bytes = GGML_EXL3_AUX_SCRATCH_BYTES;
            }
        }

        // within a device the budget is split by the size of the experts, so each group caches the same fraction of its experts
        std::vector<size_t> n_tensors(devices.size(), 0);
        size_t n_tensors_host = 0;
        for (group & g : groups) {
            const device & d = devices[g.id];
            const int32_t n_expert  = g.ref[0]->ne[2];
            const size_t  budget    = (size_t) (((double) size*d.split/split_sum - d.aux_bytes)*g.host_bytes/d.host_bytes);
            const int32_t max_slots = g.layers.size()*n_expert;
            while (g.n_slots < max_slots && alloc_size(g, g.n_slots + 1) <= budget) {
                g.n_slots++;
            }
            if (g.n_slots < n_expert_used) {
                LLAMA_LOG_WARN("%s: MoE cache budget is too small for %zu layers, they are not cached\n", __func__, g.layers.size());
                g.n_slots = 0;
                continue;
            }
            g.slru.init(model.layers.size(), n_expert, g.n_slots);
            n_tensors[g.id] += g.ref.size()*(1 + g.layers.size()) + (has_table(g) ? 1 : 0);
            n_tensors_host  += g.layers.size()*(shared ? 2 : 1);
        }
        if (n_tensors_host == 0) {
            throw std::runtime_error("MoE cache is too small to hold the experts of one token");
        }

        auto init_ctx = [](size_t n_tensors) {
            ggml_init_params params = {
                /*.mem_size   =*/ n_tensors*ggml_tensor_overhead(),
                /*.mem_buffer =*/ nullptr,
                /*.no_alloc   =*/ true,
            };
            ggml_context_ptr res(ggml_init(params));
            if (!res) {
                throw std::runtime_error("failed to create the MoE cache context");
            }
            return res;
        };
        for (size_t id = 0; id < devices.size(); ++id) {
            if (n_tensors[id] > 0) {
                auto & d = devices[id];
                d.ctx = init_ctx(n_tensors[id] + (d.aux_bytes ? 1 : 0));
                if (d.aux_bytes) {
                    d.aux_scratch = ggml_new_tensor_1d(d.ctx.get(), GGML_TYPE_I8, d.aux_bytes);
                    ggml_set_name(d.aux_scratch, "moe_cache.aux_scratch");
                    d.buf_size += d.aux_bytes;
                }
            }
        }
        ctx_host  = init_ctx(n_tensors_host);
        ctx_views = init_ctx(2);


        ggml_backend_buffer_type_t buft_host = ggml_backend_cpu_buffer_type();
        const size_t alignment_host = ggml_backend_buft_get_alignment(buft_host);

        for (size_t ig = 0; ig < groups.size(); ++ig) {
            group & g = groups[ig];
            if (g.n_slots == 0) {
                continue;
            }
            ggml_context * ctx = devices[g.id].ctx.get();
            for (const ggml_tensor * t : g.ref) {
                ggml_tensor * bank = ggml_new_tensor_3d(ctx, t->type, t->ne[0], t->ne[1], g.n_slots + 1);
                GGML_ASSERT(bank->nb[2] == t->nb[2]);
                ggml_format_name(bank, "moe_cache.%zu.%s", ig, t->name);
                g.banks.push_back(bank);
            }
            if (has_table(g)) {
                g.table_rows = (int32_t) std::min<int64_t>(16384, int64_t(g.layers.size())*g.slru.n_expert);
                g.table = ggml_new_tensor_2d(ctx, GGML_TYPE_I64, (int64_t) g.ref.size(), g.table_rows);
                ggml_format_name(g.table, "moe_cache.%zu.slot_table", ig);
            }
            for (int32_t il : g.layers) {
                layer & l = layers[il];
                l.ig      = (int32_t) ig;
                l.experts = llama_moe_cache_layer_experts(model, model.layers[il]);
                for (size_t ip = 0; ip < l.experts.size(); ++ip) {
                    ggml_tensor * bank   = g.banks[ip];
                    ggml_tensor * cached = ggml_view_3d(ctx, bank, bank->ne[0], bank->ne[1], g.n_slots, bank->nb[1], bank->nb[2], 0);
                    ggml_format_name(cached, "moe_cache.%s", l.experts[ip]->name);
                    bindings[l.experts[ip]] = { il, (int32_t) ip, cached };
                }
                l.slot_map = ggml_new_tensor_2d(ctx_host.get(), GGML_TYPE_I32, 1, g.ref[0]->ne[2]);
                ggml_format_name(l.slot_map, "moe_cache.slot_map-%d", il);
                layer_of[l.slot_map] = il;
                buf_host_size += GGML_PAD(ggml_nbytes(l.slot_map), alignment_host);
                if (shared) {
                    l.trigger = ggml_new_tensor_2d(ctx_host.get(), GGML_TYPE_I32, 1, g.ref[0]->ne[2]);
                    ggml_format_name(l.trigger, "moe_cache.route_trigger-%d", il);
                    trigger_of[l.trigger] = il;
                    buf_host_size += GGML_PAD(ggml_nbytes(l.trigger), alignment_host);
                }
            }
            devices[g.id].buf_size += alloc_size(g, g.n_slots);
        }

        if (model.hparams.no_alloc) {
            // only used to measure the memory use, see llama_context::memory_breakdown
            for (device & d : devices) {
                if (!d.ctx) {
                    continue;
                }
                d.buf.reset(ggml_backend_buft_alloc_buffer(d.buft, 0));
                for (ggml_tensor * t = ggml_get_first_tensor(d.ctx.get()); t != nullptr; t = ggml_get_next_tensor(d.ctx.get(), t)) {
                    t->buffer = d.buf.get();
                }
            }
            buf_host.reset(ggml_backend_buft_alloc_buffer(buft_host, 0));
            for (ggml_tensor * t = ggml_get_first_tensor(ctx_host.get()); t != nullptr; t = ggml_get_next_tensor(ctx_host.get(), t)) {
                t->buffer = buf_host.get();
            }
        } else {
            for (device & d : devices) {
                if (!d.ctx) {
                    continue;
                }
                d.buf.reset(ggml_backend_alloc_ctx_tensors_from_buft(d.ctx.get(), d.buft));
                if (!d.buf) {
                    throw std::runtime_error("failed to allocate the MoE cache buffers");
                }
                ggml_backend_buffer_clear(d.buf.get(), 0);
                d.buf_size = ggml_backend_buffer_get_size(d.buf.get());
            }
            buf_host.reset(ggml_backend_alloc_ctx_tensors_from_buft(ctx_host.get(), buft_host));
            if (!buf_host) {
                throw std::runtime_error("failed to allocate the MoE cache buffers");
            }
            ggml_backend_buffer_clear(buf_host.get(), 0xff); // all slots are -1
            buf_host_size = ggml_backend_buffer_get_size(buf_host.get());

            for (group & g : groups) {
                for (int32_t il : g.layers) {
                    if (layers[il].slot_map != nullptr) {
                        g.slru.slot_map[il] = (int32_t *) layers[il].slot_map->data;
                    }
                }
                build_table(g);
            }
        }

        // as weights, the ops that read the banks run on the device and the slot maps are copied with the copy callback
        for (device & d : devices) {
            if (d.buf) {
                ggml_backend_buffer_set_usage(d.buf.get(), GGML_BACKEND_BUFFER_USAGE_WEIGHTS);
            }
            if (d.aux_scratch) {
                auto * host_buft = ggml_backend_dev_host_buffer_type(ggml_backend_get_device(d.backend));
                d.aux_host_buft = host_buft;
                if (host_buft && !model.hparams.no_alloc) {
                    d.aux_host.reset(ggml_backend_buft_alloc_buffer(host_buft, d.aux_bytes));
                }
                if (!model.hparams.no_alloc && (!d.aux_host || ggml_backend_buffer_get_type(d.aux_host.get()) != host_buft)) {
                    d.aux_enabled = false;
                    d.aux_host.reset();
                    d.aux_host_buft = nullptr;
                    LLAMA_LOG_WARN("%s: pinned auxiliary staging unavailable; using ordinary fills\n", __func__);
                }
                LLAMA_LOG_INFO("%s: MoE auxiliary batching %s, %.2f MiB device scratch inside cache budget\n", __func__,
                    d.aux_enabled ? "enabled" : "reserved only", d.aux_bytes/1024.0/1024.0);
            }
            if (shared == SHARED_EARLY && d.buf && !model.hparams.no_alloc) {
                auto * host_buft = ggml_backend_dev_host_buffer_type(ggml_backend_get_device(d.backend));
                if (host_buft) {
                    d.ids_host.reset(ggml_backend_buft_alloc_buffer(host_buft, max_batch*n_expert_used*sizeof(int32_t)));
                }
                if (d.ids_host && ggml_backend_buffer_get_type(d.ids_host.get()) == host_buft) {
                    d.ids_ready = ggml_backend_event_new(ggml_backend_get_device(d.backend));
                }
                if (!d.ids_ready) {
                    d.ids_host.reset();
                    LLAMA_LOG_WARN("%s: pinned readback or events unavailable; the shared expert runs after the readback\n", __func__);
                }
            }
        }
        ggml_backend_buffer_set_usage(buf_host.get(), GGML_BACKEND_BUFFER_USAGE_WEIGHTS);

        for (size_t id = 0; id < devices.size(); ++id) {
            const device & d = devices[id];
            if (d.host_bytes == 0) {
                continue;
            }
            LLAMA_LOG_INFO("%s: %10s MoE cache size = %8.2f MiB for %.2f MiB of host experts\n", __func__,
                ggml_backend_buft_name(d.buft), d.buf_size/1024.0/1024.0, d.host_bytes/1024.0/1024.0);
            for (const group & g : groups) {
                if (g.id == (int32_t) id) {
                    if (g.exl3_bits_x2) {
                        LLAMA_LOG_INFO("%s: %2zu layers, EXL3 %.1f bpw + F16 auxiliaries: %5d slots (%.1f%%)\n", __func__,
                            g.layers.size(), 0.5*g.exl3_bits_x2, g.n_slots, 100.0*g.n_slots/(g.layers.size()*g.ref[0]->ne[2]));
                    } else {
                        LLAMA_LOG_INFO("%s: %2zu layers, %s: %5d slots (%.1f%%)\n", __func__,
                            g.layers.size(), ggml_type_name(g.ref.back()->type), g.n_slots, 100.0*g.n_slots/(g.layers.size()*g.ref[0]->ne[2]));
                    }
                }
            }
        }
        pin_host_experts(model, pin);
        if (!mtp) {
            open_trace();
        }
    }

    // header: "MOERT001", u32 n_layer, u32 n_expert, u32 n_groups, per group u32 slots, u32 bytes per expert, u32 n_layers, u32 layers[]
    // records: u8 type, then 1 = route: u8 0, u16 layer, u32 ms, u16 n_ids, u16 ids[n_ids] (n_expert_used per row, rows in order)
    //                        2 = staging attached: u8 0, u16 group, u32 base slots, u32 total slots
    //                        3 = lending level: u8 level, u16 0, u32 ms
    //                        4 = compute region: u8 level, u16 group, u32 first slot, u32 count
    //                        5 = compute detached and cache reset: u8 0, u16 0, u32 ms
    void open_trace() {
        const char * path = std::getenv("LLAMA_MOE_ROUTE_TRACE");
        if (!path || !*path || groups.empty()) {
            return;
        }
        trace = std::fopen(path, "wb");
        if (!trace) {
            throw std::runtime_error(std::string("LLAMA_MOE_ROUTE_TRACE: cannot open ") + path + ": " + std::strerror(errno));
        }
        std::setvbuf(trace, nullptr, _IOFBF, 1 << 20);
        trace_t0 = ggml_time_us();
        auto put = [&](uint32_t v) { std::fwrite(&v, sizeof(v), 1, trace); };
        std::fwrite("MOERT001", 1, 8, trace);
        put(layers.size());
        put(groups[0].slru.n_expert);
        put(groups.size());
        for (const group & g : groups) {
            size_t expert_bytes = 0;
            for (const auto * w : layers[g.layers[0]].experts) { expert_bytes += w->nb[2]; }
            put(g.n_slots);
            put(expert_bytes);
            put(g.layers.size());
            for (int32_t il : g.layers) { put(il); }
        }
        const std::string lease_path = std::string(path) + ".leases";
        lease_trace = std::fopen(lease_path.c_str(), "wb");
        if (!lease_trace) {
            std::fclose(trace);
            trace = nullptr;
            throw std::runtime_error("cannot open MoE ownership trace " + lease_path);
        }
        std::setvbuf(lease_trace, nullptr, _IOFBF, 1 << 20);
        std::fprintf(lease_trace, "MOELIF001\n");
        lending.observe = [this](const char * op, uint64_t region, const llama_memory_lending::span & span) {
            std::fprintf(lease_trace, "seq=%" PRIu64 " route=%" PRIu64 " op=%s owner=%" PRIu64 " generation=%" PRIu64 " kind=%d region=%" PRIu64 " offset=%zu bytes=%zu\n",
                ++trace_sequence, trace_routes, op, span.allocation.id, span.allocation.generation,
                int(lending.owner_kind(span.allocation)), region, span.offset, span.bytes);
        };
        LLAMA_LOG_INFO("%s: recording the selected experts of cached layers to %s\n", __func__, path);
    }

    void trace_record(uint8_t type, uint8_t flag, uint16_t field, uint32_t value) {
        std::fwrite(&type, 1, 1, trace);
        std::fwrite(&flag, 1, 1, trace);
        std::fwrite(&field, 2, 1, trace);
        std::fwrite(&value, 4, 1, trace);
    }

    uint32_t trace_ms() const { return (uint32_t) ((ggml_time_us() - trace_t0)/1000); }

    // the bank holding slot s of expert tensor ip, the slot's index in it and the slots from s to the end of that bank
    ggml_tensor * locate(const group & g, int32_t s, size_t ip, int32_t & local, int32_t & room) const {
        const int32_t base = g.base_slots ? g.base_slots : g.n_slots;
        if (s < base) {
            local = s;
            room = base - s;
            return g.banks[ip];
        }
        for (const auto & r : g.regions) {
            if (s >= r.first && s < r.first + r.count) {
                local = s - r.first;
                room = r.first + r.count - s;
                return r.banks[ip];
            }
        }
        GGML_ABORT("MoE cache slot %d is outside the banks", s);
    }

    // the address of every slot's storage of each expert tensor: base slots in the banks, then the borrowed regions
    void build_table(group & g) {
        if (!g.table || !g.n_slots) {
            return;
        }
        GGML_ASSERT(g.n_slots <= g.table_rows);
        const size_t cols = g.ref.size();
        g.table_host.assign((size_t) g.table_rows*cols, 0);
        for (int32_t s = 0; s < g.n_slots; ++s) {
            for (size_t c = 0; c < cols; ++c) {
                int32_t local, room;
                const ggml_tensor * bank = locate(g, s, c, local, room);
                g.table_host[(size_t) s*cols + c] = (int64_t) (intptr_t) ((const char *) bank->data + (size_t) local*bank->nb[2]);
            }
        }
        ggml_backend_tensor_set(g.table, g.table_host.data(), 0, (size_t) g.n_slots*cols*sizeof(int64_t));
        ctx_tables.emplace_back(ggml_init({(cols/3 + 1)*ggml_tensor_overhead(), nullptr, true}));
        if (!ctx_tables.back()) { throw std::runtime_error("failed to create the MoE cache table context"); }
        ggml_context * ctx_table = ctx_tables.back().get();
        g.table_views.clear();
        for (size_t t = 0; t < cols/3; ++t) {
            ggml_tensor * view = ggml_view_2d(ctx_table, g.table, 3, g.n_slots, g.table->nb[1], t*3*sizeof(int64_t));
            ggml_format_name(view, "%s.%zu", g.table->name, t);
            ggml_backend_view_init(view);
            g.table_views.push_back(view);
        }
        ggml_tensor * all = ggml_view_2d(ctx_table, g.table, (int64_t) cols, g.n_slots, g.table->nb[1], 0);
        ggml_format_name(all, "%s.aux", g.table->name);
        ggml_backend_view_init(all);
        g.table_views.push_back(all);
    }

    // pinned (registered) host experts: their uploads are DMA from host memory, and their pages cannot be evicted
    void pin_host_experts(const llama_model & model, size_t pin) {
        if (pin == 0) {
            return;
        }
        if (devices.size() != 1) {
            throw std::runtime_error("pinning the host experts requires one GPU backend");
        }
        std::vector<ggml_tensor *> tensors;
        size_t total = 0;
        for (const layer & l : layers) {
            for (ggml_tensor * t : l.experts) {
                tensors.push_back(t);
                total += ggml_nbytes(t);
            }
        }
        const size_t pinned = model.register_host_weights(devices[0].backend, tensors, pin);
        LLAMA_LOG_INFO("%s: pinned %.2f of %.2f MiB of host experts\n", __func__, pinned/1024.0/1024.0, total/1024.0/1024.0);
        if (pin == SIZE_MAX && pinned < total) {
            LLAMA_LOG_WARN("%s: %.2f MiB of host experts could not be pinned\n", __func__, (total - pinned)/1024.0/1024.0);
        }
    }

    ggml_backend_t backend(int32_t il) const {
        return devices[groups[layers[il].ig].id].backend;
    }

    ~impl() {
        drop_compute();
        if (staging_owner.id) { lending.retire(staging_owner, return_hooks()); }
        if (lease_trace) { std::fclose(lease_trace); }
        if (trace) { std::fclose(trace); }
        log_stats();
        for (auto & d : devices) {
            if (d.aux_host || d.ids_host) { ggml_backend_synchronize(d.backend); }
            if (d.ids_ready) { ggml_backend_event_free(d.ids_ready); }
        }
    }

    void trace_cache(const char * op, size_t ig, int32_t first, int32_t count, uint64_t region = 0) {
        if (lease_trace) {
            std::fprintf(lease_trace, "seq=%" PRIu64 " route=%" PRIu64 " op=%s group=%zu first=%d count=%d region=%" PRIu64 "\n",
                ++trace_sequence, trace_routes, op, ig, first, count, region);
        }
    }

    uint64_t register_region(size_t ig, const group::region & r, llama_memory_lending::owner owner, ggml_backend_buffer_t buffer) {
        std::vector<llama_memory_lending::span> spans;
        const uintptr_t base = (uintptr_t) ggml_backend_buffer_get_base(buffer);
        for (const auto * bank : r.banks) {
            const uintptr_t address = (uintptr_t) bank->data;
            if (address < base) { throw std::runtime_error("invalid MoE lending bank"); }
            spans.push_back({owner, address - base, ggml_nbytes(bank)});
        }
        const uint64_t id = lending.add(spans);
        trace_cache("slots", ig, r.first, r.count, id);
        return id;
    }

    llama_memory_lending::hooks return_hooks(bool drained = false) {
        return {
            [this, drained] {
                if (!drained) { ggml_backend_synchronize(devices[0].backend); }
                pending_il = -1;
                pending_sel = nullptr;
            },
            [this](const std::vector<uint64_t> & ids) {
                for (auto & g : groups) {
                    for (const auto & r : g.regions) {
                        if (!std::binary_search(ids.begin(), ids.end(), r.lease)) { continue; }
                        for (int32_t s = r.first; s < r.first + r.count; ++s) {
                            const int32_t key = g.slru.key_of[s];
                            if (key >= 0) { layers[key/g.slru.n_expert].perf.values[llama_moe_perf_data::extra_invalidated]++; }
                        }
                        g.slru.set_range(r.first, r.count, false, false);
                    }
                    g.slru.rebalance();
                }
            },
            {} // Compute and expert staging overwrite their own bytes; they do not require KV zero-on-return.
        };
    }

    void attach_staging(ggml_backend_t backend, ggml_backend_buffer_t buffer) {
        if (staging == STAGING_OFF) { return; }
        if (ctx_extra || devices.size() != 1 || devices[0].backend != backend || !buffer ||
                ggml_backend_buffer_get_type(buffer) != devices[0].buft || !devices[0].aux_enabled) {
            throw std::runtime_error("MoE staging cache requires the target's staging buffer and pinned auxiliary batching");
        }
        const size_t align = ggml_backend_buft_get_alignment(devices[0].buft);
        const size_t available = ggml_backend_buffer_get_size(buffer);
        const size_t host_bytes = devices[0].host_bytes;
        if (!align || !host_bytes || !ggml_backend_buffer_get_base(buffer)) {
            throw std::runtime_error("MoE staging cache has no usable backing buffer");
        }
        auto allocation = [&](const group & g, int32_t slots) {
            size_t total = 0;
            for (auto * t : g.ref) {
                if (!t->nb[2] || size_t(slots) + 1 > (SIZE_MAX - (align - 1))/t->nb[2]) {
                    throw std::runtime_error("MoE staging cache bank size overflow");
                }
                const size_t bytes = GGML_PAD((size_t(slots) + 1)*t->nb[2], align);
                if (bytes > SIZE_MAX - total) { throw std::runtime_error("MoE staging cache size overflow"); }
                total += bytes;
            }
            return total;
        };
        std::vector<int32_t> counts(groups.size(), 0);
        for (size_t i = 0; i < groups.size(); ++i) {
            const group & g = groups[i];
            if (g.id != 0 || !g.n_slots || !g.exl3_bits_x2 || g.ref.size() != 9 ||
                    layers.size() > size_t(INT32_MAX)/g.slru.n_expert) {
                throw std::runtime_error("MoE staging cache requires supported EXL3 base banks");
            }
            const size_t budget = (size_t) ((long double) available*g.host_bytes/host_bytes);
            const int32_t limit = std::min<int64_t>(16384 - g.n_slots, int64_t(g.layers.size())*g.slru.n_expert - g.n_slots);
            while (counts[i] < limit && allocation(g, counts[i] + 1) <= budget) { counts[i]++; }
            if (!counts[i]) { throw std::runtime_error("MoE staging cache has no room for an extension bank"); }
        }
        ctx_extra.reset(ggml_init({groups.size()*18*ggml_tensor_overhead(), nullptr, true}));
        if (!ctx_extra) { throw std::runtime_error("failed to create the MoE staging cache context"); }
        staging_owner = lending.bind(1, llama_memory_lending::kind::staging, devices[0].backend,
            (uintptr_t) ggml_backend_buffer_get_base(buffer), available, align);
        char * base = (char *) ggml_backend_buffer_get_base(buffer);
        for (size_t i = 0; i < groups.size(); ++i) {
            group & g = groups[i];
            const int32_t count = counts[i];
            g.base_slots = g.n_slots;
            group::region r;
            r.first = g.n_slots;
            r.count = count;
            r.level = 1;
            for (auto * t : g.ref) {
                auto * bank = ggml_new_tensor_3d(ctx_extra.get(), t->type, t->ne[0], t->ne[1], count + 1);
                const size_t size = GGML_PAD(ggml_backend_buft_get_alloc_size(devices[0].buft, bank), align);
                if (staging_used > available || size > available - staging_used || bank->nb[2] != t->nb[2]) {
                    throw std::runtime_error("MoE staging cache bank exceeds its backing buffer");
                }
                ggml_format_name(bank, "moe_cache.extra.%zu.%s", i, t->name);
                if (ggml_backend_tensor_alloc(buffer, bank, base + staging_used) != GGML_STATUS_SUCCESS) {
                    throw std::runtime_error("failed to bind a MoE staging cache bank");
                }
                staging_used += size;
                r.banks.push_back(bank);
            }
            r.lease = register_region(i, r, staging_owner, buffer);
            g.regions.push_back(std::move(r));
            g.n_slots += count;
            build_table(g);
            g.slru.init(layers.size(), g.slru.n_expert, g.n_slots);
            for (int32_t il : g.layers) { g.slru.slot_map[il] = (int32_t *) layers[il].slot_map->data; }
            g.slru.set_active(g.base_slots);
            LLAMA_LOG_INFO("moe_staging: group=%zu base_slots=%d extra_slots=%d total_slots=%d bits_x2=%d\n",
                i, g.base_slots, count, g.n_slots, g.exl3_bits_x2);
            if (trace) {
                trace_record(2, 0, (uint16_t) i, g.base_slots);
                std::fwrite(&g.n_slots, 4, 1, trace);
            }
        }
        ggml_backend_buffer_set_usage(buffer, GGML_BACKEND_BUFFER_USAGE_WEIGHTS);
        LLAMA_LOG_INFO("moe_staging: borrowed_bytes=%zu buffer_bytes=%zu device_payload_added=0 mode=%s\n",
            staging_used, available, staging == STAGING_ON ? "enabled" : "reserve");
    }

    // lend the borrowed regions up to a level (0: base slots only); lowering it invalidates the experts in the regions
    // it gives back, raising it clears the padding slots of the regions it adds
    void set_level(int level) {
        if (!ctx_extra) { return; }
        level = std::max(0, std::min(level, max_level()));
        if (level == lend_level) { return; }
        const int64_t start = ggml_time_us();
        ggml_backend_synchronize(devices[0].backend);
        pending_il = -1;
        pending_sel = nullptr;
        size_t invalidated = 0;
        std::vector<uint64_t> returned;
        for (auto & g : groups) {
            for (const auto & r : g.regions) {
                if (r.level > level && lending.get(r.lease).leased) {
                    for (const auto & span : lending.get(r.lease).spans) { lending.withdraw(span); }
                    returned.push_back(r.lease);
                    for (int32_t slot = r.first; slot < r.first + r.count; ++slot) { invalidated += g.slru.key_of[slot] >= 0; }
                }
            }
        }
        lending.release(returned, return_hooks(true));
        if (staging == STAGING_ON) {
            for (auto & g : groups) {
                for (const auto & r : g.regions) {
                    if (r.level > level || lending.get(r.lease).leased) { continue; }
                    for (const auto & span : lending.get(r.lease).spans) { lending.publish(span); }
                    lending.acquire(r.lease, [&r] {
                        for (auto * bank : r.banks) {
                            ggml_backend_tensor_memset(bank, 0, (bank->ne[2] - 1)*bank->nb[2], bank->nb[2]);
                        }
                    });
                    g.slru.set_range(r.first, r.count, true, true);
                }
                g.slru.rebalance();
            }
        }
        lend_level = level;
        staging_active = level > 0;
        if (trace) {
            trace_record(3, (uint8_t) level, 0, trace_ms());
            std::fflush(trace);
        }
        LLAMA_LOG_INFO("moe_staging: active=%d level=%d invalidated=%zu transition_us=%" PRId64 "\n", level > 0 ? 1 : 0, level, invalidated,
            ggml_time_us() - start);
    }

    int max_level() const {
        int level = 0;
        for (const auto & g : groups) {
            for (const auto & r : g.regions) { level = std::max(level, r.level); }
        }
        return level;
    }

    void set_staging_active(bool active) {
        set_level(active ? max_level() : 0);
    }

    // bytes of n slots of the group's banks, each bank padded and with its zero padding slot
    size_t bank_bytes(const group & g, int32_t n, size_t align) const {
        size_t total = 0;
        for (auto * t : g.ref) {
            if (!t->nb[2] || size_t(n) + 1 > (SIZE_MAX - (align - 1))/t->nb[2]) {
                throw std::runtime_error("MoE cache bank size overflow");
            }
            const size_t bytes = GGML_PAD((size_t(n) + 1)*t->nb[2], align);
            if (bytes > SIZE_MAX - total) { throw std::runtime_error("MoE cache size overflow"); }
            total += bytes;
        }
        return total;
    }

    // lend the scheduler's compute buffer above its decode use: [b_start, size) at level 1, [a_start, b_start) at level 2.
    // Requires the staging extension (split banks, slot tables, pinned auxiliary batching). Clears the cache.
    struct compute_lend {
        ggml_backend_buffer_t buffer = nullptr;
        void * base = nullptr;
        size_t size = 0, a_start = 0, b_start = 0;
    } lent;

    // give back the compute buffer: drop its regions and every cached expert (the slots of the others are renumbered)
    void drop_compute() {
        if (!lent.buffer) { return; }
        set_level(0);
        lending.retire(compute_owner, return_hooks());
        compute_owner = {};
        for (size_t i = 0; i < groups.size(); ++i) {
            group & g = groups[i];
            while (!g.regions.empty() && g.regions.back().compute) {
                g.n_slots -= g.regions.back().count;
                g.regions.pop_back();
            }
            reset_group(g);
        }
        ctx_compute.reset();
        lent = {};
        if (trace) { trace_record(5, 0, 0, trace_ms()); }
    }

    void reset_group(group & g) {
        build_table(g);
        g.slru.init(layers.size(), g.slru.n_expert, g.n_slots);
        for (int32_t il : g.layers) {
            int32_t * map = (int32_t *) layers[il].slot_map->data;
            std::fill(map, map + g.slru.n_expert, -1);
            g.slru.slot_map[il] = map;
        }
        g.slru.set_active(g.base_slots ? g.base_slots : g.n_slots);
        trace_cache("reset", &g - groups.data(), g.base_slots ? g.base_slots : g.n_slots, g.n_slots);
    }

    void attach_compute(ggml_backend_buffer_t buffer, size_t a_start, size_t b_start) {
        set_level(0);
        // the regions of an earlier buffer go first: it may be freed already
        drop_compute();
        if (!ctx_extra || staging != STAGING_ON || devices.size() != 1 || !buffer ||
                ggml_backend_buffer_get_type(buffer) != devices[0].buft) {
            return;
        }
        const size_t align = ggml_backend_buft_get_alignment(devices[0].buft);
        const size_t size = ggml_backend_buffer_get_size(buffer);
        if (!align || !ggml_backend_buffer_get_base(buffer) || a_start > SIZE_MAX - (align - 1) ||
                b_start > SIZE_MAX - (align - 1)) {
            throw std::invalid_argument("invalid MoE compute lending range");
        }
        a_start = std::min(GGML_PAD(a_start, align), size);
        b_start = std::min(GGML_PAD(std::max(a_start, b_start), align), size);
        compute_owner = lending.bind(2, llama_memory_lending::kind::compute, devices[0].backend,
            (uintptr_t) ggml_backend_buffer_get_base(buffer), size, align);
        lent = {buffer, ggml_backend_buffer_get_base(buffer), size, a_start, b_start};
        const size_t host_bytes = devices[0].host_bytes;
        ctx_compute.reset(ggml_init({groups.size()*2*g_ref_max()*ggml_tensor_overhead(), nullptr, true}));
        if (!ctx_compute) { throw std::runtime_error("failed to create the MoE compute lending context"); }
        char * base = (char *) lent.base;
        size_t lent_bytes[3] = {0, 0, 0};
        for (const auto & [level, begin, end] : { std::tuple<int, size_t, size_t>{1, b_start, size}, std::tuple<int, size_t, size_t>{2, a_start, b_start} }) {
            if (begin >= end || !host_bytes) { continue; }
            size_t used = begin;
            for (size_t i = 0; i < groups.size(); ++i) {
                group & g = groups[i];
                if (!g.base_slots) { continue; }
                const size_t budget = (size_t) ((long double) (end - begin)*g.host_bytes/host_bytes);
                const int32_t limit = std::min<int64_t>(g.table_rows - g.n_slots, int64_t(g.layers.size())*g.slru.n_expert - g.n_slots);
                int32_t count = 0;
                while (count < limit && bank_bytes(g, count + 1, align) <= budget) { count++; }
                if (!count || used > end || bank_bytes(g, count, align) > end - used) { continue; }
                group::region r;
                r.first = g.n_slots;
                r.count = count;
                r.level = level;
                r.compute = true;
                for (auto * t : g.ref) {
                    auto * bank = ggml_new_tensor_3d(ctx_compute.get(), t->type, t->ne[0], t->ne[1], count + 1);
                    GGML_ASSERT(bank->nb[2] == t->nb[2]);
                    ggml_format_name(bank, "moe_cache.lend%d.%zu.%s", level, i, t->name);
                    if (ggml_backend_tensor_alloc(buffer, bank, base + used) != GGML_STATUS_SUCCESS) {
                        throw std::runtime_error("failed to bind a MoE cache bank in the compute buffer");
                    }
                    used += GGML_PAD(ggml_nbytes(bank), align);
                    r.banks.push_back(bank);
                }
                lent_bytes[level] += bank_bytes(g, count, align);
                r.lease = register_region(i, r, compute_owner, buffer);
                g.regions.push_back(std::move(r));
                g.n_slots += count;
            }
        }
        for (size_t i = 0; i < groups.size(); ++i) {
            group & g = groups[i];
            if (!g.base_slots) { continue; }
            reset_group(g);
            for (const auto & r : g.regions) {
                if (r.compute) {
                    LLAMA_LOG_INFO("moe_lend: group=%zu level=%d first=%d slots=%d\n", i, r.level, r.first, r.count);
                    if (trace) {
                        trace_record(4, (uint8_t) r.level, (uint16_t) i, r.first);
                        std::fwrite(&r.count, 4, 1, trace);
                    }
                }
            }
        }
        lend_level = 0;
        staging_active = false;
        LLAMA_LOG_INFO("moe_lend: compute buffer %.2f MiB, level 1 [%.2f, %.2f) MiB lends %.2f MiB, level 2 [%.2f, %.2f) MiB lends %.2f MiB\n",
            size/1048576.0, b_start/1048576.0, size/1048576.0, lent_bytes[1]/1048576.0, a_start/1048576.0, b_start/1048576.0,
            lent_bytes[2]/1048576.0);
    }

    size_t g_ref_max() const {
        size_t n = 0;
        for (const auto & g : groups) { n = std::max(n, g.ref.size()); }
        return n;
    }

    // before a graph runs: give back what its compute buffer use reaches, and follow a reallocated compute buffer
    void fit_compute(ggml_backend_buffer_t buffer, size_t peak) {
        if (!lent.buffer) { return; }
        if (!buffer || buffer != lent.buffer || ggml_backend_buffer_get_base(buffer) != lent.base ||
                ggml_backend_buffer_get_size(buffer) != lent.size) {
            // the graph about to run was built for this buffer: give back the old one now (the MoE cache restarts empty),
            // the next scheduler reserve lends the new one
            LLAMA_LOG_WARN("moe_lend: the compute buffer changed under a graph, no longer lending it until the next reserve\n");
            drop_compute();
            return;
        }
        if (peak > lent.b_start) {
            set_level(0);
        } else if (peak > lent.a_start) {
            set_level(std::min(lend_level, 1));
        }
    }

    ggml_tensor * get_slot_map(int32_t il, int64_t n_tokens, int64_t n_expert_used) const {
        if (il < 0 || il >= (int32_t) layers.size() || layers[il].ig < 0) {
            return nullptr;
        }
        const layer & l = layers[il];

        // large batches use most experts of a layer, so they gain little from the cache and would evict the experts used in generation
        if (n_tokens == 0 || n_tokens > groups[l.ig].max_tokens || std::min(n_tokens*n_expert_used, l.slot_map->ne[1]) > (groups[l.ig].base_slots ? groups[l.ig].base_slots : groups[l.ig].n_slots)) {
            return nullptr;
        }
        return l.slot_map;
    }

    ggml_tensor * get_route_trigger(int32_t il, int64_t n_tokens, int64_t n_expert_used) const {
        if (!shared || get_slot_map(il, n_tokens, n_expert_used) == nullptr) {
            return nullptr;
        }
        return layers[il].trigger;
    }

    // the route trigger of layer il is copied after the split that selects its experts: read them back before the
    // following split (the shared expert) is queued, so the slot map copy waits only for the routing
    bool read_routes(ggml_backend_t backend, int32_t il, ggml_tensor * dst, ggml_cgraph * graph) {
        const layer & l = layers[il];
        device & d = devices[groups[l.ig].id];
        GGML_ASSERT(backend == d.backend);

        const ggml_tensor * route = nullptr;
        for (int i = 0; i < ggml_graph_n_nodes(graph) && route == nullptr; ++i) {
            const ggml_tensor * node = ggml_graph_node(graph, i);
            if (node->op == GGML_OP_GET_ROWS && node->src[0] == dst) {
                route = node;
            }
        }
        GGML_ASSERT(route != nullptr);
        const ggml_tensor * sel = route->src[1];
        for (int i = 0; i < ggml_graph_n_nodes(graph); ++i) {
            const ggml_tensor * node = ggml_graph_node(graph, i);
            if (node == sel || node == sel->view_src) {
                GGML_ABORT("the experts of layer %d are selected in the same split as their route trigger", il);
            }
        }
        GGML_ASSERT(ggml_is_contiguous(sel));

        pending_il  = -1;
        pending_sel = nullptr;
        if (shared != SHARED_EARLY || !d.ids_ready || ggml_nbytes(sel) > ggml_backend_buffer_get_size(d.ids_host.get())) {
            return true; // the slot map copy reads them back after the shared expert
        }
        const int64_t t0 = llama_moe_perf_enabled() ? ggml_time_us() : 0;
        ggml_backend_tensor_get_async(backend, sel, ggml_backend_buffer_get_base(d.ids_host.get()), 0, ggml_nbytes(sel));
        ggml_backend_event_record(d.ids_ready, backend);
        if (t0) {
            layers[il].perf.values[llama_moe_perf_data::submit_us] += ggml_time_us() - t0;
        }
        pending_il  = il;
        pending_sel = sel;
        return true;
    }

    ggml_tensor * get_experts(const ggml_tensor * w) const {
        const auto it = bindings.find(w);
        return it != bindings.end() ? it->second.cached : nullptr;
    }

    int64_t copy_experts(ggml_backend_t backend, const ggml_tensor * w, ggml_tensor * dst, int64_t e, int64_t last) {
        const auto it = bindings.find(w);
        if (it == bindings.end()) {
            return 0;
        }
        const binding & b = it->second;
        const group & g = groups[layers[b.il].ig];
        if (backend != devices[g.id].backend) {
            return 0;
        }

        // large batches only read the cache, so the experts used in generation stay in it
        const int32_t * slots = g.slru.slot_map[b.il];
        if (slots == nullptr || slots[e] < 0) {
            return 0;
        }
        int64_t n = 1;
        while (e + n <= last && slots[e + n] == slots[e] + n) {
            n++;
        }

        int32_t local, room;
        ggml_tensor * bank = locate(g, slots[e], b.ip, local, room);
        n = std::min<int64_t>(n, room);
        ggml_reset(ctx_views.get());
        ggml_tensor * src_view = ggml_view_3d(ctx_views.get(), bank, bank->ne[0], bank->ne[1], n, bank->nb[1], bank->nb[2], local*bank->nb[2]);
        ggml_tensor * dst_view = ggml_view_3d(ctx_views.get(), dst,  dst->ne[0],  dst->ne[1],  n, dst->nb[1],  dst->nb[2],  e*dst->nb[2]);
        ggml_backend_view_init(src_view);
        ggml_backend_view_init(dst_view);
        const int64_t t_submit = llama_moe_perf_enabled() ? ggml_time_us() : 0;
        ggml_backend_tensor_copy_async(backend, backend, src_view, dst_view);
        if (llama_moe_perf_enabled()) {
            auto & perf = layers[b.il].perf.values;
            perf[llama_moe_perf_data::d2d_bytes] += ggml_nbytes(src_view);
            perf[llama_moe_perf_data::d2d_calls]++;
            perf[llama_moe_perf_data::submit_us] += ggml_time_us() - t_submit;
        }

        stats_copy.hits  += n;
        stats_copy.bytes += ggml_nbytes(src_view);

        return n;
    }

    bool pack_aux(const layer & l, const group & g, ggml_exl3_aux_batch & batch, size_t & bytes) {
        const device & d = devices[g.id];
        if (!d.aux_enabled || !d.aux_host || !g.exl3_bits_x2 || l.experts.size() != 9 || g.banks.size() != 9 ||
                fills.empty() || fills.size() > GGML_EXL3_AUX_MAX_FILLS) {
            return false;
        }
        static constexpr int indices[] = {1, 2, 4, 5, 7, 8};
        bytes = 0;
        for (int i = 0; i < GGML_EXL3_AUX_BANKS; ++i) {
            const auto * w = l.experts[indices[i]];
            auto * bank = g.banks[indices[i]];
            if (w->type != GGML_TYPE_F16 || bank->type != GGML_TYPE_F16 || !ggml_is_contiguous(w) ||
                    w->nb[2] != bank->nb[2] || !w->nb[2] || w->nb[2]%16 || w->nb[2] > (d.aux_bytes - bytes)/fills.size()) {
                return false;
            }
            for (const auto & f : fills) {
                if (f.expert < 0 || f.expert >= w->ne[2] || f.slot < 0 || f.slot >= g.n_slots) { return false; }
            }
            bytes += w->nb[2]*fills.size();
            batch.banks[i] = bank;
            batch.table_cols[i] = indices[i];
        }
        batch.count = (int32_t) fills.size();
        batch.n_slots = g.n_slots;
        batch.slot_table = g.table_views.empty() ? nullptr : g.table_views.back();
        for (size_t i = 0; i < fills.size(); ++i) { batch.slots[i] = fills[i].slot; }
        auto * out = (uint8_t *) ggml_backend_buffer_get_base(d.aux_host.get());
        if (!out) { return false; }
        for (int ip : indices) {
            const auto * w = l.experts[ip];
            for (const auto & f : fills) {
                std::memcpy(out, (const uint8_t *) w->data + size_t(f.expert)*w->nb[2], w->nb[2]);
                out += w->nb[2];
            }
        }
        return true;
    }

    bool copy(ggml_backend_t backend, const ggml_tensor * src, ggml_tensor * dst, ggml_cgraph * graph) {
        if (const auto it_trigger = trigger_of.find(src); it_trigger != trigger_of.end()) {
            return read_routes(backend, it_trigger->second, dst, graph);
        }
        const auto it = layer_of.find(src);
        if (it == layer_of.end()) {
            return false;
        }
        const int32_t il = it->second;
        layer & l = layers[il];
        group & g = groups[l.ig];

        GGML_ASSERT(backend == devices[g.id].backend);

        // the get_rows that looks up the slots of the selected experts
        const int n_nodes = ggml_graph_n_nodes(graph);
        const ggml_tensor * lookup = nullptr;
        for (int i = 0; i < n_nodes && lookup == nullptr; ++i) {
            const ggml_tensor * node = ggml_graph_node(graph, i);
            if (node->op == GGML_OP_GET_ROWS && node->src[0] == dst) {
                lookup = node;
            }
        }
        GGML_ASSERT(lookup != nullptr);

        // the selected experts must be computed in an earlier split
        // the host weight at the lookup starts a new split after the selected experts
        const ggml_tensor * sel = lookup->src[1];
        for (int i = 0; i < n_nodes; ++i) {
            const ggml_tensor * node = ggml_graph_node(graph, i);
            if (node == sel || node == sel->view_src) {
                GGML_ABORT("the experts of layer %d are selected in the same split as their MoE cache lookup", il);
            }
        }
        GGML_ASSERT(ggml_is_contiguous(sel));

        ids.resize(ggml_nelements(sel));
        const bool observe = llama_moe_perf_enabled();
        const int64_t t_ids = observe ? ggml_time_us() : 0;
        // a pending readback was queued after the routing by the route trigger: the scheduler did not wait for the
        // backend, and waiting for the readback also waits for all earlier uploads and uses of the staging buffers
        if (pending_il == il && pending_sel == sel) {
            const device & d = devices[g.id];
            ggml_backend_event_synchronize(d.ids_ready);
            std::memcpy(ids.data(), ggml_backend_buffer_get_base(d.ids_host.get()), ggml_nbytes(sel));
        } else {
            ggml_backend_tensor_get_async(backend, sel, ids.data(), 0, ggml_nbytes(sel));
            ggml_backend_synchronize(backend);
        }
        pending_il  = -1;
        pending_sel = nullptr;
        const int64_t t_plan = observe ? ggml_time_us() : 0;

        if (trace) {
            trace_ids.assign(ids.begin(), ids.end());
            trace_record(1, 0, (uint16_t) il, trace_ms());
            const uint16_t n_ids = (uint16_t) trace_ids.size();
            std::fwrite(&n_ids, 2, 1, trace);
            std::fwrite(trace_ids.data(), 2, trace_ids.size(), trace);
            trace_routes++;
        }

        size_t n_hit = 0, n_evicted = 0;
        if (!g.slru.plan(il, ids.data(), ids.size(), fills, n_hit, n_evicted)) {
            GGML_ABORT("the MoE cache is too small for the experts selected in layer %d", il);
        }

        const int64_t t_submit = observe ? ggml_time_us() : 0;
        if (observe) {
            auto & perf = l.perf.values;
            perf[llama_moe_perf_data::id_wait_us] += t_plan - t_ids;
            perf[llama_moe_perf_data::id_bytes] += ggml_nbytes(sel);
            perf[llama_moe_perf_data::id_calls]++;
            perf[llama_moe_perf_data::plan_us] += t_submit - t_plan;
            perf[llama_moe_perf_data::hits] += n_hit;
            perf[llama_moe_perf_data::misses] += fills.size();
            perf[llama_moe_perf_data::evictions] += n_evicted;
            if (g.base_slots) {
                size_t requested = 0, missed = 0, expert_bytes = 0;
                for (auto e : g.slru.uniq) { requested += g.slru.slot_map[il][e] >= g.base_slots; }
                for (const auto & f : fills) { missed += f.slot >= g.base_slots; }
                for (const auto * w : l.experts) { expert_bytes += w->nb[2]; }
                perf[llama_moe_perf_data::extra_hits] += requested - missed;
                perf[llama_moe_perf_data::extra_misses] += missed;
                perf[llama_moe_perf_data::extra_fill_bytes] += missed*expert_bytes;
            }
            for (const auto * w : l.experts) {
                perf[llama_moe_perf_data::hit_bytes] += n_hit*w->nb[2];
            }
        }
        ggml_exl3_aux_batch aux_batch{};
        size_t aux_bytes = 0;
        bool batched = false;
        if (devices[g.id].aux_enabled && l.n_expert_used > 0 && ids.size() <= (size_t) 8*l.n_expert_used) {
            // The readback wait above also completes the previous use of the shared staging buffers.
            const int64_t t_pack = observe ? ggml_time_us() : 0;
            batched = pack_aux(l, g, aux_batch, aux_bytes);
            if (observe) { l.perf.values[llama_moe_perf_data::submit_us] += ggml_time_us() - t_pack; }
        }
        // upload the missing experts, consecutive experts going to consecutive slots are uploaded together
        size_t bytes = 0;
        auto upload_bank = [&](size_t ip) {
            const ggml_tensor * w    = l.experts[ip];
            const size_t expert_size = w->nb[2];
            for (size_t i = 0; i < fills.size();) {
                int32_t local, room;
                ggml_tensor * bank = locate(g, fills[i].slot, ip, local, room);
                size_t n = 1;
                while (i + n < fills.size() && (int32_t) n < room && fills[i + n].expert == fills[i].expert + (int32_t) n &&
                        fills[i + n].slot == fills[i].slot + (int32_t) n) {
                    n++;
                }
                const int64_t t_copy = observe ? ggml_time_us() : 0;
                ggml_backend_tensor_set_async(backend, bank, (const uint8_t *) w->data + fills[i].expert*expert_size, local*expert_size, n*expert_size);
                if (observe) {
                    l.perf.values[llama_moe_perf_data::submit_us] += ggml_time_us() - t_copy;
                }
                bytes += n*expert_size;
                if (observe) {
                    l.perf.upload(n*expert_size, true);
                }
                i += n;
            }
        };
        for (size_t ip = 0; ip < l.experts.size(); ++ip) {
            if (!batched || ip%3 == 0) { upload_bank(ip); }
        }
        if (batched) {
            auto & d = devices[g.id];
            const int64_t t_copy = observe ? ggml_time_us() : 0;
            const bool uploaded = d.aux_upload(backend, d.aux_scratch, ggml_backend_buffer_get_base(d.aux_host.get()), aux_bytes, &aux_batch);
            if (observe) { l.perf.values[llama_moe_perf_data::submit_us] += ggml_time_us() - t_copy; }
            if (uploaded) {
                bytes += aux_bytes;
                if (observe) { l.perf.upload(aux_bytes, true); }
            } else {
                for (size_t ip = 0; ip < l.experts.size(); ++ip) {
                    if (ip%3 != 0) { upload_bank(ip); }
                }
            }
        }


        stats & st = ids.size() <= (size_t) 8*n_expert_used ? stats_small : stats_large;
        st.hits   += n_hit;
        st.misses += fills.size();
        st.bytes  += bytes;

        // the next copy synchronizes the backend before it changes the slot map again
        const int64_t t_map = observe ? ggml_time_us() : 0;
        ggml_backend_tensor_set_async(backend, dst, src->data, 0, ggml_nbytes(src));
        if (observe) {
            l.perf.values[llama_moe_perf_data::submit_us] += ggml_time_us() - t_map;
        }

        return true;
    }

    void perf_flush(int64_t task, int slot, int phase, int role, bool final) {
        std::map<int, llama_moe_perf_data> delta;
        for (size_t il = 0; il < layers.size(); ++il) {
            auto & l = layers[il];
            if (l.ig < 0) {
                continue;
            }
            const int type = groups[l.ig].ref.front()->type;
            delta[type] += l.perf - l.perf_last;
            l.perf_last = l.perf;
            if (final) {
                l.perf.print(task, slot, phase, role, il, type, true);
                l.perf = l.perf_last = {};
            }
        }
        for (const auto & entry : delta) {
            entry.second.print(task, slot, phase, role, -1, entry.first, false);
        }
    }

    void log_stats() const {
        auto log = [](const char * name, const stats & st) {
            const size_t n = st.hits + st.misses;
            if (n == 0) {
                return;
            }
            LLAMA_LOG_INFO("llama_moe_cache: %s: hits = %zu, misses = %zu, hit rate = %.2f%%, uploaded = %.2f MiB\n",
                name, st.hits, st.misses, 100.0*st.hits/n, st.bytes/1024.0/1024.0);
        };
        log("ubatch <= 8", stats_small);
        log("ubatch  > 8", stats_large);
        if (stats_copy.hits > 0) {
            LLAMA_LOG_INFO("llama_moe_cache: large batches: %zu experts copied from the cache, %.2f MiB\n", stats_copy.hits, stats_copy.bytes/1024.0/1024.0);
        }
    }
};

llama_moe_cache::llama_moe_cache(const llama_model & model, const std::vector<ggml_backend_t> & backends, const std::vector<ggml_backend_buffer_type_t> & bufts, size_t size, size_t pin, bool aux_allowed, bool mtp) :
    pimpl(new impl(model, backends, bufts, size, pin, aux_allowed, mtp)) {
}

llama_moe_cache::~llama_moe_cache() = default;

void llama_moe_cache::perf_flush(int64_t task, int slot, int phase, int role, bool final) {
    pimpl->perf_flush(task, slot, phase, role, final);
}

ggml_backend_t llama_moe_cache::backend(int32_t il) const {
    return pimpl->backend(il);
}

ggml_tensor * llama_moe_cache::get_slot_map(int32_t il, int64_t n_tokens, int64_t n_expert_used) const {
    return pimpl->get_slot_map(il, n_tokens, n_expert_used);
}

ggml_tensor * llama_moe_cache::get_route_trigger(int32_t il, int64_t n_tokens, int64_t n_expert_used) const {
    return pimpl->get_route_trigger(il, n_tokens, n_expert_used);
}

bool llama_moe_cache::shared_early() const {
    return pimpl->shared != impl::SHARED_OFF;
}

ggml_tensor * llama_moe_cache::get_experts(const ggml_tensor * w) const {
    return pimpl->get_experts(w);
}

ggml_tensor * llama_moe_cache::get_slot_table(const ggml_tensor * w) const {
    const auto it = pimpl->bindings.find(w);
    if (it == pimpl->bindings.end() || it->second.ip % 3) { return nullptr; }
    const auto & g = pimpl->groups[pimpl->layers[it->second.il].ig];
    const size_t t = it->second.ip/3;
    return t + 1 < g.table_views.size() ? g.table_views[t] : nullptr;
}

bool llama_moe_cache::wants_staging() const {
    return pimpl->staging != impl::STAGING_OFF;
}

void llama_moe_cache::attach_staging(ggml_backend_t backend, ggml_backend_buffer_t buffer) {
    pimpl->attach_staging(backend, buffer);
}

void llama_moe_cache::set_staging_active(bool active) {
    pimpl->set_staging_active(active);
}

void llama_moe_cache::set_lend_level(int level) {
    pimpl->set_level(level);
}

void llama_moe_cache::attach_compute(ggml_backend_buffer_t buffer, size_t a_start, size_t b_start) {
    pimpl->attach_compute(buffer, a_start, b_start);
}

void llama_moe_cache::retire_compute(ggml_backend_buffer_t buffer) {
    if (buffer == pimpl->lent.buffer) { pimpl->drop_compute(); }
}

bool llama_moe_cache::lends_compute() const {
    return pimpl->lent.buffer != nullptr;
}

void llama_moe_cache::fit_compute(ggml_backend_buffer_t buffer, size_t planned_size) {
    pimpl->fit_compute(buffer, planned_size);
}

bool llama_moe_cache::copy(ggml_backend_t backend, const ggml_tensor * src, ggml_tensor * dst, ggml_cgraph * graph) {
    return pimpl->copy(backend, src, dst, graph);
}

bool llama_moe_cache::copy_ordered(ggml_backend_t backend, const ggml_tensor * src, ggml_tensor * dst, ggml_cgraph * graph) {
    if (pimpl->trigger_of.count(src)) {
        return pimpl->copy(backend, src, dst, graph);
    }
    const auto it = pimpl->layer_of.find(src);
    return it != pimpl->layer_of.end() && it->second == pimpl->pending_il && pimpl->copy(backend, src, dst, graph);
}

int64_t llama_moe_cache::copy_experts(ggml_backend_t backend, const ggml_tensor * w, ggml_tensor * dst, int64_t e, int64_t last) {
    return pimpl->copy_experts(backend, w, dst, e, last);
}

std::map<ggml_backend_buffer_type_t, size_t> llama_moe_cache::memory_breakdown() const {
    std::map<ggml_backend_buffer_type_t, size_t> res;
    for (const auto & d : pimpl->devices) {
        if (d.buf) {
            res[ggml_backend_buffer_get_type(d.buf.get())] += d.buf_size;
        }
        if (d.aux_host_buft) {
            res[d.aux_host_buft] += d.aux_bytes;
        }
        if (d.ids_host) {
            res[ggml_backend_buffer_get_type(d.ids_host.get())] += ggml_backend_buffer_get_size(d.ids_host.get());
        }
    }
    if (pimpl->buf_host) {
        res[ggml_backend_buffer_get_type(pimpl->buf_host.get())] += pimpl->buf_host_size;
    }
    return res;
}
