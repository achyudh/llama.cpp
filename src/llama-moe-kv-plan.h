#pragma once

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <vector>

struct llama_moe_kv_plan {
    struct group {
        std::array<size_t, 9> stride;
        size_t layers;
        size_t limit;
    };
    struct component {
        size_t tensor, offset, bytes;
    };
    struct bank {
        size_t group, count;
        std::array<component, 9> parts;
    };
    std::vector<bank> banks;
    size_t auxiliary_reserve = 0;

    static llama_moe_kv_plan build(size_t tensors, size_t bytes, size_t alignment, const std::vector<group> & groups) {
        if (tensors < 6 || tensors > 4096 || tensors % 3 || !alignment || bytes % alignment || bytes > SIZE_MAX/3 || groups.empty()) {
            throw std::invalid_argument("unsupported KV lending geometry");
        }
        for (const auto & g : groups) {
            if (!g.layers || g.limit > 16384 || g.stride[0] == 0 || g.stride[3] != g.stride[0] || g.stride[6] != g.stride[0]) {
                throw std::invalid_argument("unsupported KV expert geometry");
            }
            for (size_t stride : g.stride) {
                if (!stride || stride % alignment || stride > bytes) { throw std::invalid_argument("invalid KV bank stride"); }
            }
        }
        auto trial = [&](size_t reserve) {
            llama_moe_kv_plan plan;
            plan.auxiliary_reserve = 3*reserve;
            std::array<size_t, 3> aux_end{bytes, bytes, bytes};
            std::vector<size_t> ends(tensors/3, bytes), counts(groups.size(), 0);
            ends.back() -= reserve;
            for (;;) {
                size_t gi = groups.size(), ti = 0, n = 0;
                for (size_t g = 0; g < groups.size(); ++g) {
                    for (size_t t = 0; t < ends.size(); ++t) {
                        size_t count = ends[t]/groups[g].stride[0];
                        count = count > 0 ? std::min(count - 1, groups[g].limit - counts[g]) : 0;
                        if (!count) { continue; }
                        const bool lower = gi == groups.size() ||
                            (long double) counts[g]/groups[g].layers < (long double) counts[gi]/groups[gi].layers;
                        const bool equal = gi != groups.size() &&
                            (long double) counts[g]/groups[g].layers == (long double) counts[gi]/groups[gi].layers;
                        if (lower || (equal && ends[t] > ends[ti])) { gi = g; ti = t; n = count; }
                    }
                }
                if (!n) { return plan; }
                const auto & g = groups[gi];
                bank next{gi, n, {}};
                auto pending = aux_end;
                for (size_t ip : {1u,2u,4u,5u,7u,8u}) {
                    if (n + 1 > bytes/g.stride[ip]) { return llama_moe_kv_plan{}; }
                    const size_t size = (n + 1)*g.stride[ip];
                    const size_t p = std::max_element(pending.begin(), pending.end()) - pending.begin();
                    if (size > pending[p] - (bytes - reserve)) { return llama_moe_kv_plan{}; }
                    pending[p] -= size;
                    next.parts[ip] = {tensors - 3 + p, pending[p], size};
                }
                const size_t size = (n + 1)*g.stride[0];
                ends[ti] -= size;
                for (size_t ip : {0u,3u,6u}) { next.parts[ip] = {3*ti + ip/3, ends[ti], size}; }
                aux_end = pending;
                counts[gi] += n;
                plan.banks.push_back(next);
            }
        };
        // Find the smallest aligned reserve that can hold the complete packed-bank plan's auxiliaries.
        size_t low = 0, high = bytes/alignment;
        while (low + 1 < high) {
            const size_t mid = low + (high - low)/2;
            auto plan = trial(mid*alignment);
            if (!plan.banks.empty()) { high = mid; } else { low = mid; }
        }
        auto result = trial(high*alignment);
        if (result.banks.empty()) { throw std::runtime_error("KV storage cannot hold an expert bank plan"); }
        return result;
    }
};
