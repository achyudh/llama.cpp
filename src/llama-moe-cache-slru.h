#pragma once

#include "ggml.h"

#include <algorithm>
#include <cstdint>
#include <vector>

// SLRU of the experts of a group of layers, the slot of each expert is kept in the slot map of its layer
struct llama_moe_cache_slru {
    int32_t n_expert = 0;
    int32_t n_slots  = 0;
    int32_t active_slots = 0;
    int32_t suspended_slots = 0;
    std::vector<uint8_t> active_mask;
    std::vector<uint8_t> valid_mask;

    std::vector<int32_t *> slot_map; // [n_layer] data of the slot maps, -1 if the expert is not cached
    std::vector<int32_t>   key_of;   // [n_slots] il*n_expert + expert, -1 if empty

    // probation and protected LRU lists; hits promote slots to the protected half
    std::vector<int32_t> prev;
    std::vector<int32_t> next;
    int32_t head[2] = { -1, -1 };
    int32_t tail[2] = { -1, -1 };
    std::vector<bool> is_protected;
    int32_t n_protected = 0;

    std::vector<uint32_t> seen; // [n_expert]
    uint32_t seen_gen = 0;
    std::vector<int32_t> uniq;

    void init(int32_t n_layer, int32_t n_expert, int32_t n_slots) {
        GGML_ASSERT(n_layer > 0 && n_expert > 0 && n_slots >= 0 && n_layer <= INT32_MAX/n_expert);
        this->n_expert = n_expert;
        this->n_slots  = n_slots;
        active_slots = n_slots;
        suspended_slots = 0;
        active_mask.assign(n_slots, 1);
        valid_mask.assign(n_slots, 1);
        slot_map.assign(n_layer, nullptr);
        key_of.assign(n_slots, -1);
        prev.resize(n_slots);
        next.resize(n_slots);
        for (int32_t s = 0; s < n_slots; ++s) {
            prev[s] = s - 1;
            next[s] = s + 1 < n_slots ? s + 1 : -1;
        }
        head[0] = n_slots > 0 ? 0 : -1;
        tail[0] = n_slots - 1;
        head[1] = tail[1] = -1;
        n_protected = 0;
        seen_gen = 0;
        uniq.clear();
        is_protected.assign(n_slots, false);
        seen.assign(n_expert, 0);
    }

    // move slot s to the most recently used end of a segment
    void move(int32_t s, bool protect) {
        const bool from = is_protected[s];
        if (from == protect && s == tail[from]) {
            return;
        }
        if (prev[s] >= 0) {
            next[prev[s]] = next[s];
        } else {
            head[from] = next[s];
        }
        if (next[s] >= 0) {
            prev[next[s]] = prev[s];
        } else {
            tail[from] = prev[s];
        }

        prev[s] = tail[protect];
        next[s] = -1;
        if (tail[protect] >= 0) {
            next[tail[protect]] = s;
        } else {
            head[protect] = s;
        }
        tail[protect] = s;
        is_protected[s] = protect;
        n_protected += int(protect) - int(from);
    }

    void touch(int32_t s) {
        move(s, true);
        if (n_protected > active_slots/2) {
            move(head[1], false);
        }
    }

    int32_t victim(int32_t il) const {
        for (int segment = 0; segment < 2; ++segment) {
            for (int32_t s = head[segment]; s >= 0; s = next[s]) {
                if (!active_mask[s]) { continue; }
                const int32_t key = key_of[s];
                // keep every expert selected by this request, including earlier fills
                if (key < 0 || key / n_expert != il || seen[key % n_expert] != seen_gen) {
                    return s;
                }
            }
        }
        GGML_ABORT("no SLRU victim");
    }

    size_t invalidate(int32_t s) {
        const int32_t key = key_of[s];
        if (key >= 0) {
            slot_map[key / n_expert][key % n_expert] = -1;
            key_of[s] = -1;
        }
        move(s, false);
        return key >= 0;
    }

    // Valid storage can stay resident while it is excluded from admission and replacement.
    size_t set_range(int32_t first, int32_t count, bool active, bool valid) {
        GGML_ASSERT(first >= 0 && count >= 0 && first <= n_slots && count <= n_slots - first && (!active || valid));
        size_t removed = 0;
        for (int32_t s = first; s < first + count; ++s) {
            if (valid_mask[s] && !valid) {
                removed += invalidate(s);
            } else if (active_mask[s] && !active) {
                move(s, false);
            }
            suspended_slots += int(valid && !active) - int(valid_mask[s] && !active_mask[s]);
            active_slots += int(active) - int(active_mask[s]);
            active_mask[s] = active;
            valid_mask[s] = valid;
        }
        return removed;
    }

    // Rebalance once after the complete ownership wave, preserving legacy prefix ordering.
    void rebalance() {
        while (n_protected > active_slots/2) { move(head[1], false); }
    }

    // Legacy prefix transitions only; independent owners use set_range.
    size_t set_active(int32_t count) {
        GGML_ASSERT(count >= 0 && count <= n_slots);
        size_t removed = 0;
        if (count < active_slots) {
            removed = set_range(count, active_slots - count, false, false);
        } else if (count > active_slots) {
            set_range(active_slots, count - active_slots, true, true);
        }
        rebalance();
        return removed;
    }

    struct fill {
        int32_t expert;
        int32_t slot;
    };

    // give a slot to each expert selected by ids in layer il, evicting from probation first
    // returns false if the ids select more distinct experts than there are slots
    bool plan(int32_t il, const int32_t * ids, size_t n_ids, std::vector<fill> & fills, size_t & n_hit, size_t & n_evicted) {
        fills.clear();
        n_hit = 0;
        n_evicted = 0;

        if (++seen_gen == 0) {
            std::fill(seen.begin(), seen.end(), 0);
            seen_gen = 1;
        }
        uniq.clear();
        for (size_t i = 0; i < n_ids; ++i) {
            GGML_ASSERT(ids[i] >= 0 && ids[i] < n_expert);
            if (seen[ids[i]] != seen_gen) {
                seen[ids[i]] = seen_gen;
                uniq.push_back(ids[i]);
            }
        }
        if (uniq.size() > (size_t) active_slots) {
            return false;
        }

        int32_t * slots = slot_map[il];

        if (suspended_slots) {
            for (int32_t e : uniq) {
                if (slots[e] >= 0 && !active_mask[slots[e]]) { invalidate(slots[e]); }
            }
        }
        // promote hits before choosing victims for misses
        for (int32_t e : uniq) {
            if (slots[e] >= 0) {
                touch(slots[e]);
                n_hit++;
            }
        }
        // sorted misses usually get consecutive slots, so the uploads can be merged
        std::sort(uniq.begin(), uniq.end());
        for (int32_t e : uniq) {
            if (slots[e] >= 0) {
                continue;
            }
            const int32_t s = victim(il);
            if (key_of[s] >= 0) {
                n_evicted++;
                slot_map[key_of[s] / n_expert][key_of[s] % n_expert] = -1;
            }
            key_of[s] = il*n_expert + e;
            slots[e] = s;
            move(s, false);
            fills.push_back({ e, s });
        }
        return true;
    }
};
