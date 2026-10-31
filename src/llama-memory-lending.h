#pragma once

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <map>
#include <stdexcept>
#include <utility>
#include <vector>

// Host control path only. Owners publish bytes after their own readers and writers finish.
class llama_memory_lending {
public:
    enum class kind { staging, compute, resident, streamed };
    struct owner {
        uint64_t id = 0;
        uint64_t generation = 0;
        bool operator==(const owner & other) const { return id == other.id && generation == other.generation; }
    };
    struct span {
        owner allocation;
        size_t offset;
        size_t bytes;
    };
    struct region {
        std::vector<span> spans;
        bool leased = false;
        bool admitted = false;
        bool returning = false;
    };
    struct hooks {
        std::function<void()> drain;
        std::function<void(const std::vector<uint64_t> &)> invalidate;
        std::function<void(const std::vector<span> &)> restore;
    };
    using observer = std::function<void(const char *, uint64_t, const span &)>;
    observer observe;

    owner bind(uint64_t id, kind type, const void * domain, uintptr_t base, size_t bytes, size_t alignment) {
        if (!id || allocations.count(id) || !domain || !base || !bytes || bytes > UINTPTR_MAX - base ||
                !alignment || base % alignment || next_generation == UINT64_MAX) {
            throw std::invalid_argument("invalid lending allocation");
        }
        for (const auto & item : allocations) {
            const auto & a = item.second;
            if (a.domain == domain && base < a.base + a.bytes && a.base < base + bytes) {
                throw std::invalid_argument("overlapping lending allocations");
            }
        }
        owner token{id, ++next_generation};
        allocation a{token, type, domain, base, bytes, alignment, {}, {}};
        allocations.emplace(id, std::move(a));
        emit("bind", 0, {token, 0, bytes});
        return token;
    }

    // Publishing is an owner promise, not a test for unused logical rows or pages.
    void publish(span range) {
        auto & a = checked(range);
        size_t begin = range.offset, end = begin + range.bytes;
        auto it = a.available.lower_bound(begin);
        if (it != a.available.begin() && std::prev(it)->second >= begin) { --it; }
        while (it != a.available.end() && it->first <= end) {
            begin = std::min(begin, it->first);
            end = std::max(end, it->second);
            it = a.available.erase(it);
        }
        a.available.emplace(begin, end);
        emit("publish", 0, range);
    }

    uint64_t add(const std::vector<span> & spans) {
        if (spans.empty() || next_region == UINT64_MAX) { throw std::invalid_argument("invalid lending region"); }
        for (size_t i = 0; i < spans.size(); ++i) {
            const auto & s = spans[i];
            const auto & a = checked(s);
            if (s.offset % a.alignment) { throw std::invalid_argument("unaligned lending component"); }
            auto it = a.index.lower_bound(s.offset);
            if ((it != a.index.end() && it->first < s.offset + s.bytes) ||
                    (it != a.index.begin() && std::prev(it)->second.end > s.offset)) {
                throw std::invalid_argument("overlapping lending regions");
            }
            for (size_t j = 0; j < i; ++j) {
                const auto & p = spans[j];
                if (p.allocation == s.allocation && s.offset < p.offset + p.bytes && p.offset < s.offset + s.bytes) {
                    throw std::invalid_argument("overlapping lending components");
                }
            }
        }
        const uint64_t id = ++next_region;
        regions.emplace(id, region{spans, false, false, false});
        for (const auto & s : spans) {
            allocations.at(s.allocation.id).index.emplace(s.offset, entry{s.offset + s.bytes, id});
            emit("add", id, s);
        }
        return id;
    }

    const region & get(uint64_t id) const { return regions.at(id); }
    kind owner_kind(owner token) { return checked({token, 0, 1}).type; }

    void acquire(uint64_t id, const std::function<void()> & initialize) {
        auto & r = regions.at(id);
        if (r.returning) { throw std::runtime_error("lending region is quarantined"); }
        if (r.leased) { return; }
        for (const auto & s : r.spans) {
            const auto & a = checked(s);
            auto it = a.available.upper_bound(s.offset);
            if (it == a.available.begin() || (--it)->second < s.offset + s.bytes) {
                throw std::runtime_error("lending region is not published");
            }
        }
        if (initialize) { initialize(); }
        r.leased = r.admitted = true;
        emit("acquire", id, r.spans.front());
    }

    void admit(uint64_t id, bool active) {
        auto & r = regions.at(id);
        if (!r.leased || r.returning) { throw std::runtime_error("lending region is not held"); }
        if (r.admitted == active) { return; }
        r.admitted = active;
        emit(active ? "resume" : "suspend", id, r.spans.front());
    }

    void release(const std::vector<uint64_t> & ids, const hooks & h) {
        std::vector<uint64_t> selected;
        std::vector<span> spans;
        for (uint64_t id : ids) {
            auto & r = regions.at(id);
            if (r.returning) { throw std::runtime_error("lending region is quarantined"); }
            if (!r.leased) { continue; }
            selected.push_back(id);
            spans.insert(spans.end(), r.spans.begin(), r.spans.end());
        }
        std::sort(selected.begin(), selected.end());
        if (std::adjacent_find(selected.begin(), selected.end()) != selected.end()) {
            throw std::invalid_argument("duplicate lending release");
        }
        if (selected.empty()) { return; }
        if (!h.drain || !h.invalidate) { throw std::invalid_argument("missing lending return hooks"); }
        for (uint64_t id : selected) { auto & r = regions.at(id); r.admitted = false; r.returning = true; }
        h.drain();
        emit("drain", 0, spans.front());
        h.invalidate(selected);
        emit("invalidate", 0, spans.front());
        // A failed fence or restore leaves the regions quarantined; never make stale bytes lendable again.
        if (h.restore) { h.restore(coalesce(std::move(spans))); }
        for (uint64_t id : selected) {
            auto & r = regions.at(id);
            r.leased = r.returning = false;
            emit("release", id, r.spans.front());
        }
    }

    void withdraw(span range) {
        auto & a = checked(range);
        const size_t end = range.offset + range.bytes;
        std::vector<std::pair<size_t, size_t>> survivors;
        auto it = a.available.lower_bound(range.offset);
        if (it != a.available.begin() && std::prev(it)->second > range.offset) { --it; }
        while (it != a.available.end() && it->first < end) {
            if (it->first < range.offset) { survivors.emplace_back(it->first, range.offset); }
            if (it->second > end) { survivors.emplace_back(end, it->second); }
            it = a.available.erase(it);
        }
        for (auto p : survivors) { a.available.emplace(p); }
    }

    void claim(span range, const hooks & h) {
        withdraw(range);
        auto & a = checked(range);
        const size_t end = range.offset + range.bytes;
        std::vector<uint64_t> ids;
        auto pos = a.index.lower_bound(range.offset);
        if (pos != a.index.begin() && std::prev(pos)->second.end > range.offset) { --pos; }
        for (; pos != a.index.end() && pos->first < end; ++pos) { ids.push_back(pos->second.region); }
        std::sort(ids.begin(), ids.end());
        ids.erase(std::unique(ids.begin(), ids.end()), ids.end());
        emit("claim", 0, range);
        release(ids, h);
        emit("claimed", 0, range);
    }

    void retire(owner token, const hooks & h) {
        auto & a = checked({token, 0, 1});
        claim({token, 0, a.bytes}, h);
        std::vector<uint64_t> ids;
        for (const auto & item : a.index) { ids.push_back(item.second.region); }
        std::sort(ids.begin(), ids.end());
        ids.erase(std::unique(ids.begin(), ids.end()), ids.end());
        for (uint64_t id : ids) {
            for (const auto & s : regions.at(id).spans) { allocations.at(s.allocation.id).index.erase(s.offset); }
            regions.erase(id);
        }
        emit("retire", 0, {token, 0, a.bytes});
        allocations.erase(token.id);
    }

    static std::vector<span> coalesce(std::vector<span> spans) {
        std::sort(spans.begin(), spans.end(), [](const span & a, const span & b) {
            if (a.allocation.id != b.allocation.id) { return a.allocation.id < b.allocation.id; }
            if (a.allocation.generation != b.allocation.generation) { return a.allocation.generation < b.allocation.generation; }
            return a.offset < b.offset;
        });
        std::vector<span> result;
        for (const auto & s : spans) {
            if (!result.empty() && result.back().allocation == s.allocation && result.back().offset + result.back().bytes == s.offset) {
                result.back().bytes += s.bytes;
            } else { result.push_back(s); }
        }
        return result;
    }

private:
    struct entry { size_t end; uint64_t region; };
    struct allocation {
        owner token;
        kind type;
        const void * domain;
        uintptr_t base;
        size_t bytes, alignment;
        std::map<size_t, size_t> available;
        std::map<size_t, entry> index;
    };
    std::map<uint64_t, allocation> allocations;
    std::map<uint64_t, region> regions;
    uint64_t next_generation = 0, next_region = 0;

    allocation & checked(const span & s) {
        auto it = allocations.find(s.allocation.id);
        if (it == allocations.end() || !(it->second.token == s.allocation)) { throw std::invalid_argument("stale lending generation"); }
        auto & a = it->second;
        if (!s.bytes || s.offset > a.bytes || s.bytes > a.bytes - s.offset) {
            throw std::invalid_argument("invalid lending span");
        }
        return a;
    }
    void emit(const char * op, uint64_t id, const span & s) { if (observe) { observe(op, id, s); } }
};
