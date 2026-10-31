#include "../src/llama-memory-lending.h"
#include "../src/llama-moe-cache-slru.h"

#undef NDEBUG
#include <cassert>
#include <cstdio>
#include <string>

using lending = llama_memory_lending;

struct cache_fixture {
    llama_moe_cache_slru cache;
    std::vector<std::vector<int32_t>> maps;
    std::vector<llama_moe_cache_slru::fill> fills;
    size_t hits = 0, evictions = 0;

    cache_fixture(int layers, int experts, int slots) : maps(layers, std::vector<int32_t>(experts, -1)) {
        cache.init(layers, experts, slots);
        for (int i = 0; i < layers; ++i) { cache.slot_map[i] = maps[i].data(); }
    }
    bool plan(int layer, const std::vector<int32_t> & ids) {
        return cache.plan(layer, ids.data(), ids.size(), fills, hits, evictions);
    }
    void check() const {
        int active = 0, suspended = 0, protected_count = 0, listed = 0;
        std::vector<bool> found(cache.n_slots, false);
        for (int segment = 0; segment < 2; ++segment) {
            int previous = -1;
            for (int s = cache.head[segment]; s >= 0; s = cache.next[s]) {
                assert(s < cache.n_slots && !found[s] && cache.prev[s] == previous);
                assert(cache.is_protected[s] == bool(segment));
                assert(!segment || cache.active_mask[s]);
                found[s] = true; previous = s; listed++;
            }
            assert(previous == cache.tail[segment]);
        }
        assert(listed == cache.n_slots);
        for (int s = 0; s < cache.n_slots; ++s) {
            active += cache.active_mask[s];
            suspended += cache.valid_mask[s] && !cache.active_mask[s];
            protected_count += cache.is_protected[s];
            assert(!cache.active_mask[s] || cache.valid_mask[s]);
            int key = cache.key_of[s];
            if (key >= 0) {
                assert(cache.valid_mask[s]);
                assert(maps[key/cache.n_expert][key%cache.n_expert] == s);
            }
        }
        assert(suspended == cache.suspended_slots);
        assert(active == cache.active_slots && protected_count == cache.n_protected && protected_count <= active/2);
        for (size_t il = 0; il < maps.size(); ++il) {
            for (int e = 0; e < cache.n_expert; ++e) {
                int s = maps[il][e];
                assert(s == -1 || (s >= 0 && s < cache.n_slots && cache.key_of[s] == int(il)*cache.n_expert + e));
            }
        }
    }
};

template<class F> static void rejected(F f) {
    bool failed = false;
    try { f(); } catch (const std::exception &) { failed = true; }
    assert(failed);
}

static void test_activity() {
    cache_fixture f(2, 32, 12);
    assert(f.plan(0, {0,1,2,3,4,5,6,7,8,9,10,11}));
    auto original = f.maps;
    assert(f.cache.set_range(3, 3, false, true) == 0);
    f.cache.rebalance(); f.check();
    assert(f.maps == original);
    assert(f.plan(1, {0,1,2,3}));
    for (int s = 3; s < 6; ++s) { assert(f.cache.key_of[s] == s); }
    // A selected suspended expert must move to eligible storage, not return a suspended slot.
    assert(f.plan(0, {3,4,4}));
    assert(f.hits == 0 && f.fills.size() == 2);
    assert(f.maps[0][3] < 3 || f.maps[0][3] >= 6);
    assert(f.maps[0][4] < 3 || f.maps[0][4] >= 6);
    assert(f.cache.key_of[3] == -1 && f.cache.key_of[4] == -1);
    int survivor = f.maps[0][5];
    assert(f.cache.set_range(7, 2, false, false) <= 2);
    assert(f.maps[0][5] == survivor);
    f.cache.set_range(3, 3, true, true); f.cache.rebalance(); f.check();
    assert(f.plan(0, {5,5,5}) && f.hits == 1 && f.fills.empty());
    auto before = f.maps;
    std::vector<int32_t> too_many; for (int i = 0; i < 20; ++i) { too_many.push_back(i); }
    assert(!f.plan(1, too_many)); assert(before == f.maps); f.check();
    f.cache.seen_gen = UINT32_MAX;
    assert(f.plan(1, {7,8,7})); assert(f.cache.seen_gen == 1); f.check();
}

static uint32_t random_value(uint32_t & state) { state = state*1664525u + 1013904223u; return state; }

static uint64_t prefix_run() {
    cache_fixture f(4, 64, 96);
    uint32_t rng = 42;
    uint64_t digest = 14695981039346656037ull;
    auto hash = [&](uint32_t value) { digest = (digest ^ value)*1099511628211ull; };
    for (int step = 0; step < 20000; ++step) {
        if (step % 17 == 0) { hash(f.cache.set_active(16 + random_value(rng)%81)); }
        int il = random_value(rng)%4, n = 1 + random_value(rng)%16;
        std::vector<int32_t> ids; for (int i = 0; i < n; ++i) { ids.push_back(random_value(rng)%64); }
        assert(f.plan(il, ids)); hash(f.hits); hash(f.evictions);
        for (auto fill : f.fills) { hash(fill.expert); hash(fill.slot); }
        for (int32_t key : f.cache.key_of) { hash(key); }
        for (int32_t item : f.cache.prev) { hash(item); }
        for (int32_t item : f.cache.next) { hash(item); }
        for (auto & map : f.maps) { for (int32_t s : map) { hash(s); } }
        hash(f.cache.n_protected);
        f.check();
    }
    return digest;
}

static void test_independent_holes() {
    cache_fixture f(4, 64, 96);
    uint32_t rng = 123;
    for (int step = 0; step < 10000; ++step) {
        if (step%7 == 0) {
            int first = 16 + random_value(rng)%72;
            bool active = random_value(rng)%3 == 0;
            bool valid = active || random_value(rng)%2;
            f.cache.set_range(first, 8, active, valid); f.cache.rebalance();
        }
        int il = random_value(rng)%4;
        std::vector<int32_t> ids;
        for (int i = 0; i < 16; ++i) { ids.push_back(random_value(rng)%64); }
        assert(f.plan(il, ids));
        for (auto e : ids) { assert(f.cache.active_mask[f.maps[il][e]]); }
        f.check();
    }
}

static void test_owners() {
    lending ledger;
    int domain = 0;
    auto a = ledger.bind(1, lending::kind::resident, &domain, 0x10000, 4096, 128);
    auto b = ledger.bind(2, lending::kind::compute, &domain, 0x20000, 4096, 128);
    rejected([&] { ledger.bind(3, lending::kind::streamed, &domain, 0x10800, 4096, 128); });
    rejected([&] { ledger.bind(3, lending::kind::streamed, &domain, UINTPTR_MAX - 127, 4096, 128); });
    rejected([&] { ledger.add({{a, 4096, 128}}); });
    rejected([&] { ledger.add({{a, 1, 128}}); });
    rejected([&] { ledger.add({{a, 0, SIZE_MAX}}); });
    const uint64_t r1 = ledger.add({{a, 0, 256}, {b, 0, 128}});
    const uint64_t r2 = ledger.add({{a, 256, 256}});
    const uint64_t r3 = ledger.add({{b, 256, 256}});
    rejected([&] { ledger.add({{a, 128, 256}}); });
    rejected([&] { ledger.add({{a, 1024, 256}, {a, 1152, 256}}); });
    // Failed registration does not reserve earlier components.
    ledger.add({{a, 1024, 256}});
    rejected([&] { ledger.acquire(r1, {}); });
    ledger.publish({a, 0, 4096}); ledger.publish({b, 0, 4096});
    int initialized = 0;
    for (uint64_t r : {r1, r2, r3}) { ledger.acquire(r, [&] { initialized++; }); }
    ledger.acquire(r1, [&] { assert(false); }); assert(initialized == 3);
    ledger.admit(r2, false); assert(ledger.get(r2).leased && !ledger.get(r2).admitted);
    ledger.admit(r2, true);
    std::vector<int> order;
    lending::hooks h{
        [&] { assert(!ledger.get(r1).admitted && ledger.get(r1).returning); order.push_back(1); },
        [&](const std::vector<uint64_t> & ids) { assert(ids == std::vector<uint64_t>{r1}); order.push_back(2); },
        [&](const std::vector<lending::span> & spans) { assert(spans.size() == 2 && order.back() == 2); order.push_back(3); }
    };
    ledger.claim({a, 1, 1}, h);
    assert(order == std::vector<int>({1,2,3}));
    assert(!ledger.get(r1).leased && ledger.get(r2).leased && ledger.get(r3).leased);
    rejected([&] { ledger.acquire(r1, {}); });
    ledger.publish({a, 0, 128}); ledger.acquire(r1, {});
    rejected([&] { ledger.retire(a, {}); });
}

static void test_retirement_and_failure() {
    lending ledger;
    int domain = 0;
    auto a = ledger.bind(1, lending::kind::streamed, &domain, 0x10000, 4096, 128);
    auto b = ledger.bind(2, lending::kind::resident, &domain, 0x20000, 4096, 128);
    auto r = ledger.add({{a,0,256}}), survivor = ledger.add({{b,0,256}});
    // A pending writeback slot has no published extent, even if it has no logical page.
    rejected([&] { ledger.acquire(r, {}); });
    ledger.publish({a,0,4096}); ledger.publish({b,0,4096});
    ledger.acquire(r, {}); ledger.acquire(survivor, {});
    int drains = 0, invalidates = 0;
    lending::hooks h{[&] { drains++; }, [&](const std::vector<uint64_t> & ids) { assert(ids == std::vector<uint64_t>{r}); invalidates++; }, {}};
    ledger.retire(a, h);
    assert(drains == 1 && invalidates == 1 && ledger.get(survivor).leased);
    auto fresh = ledger.bind(1, lending::kind::streamed, &domain, 0x10000, 4096, 128);
    assert(fresh.generation > a.generation);
    rejected([&] { ledger.publish({a,0,128}); });
    rejected([&] { ledger.claim({a,0,128}, h); });
    rejected([&] { ledger.retire(a, h); });
    auto replacement = ledger.add({{fresh,0,256}});
    assert(replacement > survivor);
    ledger.publish({fresh,0,4096}); ledger.acquire(replacement, {});
    rejected([&] { ledger.claim({fresh,0,128}, {[&] { drains++; }, [&](const std::vector<uint64_t> &) { invalidates++; }, [](const std::vector<lending::span> &) { throw std::runtime_error("failed restore"); }}); });
    assert(ledger.get(replacement).returning && !ledger.get(replacement).admitted);
    rejected([&] { ledger.claim({fresh,0,128}, h); });
    rejected([&] { ledger.acquire(replacement, {}); });
    rejected([&] { ledger.admit(replacement, true); });
    rejected([&] { ledger.retire(fresh, h); });
    assert(ledger.get(survivor).leased);
}

static void test_zero_and_padding() {
    lending ledger;
    int domain = 0;
    std::vector<uint8_t> bytes(4096, 0xa5);
    auto owner = ledger.bind(1, lending::kind::resident, &domain, 0x10000, bytes.size(), 128);
    ledger.publish({owner,0,bytes.size()});
    auto r1 = ledger.add({{owner,0,128},{owner,512,128}});
    auto r2 = ledger.add({{owner,128,128},{owner,640,128}});
    // Shared padding is a separate bank-lifetime reservation, not duplicated per expert.
    auto pad = ledger.add({{owner,256,128},{owner,768,128}});
    for (auto r : {r1,r2,pad}) { ledger.acquire(r, {}); }
    std::vector<uint64_t> invalidated;
    size_t zero_calls = 0;
    ledger.claim({owner,0,256}, {[]{}, [&](const std::vector<uint64_t> & ids) { invalidated = ids; }, [&](const std::vector<lending::span> & spans) {
        for (auto s : spans) { zero_calls++; std::fill(bytes.begin()+s.offset, bytes.begin()+s.offset+s.bytes, 0); }
    }});
    assert(invalidated == std::vector<uint64_t>({r1,r2}) && zero_calls == 2 && ledger.get(pad).leased);
    for (size_t i = 0; i < bytes.size(); ++i) { assert(bytes[i] == (i < 256 || (i >= 512 && i < 768) ? 0 : 0xa5)); }
    // Claims with no borrowed bytes require no fence or initialization.
    ledger.claim({owner,1025,1}, {});
    const auto before_retire = bytes;
    ledger.retire(owner, {[]{}, [&](const std::vector<uint64_t> & ids) { assert(ids == std::vector<uint64_t>{pad}); }, {}});
    assert(bytes == before_retire && zero_calls == 2);
}

static void test_borrower_return() {
    cache_fixture f(1, 32, 12);
    f.cache.set_active(4);
    lending ledger;
    int domain = 0;
    auto a = ledger.bind(1, lending::kind::resident, &domain, 0x10000, 4096, 128);
    auto b = ledger.bind(2, lending::kind::compute, &domain, 0x20000, 4096, 128);
    auto ra = ledger.add({{a,0,512}}), rb = ledger.add({{b,0,512}});
    ledger.publish({a,0,512}); ledger.publish({b,0,512});
    ledger.acquire(ra, {}); ledger.acquire(rb, {});
    f.cache.set_range(4, 8, true, true); f.cache.rebalance();
    assert(f.plan(0, {0,1,2,3,4,5,6,7,8,9,10,11}));
    const auto before = f.maps;
    ledger.admit(ra, false); f.cache.set_range(4, 4, false, true); f.cache.rebalance();
    assert(before == f.maps);
    bool drained = false, restored = false;
    ledger.claim({a,128,128}, {
        [&] { assert(!ledger.get(ra).admitted); drained = true; },
        [&](const std::vector<uint64_t> & ids) {
            assert(drained && ids == std::vector<uint64_t>{ra});
            assert(f.cache.set_range(4, 4, false, false) == 4); f.cache.rebalance();
        },
        [&](const std::vector<lending::span> & spans) {
            assert(drained && spans.size() == 1 && spans[0].bytes == 512);
            for (int i = 4; i < 8; ++i) { assert(f.maps[0][i] == -1); }
            restored = true;
        }
    });
    assert(restored && ledger.get(rb).leased);
    for (int i : {0,1,2,3,8,9,10,11}) { assert(f.maps[0][i] == before[0][i]); }
    f.check();
}

static void test_composite_claim() {
    lending ledger;
    int domain = 0;
    auto owner = ledger.bind(1, lending::kind::resident, &domain, 0x10000, 9*4096, 128);
    ledger.publish({owner,0,9*4096});
    std::vector<lending::span> spans;
    for (uint64_t i = 0; i < 9; ++i) {
        spans.push_back({owner,i*4096+1024,256});
    }
    auto r = ledger.add(spans);
    ledger.acquire(r, {});
    int drains = 0, invalidated = 0;
    lending::hooks h{[&] { drains++; }, [&](const std::vector<uint64_t> & ids) { assert(ids == std::vector<uint64_t>{r}); invalidated++; }, {}};
    // Any one of the nine components can revoke the logical expert.
    ledger.claim({owner,8*4096,1024}, {});
    assert(ledger.get(r).leased && drains == 0);
    ledger.claim({owner,8*4096,1025}, h);
    assert(!ledger.get(r).leased && drains == 1 && invalidated == 1);
    ledger.claim({owner,8*4096,1025}, {});
    ledger.claim({owner,8*4096,512}, {});
    assert(drains == 1);
    rejected([&] { ledger.acquire(r, {}); });
    // Only the claimed interval was withdrawn; the owner must explicitly republish it.
    auto survivor = ledger.add({{owner,0,128}});
    ledger.acquire(survivor, {});
    ledger.publish({owner,8*4096,1025});
    ledger.acquire(r, {});
    assert(ledger.get(r).leased && ledger.get(survivor).leased);
}

int main() {
    test_activity();
    test_independent_holes();
    test_composite_claim();
    test_borrower_return();
    const auto digest = prefix_run();
    std::printf("legacy prefix digest: %llu\n", (unsigned long long) digest);
    // This fixture is pinned against the pre-KV1 implementation at cffeadb2b.
    assert(digest == 2145768551506956810ull);
    test_owners();
    test_retirement_and_failure();
    test_zero_and_padding();
    std::puts("MoE independent activity and ownership tests passed");
}
