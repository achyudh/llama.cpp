#include "ggml-cuda/exl3.cuh"
#include "ggml-exl3.h"
#include "gguf.h"

#undef NDEBUG
#include <algorithm>
#include <array>
#include <cassert>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <string>
#include <vector>

static __global__ void codebook_kernel(uint16_t * dst) {
    const unsigned i = blockIdx.x * blockDim.x + threadIdx.x;
    dst[i] = __half_as_ushort(ggml_cuda_exl3_mul1(i));
}

template<int bits>
static __global__ void tile_kernel(const uint32_t * p, uint16_t * dst) {
    const int i = threadIdx.x, lane = i / 8, j = i % 8;
    const int row = (lane & 3) * 2 + (j & 1) + ((j & 2) ? 8 : 0);
    const int col = lane / 4 + ((j & 4) ? 8 : 0);
    dst[row * 16 + col] = __half_as_ushort(ggml_cuda_exl3_weight<bits>(p, i));
}

template<int bits>
static __global__ void fragment_kernel(const uint32_t * p, uint16_t * dst) {
    const int lane = threadIdx.x;
    half w[8];
    ggml_cuda_exl3_fragment<bits>(p, lane, w);
#pragma unroll
    for (int j = 0; j < 8; ++j) {
        const int row = (lane & 3) * 2 + (j & 1) + ((j & 2) ? 8 : 0);
        const int col = lane / 4 + ((j & 4) ? 8 : 0);
        dst[row * 16 + col] = __half_as_ushort(w[j]);
    }
}

static void codec(const char * path) {
    std::ifstream in(path, std::ios::binary);
    assert(in);
    uint16_t * device;
    uint32_t * packed;
    CUDA_CHECK(cudaMalloc((void **) &device, 65536 * 2));
    CUDA_CHECK(cudaMalloc((void **) &packed, 256));
    std::vector<uint16_t> expected(65536), actual(65536);
    assert(in.read((char *) expected.data(), 65536 * 2));
    codebook_kernel<<<256, 256>>>(device);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaMemcpy(actual.data(), device, 65536 * 2, cudaMemcpyDeviceToHost));
    assert(actual == expected);
    unsigned bits, length, count = 0;
    while (in.read((char *) &bits, 4)) {
        assert(in.read((char *) &length, 4));
        assert(length == ggml_exl3_tile_size(bits) && length <= 256);
        std::vector<char> p(length);
        assert(in.read(p.data(), length));
        assert(in.read((char *) expected.data(), 512));
        CUDA_CHECK(cudaMemcpy(packed, p.data(), length, cudaMemcpyHostToDevice));
#define TILE_CASE(b) case b: tile_kernel<b><<<1, 256>>>(packed, device); fragment_kernel<b><<<1, 32>>>(packed, device + 256); break
        switch (bits) {
            TILE_CASE(2); TILE_CASE(3); TILE_CASE(4); TILE_CASE(5); TILE_CASE(6); TILE_CASE(7);
            TILE_CASE(8); TILE_CASE(10); TILE_CASE(12); TILE_CASE(14); TILE_CASE(16);
            default: assert(false);
        }
#undef TILE_CASE
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaMemcpy(actual.data(), device, 1024, cudaMemcpyDeviceToHost));
        assert(std::equal(actual.begin(), actual.begin() + 256, expected.begin()));
        assert(std::equal(actual.begin() + 256, actual.begin() + 512, expected.begin()));
        ++count;
    }
    assert(in.eof() && in.gcount() == 0 && count > 0);
    CUDA_CHECK(cudaFree(packed)); CUDA_CHECK(cudaFree(device));
    std::printf("{\"codebook_entries\":65536,\"exact_tiles\":%u,\"exact_fragments\":%u}\n", count, count);
}

static void save_output(const std::string & path, ggml_tensor * out) {
    std::vector<float> values(ggml_nelements(out));
    ggml_backend_tensor_get(out, values.data(), 0, ggml_nbytes(out));
    std::ofstream f(path, std::ios::binary);
    assert(f.write((const char *) values.data(), ggml_nbytes(out)));
}

static double time_stage(const ggml_cuda_exl3_args & a, ggml_cuda_exl3_mode mode, ggml_cuda_exl3_stage stage, cudaStream_t stream) {
    hipEvent_t start, stop;
    CUDA_CHECK(hipEventCreate(&start)); CUDA_CHECK(hipEventCreate(&stop));
    for (int i = 0; i < 10; ++i) { ggml_cuda_exl3_launch(a, mode, stage, stream); }
    hipGraph_t graph;
    hipGraphExec_t executable;
    CUDA_CHECK(hipStreamBeginCapture(stream, hipStreamCaptureModeThreadLocal));
    for (int i = 0; i < 50; ++i) { ggml_cuda_exl3_launch(a, mode, stage, stream); }
    CUDA_CHECK(hipStreamEndCapture(stream, &graph));
    CUDA_CHECK(hipGraphInstantiate(&executable, graph, nullptr, nullptr, 0));
    std::vector<double> samples;
    for (int group = 0; group < 7; ++group) {
        CUDA_CHECK(hipEventRecord(start, stream));
        CUDA_CHECK(hipGraphLaunch(executable, stream));
        CUDA_CHECK(hipEventRecord(stop, stream)); CUDA_CHECK(hipEventSynchronize(stop));
        float ms;
        CUDA_CHECK(hipEventElapsedTime(&ms, start, stop));
        samples.push_back(ms * 1000 / 50);
    }
    CUDA_CHECK(hipGraphExecDestroy(executable)); CUDA_CHECK(hipGraphDestroy(graph));
    CUDA_CHECK(hipEventDestroy(start)); CUDA_CHECK(hipEventDestroy(stop));
    std::sort(samples.begin(), samples.end());
    return samples[samples.size() / 2];
}

static void projection(const char * bank, const char * input, const char * prefix) {
    ggml_context * weights = nullptr, * inputs = nullptr;
    auto * meta = gguf_init_from_file(bank, {false, &weights});
    auto * data = gguf_init_from_file(input, {false, &inputs});
    ggml_exl3_bundle b = {};
    assert(meta && data && ggml_exl3_load_bundle(meta, weights, &b));
    auto * backend = ggml_backend_cuda_init(0);
    assert(backend);
    auto * ctx = ggml_init({4 * 1024 * 1024, nullptr, true});
    std::vector<std::pair<ggml_tensor *, ggml_tensor *>> copies;
    auto copy = [&](ggml_tensor * source) -> ggml_tensor * {
        if (!source) { return nullptr; }
        auto * tensor = ggml_dup_tensor(ctx, source);
        copies.emplace_back(source, tensor);
        return tensor;
    };
    auto * p = copy(b.packed), * su = copy(b.suh), * sv = copy(b.svh);
    auto * x = copy(ggml_get_tensor(inputs, "input")), * ids = copy(ggml_get_tensor(inputs, "ids"));
    auto * out = ggml_mul_mat_exl3(ctx, p, su, sv, x, ids, b.bits_x2);
    assert(out && ggml_backend_dev_supports_op(ggml_backend_get_device(backend), out));
    auto * graph = ggml_new_graph(ctx);
    ggml_build_forward_expand(graph, out);
    auto * buffer = ggml_backend_alloc_ctx_tensors(ctx, backend);
    assert(buffer);
    for (auto & pair : copies) { ggml_backend_tensor_set(pair.second, pair.first->data, 0, ggml_nbytes(pair.first)); }
    const int assignments = out->ne[1] * out->ne[2];
    const int k = su->ne[0], n = sv->ne[0];
    void * xh;
    float * yh;
    CUDA_CHECK(cudaMalloc(&xh, size_t(assignments) * k * sizeof(float)));
    CUDA_CHECK(cudaMalloc((void **) &yh, size_t(assignments) * n * sizeof(float)));
    cudaStream_t stream;
    CUDA_CHECK(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
    const ggml_cuda_exl3_args a = {(const uint32_t *) p->data, (const half *) su->data, (const half *) sv->data,
        (const float *) x->data, ids ? (const int32_t *) ids->data : nullptr, (float *) out->data, xh, yh,
        k, n, int(p->ne[2]), assignments, ids ? int(ids->ne[0]) : 1, int(x->ne[1]), b.bits_x2};
    std::printf("{\"k\":%d,\"n\":%d,\"assignments\":%d,\"timing_us\":{", k, n, assignments);
    for (auto mode : {GGML_EXL3_F32, GGML_EXL3_F16_DOT, GGML_EXL3_F32_FRAGMENT, GGML_EXL3_F32_ROWS}) {
        const char * name = mode == GGML_EXL3_F32 ? "f32" : mode == GGML_EXL3_F16_DOT ? "dot" :
            mode == GGML_EXL3_F32_FRAGMENT ? "fragment" : "rows";
        ggml_cuda_exl3_launch(a, mode, GGML_EXL3_ALL, stream);
        CUDA_CHECK(cudaStreamSynchronize(stream));
        save_output(std::string(prefix) + "." + name + ".bin", out);
        double times[4];
        for (int s = 0; s < 4; ++s) { times[s] = time_stage(a, mode, (ggml_cuda_exl3_stage) s, stream); }
        std::printf("%s\"%s\":{\"input\":%.6f,\"matrix\":%.6f,\"output\":%.6f,\"total\":%.6f}",
            mode == GGML_EXL3_F32 ? "" : ",", name, times[0], times[1], times[2], times[3]);
    }
    // Replay the real GGML graph with changed inputs and routing, then restore the fixture.
    assert(ggml_backend_graph_compute(backend, graph) == GGML_STATUS_SUCCESS);
    save_output(std::string(prefix) + ".graph.bin", out);
    for (int pass = 0; pass < 4; ++pass) {
        std::vector<float> changed(ggml_nelements(x), 0.25f);
        ggml_backend_tensor_set(x, pass == 3 ? copies[3].first->data : changed.data(), 0, ggml_nbytes(x));
        if (ids) {
            const int32_t id = pass == 1 ? -1 : pass == 2 ? int(p->ne[2]) : int(p->ne[2] - 1);
            std::vector<int32_t> selected(ggml_nelements(ids), id);
            ggml_backend_tensor_set(ids, pass == 3 ? copies[4].first->data : selected.data(), 0, ggml_nbytes(ids));
        }
        assert(ggml_backend_graph_compute(backend, graph) == GGML_STATUS_SUCCESS);
        save_output(std::string(prefix) + ".replay-" + std::to_string(pass) + ".bin", out);
    }
    std::puts("}}");
    CUDA_CHECK(cudaStreamDestroy(stream)); CUDA_CHECK(cudaFree(xh)); CUDA_CHECK(cudaFree(yh));
    ggml_backend_buffer_free(buffer); ggml_free(ctx); ggml_backend_free(backend);
    gguf_free(meta); gguf_free(data); ggml_free(weights); ggml_free(inputs);
}

static void split_bank_projection(bool slot_order = false) {
    auto * backend = ggml_backend_cuda_init(0);
    assert(backend);
    bool exact = true;
    for (int k : slot_order ? std::vector<int>{640, 2560} : std::vector<int>{128}) {
        for (int rate : {7, 8}) {
            for (int rows : slot_order ? std::vector<int>{8, 16, 32} : std::vector<int>{1, 4, 8, 16, 32}) {
                const int experts = slot_order ? 12 : 5, used = slot_order ? 10 : 4, base_count = slot_order ? 5 : 2;
                auto * ctx = ggml_init({4*1024*1024, nullptr, true});
                auto make_bank = [&](int count) {
                    ggml_exl3_bundle b{};
                    b.packed = ggml_new_tensor_3d(ctx, GGML_TYPE_I16, k/16*(256/16)*ggml_exl3_tile_size(rate)/2, 1, count);
                    b.suh = ggml_new_tensor_3d(ctx, GGML_TYPE_F16, k, 1, count);
                    b.svh = ggml_new_tensor_3d(ctx, GGML_TYPE_F16, 256, 1, count);
                    return b;
                };
                auto full = make_bank(experts), rebased = make_bank(experts), base = make_bank(base_count), extra = make_bank(experts - base_count);
                auto * x = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, k, used, rows);
                auto * ids = ggml_new_tensor_2d(ctx, GGML_TYPE_I32, used, rows);
                auto * ordinary = ggml_mul_mat_exl3(ctx, full.packed, full.suh, full.svh, x, ids, rate);
                auto * mapped_ids = ggml_new_tensor_2d(ctx, GGML_TYPE_I32, used, rows);
                auto * permuted = ggml_mul_mat_exl3(ctx, rebased.packed, rebased.suh, rebased.svh, x, mapped_ids, rate);
                auto * split = ggml_mul_mat_exl3(ctx, base.packed, base.suh, base.svh, x, mapped_ids, rate);
                // the base and extension banks behind one slot table, as a MoE cache with a borrowed extension
                auto * table = ggml_new_tensor_2d(ctx, GGML_TYPE_I64, 3, experts);
                assert(ggml_exl3_set_slots(split, table));
                for (auto * op : {ordinary, permuted, split}) { assert(ggml_exl3_set_cache_order(op)); }
                auto * gfull = ggml_new_graph(ctx), * gsplit = ggml_new_graph(ctx), * gmapped = ggml_new_graph(ctx);
                ggml_build_forward_expand(gfull, ordinary); ggml_build_forward_expand(gmapped, permuted); ggml_build_forward_expand(gsplit, split);
                auto * buffer = ggml_backend_alloc_ctx_tensors(ctx, backend);
                assert(buffer && ggml_backend_dev_supports_op(ggml_backend_get_device(backend), split));
                {
                    std::vector<int64_t> addresses(3*experts);
                    for (int s = 0; s < experts; ++s) {
                        const auto & b = s < base_count ? base : extra;
                        const int local = s < base_count ? s : s - base_count;
                        const ggml_tensor * parts[3] = {b.packed, b.suh, b.svh};
                        for (int c = 0; c < 3; ++c) { addresses[3*s + c] = (int64_t) (intptr_t) ((const char *) parts[c]->data + size_t(local)*parts[c]->nb[2]); }
                    }
                    ggml_backend_tensor_set(table, addresses.data(), 0, ggml_nbytes(table));
                }
                for (auto banks : {std::array<ggml_tensor *, 4>{full.packed, base.packed, extra.packed, rebased.packed},
                                   std::array<ggml_tensor *, 4>{full.suh, base.suh, extra.suh, rebased.suh},
                                   std::array<ggml_tensor *, 4>{full.svh, base.svh, extra.svh, rebased.svh}}) {
                    std::vector<uint8_t> raw(ggml_nbytes(banks[0]));
                    if (banks[0]->type == GGML_TYPE_I16) {
                        for (size_t i = 0; i < raw.size(); ++i) { raw[i] = uint8_t(i*23+i/317); }
                    } else {
                        for (size_t i = 0; i < raw.size()/2; ++i) {
                            const auto v = ggml_fp32_to_fp16(float(int(i%11)-5)/8);
                            std::memcpy(raw.data()+i*2, &v, 2);
                        }
                    }
                    ggml_backend_tensor_set(banks[0], raw.data(), 0, raw.size());
                    std::vector<uint8_t> reordered(raw.size());
                    for (int expert = 0; expert < experts; ++expert) {
                        const int slot = slot_order ? experts - 1 - expert : expert;
                        std::memcpy(reordered.data() + slot*banks[0]->nb[2], raw.data() + expert*banks[0]->nb[2], banks[0]->nb[2]);
                    }
                    ggml_backend_tensor_set(banks[3], reordered.data(), 0, reordered.size());
                    ggml_backend_tensor_set(banks[1], reordered.data(), 0, ggml_nbytes(banks[1]));
                    ggml_backend_tensor_set(banks[2], reordered.data()+ggml_nbytes(banks[1]), 0, ggml_nbytes(banks[2]));
                }
                std::vector<float> input(ggml_nelements(x)), expected(ggml_nelements(ordinary)), actual(expected.size());
                std::vector<int32_t> selected(ggml_nelements(ids)), mapped(selected.size());
                for (int pass = 0; pass < 6; ++pass) {
                    for (size_t i = 0; i < input.size(); ++i) { input[i] = float(int((i+pass)%19)-9)/16; }
                    for (size_t i = 0; i < selected.size(); ++i) { selected[i] = pass == 1 ? 0 : pass == 2 ? experts - 1 : int((i*7+pass)%experts); }
                    if (pass == 4) { selected[0] = -1; }
                    if (pass == 5) { selected[0] = experts; }
                    for (size_t i = 0; i < selected.size(); ++i) { mapped[i] = slot_order && selected[i] >= 0 && selected[i] < experts ? experts - 1 - selected[i] : selected[i]; }
                    ggml_backend_tensor_set(mapped_ids, mapped.data(), 0, ggml_nbytes(mapped_ids));
                    ggml_backend_tensor_set(x, input.data(), 0, ggml_nbytes(x));
                    ggml_backend_tensor_set(ids, selected.data(), 0, ggml_nbytes(ids));
                    assert(ggml_backend_graph_compute(backend, gfull) == GGML_STATUS_SUCCESS);
                    assert(ggml_backend_graph_compute(backend, gmapped) == GGML_STATUS_SUCCESS);
                    assert(ggml_backend_graph_compute(backend, gsplit) == GGML_STATUS_SUCCESS);
                    ggml_backend_tensor_get(ordinary, expected.data(), 0, ggml_nbytes(ordinary));
                    ggml_backend_tensor_get(split, actual.data(), 0, ggml_nbytes(split));
                    std::vector<float> relocated(actual.size());
                    ggml_backend_tensor_get(permuted, relocated.data(), 0, ggml_nbytes(permuted));
                    assert(std::memcmp(relocated.data(), actual.data(), ggml_nbytes(split)) == 0);
                    size_t changed = 0;
                    float max_delta = 0;
                    for (size_t i = 0; i < expected.size(); ++i) {
                        changed += std::memcmp(&expected[i], &actual[i], sizeof(float)) != 0;
                        max_delta = std::max(max_delta, std::abs(expected[i] - actual[i]));
                    }
                    if (changed) {
                        std::printf("slot order: k=%d rate=%d rows=%d pass=%d changed=%zu/%zu max_delta=%g\n", k, rate, rows, pass, changed, expected.size(), max_delta);
                        exact = false;
                    }
                }
                ggml_backend_buffer_free(buffer); ggml_free(ctx);
            }
        }
    }
    ggml_backend_free(backend);
    std::fflush(stdout);
    assert(exact);
    std::puts("split EXL3 banks: exact GEMV/GEMM, changing captured routes and invalid-ID bounds");
}

static void auxiliary_upload() {
    auto * backend = ggml_backend_cuda_init(0);
    auto * reg = ggml_backend_dev_backend_reg(ggml_backend_get_device(backend));
    auto upload = (ggml_exl3_aux_upload_t) ggml_backend_reg_get_proc_address(reg, "ggml_backend_exl3_aux_upload");
    assert(upload);
    for (int base_slots : {0, 37}) {
        auto * ctx = ggml_init({1024*1024, nullptr, true});
        auto * scratch = ggml_new_tensor_1d(ctx, GGML_TYPE_I8, GGML_EXL3_AUX_SCRATCH_BYTES);
        ggml_exl3_aux_batch batch{};
        batch.n_slots = 83;
        const int widths[] = {2560, 640, 2560, 640, 640, 2560};
        std::vector<ggml_tensor *> banks;
        for (int i = 0; i < GGML_EXL3_AUX_BANKS; ++i) {
            batch.banks[i] = ggml_new_tensor_3d(ctx, GGML_TYPE_F16, widths[i], 1, (base_slots ? base_slots : batch.n_slots)+1);
            banks.push_back(batch.banks[i]);
        }
        std::vector<ggml_tensor *> extra_banks;
        ggml_tensor * table = nullptr;
        if (base_slots) {
            for (int i = 0; i < GGML_EXL3_AUX_BANKS; ++i) {
                extra_banks.push_back(ggml_new_tensor_3d(ctx, GGML_TYPE_F16, widths[i], 1, batch.n_slots-base_slots+1));
                banks.push_back(extra_banks.back());
            }
            // columns in reverse bank order, so a column/bank mix-up shows
            table = ggml_new_tensor_2d(ctx, GGML_TYPE_I64, GGML_EXL3_AUX_BANKS, batch.n_slots);
            batch.slot_table = table;
            for (int i = 0; i < GGML_EXL3_AUX_BANKS; ++i) { batch.table_cols[i] = GGML_EXL3_AUX_BANKS - 1 - i; }
        }
        auto * buffer = ggml_backend_alloc_ctx_tensors(ctx, backend);
        std::vector<int64_t> addresses;
        if (table) {
            addresses.resize(GGML_EXL3_AUX_BANKS*batch.n_slots);
            for (int s = 0; s < batch.n_slots; ++s) {
                for (int i = 0; i < GGML_EXL3_AUX_BANKS; ++i) {
                    const auto * bank = s < base_slots ? batch.banks[i] : extra_banks[i];
                    const int local = s < base_slots ? s : s - base_slots;
                    addresses[size_t(s)*GGML_EXL3_AUX_BANKS + batch.table_cols[i]] = (int64_t) (intptr_t) ((const char *) bank->data + size_t(local)*bank->nb[2]);
                }
            }
            ggml_backend_tensor_set(table, addresses.data(), 0, ggml_nbytes(table));
        }
        auto * host_buft = ggml_backend_dev_host_buffer_type(ggml_backend_get_device(backend));
        auto * host = ggml_backend_buft_alloc_buffer(host_buft, GGML_EXL3_AUX_SCRATCH_BYTES);
        assert(buffer && host && ggml_backend_buffer_get_type(host) == host_buft);
        auto * payload = (uint16_t *) ggml_backend_buffer_get_base(host);
        for (int count : {0, 1, 3, 10, 80}) {
            ggml_backend_buffer_clear(buffer, 0x5a);
            if (table) { ggml_backend_tensor_set(table, addresses.data(), 0, ggml_nbytes(table)); } // the clear overwrote it
            batch.count = count;
            for (int i = 0; i < count; ++i) { batch.slots[i] = (i*17+3)%batch.n_slots; }
            std::vector<std::vector<uint16_t>> expected;
            for (auto * bank : banks) { expected.emplace_back(ggml_nelements(bank), 0x5a5a); }
            size_t words = 0;
            for (int i = 0; i < GGML_EXL3_AUX_BANKS; ++i) {
                for (int j = 0; j < count; ++j) {
                    const bool extra = base_slots && batch.slots[j] >= base_slots;
                    const int bank = i + (extra ? GGML_EXL3_AUX_BANKS : 0);
                    const int slot = batch.slots[j] - (extra ? base_slots : 0);
                    for (int k = 0; k < widths[i]; ++k) {
                        const uint16_t value = uint16_t(i*32713+j*131+k*17);
                        payload[words++] = value;
                        expected[bank][slot*widths[i]+k] = value;
                    }
                }
            }
            assert(upload(backend, scratch, payload, words*2, &batch));
            ggml_backend_synchronize(backend);
            auto check = [&] {
                for (size_t i = 0; i < banks.size(); ++i) {
                    std::vector<uint16_t> actual(expected[i].size());
                    ggml_backend_tensor_get(banks[i], actual.data(), 0, actual.size()*2);
                    assert(actual == expected[i]);
                }
            };
            check();
            if (count) {
                auto invalid = batch;
                invalid.count = GGML_EXL3_AUX_MAX_FILLS+1;
                assert(!upload(backend, scratch, payload, words*2, &invalid));
                invalid = batch; invalid.slots[0] = -1;
                assert(!upload(backend, scratch, payload, words*2, &invalid));
                invalid.slots[0] = batch.n_slots;
                assert(!upload(backend, scratch, payload, words*2, &invalid));
                invalid = batch; invalid.banks[1] = invalid.banks[0];
                assert(!upload(backend, scratch, payload, words*2, &invalid));
                assert(!upload(backend, scratch, payload, words*2-2, &batch));
                assert(!upload(backend, scratch, payload, GGML_EXL3_AUX_SCRATCH_BYTES+1, &batch));
                if (count > 1) {
                    invalid = batch; invalid.slots[1] = invalid.slots[0];
                    assert(!upload(backend, scratch, payload, words*2, &invalid));
                }
                if (base_slots) {
                    invalid = batch; invalid.table_cols[1] = invalid.table_cols[0];
                    assert(!upload(backend, scratch, payload, words*2, &invalid));
                    invalid = batch; invalid.table_cols[0] = GGML_EXL3_AUX_BANKS;
                    assert(!upload(backend, scratch, payload, words*2, &invalid));
                    invalid = batch; invalid.n_slots = batch.n_slots - 1;
                    assert(!upload(backend, scratch, payload, words*2, &invalid));
                }
                ggml_backend_synchronize(backend); check();
            }
        }
        ggml_backend_buffer_free(host); ggml_backend_buffer_free(buffer); ggml_free(ctx);
    }
    ggml_backend_free(backend);
    std::puts("auxiliary uploads: exact base/extension bytes and guards, malformed requests rejected");
}

int main(int argc, char ** argv) {
    CUDA_CHECK(cudaSetDevice(0));
    hipDeviceProp_t prop;
    CUDA_CHECK(hipGetDeviceProperties(&prop, 0));
    assert(std::strncmp(prop.gcnArchName, "gfx1201", 7) == 0);
    if (argc == 2 && std::strcmp(argv[1], "--split-banks") == 0) { split_bank_projection(); }
    else if (argc == 2 && std::strcmp(argv[1], "--slot-order") == 0) { split_bank_projection(true); }
    else if (argc == 2 && std::strcmp(argv[1], "--aux-upload") == 0) { auxiliary_upload(); }
    else if (argc == 3 && std::strcmp(argv[1], "--codec") == 0) { codec(argv[2]); }
    else if (argc == 5) { projection(argv[2], argv[3], argv[4]); }
    else { std::fprintf(stderr, "usage: %s --split-banks | --slot-order | --aux-upload | --codec FIXTURE | --bundle BANK INPUT OUTPUT_PREFIX\n", argv[0]); return 1; }
    return 0;
}
