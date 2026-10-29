#include "ggml-cuda/exl3.cuh"
#include "ggml-exl3.h"
#include "gguf.h"

#undef NDEBUG
#include <algorithm>
#include <cassert>
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

static void auxiliary_upload() {
    auto * backend = ggml_backend_cuda_init(0);
    auto * reg = ggml_backend_dev_backend_reg(ggml_backend_get_device(backend));
    auto upload = (ggml_exl3_aux_upload_t) ggml_backend_reg_get_proc_address(reg, "ggml_backend_exl3_aux_upload");
    assert(upload);
    auto * ctx = ggml_init({1024*1024, nullptr, true});
    auto * scratch = ggml_new_tensor_1d(ctx, GGML_TYPE_I8, GGML_EXL3_AUX_SCRATCH_BYTES);
    ggml_exl3_aux_batch batch{};
    batch.n_slots = 83;
    const int widths[] = {2560, 640, 2560, 640, 640, 2560};
    for (int i = 0; i < GGML_EXL3_AUX_BANKS; ++i) {
        batch.banks[i] = ggml_new_tensor_3d(ctx, GGML_TYPE_F16, widths[i], 1, batch.n_slots+1);
    }
    auto * buffer = ggml_backend_alloc_ctx_tensors(ctx, backend);
    auto * host_buft = ggml_backend_dev_host_buffer_type(ggml_backend_get_device(backend));
    auto * host = ggml_backend_buft_alloc_buffer(host_buft, GGML_EXL3_AUX_SCRATCH_BYTES);
    assert(buffer && host && ggml_backend_buffer_get_type(host) == host_buft);
    auto * payload = (uint16_t *) ggml_backend_buffer_get_base(host);
    for (int count : {0, 1, 3, 10, 80}) {
        ggml_backend_buffer_clear(buffer, 0x5a);
        batch.count = count;
        for (int i = 0; i < count; ++i) { batch.slots[i] = (i*17+3)%batch.n_slots; }
        std::vector<std::vector<uint16_t>> expected;
        size_t words = 0;
        for (int i = 0; i < GGML_EXL3_AUX_BANKS; ++i) {
            expected.emplace_back(ggml_nelements(batch.banks[i]), 0x5a5a);
            for (int j = 0; j < count; ++j) {
                for (int k = 0; k < widths[i]; ++k) {
                    const uint16_t value = uint16_t(i*32713+j*131+k*17);
                    payload[words++] = value;
                    expected.back()[batch.slots[j]*widths[i]+k] = value;
                }
            }
        }
        assert(upload(backend, scratch, payload, words*2, &batch));
        ggml_backend_synchronize(backend);
        auto check = [&] {
            for (int i = 0; i < GGML_EXL3_AUX_BANKS; ++i) {
                std::vector<uint16_t> actual(expected[i].size());
                ggml_backend_tensor_get(batch.banks[i], actual.data(), 0, actual.size()*2);
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
            ggml_backend_synchronize(backend);
            check();
        }
    }
    ggml_backend_buffer_free(host); ggml_backend_buffer_free(buffer); ggml_free(ctx); ggml_backend_free(backend);
    std::puts("auxiliary uploads: exact bytes, untouched slots and guards, malformed requests rejected");
}

int main(int argc, char ** argv) {
    CUDA_CHECK(cudaSetDevice(0));
    hipDeviceProp_t prop;
    CUDA_CHECK(hipGetDeviceProperties(&prop, 0));
    assert(std::strncmp(prop.gcnArchName, "gfx1201", 7) == 0);
    if (argc == 2 && std::strcmp(argv[1], "--aux-upload") == 0) { auxiliary_upload(); }
    else if (argc == 3 && std::strcmp(argv[1], "--codec") == 0) { codec(argv[2]); }
    else if (argc == 5) { projection(argv[2], argv[3], argv[4]); }
    else { std::fprintf(stderr, "usage: %s --aux-upload | --codec FIXTURE | --bundle BANK INPUT OUTPUT_PREFIX\n", argv[0]); return 1; }
    return 0;
}
