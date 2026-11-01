#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <limits>
#include <stdexcept>
#include <string>
#include <tuple>
#include <vector>

#define GGML_USE_HIP 1
#define GGML_UNUSED_VARS(...) ((void) std::tie(__VA_ARGS__))
#define NO_DEVICE_CODE __builtin_trap()
using std::min;
#include "../reference/gemm-wmma.standalone.cuh"

static void check(hipError_t e, const char * what) {
    if (e != hipSuccess) {
        throw std::runtime_error(std::string(what) + ": " + hipGetErrorString(e));
    }
}
#define HIP(x) check((x), #x)

struct shape {
    unsigned m, n, k;
    unsigned gm() const {
        unsigned rows = (m + 127)/128, g = 1;
        while (g < 8 && 2*g <= rows) { g *= 2; }
        return g;
    }
    unsigned group_log2() const {
        unsigned g = gm(), r = 0;
        while (g >>= 1) { ++r; }
        return r;
    }
    void validate() const {
        if (!m || !n || !k || k % 128 || m > 65536 || n > 65536 || k > 65536) {
            throw std::runtime_error("expected positive m,n,k <= 65536 and k divisible by 128");
        }
        const uint64_t mp = ((uint64_t(m)+127)/128)*128;
        const uint64_t np = ((uint64_t(n)+127)/128)*128;
        // Bounds include addresses from masked or zero-filled edge lanes.
        if (mp*k*2 + 64 >= UINT32_MAX || np*k*2 + 64 >= UINT32_MAX || mp*np*4 >= UINT32_MAX) {
            throw std::runtime_error("32-bit buffer offset overflow");
        }
    }
};

static uint32_t next_random(uint32_t & s) {
    s ^= s << 13; s ^= s >> 17; s ^= s << 5;
    return s;
}

struct allocations {
    shape dims;
    unsigned banks;
    std::vector<half> hx, hw;
    half * x = nullptr, * w = nullptr;
    float * reference = nullptr, * candidate = nullptr;
    size_t nx, nw, ny;
    allocations(shape d, unsigned count, bool dyadic) : dims(d), banks(count), nx(size_t(d.m)*d.k), nw(size_t(d.n)*d.k), ny(size_t(d.m)*d.n) {
        hx.resize(nx*banks); hw.resize(nw*banks);
        uint32_t seed = 0x6a09e667;
        for (auto * vec : {&hx, &hw}) {
            for (auto & value : *vec) {
                uint32_t bits = next_random(seed);
                float v = dyadic ? float(int(bits % 33)-16)/16.0f : float(int(bits % 65537)-32768)/32768.0f;
                value = __float2half(v);
            }
        }
        HIP(hipMalloc(&x, nx*banks*sizeof(half)));
        HIP(hipMalloc(&w, nw*banks*sizeof(half)));
        HIP(hipMalloc(&reference, ny*banks*sizeof(float)));
        HIP(hipMalloc(&candidate, ny*banks*sizeof(float)));
        HIP(hipMemcpy(x, hx.data(), hx.size()*sizeof(half), hipMemcpyHostToDevice));
        HIP(hipMemcpy(w, hw.data(), hw.size()*sizeof(half), hipMemcpyHostToDevice));
    }
    ~allocations() {
        (void) hipFree(candidate); (void) hipFree(reference); (void) hipFree(w); (void) hipFree(x);
    }
    allocations(const allocations &) = delete;
    allocations & operator=(const allocations &) = delete;
};

static void launch(bool candidate, hipFunction_t function, allocations & a, unsigned bank, hipStream_t stream) {
    auto d = a.dims;
    auto * x = a.x + a.nx*bank;
    auto * w = a.w + a.nw*bank;
    auto * y = (candidate ? a.candidate : a.reference) + a.ny*bank;
    if (candidate) {
        unsigned group_log2 = d.group_log2();
        void * args[] = {&x, &w, &y, &d.m, &d.n, &d.k, &group_log2};
        HIP(hipModuleLaunchKernel(function, ((d.n+127)/128)*d.gm(), (((d.m+127)/128)+d.gm()-1)/d.gm(), 1,
                                  128, 1, 1, 0, stream, args, nullptr));
    } else {
        hipLaunchKernelGGL((gemm_f16_wmma<128, false>), dim3(((d.m+127)/128)*((d.n+127)/128)), dim3(128), 0, stream,
                           x, w, nullptr, y, int(d.m), int(d.n), int(d.k), 0);
        HIP(hipGetLastError());
    }
}

static void compare(allocations & a, bool dyadic, const char * phase) {
    std::vector<float> ref(a.ny), got(a.ny);
    double worst = 0;
    uint64_t checked = 0;
    for (unsigned bank = 0; bank < a.banks; ++bank) {
        HIP(hipMemcpy(ref.data(), a.reference + bank*a.ny, a.ny*sizeof(float), hipMemcpyDeviceToHost));
        HIP(hipMemcpy(got.data(), a.candidate + bank*a.ny, a.ny*sizeof(float), hipMemcpyDeviceToHost));
        for (size_t i = 0; i < a.ny; ++i) {
            if (!std::isfinite(got[i]) || !std::isfinite(ref[i]) || std::memcmp(&got[i], &ref[i], sizeof(float))) {
                std::fprintf(stderr, "mismatch %s bank=%u row=%zu col=%zu hip=%a asm=%a\n", phase, bank, i/a.dims.n, i%a.dims.n, ref[i], got[i]);
                throw std::runtime_error("exact HIP/assembly gate failed");
            }
        }
        // Check all small outputs and a deterministic sample of each large output against FP64.
        size_t probes = a.ny <= 20000 ? a.ny : 257;
        for (size_t p = 0; p < probes; ++p) {
            size_t i = probes == a.ny ? p : p*(a.ny-1)/(probes-1);
            size_t row = i/a.dims.n, col = i%a.dims.n;
            double expected = 0;
            for (unsigned k = 0; k < a.dims.k; ++k) {
                expected += double(__half2float(a.hx[bank*a.nx + row*a.dims.k+k])) *
                            double(__half2float(a.hw[bank*a.nw + col*a.dims.k+k]));
            }
            double error = std::abs(double(got[i])-expected);
            worst = std::max(worst, error);
            if ((dyadic && error != 0) || (!dyadic && error > 2e-4 + 2e-4*std::abs(expected))) {
                throw std::runtime_error("independent FP64 reference gate failed");
            }
            ++checked;
        }
    }
    std::printf("gate phase=%s m=%u n=%u k=%u banks=%u exact=pass fp64_probes=%llu max_abs=%.9g\n",
                phase, a.dims.m, a.dims.n, a.dims.k, a.banks, (unsigned long long)checked, worst);
}

static void qualify(hipFunction_t f, allocations & a, bool dyadic, hipStream_t stream, bool graph) {
    HIP(hipMemsetAsync(a.reference, 0xff, a.ny*a.banks*sizeof(float), stream));
    HIP(hipMemsetAsync(a.candidate, 0xff, a.ny*a.banks*sizeof(float), stream));
    for (unsigned b = 0; b < a.banks; ++b) { launch(false, f, a, b, stream); launch(true, f, a, b, stream); }
    HIP(hipStreamSynchronize(stream));
    compare(a, dyadic, "direct");
    for (int repeat = 0; repeat < 3; ++repeat) {
        HIP(hipMemsetAsync(a.candidate, 0xff, a.ny*a.banks*sizeof(float), stream));
        for (unsigned b = 0; b < a.banks; ++b) { launch(true, f, a, b, stream); }
        HIP(hipStreamSynchronize(stream));
        compare(a, dyadic, "repeat");
    }
    if (graph) {
        hipGraph_t g;
        hipGraphExec_t exec;
        HIP(hipStreamBeginCapture(stream, hipStreamCaptureModeThreadLocal));
        for (unsigned b = 0; b < a.banks; ++b) { launch(true, f, a, b, stream); }
        HIP(hipStreamEndCapture(stream, &g));
        HIP(hipGraphInstantiate(&exec, g, nullptr, nullptr, 0));
        for (int repeat = 0; repeat < 3; ++repeat) {
            HIP(hipMemsetAsync(a.candidate, 0xff, a.ny*a.banks*sizeof(float), stream));
            HIP(hipGraphLaunch(exec, stream));
            HIP(hipStreamSynchronize(stream));
            compare(a, dyadic, "graph");
        }
        HIP(hipGraphExecDestroy(exec)); HIP(hipGraphDestroy(g));
    }
}

static void benchmark(hipFunction_t f, allocations & a, hipStream_t stream, unsigned rounds, unsigned reps, std::ofstream & csv) {
    for (unsigned warm = 0; warm < 8; ++warm) {
        for (unsigned b = 0; b < a.banks; ++b) { launch(false, f, a, b, stream); launch(true, f, a, b, stream); }
    }
    HIP(hipStreamSynchronize(stream));
    hipEvent_t begin, end;
    HIP(hipEventCreate(&begin)); HIP(hipEventCreate(&end));
    std::vector<float> times[2];
    for (unsigned round = 0; round < rounds; ++round) {
        for (unsigned order = 0; order < 4; ++order) {
            // Reverse the ABBA cycle every other round.
            bool candidate = ((order == 1 || order == 2) != bool(round % 2));
            HIP(hipEventRecord(begin, stream));
            for (unsigned r = 0; r < reps; ++r) {
                for (unsigned b = 0; b < a.banks; ++b) { launch(candidate, f, a, b, stream); }
            }
            HIP(hipEventRecord(end, stream)); HIP(hipEventSynchronize(end));
            float ms;
            HIP(hipEventElapsedTime(&ms, begin, end));
            ms /= reps*a.banks;
            times[candidate].push_back(ms);
            csv << a.dims.m << ',' << a.dims.n << ',' << a.dims.k << ',' << round << ',' << order << ','
                << (candidate ? "assembly" : "hip") << ',' << a.banks << ',' << reps << ',' << ms << '\n';
            csv.flush();
        }
    }
    for (int arm = 0; arm < 2; ++arm) {
        auto & t = times[arm]; std::sort(t.begin(), t.end());
        float ms = (t[(t.size()-1)/2] + t[t.size()/2])/2;
        std::printf("timing arm=%s m=%u n=%u k=%u median_ms=%.6f tflops=%.3f\n", arm ? "assembly" : "hip",
                    a.dims.m, a.dims.n, a.dims.k, ms, 2.0*a.dims.m*a.dims.n*a.dims.k/(ms*1e9));
    }
    HIP(hipEventDestroy(end)); HIP(hipEventDestroy(begin));
}

int main(int argc, char ** argv) try {
    std::string mode, codeobject, symbol, samples;
    unsigned rounds = 12, reps = 4, banks = 8;
    bool graph = false;
    std::vector<shape> shapes;
    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "--help") {
            std::puts("ASM0 isolated gfx1201 FP16 WMMA experiment (unqualified)\n"
                      "--check|--bench --codeobject FILE --symbol asm0_strict|asm0_pipeline\n"
                      "[--shape M N K] [--banks 1..8] [--rounds 1..100] [--reps 1..100] [--graph]\n"
                      "--bench requires --samples FILE; all shapes pass exact and FP64 gates before timing.\n"
                      "--check defaults to small/tail suite; --bench defaults to three historical shapes.\n"
                      "--help and --describe do not initialize HIP.");
            return 0;
        }
        if (arg == "--describe") {
            std::puts("target=gfx1201 precision=f16xf16->f32 tile=128x128x32 threads=128 lds=24576 vgpr=240 scratch=0 device_qualified=false");
            return 0;
        }
        auto value = [&]() -> const char * {
            if (++i >= argc) { throw std::runtime_error("missing argument value"); }
            return argv[i];
        };
        auto number = [&]() -> unsigned {
            const char * s = value();
            char * end = nullptr;
            unsigned long result = std::strtoul(s, &end, 10);
            if (!*s || *end || result > 65536) { throw std::runtime_error("invalid integer argument"); }
            return unsigned(result);
        };
        if (arg == "--check" || arg == "--bench") { mode = arg; }
        else if (arg == "--codeobject") { codeobject = value(); }
        else if (arg == "--symbol") { symbol = value(); }
        else if (arg == "--samples") { samples = value(); }
        else if (arg == "--banks") { banks = number(); }
        else if (arg == "--rounds") { rounds = number(); }
        else if (arg == "--reps") { reps = number(); }
        else if (arg == "--graph") { graph = true; }
        else if (arg == "--shape") { shape d; d.m = number(); d.n = number(); d.k = number(); shapes.push_back(d); }
        else { throw std::runtime_error("unknown argument " + arg); }
    }
    if (mode.empty() || codeobject.empty() || (symbol != "asm0_strict" && symbol != "asm0_pipeline")) {
        throw std::runtime_error("specify mode, codeobject and known symbol; see --help");
    }
    if (!banks || banks > 8 || !rounds || rounds > 100 || !reps || reps > 100) { throw std::runtime_error("invalid benchmark count"); }
    if (shapes.empty()) {
        if (mode == "--bench") { shapes = {{4096, 12288, 2560}, {4096, 2560, 6144}, {512, 12288, 2560}}; }
        else { shapes = {{1,1,128}, {15,17,128}, {17,15,256}, {127,129,128}, {129,127,384}, {129,129,128}, {257,17,256}, {1025,1,128}}; }
    }
    for (const auto & d : shapes) { d.validate(); }
    std::ofstream csv;
    if (mode == "--bench") {
        if (samples.empty()) { throw std::runtime_error("--bench requires --samples"); }
        csv.open(samples, std::ios::out | std::ios::app);
        if (!csv) { throw std::runtime_error("cannot open samples"); }
        if (csv.tellp() == 0) { csv << "m,n,k,round,order,arm,banks,reps,ms\n"; }
        csv.precision(9);
    }
    hipDeviceProp_t props;
    HIP(hipGetDeviceProperties(&props, 0));
    if (std::string(props.gcnArchName).substr(0, 7) != "gfx1201") { throw std::runtime_error("requires gfx1201"); }
    std::printf("device=%s arch=%s symbol=%s\n", props.name, props.gcnArchName, symbol.c_str());
    hipModule_t module;
    hipFunction_t function;
    HIP(hipModuleLoad(&module, codeobject.c_str())); HIP(hipModuleGetFunction(&function, module, symbol.c_str()));
    hipStream_t stream; HIP(hipStreamCreateWithFlags(&stream, hipStreamNonBlocking));
    for (const auto & d : shapes) {
        for (bool dyadic : {true, false}) {
            allocations a(d, mode == "--bench" ? banks : 1, dyadic);
            qualify(function, a, dyadic, stream, graph);
            if (mode == "--bench" && !dyadic) { benchmark(function, a, stream, rounds, reps, csv); }
        }
    }
    HIP(hipStreamDestroy(stream)); HIP(hipModuleUnload(module));
    return 0;
} catch (const std::exception & e) {
    std::fprintf(stderr, "ASM0: %s\n", e.what()); return 1;
}
