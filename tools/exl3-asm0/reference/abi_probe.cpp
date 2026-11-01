#include <hip/hip_runtime.h>
extern "C" __global__ void asm0_abi_probe(const void * a, const void * b, float * c, unsigned m, unsigned n, unsigned k, unsigned group_log2) {
    if (threadIdx.x == 0) c[blockIdx.y * n + blockIdx.x] = float(m + k + group_log2) + float((unsigned long long)a == (unsigned long long)b);
    __syncthreads();
}
