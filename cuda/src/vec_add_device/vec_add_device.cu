#include"vec_add_device/vec_add_device.h"

__global__ void vec_add_kernel(const uint64_t n, float *c, const float *a, const float *b) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
 
    if (i < n) {
        c[i] = a[i] + b[i];
    }
}

void vec_add_device(cudaStream_t stream, const uint64_t n, float *d_c, const float *d_a, const float *d_b) {
    uint64_t blockSize = 1;
    uint64_t gridSize = n;
    vec_add_kernel<<<gridSize, blockSize, 0, stream>>>(n, d_c, d_a, d_b);
}