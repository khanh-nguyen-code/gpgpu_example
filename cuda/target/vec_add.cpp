#include<cstdint>
#include<random>
#include<vector>
#include<iostream>
#include<cuda_runtime_api.h>
#include"vec_add_device/vec_add_device.h"
#include"cuda_util/cuda_util.h"

const size_t n = 1024;

std::default_random_engine random_engine;

template<typename T>
void random_seed(const T seed) {
    random_engine = std::default_random_engine(seed);
}
template<typename T>
void random_normal_inplace(const size_t n, T* vec) {
    std::normal_distribution<T> dist(0.0, 1.0);
    for (size_t i=0; i<n; i++) {
        vec[i] = dist(random_engine);
    }
}

template<typename T>
void vec_add_inplace(const size_t n, T* c, const T* a, const T* b) {
    for (size_t i=0; i<n; i++) {
        c[i] = a[i] + b[i];
    }
}

template<typename T>
T vec_diff(const size_t n, const T* a, const T* b) {
    auto max = [](T a, T b) -> T {
        return (a >= b) ? a : b;
    };
    auto abs = [&max](T x) -> T {
        return max(x, -x);
    };
    T max_diff = 0;
    for (size_t i=0; i<n; i++) {
        T diff = abs(a[i] - b[i]);
        if (diff > max_diff) {
            max_diff = diff;
        }
    }
    return max_diff;
}




int main( int argc, char* argv[] ) {

    float* a = (float*)std::malloc(n * sizeof(float));
    float* b = (float*)std::malloc(n * sizeof(float));
    float* c = (float*)std::malloc(n * sizeof(float));
    
    random_normal_inplace<float>(n, a);
    random_normal_inplace<float>(n, b);

    float *d_a, *d_b, *d_c;
    
    cudaOk(cudaMalloc((void**)&d_a, n * sizeof(float)));
    cudaOk(cudaMalloc((void**)&d_b, n * sizeof(float)));
    cudaOk(cudaMalloc((void**)&d_c, n * sizeof(float)));
 
    cudaStream_t stream;
    cudaOk(cudaStreamCreate(&stream));
    
    cudaOk(cudaMemcpyAsync(d_a, a, n * sizeof(float), cudaMemcpyHostToDevice, stream));
    cudaOk(cudaMemcpyAsync(d_b, b, n * sizeof(float), cudaMemcpyHostToDevice, stream));

    cudaEvent_t writeEvent;
    cudaOk(cudaEventCreate(&writeEvent));
    cudaOk(cudaEventRecord(writeEvent, stream));
    cudaOk(cudaStreamWaitEvent(stream, writeEvent));
    cudaOk(cudaEventDestroy(writeEvent));

    vec_add_device(stream, n, d_c, d_a, d_b);

    cudaEvent_t kernelEvent;
    cudaOk(cudaEventCreate(&kernelEvent));
    cudaOk(cudaEventRecord(kernelEvent, stream));
    cudaOk(cudaStreamWaitEvent(stream, kernelEvent));
    cudaOk(cudaEventDestroy(kernelEvent));

    cudaOk(cudaMemcpyAsync(c, d_c, n * sizeof(float), cudaMemcpyDeviceToHost, stream));
 
    cudaOk(cudaStreamSynchronize(stream));

    cudaOk(cudaStreamDestroy(stream));

    cudaOk(cudaFree(d_a));
    cudaOk(cudaFree(d_b));
    cudaOk(cudaFree(d_c));

    float* c_host = (float*)std::malloc(n * sizeof(float));

    vec_add_inplace(n, c_host, a, b);
 
    std::cout << "max diff: " << vec_diff(n, c, c_host) << std::endl;

    std::free(a);
    std::free(b);
    std::free(c);
    std::free(c_host);
    return 0;
}
