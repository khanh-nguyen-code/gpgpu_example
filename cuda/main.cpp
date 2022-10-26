#include<cstdint>
#include<random>
#include<vector>
#include<iostream>
#include<cuda_runtime_api.h>
#include"vec_add_device.h"

const int n = 1024;

std::default_random_engine random_engine;

template<typename T>
void random_seed(T seed) {
    random_engine = std::default_random_engine(seed);
}

template<typename T>
std::vector<T> random_normal(int size) {
    std::vector<T> vec(size);
    std::normal_distribution<T> dist(0.0, 1.0);
    for (int i=0; i<size; i++) {
        vec[i] = dist(random_engine);
    }
    return vec;
}

template<typename T>
std::vector<T> vec_add(const std::vector<T>& a, const std::vector<T>& b) {
    if (a.size() != b.size()) {
        std::cerr << "size error" << std::endl;
        std::exit(1);
    }

    std::vector<T> c(a.size());

    for (uint64_t i=0; i<c.size(); i++) {
        c[i] = a[i] + b[i];
    }
    return c;
}

template<typename T>
T vec_diff(const std::vector<T>& a, const std::vector<T>& b) {
    auto max = [](T a, T b) -> T {
        return (a >= b) ? a : b;
    };
    auto abs = [&max](T x) -> T {
        return max(x, -x);
    };
    if (a.size() != b.size()) {
        return std::numeric_limits<T>::infinity();
    }
    T max_diff = 0;
    for (size_t i=0; i<a.size(); i++) {
        T diff = abs(a[i] - b[i]);
        if (diff > max_diff) {
            max_diff = diff;
        }
    }
    return max_diff;
}

void cudaOk(cudaError_t err) {
    if (err != cudaSuccess) {
        std::cerr << "error: " << err << std::endl;
        std::exit(1);
    }
}

int main( int argc, char* argv[] ) {

    std::vector<float> a = random_normal<float>(n);
    std::vector<float> b = random_normal<float>(n);
    std::vector<float> c(n);

    cudaStream_t stream;
    cudaOk(cudaStreamCreate(&stream));


    float *d_a, *d_b, *d_c;
    
    cudaOk(cudaMalloc((void**)&d_a, a.size() * sizeof(float)));
    cudaOk(cudaMalloc((void**)&d_b, b.size() * sizeof(float)));
    cudaOk(cudaMalloc((void**)&d_c, c.size() * sizeof(float)));
 
    cudaOk(cudaMemcpyAsync(d_a, a.data(), a.size() * sizeof(float), cudaMemcpyHostToDevice, stream));
    cudaOk(cudaMemcpyAsync(d_b, b.data(), b.size() * sizeof(float), cudaMemcpyHostToDevice, stream));


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

    cudaOk(cudaMemcpyAsync(c.data(), d_c, c.size() * sizeof(float), cudaMemcpyDeviceToHost, stream));
 
    cudaOk(cudaStreamSynchronize(stream));

    cudaOk(cudaStreamDestroy(stream));

    cudaOk(cudaFree(d_a));
    cudaOk(cudaFree(d_b));
    cudaOk(cudaFree(d_c));

    auto c_host = vec_add(a, b);
 
    std::cout << "max diff: " << vec_diff(c, c_host) << std::endl;

    return 0;
}
