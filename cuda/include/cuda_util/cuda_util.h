#ifndef __CUDA_UTIL_H__
#define __CUDA_UTIL_H__

#include<cuda_runtime_api.h>
#include<iostream>
#define cudaOk(err, ...) cudaAssert(err, __FILE__, __LINE__, true);

const cudaError_t cudaAssert(const cudaError_t err, const char* file, const int line, bool abort) {
    if (err != cudaSuccess) {
        std::cerr << file << ":" << line << " error: " << err << " " << cudaGetErrorString(err) << std::endl;
        if (abort) {
            cudaDeviceReset();
            std::exit(1);
        }
    }
    return err;
}

#endif // __CUDA_UTIL_H__