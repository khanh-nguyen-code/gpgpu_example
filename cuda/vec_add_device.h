#ifndef __VEC_ADD_DEVICE_H__
#define __VEC_ADD_DEVICE_H__

void vec_add_device(cudaStream_t stream, const uint64_t n, float *d_c, const float *d_a, const float *d_b);

#endif // __VEC_ADD_DEVICE_H__