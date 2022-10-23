__kernel void vector_add(__global float *c, __global const float *a, __global const float *b) {
    int i = get_global_id(0);
    c[i] = a[i] + b[i];
}