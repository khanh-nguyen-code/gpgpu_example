__kernel void vector_add(int n, __global float *c, __global const float *a, __global const float *b) {
    int i = get_global_id(0);
    if (i < n) {
        c[i] = a[i] + b[i];
    }
}
