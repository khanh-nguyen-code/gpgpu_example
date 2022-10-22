__kernel void vector_add(int n, __global double *c, __global const double *a, __global const double *b) {
    int i = get_global_id(0);
    if (i < n) {
        c[i] = a[i] + b[i];
    }
}
