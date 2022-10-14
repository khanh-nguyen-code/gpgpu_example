__kernel void matmul(const int k_d, __global double *c, __global const double *a, __global const double *b) {
    int i = get_global_id(0); // row
    int j = get_global_id(1); // column
    int i_d = get_global_size(0);
    int j_d = get_global_size(1);
    // dot
    double sum = 0.0;
    for (int k=0; k<k_d; k++) {
        sum += a[i * k_d + k] * b[k * j_d + j];
    }
    // write
    c[i * j_d + j] = sum;
}