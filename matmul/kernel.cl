__kernel void matmul(__global double *c, __global const double *a, __global double *b, __local double *tmp) {
    int i = get_global_id(0); // row
    int j = get_global_id(2); // col
    int k = get_local_id(1);
    int i_d = get_global_size(0);
    int j_d = get_global_size(2);
    int k_d = get_local_size(1);
    // calculate
    tmp[k] = a[i * k_d + k] * b[k * j_d + j];
    barrier(CLK_LOCAL_MEM_FENCE);
    if (k == 0) {
        // reduce sum
        double sum = 0.0;
        for (int k=0; k<k_d; k++) {
            sum += tmp[k];
        }
        // write
        c[i * j_d + j] = sum;
    }
}