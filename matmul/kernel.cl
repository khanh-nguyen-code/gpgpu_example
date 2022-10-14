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
    // reduce sum
    int work_size = k_d;
    while (work_size > 1) {
        int step_size = (work_size + 1) / 2;
        if (k < step_size) {
            int k2 = k + step_size;
            if (k2 < work_size) {
                tmp[k] = tmp[k] + tmp[k2];
            }
        }
        work_size = step_size;
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    // write
    c[i * j_d + j] = tmp[0];
}