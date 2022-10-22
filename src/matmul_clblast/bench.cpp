#include<iostream>
#include<vector>
#include<omp.h>
#include"util.h"
#include"high_precision_timer.h"

#define CL_TARGET_OPENCL_VERSION 200
#include"cl_util.h"
#include<clblast.h>
const int platform_id = 0;
const int device_id = 0;


cl_int code;
// (m x k) (k x n) -> (m x n)
const int m = 2048;
const int n = 2048;
const int k = 2048;
const double eps = 1e-6;


void matmul_clblast(
    cl_mem c, cl_mem a, cl_mem b, int m, int n, int k,
    cl_command_queue* queue, cl_event* event
) {
    // c <- alpha A B + beta C
    double alpha = 1.0;
    double beta = 0.0;
    clblast::StatusCode code = clblast::Gemm<double>(
        clblast::Layout::kRowMajor, clblast::Transpose::kNo, clblast::Transpose::kNo,
        m, n, k, 
        alpha,
        a, 0, k,
        b, 0, n,
        beta,
        c, 0, n,
        queue, event
    );
    if (code != clblast::StatusCode::kSuccess) {
        std::printf("clblast_gemm: %d\n", code);
        std::exit(1);
    }
}

bool vector_cmp(const std::vector<double>& a, const std::vector<double>& b) {
    if (a.size() != b.size()) {
        return false;
    }
    for (int i=0; i<a.size(); i++) {
        double diff = a[i] - b[i];
        diff = (diff >= 0) ? diff : -diff;
        if (diff > eps) {
            return false;
        }
    }
    return true;
}

void matmul(double* c, const double *a, const double* b, int m, int n, int k) {
    int m_i, n_i;
    # pragma omp parallel for shared(a, b, c) private(m_i, n_i) collapse(2)
    for (m_i=0; m_i < m; m_i++)
    for (n_i=0; n_i < n; n_i++) {
        double acc = 0;
        for (int k_i=0; k_i < k; k_i++) {
            acc += a[m_i * k + k_i] * b[k_i * n + n_i];
        }
        c[m_i * n + n_i] = acc;
    }
}


int main() {
    util::random_seed(1234);
    std::vector<double> a = util::random_normal(m * k);
    std::vector<double> b = util::random_normal(k * n);
    std::vector<double> c(m * n);

    cl_device_id device = cl_util::get_device(platform_id, device_id);
    cl_context context = clCreateContext(nullptr, 1, &device, nullptr, nullptr, &code);
    cl_util::assert("create_context", code);
    cl_command_queue queue = clCreateCommandQueueWithProperties(context, device, nullptr, &code);
    cl_util::assert("create_command_queue_with_properties", code);
    
    cl_mem a_buf = clCreateBuffer(context, CL_MEM_READ_ONLY, a.size() * sizeof(double), nullptr, &code);
    cl_util::assert("create_buffer_with_properties", code);
    cl_mem b_buf = clCreateBuffer(context, CL_MEM_READ_ONLY, b.size() * sizeof(double), nullptr, &code);
    cl_util::assert("create_buffer_with_properties", code);
    cl_mem c_buf = clCreateBuffer(context, CL_MEM_WRITE_ONLY, c.size() * sizeof(double), nullptr, &code);
    cl_util::assert("create_buffer_with_properties", code);
    
    // queue
    std::vector<cl_event> write_event(2);
    code = clEnqueueWriteBuffer(queue, a_buf, CL_NON_BLOCKING, 0, a.size() * sizeof(double), a.data(), 0, nullptr, &write_event[0]);
    cl_util::assert("enqueue_write_buffer", code);
    code = clEnqueueWriteBuffer(queue, b_buf, CL_NON_BLOCKING, 0, b.size() * sizeof(double), b.data(), 0, nullptr, &write_event[1]);
    cl_util::assert("enqueue_write_buffer", code);
    code = clWaitForEvents(write_event.size(), write_event.data());
    cl_util::assert("wait_for_events", code);

    auto t1 = timer::now();
    
    cl_event kernel_event;
    matmul_clblast(c_buf, a_buf, b_buf, m, n, k, &queue, &kernel_event);

    code = clWaitForEvents(1, &kernel_event);
    cl_util::assert("wait_for_events", code);
    auto t2 = timer::now();
    
    cl_event read_event;
    code = clEnqueueReadBuffer(queue, c_buf, CL_NON_BLOCKING, 0, c.size() * sizeof(double), c.data(), 1, &kernel_event, &read_event);
    cl_util::assert("enqueue_read_buffer", code);

    code = clWaitForEvents(1, &read_event);
    cl_util::assert("wait_for_events", code);
    
    clFinish(queue);
    for (auto& event: write_event) {
        clReleaseEvent(event);
    }
    clReleaseEvent(kernel_event);
    clReleaseEvent(read_event); 
    // compare
    std::vector<double> c_host(c.size());
    auto t3 = timer::now();
    matmul(c_host.data(), a.data(), b.data(), m, n, k);
    auto t4 = timer::now();
    if (vector_cmp(c, c_host)) {
        std::printf("result ok\n");
        std::printf("cl  time:\t%lld\n", (t2-t1));
        std::printf("omp time:\t%lld\n", (t4-t3));
    } else {
        std::printf("error\n");
    }

    // clean up
    clReleaseMemObject(a_buf);
    clReleaseMemObject(b_buf);
    clReleaseMemObject(c_buf);
    clReleaseCommandQueue(queue);
    clReleaseContext(context);
}