#include<iostream>
#include<vector>
#include<omp.h>
#include"util.h"

#define CL_TARGET_OPENCL_VERSION 300 
#include"cl_util/cl_util.h"
#include<clblast.h>



cl_int code;
// (m x k) (k x n) -> (m x n)
const int m = 512;
const int n = 512;
const int k = 512;


void matmul_clblast(
    cl_mem c, cl_mem a, cl_mem b, int m, int n, int k,
    cl_command_queue* queue, cl_event* event
) {
    // c <- alpha A B + beta C
    float alpha = 1.0;
    float beta = 0.0;
    clblast::StatusCode code = clblast::Gemm<float>(
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

template<typename T>
T vector_diff(const std::vector<T>& a, const std::vector<T>& b) {
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

template<typename T>
void matmul(T* c, const T* a, const T* b, int m, int n, int k) {
    int m_i, n_i;
    # pragma omp parallel for shared(a, b, c) private(m_i, n_i) collapse(2)
    for (m_i=0; m_i < m; m_i++)
    for (n_i=0; n_i < n; n_i++) {
        T acc = 0;
        for (int k_i=0; k_i < k; k_i++) {
            acc += a[m_i * k + k_i] * b[k_i * n + n_i];
        }
        c[m_i * n + n_i] = acc;
    }
}


int main(int argc, char** argv) {
    int platform_id = 0;
    int device_id = 0;
    if (argc >= 3) {
        platform_id = std::atoi(argv[1]);
        device_id = std::atoi(argv[2]);
    }
    util::random_seed(1234);
    std::vector<float> a = util::random_normal<float>(m * k);
    std::vector<float> b = util::random_normal<float>(k * n);
    std::vector<float> c(m * n);

    cl_device_id device = cl_util::get_device(platform_id, device_id);
    cl_context context = clCreateContext(nullptr, 1, &device, nullptr, nullptr, &code);
    cl_util::code_ok("create_context", code);
    cl_command_queue queue = clCreateCommandQueue(context, device, 0, &code);
    cl_util::code_ok("create_command_queue_with_properties", code);
    
    cl_mem a_buf = clCreateBuffer(context, CL_MEM_READ_ONLY, a.size() * sizeof(double), nullptr, &code);
    cl_util::code_ok("create_buffer_with_properties", code);
    cl_mem b_buf = clCreateBuffer(context, CL_MEM_READ_ONLY, b.size() * sizeof(double), nullptr, &code);
    cl_util::code_ok("create_buffer_with_properties", code);
    cl_mem c_buf = clCreateBuffer(context, CL_MEM_WRITE_ONLY, c.size() * sizeof(double), nullptr, &code);
    cl_util::code_ok("create_buffer_with_properties", code);
    
    // queue
    std::vector<cl_event> write_event(2);
    code = clEnqueueWriteBuffer(queue, a_buf, CL_NON_BLOCKING, 0, a.size() * sizeof(double), a.data(), 0, nullptr, &write_event[0]);
    cl_util::code_ok("enqueue_write_buffer", code);
    code = clEnqueueWriteBuffer(queue, b_buf, CL_NON_BLOCKING, 0, b.size() * sizeof(double), b.data(), 0, nullptr, &write_event[1]);
    cl_util::code_ok("enqueue_write_buffer", code);
    
    cl_event kernel_event;
    matmul_clblast(c_buf, a_buf, b_buf, m, n, k, &queue, &kernel_event);

    cl_event read_event;
    code = clEnqueueReadBuffer(queue, c_buf, CL_NON_BLOCKING, 0, c.size() * sizeof(double), c.data(), 1, &kernel_event, &read_event);
    cl_util::code_ok("enqueue_read_buffer", code);
    
    code = clWaitForEvents(1, &read_event);
    cl_util::code_ok("wait_for_events", code);

    clFinish(queue);
    for (auto& event: write_event) {
        clReleaseEvent(event);
    }
    clReleaseEvent(kernel_event);
    clReleaseEvent(read_event); 
    // compare
    std::vector<float> c_host(c.size());
    matmul<float>(c_host.data(), a.data(), b.data(), m, n, k);
    std::printf("max_diff %f\n", vector_diff<float>(c, c_host));

    // clean up
    clReleaseMemObject(a_buf);
    clReleaseMemObject(b_buf);
    clReleaseMemObject(c_buf);
    clReleaseCommandQueue(queue);
    clReleaseContext(context);
}