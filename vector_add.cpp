#include<iostream>
#include<vector>
#include"src/kernel/vector_add.cl.h"
#include"util.h"

#define CL_TARGET_OPENCL_VERSION 120 
#include"cl_util.h"
const int platform_id = 0;
const int device_id = 0;


cl_int code;
const uint64_t n = 1024;

template<typename T>
std::vector<T> vector_add(const std::vector<T>& a, const std::vector<T>& b) {
    std::vector<T> c;
    for (size_t i=0;; i++) {
        if (i >= a.size() or i >= b.size()) {
            break;
        }
        c.push_back(a[i] + b[i]);
    }
    return c;
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

int main() {
    util::random_seed(1234);
    std::vector<float> a = util::random_normal<float>(n);
    std::vector<float> b = util::random_normal<float>(n);
    std::vector<float> c(n);

    cl_device_id device = cl_util::get_device(platform_id, device_id);
    cl_context context = clCreateContext(nullptr, 1, &device, nullptr, nullptr, &code);
    cl_util::code_ok("create_context", code);
    cl_command_queue queue = clCreateCommandQueue(context, device, CL_QUEUE_PROFILING_ENABLE | 0, &code);
    cl_util::code_ok("create_command_queue", code);
    
    const std::string source(reinterpret_cast<const char*>(&src_kernel_vector_add_cl[0]), src_kernel_vector_add_cl_len);
    cl_kernel kernel = cl_util::create_kernel(context, source, "vector_add");

    cl_mem a_buf = clCreateBuffer(context, CL_MEM_READ_ONLY, a.size() * sizeof(float), nullptr, &code);
    cl_util::code_ok("create_buffer_with_properties", code);
    cl_mem b_buf = clCreateBuffer(context, CL_MEM_READ_ONLY, b.size() * sizeof(float), nullptr, &code);
    cl_util::code_ok("create_buffer_with_properties", code);
    cl_mem c_buf = clCreateBuffer(context, CL_MEM_WRITE_ONLY, c.size() * sizeof(float), nullptr, &code);
    cl_util::code_ok("create_buffer_with_properties", code);
    
    // queue
    std::vector<cl_event> write_event(2);
    code = clEnqueueWriteBuffer(queue, a_buf, CL_NON_BLOCKING, 0, a.size() * sizeof(float), a.data(), 0, nullptr, &write_event[0]);
    cl_util::code_ok("enqueue_write_buffer", code);
    code = clEnqueueWriteBuffer(queue, b_buf, CL_NON_BLOCKING, 0, b.size() * sizeof(float), b.data(), 0, nullptr, &write_event[1]);
    cl_util::code_ok("enqueue_write_buffer", code);

    cl_event kernel_event;
    code = clSetKernelArg(kernel, 0, sizeof(cl_mem), &c_buf);
    cl_util::code_ok("set_kernel_arg", code);
    code = clSetKernelArg(kernel, 1, sizeof(cl_mem), &a_buf);
    cl_util::code_ok("set_kernel_arg", code);
    code = clSetKernelArg(kernel, 2, sizeof(cl_mem), &b_buf);
    cl_util::code_ok("set_kernel_arg", code);

    std::vector<size_t> global_work_offset = {0};
    std::vector<size_t> global_work_size = {n};
    std::vector<size_t> local_work_size = {1};
    size_t work_dim = global_work_size.size();
    code = clEnqueueNDRangeKernel(
        queue, kernel, work_dim, global_work_offset.data(),
        global_work_size.data(), local_work_size.data(),
        write_event.size(), write_event.data(), &kernel_event
    );
    cl_util::code_ok("enqueue_ndrange_kernel", code);


    cl_event read_event;
    code = clEnqueueReadBuffer(queue, c_buf, CL_NON_BLOCKING, 0, c.size() * sizeof(float), c.data(), 1, &kernel_event, &read_event);
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
    std::vector<float> c_host = vector_add(a, b);
    std::printf("max_diff %f\n", vector_diff<float>(c, c_host));

    // clean up
    clReleaseMemObject(a_buf);
    clReleaseMemObject(b_buf);
    clReleaseMemObject(c_buf);
    clReleaseKernel(kernel);
    clReleaseCommandQueue(queue);
    clReleaseContext(context);
}