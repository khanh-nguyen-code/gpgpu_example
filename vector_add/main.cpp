#include<iostream>
#include<vector>
#include"kernel.cl.h"
#include"util.h"

#define CL_TARGET_OPENCL_VERSION 200
#include"cl_util.h"
const int platform_id = 0;
const int device_id = 0;


cl_int code;
const cl_uint MAX_SIZE = 256;
const int n = 1024;
const double eps = 1e-6;

std::vector<double> vector_add(const std::vector<double>& a, const std::vector<double>& b) {
    std::vector<double> c;
    for (int i=0;; i++) {
        if (i >= a.size() or i >= b.size()) {
            break;
        }
        c.push_back(a[i] + b[i]);
    }
    return c;
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

int main() {
    util::random_seed(1234);
    std::vector<double> a = util::random_normal(n);
    std::vector<double> b = util::random_normal(n);
    std::vector<double> c(n);

    cl_device_id device = cl_util::get_device(platform_id, device_id);
    cl_context context = clCreateContext(nullptr, 1, &device, nullptr, nullptr, &code);
    if (code != CL_SUCCESS) {
        std::printf("create_context: %d\n", code);
        std::exit(1);
    }
    cl_command_queue queue = clCreateCommandQueueWithProperties(context, device, nullptr, &code);
    if (code != CL_SUCCESS) {
        std::printf("create_command_queue_with_properties: %d\n", code);
        std::exit(1);
    }
    
    const std::string source(reinterpret_cast<char*>(&kernel_cl[0]), kernel_cl_len);
    cl_kernel kernel = cl_util::create_kernel(context, source, "vector_add");

    cl_mem a_buf = clCreateBuffer(context, CL_MEM_READ_ONLY, a.size() * sizeof(double), nullptr, &code);
    if (code != CL_SUCCESS) {
        std::printf("create_buffer_with_properties: %d\n", code);
        std::exit(1);
    }
    cl_mem b_buf = clCreateBuffer(context, CL_MEM_READ_ONLY, b.size() * sizeof(double), nullptr, &code);
    if (code != CL_SUCCESS) {
        std::printf("create_buffer_with_properties: %d\n", code);
        std::exit(1);
    }
    cl_mem c_buf = clCreateBuffer(context, CL_MEM_WRITE_ONLY, c.size() * sizeof(double), nullptr, &code);
    if (code != CL_SUCCESS) {
        std::printf("create_buffer_with_properties: %d\n", code);
        std::exit(1);
    }

    // queue
    std::vector<cl_event> write_event(2);
    code = clEnqueueWriteBuffer(queue, a_buf, CL_NON_BLOCKING, 0, a.size() * sizeof(double), a.data(), 0, nullptr, &write_event[0]);
    if (code != CL_SUCCESS) {
        std::printf("enqueue_write_buffer: %d\n", code);
        std::exit(1);
    }
    code = clEnqueueWriteBuffer(queue, b_buf, CL_NON_BLOCKING, 0, b.size() * sizeof(double), b.data(), 0, nullptr, &write_event[1]);
    if (code != CL_SUCCESS) {
        std::printf("enqueue_write_buffer: %d\n", code);
        std::exit(1);
    }

    cl_event kernel_event;
    code = clSetKernelArg(kernel, 0, sizeof(int), &n);
    if (code != CL_SUCCESS) {
        std::printf("set_kernel_arg: %d\n", code);
        std::exit(1);
    }
    code = clSetKernelArg(kernel, 1, sizeof(cl_mem), &c_buf);
    if (code != CL_SUCCESS) {
        std::printf("set_kernel_arg: %d\n", code);
        std::exit(1);
    }
    code = clSetKernelArg(kernel, 2, sizeof(cl_mem), &a_buf);
    if (code != CL_SUCCESS) {
        std::printf("set_kernel_arg: %d\n", code);
        std::exit(1);
    }
    code = clSetKernelArg(kernel, 3, sizeof(cl_mem), &b_buf);
    if (code != CL_SUCCESS) {
        std::printf("set_kernel_arg: %d\n", code);
        std::exit(1);
    }

    std::vector<size_t> global_work_size = {n};
    std::vector<size_t> local_work_size = {1};
    if (global_work_size.size() != local_work_size.size()) {
        std::printf("dim: %d %d\n", global_work_size.size(), local_work_size.size());
        std::exit(1);
    }
    size_t work_dim = global_work_size.size();
    code = clEnqueueNDRangeKernel(
        queue, kernel, work_dim, nullptr,
        global_work_size.data(), local_work_size.data(),
        write_event.size(), write_event.data(), &kernel_event
    );
    if (code != CL_SUCCESS) {
        std::printf("enqueue_ndrange_kernel: %d\n", code);
        std::exit(1);
    }


    cl_event read_event;
    code = clEnqueueReadBuffer(queue, c_buf, CL_NON_BLOCKING, 0, c.size() * sizeof(double), c.data(), 1, &kernel_event, &read_event);
    if (code != CL_SUCCESS) {
        std::printf("enqueue_read_buffer: %d\n", code);
        std::exit(1);
    }

    code = clWaitForEvents(1, &read_event);
    if (code != CL_SUCCESS) {
        std::printf("wait_for_events: %d\n", code);
        std::exit(1);
    }
    clFinish(queue);
    for (auto& event: write_event) {
        clReleaseEvent(event);
    }
    clReleaseEvent(kernel_event);
    clReleaseEvent(read_event); 
    // compare
    std::vector<double> c_host = vector_add(a, b);
    if (vector_cmp(c, c_host)) {
        std::printf("result ok\n");
    } else {
        std::printf("error\n");
    }

    // clean up
    clReleaseMemObject(a_buf);
    clReleaseMemObject(b_buf);
    clReleaseMemObject(c_buf);
    clReleaseKernel(kernel);
    clReleaseCommandQueue(queue);
    clReleaseContext(context);
}