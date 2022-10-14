#include<iostream>
#define CL_TARGET_OPENCL_VERSION 200
#include<CL/cl.hpp>
#include"util.h"

const int platform_id = 0;
const int device_id = 0;

const double eps = 1e-6;


void print(const double *a, const int m, const int n) {
    for (int i=0; i<m; i++) {
        for (int j=0; j<n; j++) {
            std::cout << a[i * n + j] << "\t";
        }
        std::cout << "\n";
    }
    std::cout << std::endl;
}

void matmul(double *c, const double *a, const double *b, const int d0, const int d1, const int d2) {
    for (int i=0; i<d0; i++) {
        for (int j=0; j<d2; j++) {
            c[i * d2 + j] = 0.0;
        }
    }
    for (int i=0; i<d0; i++) {
        for (int j=0; j<d2; j++) {
            for (int k=0; k<d1; k++) {
                c[i * d2 + j] += a[i * d1 + k] * b[k * d2 + j];
            }
        }
    }
}

int main() {
    // choose default platform and defaul device
    std::vector<cl::Platform> platform_list;
    cl::Platform::get(&platform_list);
    if (platform_list.size() == 0) {
        std::cerr << "no platform found" << std::endl;
        std::exit(1);
    }
    cl::Platform& platform = platform_list[platform_id];
    std::cout << "choosing platform [" << platform_id << "] " << platform.getInfo<CL_PLATFORM_NAME>() << std::endl;
    std::vector<cl::Device> device_list;
    platform.getDevices(CL_DEVICE_TYPE_ALL, &device_list);
    if (device_list.size() == 0) {
        std::cerr << "no device found" << std::endl;
        std::exit(1);
    }
    cl::Device& device = device_list[device_id];
    std::cout << "choosing device [" << device_id << "] " << device.getInfo<CL_DEVICE_NAME>() << std::endl;

    // create context
    cl::Context context({device});
    cl::Program::Sources source_list;

    // load kernel
    std::string kernel_str = util::read("kernel.cl");
    source_list.push_back({kernel_str.c_str(), kernel_str.length()});

    // build kernel
    cl::Program program(context, source_list);
    if (program.build({device}) != CL_SUCCESS) {
        std::cerr << "build error: " << program.getBuildInfo<CL_PROGRAM_BUILD_LOG>(device) << std::endl;
        std::exit(1);
    }

    // make dummy data
    const int d0 = 3;
    const int d1 = 4;
    const int d2 = 5;
    double* a = (double*) std::malloc(d0*d1 * sizeof(double));
    double* b = (double*) std::malloc(d1*d2 * sizeof(double));
    double* c = (double*) std::malloc(d0*d2 * sizeof(double));
    for (int i=0; i<d0*d1; i++) {
        a[i] = (double) i;
    }
    for (int i=0; i<d1*d2; i++) {
        b[i] = (double) i;
    }
    // make queue
    cl::CommandQueue queue(context, device, CL_QUEUE_OUT_OF_ORDER_EXEC_MODE_ENABLE);

    // make device buffer
    cl::Buffer a_buf(context, CL_MEM_READ_ONLY, d0*d1 * sizeof(double));
    cl::Buffer b_buf(context, CL_MEM_READ_ONLY, d1*d2 * sizeof(double));
    cl::Buffer c_buf(context, CL_MEM_WRITE_ONLY, d0*d2 * sizeof(double));

    // enqueue
    // (1) enqueue write data to device buffer
    std::vector<cl::Event> write_event_list(2);
    queue.enqueueWriteBuffer(a_buf, CL_FALSE, 0, d0*d1 * sizeof(double), a, nullptr, &write_event_list[0]);
    queue.enqueueWriteBuffer(b_buf, CL_FALSE, 0, d1*d2 * sizeof(double), b, nullptr, &write_event_list[1]);
    queue.enqueueBarrierWithWaitList(&write_event_list, nullptr);
    // (2) enqueue kernel
    cl::Event kernel_event;
    cl::Kernel kernel(program, "matmul");
    kernel.setArg(0, c_buf);
    kernel.setArg(1, a_buf);
    kernel.setArg(2, b_buf);
    kernel.setArg(3, cl::Local(d1 * sizeof(double)));
    cl::NDRange global_size(d0, d1, d2);
    cl::NDRange local_size(1, d1, 1);
    queue.enqueueNDRangeKernel(kernel, cl::NullRange, global_size, local_size, nullptr, &kernel_event);
    queue.enqueueBarrierWithWaitList(nullptr, &kernel_event);
    // (3) enqueue read data from device_buffer
    cl::Event read_event;
    queue.enqueueReadBuffer(c_buf, CL_FALSE, 0, d0*d2 * sizeof(double), c, nullptr, &read_event);
    queue.enqueueBarrierWithWaitList(nullptr, &read_event);
    // finish
    queue.finish();
    
    // output
    double* d = (double*) std::malloc(d0*d2 * sizeof(double));
    print(a, d0, d1);
    print(b, d1, d2);
    print(c, d0, d2);
    matmul(d, a, b, d0, d1, d2);
    print(d, d0, d2);

    // free
    std::free(a);
    std::free(b);
    std::free(c);
    std::free(d);
}
