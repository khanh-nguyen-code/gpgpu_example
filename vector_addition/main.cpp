#include<iostream>
#define CL_TARGET_OPENCL_VERSION 200
#include<CL/cl.hpp>

const int platform_id = 0;
const int device_id = 0;

const int n = 4;
const double eps = 1e-6;
const bool print_cl = false;

const std::string kernel_str = R""""(
__kernel void vector_add(__global double *c, __global const double *a, __global const double *b) {
    int i = get_global_id(0);
    c[i] = a[i] + b[i];
}
)"""";

int main() {
    // list all platforms and devices
    if (print_cl) {
        std::vector<cl::Platform> platform_list;
        cl::Platform::get(&platform_list);
        std::cout << "platform list:" << std::endl;
        for (int i=0; i<platform_list.size(); i++) {
            cl::Platform& platform = platform_list[i];
            std::cout << "\t[" << i << "] " << platform.getInfo<CL_PLATFORM_NAME>() << std::endl;
            std::vector<cl::Device> device_list;
            platform.getDevices(CL_DEVICE_TYPE_ALL, &device_list);
            std::cout << "\tdevice list:" << std::endl;
            for (int i=0; i<device_list.size(); i++) {
                cl::Device& device = device_list[i];
                std::cout << "\t\t[" << i << "]" << std::endl;
                std::cout << "\t\t\tName:\t" << device.getInfo<CL_DEVICE_NAME>() << std::endl;
                std::cout << "\t\t\tVendor:\t" << device.getInfo<CL_DEVICE_VENDOR>() << std::endl;
                std::cout << "\t\t\tType:\t" << device.getInfo<CL_DEVICE_TYPE>() << std::endl;
                std::cout << "\t\t\tMax Compute Units:\t" << device.getInfo<CL_DEVICE_MAX_COMPUTE_UNITS>() << std::endl;
                std::cout << "\t\t\tMax Clock Frequency:\t" << device.getInfo<CL_DEVICE_MAX_CLOCK_FREQUENCY>() << std::endl;
                std::cout << "\t\t\tGlobal Memory:\t" << device.getInfo<CL_DEVICE_GLOBAL_MEM_SIZE>() / (1024 * 1024) << " MB" << std::endl;
                std::cout << "\t\t\tAllocateable Memory:\t" << device.getInfo<CL_DEVICE_MAX_MEM_ALLOC_SIZE>() / (1024 * 1024) << " MB" << std::endl;
                std::cout << "\t\t\tLocal Memory:\t" << device.getInfo<CL_DEVICE_LOCAL_MEM_SIZE>() / (1024) << " KB" << std::endl;
            }
        }
    }

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
    // std::string kernel_str = read("kernel.cl");
    source_list.push_back({kernel_str.c_str(), kernel_str.length()});

    // build kernel
    cl::Program program(context, source_list);
    if (program.build({device}) != CL_SUCCESS) {
        std::cerr << "build error: " << program.getBuildInfo<CL_PROGRAM_BUILD_LOG>(device) << std::endl;
        std::exit(1);
    }

    // make dummy data
    double* a = (double*) std::malloc(n * sizeof(double));
    double* b = (double*) std::malloc(n * sizeof(double));
    double* c = (double*) std::malloc(n * sizeof(double));
    for (int i=0; i<n; i++) {
        a[i] = (double) (i+1);
        b[i] = (double) (i+2);
    }
    // make queue
    cl::CommandQueue queue(context, device, CL_QUEUE_OUT_OF_ORDER_EXEC_MODE_ENABLE);

    // make device buffer
    cl::Buffer a_buf(context, CL_MEM_READ_ONLY, n * sizeof(double));
    cl::Buffer b_buf(context, CL_MEM_READ_ONLY, n * sizeof(double));
    cl::Buffer c_buf(context, CL_MEM_WRITE_ONLY, n * sizeof(double));

    // enqueue
    // (1) enqueue write data to device buffer
    std::vector<cl::Event> write_event_list(2);
    queue.enqueueWriteBuffer(a_buf, CL_FALSE, 0, n * sizeof(double), a, nullptr, &write_event_list[0]);
    queue.enqueueWriteBuffer(b_buf, CL_FALSE, 0, n * sizeof(double), b, nullptr, &write_event_list[1]);
    queue.enqueueBarrierWithWaitList(&write_event_list, nullptr);
    // (2) enqueue kernel
    cl::Event kernel_event;
    cl::Kernel kernel(program, "vector_add");
    kernel.setArg(0, c_buf);
    kernel.setArg(1, a_buf);
    kernel.setArg(2, b_buf);
    cl::NDRange global_size(n);
    cl::NDRange local_size = cl::NullRange;
    queue.enqueueNDRangeKernel(kernel, cl::NullRange, global_size, local_size, nullptr, &kernel_event);
    queue.enqueueBarrierWithWaitList(nullptr, &kernel_event);
    // (3) enqueue read data from device_buffer
    cl::Event read_event;
    queue.enqueueReadBuffer(c_buf, CL_FALSE, 0, n * sizeof(double), c, nullptr, &read_event);
    queue.enqueueBarrierWithWaitList(nullptr, &read_event);
    // finish
    queue.finish();
    
    // output
    for (int i=0; i<n; i++) {
        double diff = c[i] - (a[i] + b[i]);
        diff = (diff >= 0) ? diff : -diff;
        if (diff > eps) {
            std::cerr << "wrong result" << std::endl;
            std::exit(1);
        }
    }
    std::cout << "result ok" << std::endl;

    // free
    std::free(a);
    std::free(b);
    std::free(c);
}