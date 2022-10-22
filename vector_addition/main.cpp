#include<iostream>
#include<vector>
#include"cl_util.h"
#include"util.h"

cl_int code;
const int platform_id = 0;
const int device_id = 0;

const int n = 1024;
const double eps = 1e-6;

cl_device_id get_device(int platform_id, int device_id) {
    std::vector<cl_platform_id> platform_list = cl_util::read_list<cl_platform_id>([&](cl_uint size, cl_platform_id* buffer, cl_uint* size_ret) -> cl_int {
        return clGetPlatformIDs(size, buffer, size_ret);
    }, platform_id+1, "get_platform_ids");

    std::vector<cl_device_id> device_list = cl_util::read_list<cl_device_id>([&](cl_uint size, cl_device_id* buffer, cl_uint* size_ret) {
        return clGetDeviceIDs(platform_list[platform_id], CL_DEVICE_TYPE_ALL, size, buffer, size_ret);
    }, device_id+1, "get_device_ids");
    
    return device_list[device_id];
}

int main() {
    util::random_seed(1234);

    cl_device_id device = get_device(platform_id, device_id);
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
    
    const std::string source = util::read("kernel.cl");
    cl_kernel kernel = cl_util::create_kernel(context, source, "vector_add");

    std::vector<double> a = util::random_normal(3);
    std::vector<double> b = util::random_normal(3);

    std::cout << a;
}
