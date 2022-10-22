#ifndef __CL_UTIL__
#define __CL_UTIL__

#include<CL/opencl.h>
#include<vector>
#include<functional>
#include<string>

namespace cl_util {
const size_t BUFFER_SIZE = 1024;
std::string read_string(
    const std::function<cl_int(size_t, char*, size_t*)>& reader,
    const char* name = ""
) {
    std::vector<char> buffer(BUFFER_SIZE);
    size_t size;
    cl_int code = reader(BUFFER_SIZE, buffer.data(), &size);
    if (code != CL_SUCCESS) {
        std::printf("%s %d\n", name, code);
        std::exit(1);
    }
    buffer.resize(size);
    buffer.push_back('\0');
    return std::string(buffer.data());
}

template<typename T>
std::vector<T> read_list(
    const std::function<cl_int(cl_uint, T*, cl_uint*)>& reader,
    const char* name = ""
) {
    std::vector<T> buffer(BUFFER_SIZE);
    cl_uint size;
    cl_int code = reader(BUFFER_SIZE, buffer.data(), &size);
    if (code != CL_SUCCESS) {
        std::printf("%s %d\n", name, code);
        std::exit(1);
    }
    buffer.resize(size);
    return buffer;
}
cl_kernel create_kernel(cl_context context, std::string source, const char* name = "") {
    cl_int code;
    const char* source_c = source.c_str();
    size_t source_length = source.length();
    cl_program program = clCreateProgramWithSource(context, 1, &source_c, &source_length, &code);
    if (code != CL_SUCCESS) {
        std::printf("%s create_program_with_source %d\n", name, code);
        std::exit(1);
    }
    const char* options = "-Werror";
    code = clBuildProgram(program, 0, nullptr, options, nullptr, nullptr);
    if (code != CL_SUCCESS) {
        clReleaseProgram(program);
        std::printf("%s build_program %d\n", name, code);
        std::exit(1);
    }
    cl_kernel kernel = clCreateKernel(program, name, &code);
    clReleaseProgram(program);
    if (code != CL_SUCCESS) {
        std::printf("%s create_kernel %d\n", name, code);
        std::exit(1);
    }
    return kernel;
}
cl_device_id get_device(int platform_id, int device_id) {
    std::vector<cl_platform_id> platform_list = read_list<cl_platform_id>([&](cl_uint size, cl_platform_id* buffer, cl_uint* size_ret) -> cl_int {
        return clGetPlatformIDs(size, buffer, size_ret);
    }, "get_platform_ids");
    cl_platform_id platform = platform_list[platform_id];

    std::string platform_name = read_string([&](size_t size, char* buffer, size_t* size_ret) -> cl_int {
        return clGetPlatformInfo(platform, CL_PLATFORM_NAME, size, buffer, size_ret);
    }, "get_platform_name");
    std::string platform_vendor = read_string([&](size_t size, char* buffer, size_t* size_ret) -> cl_int {
        return clGetPlatformInfo(platform, CL_PLATFORM_VENDOR, size, buffer, size_ret);
    }, "get_platform_vendor");
    std::string platform_version = read_string([&](size_t size, char* buffer, size_t* size_ret) -> cl_int {
        return clGetPlatformInfo(platform, CL_PLATFORM_VERSION, size, buffer, size_ret);
    }, "get_platform_version");

    std::printf("choosing platform: %s (%s) %s\n", platform_name.c_str(), platform_vendor.c_str(), platform_version.c_str());
    

    std::vector<cl_device_id> device_list = read_list<cl_device_id>([&](cl_uint size, cl_device_id* buffer, cl_uint* size_ret) {
        return clGetDeviceIDs(platform, CL_DEVICE_TYPE_ALL, size, buffer, size_ret);
    }, "get_device_ids");
    
    cl_device_id device = device_list[device_id];
    std::string device_name = read_string([&](size_t size, char* buffer, size_t* size_ret) -> cl_int {
        return clGetDeviceInfo(device, CL_DEVICE_NAME, size, buffer, size_ret);
    }, "get_device_name");
    std::string device_vendor = read_string([&](size_t size, char* buffer, size_t* size_ret) -> cl_int {
        return clGetDeviceInfo(device, CL_DEVICE_VENDOR, size, buffer, size_ret);
    }, "get_device_vendor");
    std::string device_version = read_string([&](size_t size, char* buffer, size_t* size_ret) -> cl_int {
        return clGetDeviceInfo(device, CL_DEVICE_VERSION, size, buffer, size_ret);
    }, "get_device_version");
    

    std::printf("choosing device: %s (%s) %s\n", device_name.c_str(), device_vendor.c_str(), device_version.c_str());
    return device_list[device_id];
}
}
#endif //__CL_UTIL__

