#include"cl_util.h"
#include<vector>
#include<cstdio>
#include<functional>
#include<string>
#include<tuple>

const cl_uint MAX_SIZE = 256;

std::string read_string(const std::function<cl_int(size_t, char*, size_t*)>& reader, const size_t max_size, const char* name) {
    std::vector<char> buffer(max_size);
    size_t size;
    cl_int code = reader(max_size, buffer.data(), &size);
    if (code != CL_SUCCESS) {
        std::printf("%s %d\n", name, code);
        std::exit(1);
    }
    buffer.resize(size);
    buffer.push_back('\0');
    return std::string(buffer.data());
}

template<typename T>
std::vector<T> read_list(const std::function<cl_int(cl_uint, T*, cl_uint*)>& reader, const cl_uint max_size, const char* name) {
    std::vector<T> buffer(max_size);
    cl_uint size;
    cl_int code = reader(max_size, buffer.data(), &size);
    if (code != CL_SUCCESS) {
        std::printf("%s %d\n", name, code);
        std::exit(1);
    }
    buffer.resize(size);
    return buffer;
}

int main(int argc, char** argv) {


    std::vector<cl_platform_id> platform_list = read_list<cl_platform_id>([&](cl_uint size, cl_platform_id* buffer, cl_uint* size_ret) -> cl_int {
        return clGetPlatformIDs(size, buffer, size_ret);
    }, MAX_SIZE, "get_platform_ids");
    std::printf("platform list:\n");
    for (int i=0; i<platform_list.size(); i++) {
        cl_platform_id platform = platform_list[i];
        // platform
        std::string platform_name = read_string([&](size_t size, char* buffer, size_t* size_ret) -> cl_int {
            return clGetPlatformInfo(platform, CL_PLATFORM_NAME, size, buffer, size_ret);
        }, MAX_SIZE, "get_platform_name");
        std::string platform_vendor = read_string([&](size_t size, char* buffer, size_t* size_ret) -> cl_int {
            return clGetPlatformInfo(platform, CL_PLATFORM_VENDOR, size, buffer, size_ret);
        }, MAX_SIZE, "get_platform_vendor");

        std::printf("\t[%d]\t%s (%s)\n", i, platform_name.c_str(), platform_vendor.c_str());

        std::vector<cl_device_id> device_list = read_list<cl_device_id>([&](cl_uint size, cl_device_id* buffer, cl_uint* size_ret) {
            return clGetDeviceIDs(platform, CL_DEVICE_TYPE_ALL, size, buffer, size_ret);
        }, MAX_SIZE, "get_device_ids");
        for (cl_uint j=0; j<device_list.size(); j++) {
            cl_device_id device = device_list[j];
            // device
            std::string device_name = read_string([&](size_t size, char* buffer, size_t* size_ret) -> cl_int {
                return clGetDeviceInfo(device, CL_DEVICE_NAME, size, buffer, size_ret);
            }, MAX_SIZE, "get_device_name");
            std::string device_vendor = read_string([&](size_t size, char* buffer, size_t* size_ret) -> cl_int {
                return clGetDeviceInfo(device, CL_DEVICE_VENDOR, size, buffer, size_ret);
            }, MAX_SIZE, "get_device_vendor");
            

            std::printf("\t\t[%d] %s (%s)\n", j, device_name.c_str(), device_vendor.c_str());
        }
    }
}