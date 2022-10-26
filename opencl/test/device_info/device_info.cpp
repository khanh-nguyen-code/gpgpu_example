#include<vector>
#include<cstdio>

#define CL_TARGET_OPENCL_VERSION 300
#include"cl_util/cl_util.h"

int main(int argc, char** argv) {
    std::vector<cl_platform_id> platform_list = cl_util::read_list<cl_platform_id>([&](cl_uint size, cl_platform_id* buffer, cl_uint* size_ret) -> cl_int {
        return clGetPlatformIDs(size, buffer, size_ret);
    }, "get_platform_ids");
    std::printf("platform list:\n");
    for (int i=0; i<platform_list.size(); i++) {
        cl_platform_id platform = platform_list[i];
        // platform
        std::string platform_name = cl_util::read_string([&](size_t size, char* buffer, size_t* size_ret) -> cl_int {
            return clGetPlatformInfo(platform, CL_PLATFORM_NAME, size, buffer, size_ret);
        }, "get_platform_name");
        std::string platform_vendor = cl_util::read_string([&](size_t size, char* buffer, size_t* size_ret) -> cl_int {
            return clGetPlatformInfo(platform, CL_PLATFORM_VENDOR, size, buffer, size_ret);
        }, "get_platform_vendor");
        std::string platform_version = cl_util::read_string([&](size_t size, char* buffer, size_t* size_ret) -> cl_int {
            return clGetPlatformInfo(platform, CL_PLATFORM_VERSION, size, buffer, size_ret);
        }, "get_platform_version");

        std::printf("\t[%d]\t%s (%s) %s\n", i, platform_name.c_str(), platform_vendor.c_str(), platform_version.c_str());

        std::vector<cl_device_id> device_list = cl_util::read_list<cl_device_id>([&](cl_uint size, cl_device_id* buffer, cl_uint* size_ret) {
            return clGetDeviceIDs(platform, CL_DEVICE_TYPE_ALL, size, buffer, size_ret);
        }, "get_device_ids");
        for (cl_uint j=0; j<device_list.size(); j++) {
            cl_device_id device = device_list[j];
            // device
            std::string device_name = cl_util::read_string([&](size_t size, char* buffer, size_t* size_ret) -> cl_int {
                return clGetDeviceInfo(device, CL_DEVICE_NAME, size, buffer, size_ret);
            }, "get_device_name");
            std::string device_vendor = cl_util::read_string([&](size_t size, char* buffer, size_t* size_ret) -> cl_int {
                return clGetDeviceInfo(device, CL_DEVICE_VENDOR, size, buffer, size_ret);
            }, "get_device_vendor");
            std::string device_version = cl_util::read_string([&](size_t size, char* buffer, size_t* size_ret) -> cl_int {
                return clGetDeviceInfo(device, CL_DEVICE_VERSION, size, buffer, size_ret);
            }, "get_device_version");
            

            std::printf("\t\t[%d] %s (%s) %s\n", j, device_name.c_str(), device_vendor.c_str(), device_version.c_str());
        }
    }
}