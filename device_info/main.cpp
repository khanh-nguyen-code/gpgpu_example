#include"cl_util.h"
#include<cstdlib>
#include<cstdio>

const cl_uint MAX_SIZE = 256;
cl_int code;

int main(int argc, char** argv) {
    cl_platform_id *platform_list = (cl_platform_id*) std::malloc(MAX_SIZE * sizeof(cl_platform_id));
    cl_uint platform_list_count;
    code = clGetPlatformIDs(MAX_SIZE, platform_list, &platform_list_count);
    if (code != CL_SUCCESS) {
        std::exit(1);
    }
    std::printf("platform list:\n");
    for (cl_uint i=0; i<platform_list_count; i++) {
        cl_platform_id platform_id = platform_list[i];
        // platform_name
        char* platform_name = (char*) std::malloc((MAX_SIZE+1) * sizeof(char));
        size_t platform_name_count;
        clGetPlatformInfo(platform_id, CL_PLATFORM_NAME, MAX_SIZE, platform_name, &platform_name_count);
        if (code != CL_SUCCESS) {
            std::exit(1);
        }
        platform_name[platform_name_count] = '\0';
        // platform_vendor
        char* platform_vendor = (char*) std::malloc((MAX_SIZE+1) * sizeof(char));
        size_t platform_vendor_count;
        clGetPlatformInfo(platform_id, CL_PLATFORM_VENDOR, MAX_SIZE, platform_vendor, &platform_vendor_count);
        if (code != CL_SUCCESS) {
            std::exit(1);
        }
        platform_vendor[platform_vendor_count] = '\0';

        std::printf("\t[%d]\t%s (%s)\n", i, platform_name, platform_vendor);
        std::free(platform_name);
        std::free(platform_vendor);

        // device
        cl_device_id* device_list = (cl_device_id*) std::malloc(MAX_SIZE * sizeof(cl_device_id));
        cl_uint device_list_count;
        clGetDeviceIDs(platform_id, CL_DEVICE_TYPE_ALL, MAX_SIZE, device_list, &device_list_count);
        if (code != CL_SUCCESS) {
            std::exit(1);
        }

        for (cl_uint j=0; j<device_list_count; j++) {
            cl_device_id device_id = device_list[j];
            // device_name
            char* device_name = (char*) std::malloc((MAX_SIZE+1) * sizeof(char));
            size_t device_name_count;
            clGetDeviceInfo(device_id, CL_DEVICE_NAME, MAX_SIZE, device_name, &device_name_count);
            if (code != CL_SUCCESS) {
                std::exit(1);
            }
            device_name[device_name_count] = '\0';
            // device_vendor
            char* device_vendor = (char*) std::malloc((MAX_SIZE+1) * sizeof(char));
            size_t device_vendor_count;
            clGetDeviceInfo(device_id, CL_DEVICE_VENDOR, MAX_SIZE, device_vendor, &device_vendor_count);
            if (code != CL_SUCCESS) {
                std::exit(1);
            }
            device_vendor[device_vendor_count] = '\0';

            std::printf("\t\t[%d] %s (%s)\n", j, device_name, device_vendor);
            std::free(device_name);
            std::free(device_vendor);
        }
        std::free(device_list);
    }
    std::free(platform_list);
}