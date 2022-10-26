#ifndef __CL_UTIL__
#define __CL_UTIL__

#include<CL/opencl.h>
#include<vector>
#include<functional>
#include<string>

namespace cl_util {
void code_ok(const char* name, cl_int code);
std::string read_string(
    const std::function<cl_int(size_t, char*, size_t*)>& reader,
    const char* name = ""
);
template<typename T>
std::vector<T> read_list(
    const std::function<cl_int(cl_uint, T*, cl_uint*)>& reader,
    const char* name = ""
);
cl_kernel create_kernel(cl_context context, std::string source, const char* name = "");
cl_device_id get_device(int platform_id, int device_id);
}
#endif //__CL_UTIL__

