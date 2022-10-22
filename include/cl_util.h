#ifndef __CL_UTIL__
#define __CL_UTIL__

#include<CL/opencl.h>
#include<vector>
#include<functional>
#include<string>

namespace cl_util {
std::string read_string(
    const std::function<cl_int(size_t, char*, size_t*)>& reader,
    const size_t max_size,
    const char* name = ""
) {
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
std::vector<T> read_list(
    const std::function<cl_int(cl_uint, T*, cl_uint*)>& reader,
    const cl_uint max_size,
    const char* name = ""
) {
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
}
#endif //__CL_UTIL__

