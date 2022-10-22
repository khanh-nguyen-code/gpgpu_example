#include<iostream>
#include"cl_util.h"

int main() {
    // list all platforms and devices
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
            std::vector<size_t> sizes = device.getInfo<CL_DEVICE_MAX_WORK_ITEM_SIZES>();
            std::cout << "\t\t\tSizes:\t (";
            for (size_t size: sizes) {
                std::cout << size << ",";
            }
            std::cout << ")" << std::endl;
        }
    }
}
