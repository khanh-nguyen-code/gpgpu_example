cmake_minimum_required(VERSION 3.24)
project("example")

message(${CMAKE_SYSTEM_NAME})

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_BUILD_TYPE Debug)

if (${CMAKE_SYSTEM_NAME} MATCHES "Darwin")
    # include directories
    include_directories(
        "include"
        "/opt/homebrew/include"
        "/opt/homebrew/opt/libomp/include"
    )
    # vendor library directories
    link_directories(
        "/opt/homebrew/lib"
        "/opt/homebrew/opt/libomp/lib"
    )

    # vendor: list of vendor packages to be linked with library and target
    set(vendor
        "-framework OpenCL"
        "omp"
        "clblast"
    )
elseif (${CMAKE_SYSTEM_NAME} MATCHES "Linux")
    # include directories
    include_directories(
        "include"
    )
    # vendor library directories
    link_directories(
    )

    # vendor: list of vendor packages to be linked with library and target
    set(vendor
        "OpenCL"
        "-fopenmp"
        "clblast"
    )
else()
    message(FATAL_ERROR "does not support system: " ${CMAKE_SYSTEM_NAME})
endif()


# library_list: placeholder for all building libraries
set(library_list)  

# generate library build 
file(GLOB library_dir_list "src/*")
foreach(library_dir ${library_dir_list})
    get_filename_component(library ${library_dir} NAME_WE)
    message("generating library ... " ${library})

    file(GLOB_RECURSE library_src ${library_dir}/*.cpp)
    add_library(${library} SHARED ${library_src})
    target_link_libraries(${library} ${vendor})

    list(APPEND library_list ${library})
    message("generating library ... " ${library} " done")
endforeach()



# generate target build 
file(GLOB target_src_list "*.cpp")
foreach(target_src ${target_src_list})
    get_filename_component(target_out ${target_src} NAME_WE)
    message("generating target ... " ${target_out})

    add_executable(${target_out} ${target_src})
    target_link_libraries(${target_out} ${library_list} ${vendor})

    message("generating target ... " ${target_out} " done")
endforeach()


