#ifndef __UTIL__
#define __UTIL__

#include<string>
#include<iostream>

namespace util{
    const int max_read_size = 1024*1024;
    std::string read(const std::string& path) {
        std::FILE* fp = std::fopen(path.c_str(), "r");
        if (!fp) {
            std::cerr << "cannot open file:" << path << std::endl;
            std::exit(1);
        }
        char* content = (char*) std::malloc(max_read_size * sizeof(char));
        int size = std::fread(content, 1, max_read_size, fp);
        std::fclose(fp);
        return std::string(content);
    }
}
#endif