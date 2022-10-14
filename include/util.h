#ifndef __UTIL__
#define __UTIL__

#include<string>
#include<fstream>
#include<sstream>

namespace util{
    std::string read(const std::string& path) {
        std::ifstream t(path);
        std::stringstream buffer;
        buffer << t.rdbuf();
        return buffer.str();
    }
}
#endif