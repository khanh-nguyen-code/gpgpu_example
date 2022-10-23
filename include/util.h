#ifndef __UTIL__
#define __UTIL__

#include<string>
#include<fstream>
#include<sstream>
#include<random>
#include<iostream>

template<typename T>
std::ostream& operator<<(std::ostream& out, const std::vector<T>& vec) {
    out << "[";
    for (int i=0; i<vec.size(); i++) {
        out << vec[i] << ", ";
    }
    out << "]";
    return out;
}
namespace util{
    std::string read(const std::string& path) {
        std::ifstream t(path);
        std::stringstream buffer;
        buffer << t.rdbuf();
        return buffer.str();
    }
    std::default_random_engine random_engine;
    template<typename T>
    void random_seed(T seed) {
        random_engine = std::default_random_engine(seed);
    }
    template<typename T>
    std::vector<T> random_normal(int size) {
        std::vector<T> vec(size);
        std::normal_distribution<T> dist(0.0, 1.0);
        for (int i=0; i<size; i++) {
            vec[i] = dist(random_engine);
        }
        return vec;
    }
}
#endif