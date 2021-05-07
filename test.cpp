#include "Matrix.h"
#include <chrono>
#include <complex>
#include <iostream>
#include <random>
#include <numeric>

namespace fc = frozenca;

int main() {
    fc::Mat<float> A{{-2, 2, 3, 4},
                     {-9, 7, 5, 5},
                     {-5, 2, 6, 6},
                     {-7, 2, 8, 9}};

//    auto H = fc::eigenval(A);
//
//    for (const auto& v : H) {
//        std::cout << v << '\n';
//    }
//    std::cout << '\n';

    auto H = fc::eigenvec(A);

    fc::Mat<std::complex<float>> AC = A;

    for (const auto& [v, V] : H) {
        std::cout << v << ' ' << V << '\n';
        std::cout << fc::normalize(fc::dot(AC, V)) << '\n';
    }
    std::cout << '\n';

    fc::Mat<float> B{{0, 1, 1, 0},
                     {1, 0, 0, 1},
                     {0, 0, 0, 1},
                     {0, 0, -1, 0}};




}
