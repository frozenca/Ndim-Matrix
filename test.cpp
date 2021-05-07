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

    auto H = fc::eigenval(A);

    for (const auto& h : H) {
        std::cout << h << '\n';
    }
    std::cout << '\n';

    fc::Mat<float> B{{0, 1, 1, 0},
                     {1, 0, 0, 1},
                     {0, 0, 0, 1},
                     {0, 0, -1, 0}};

    auto H2 = fc::eigenval(B);

    for (const auto& h : H2) {
        std::cout << h << '\n';
    }
    std::cout << '\n';



}
