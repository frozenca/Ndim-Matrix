#include "Matrix.h"
#include <chrono>
#include <complex>
#include <iostream>
#include <random>
#include <numeric>

namespace fc = frozenca;

int main() {
    fc::Mat<float> A{{4, 1, -2, 2},
                     {1, 2, 0, 1},
                     {-2, 0, 3, -2},
                     {2, 1, -2, -1}};

    auto H = Hessenberg(A, true);

    std::cout << H << '\n';




}
