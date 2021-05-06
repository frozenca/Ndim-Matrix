#include "Matrix.h"
#include <chrono>
#include <iostream>
#include <random>
#include <numeric>

namespace fc = frozenca;

int main() {
    fc::Mat<float> A {{1, 1, 1, 1, 1, 1},
                      {32, 16, 8, 4, 2, 1},
                      {243, 81, 27, 9, 3, 1},
                      {1024, 256, 64, 16, 4, 1},
                      {3125, 625, 125, 25, 5, 1},
                      {7776, 1296, 216, 36, 6, 1}};

    auto B = fc::inv(A);
    std::cout << B << '\n';





}
