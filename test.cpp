#include "Matrix.h"
#include <chrono>
#include <complex>
#include <iostream>
#include <random>
#include <numeric>

namespace fc = frozenca;

int main() {
    {
        fc::Mat<float> A{{1, 0, 0, 0, 2},
                         {0, 0, 3, 0, 0},
                         {0, 0, 0, 0, 0},
                         {0, 2, 0, 0, 0}};

        auto[U, Sigma, V] = fc::SVD(A);
        std::cout << U << '\n';
        std::cout << Sigma << '\n';
        std::cout << V << '\n';
    }
    {
        fc::Mat<std::complex<float>> A{{1, 0, 0, 0, 2},
                         {0, 0, 3, 0, 0},
                         {0, 0, 0, 0, 0},
                         {0, 2, 0, 0, 0}};

        auto[U, Sigma, V] = fc::SVD(A);
        std::cout << U << '\n';
        std::cout << Sigma << '\n';
        std::cout << V << '\n';
    }




}
