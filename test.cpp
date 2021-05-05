#include "Matrix.h"
#include <iostream>
#include <numeric>

int main() {
    frozenca::Matrix<float, 2> A {{2, 0, 2, 0.6},
                                  {3, 3, 4, -2},
                                  {5, 5, 4, 2},
                                  {-1, -2, 3.4, -1}};

    auto [P, L, U] = frozenca::LUP(A);
    for (auto p : P) {
        std::cout << p << ' ';
    }
    std::cout << '\n';

    std::cout << L << '\n';
    std::cout << U << '\n';



}
