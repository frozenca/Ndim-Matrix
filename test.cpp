#include "Matrix.h"
#include <iostream>
#include <numeric>

int main() {
    frozenca::Matrix<float, 2> A {{4, 12, -16},
                                  {12, 37, -43},
                                  {-16, -43, 98}};

    auto [L, L_] = frozenca::Cholesky(A);

    std::cout << L << '\n';
    std::cout << L_ << '\n';

}
