#include "Matrix.h"
#include <iostream>
#include <numeric>

int main() {
    frozenca::Matrix<int, 3> A(2, 2, 3);
    frozenca::Matrix<int, 3> B(2, 3, 2);
    std::iota(std::begin(A), std::end(A), 0);
    std::iota(std::begin(B), std::end(B), 0);
    auto C = frozenca::dot(A, B);
    std::cout << C << '\n';


}
