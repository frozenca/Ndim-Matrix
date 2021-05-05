#include "Matrix.h"
#include <iostream>
#include <numeric>

int main() {
    frozenca::Matrix<int, 1> A(3);
    frozenca::Matrix<int, 1> B(3);
    std::iota(std::begin(A), std::end(A), 0);
    std::iota(std::begin(B), std::end(B), 0);
    auto C = frozenca::dot(A, B);
    std::cout << C << '\n';
    auto D = frozenca::matmul(A, B);
    std::cout << D << '\n';


}
