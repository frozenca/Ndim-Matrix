#include "Matrix.h"
#include <iostream>
#include <numeric>

int main() {
    frozenca::Matrix<int, 4> mat (2, 2, 3, 3);
    std::iota(std::begin(mat), std::end(mat), 1);
    std::cout << mat << '\n';

    auto mat2 = frozenca::reshape<int, 2, 4>(std::move(mat), {6, 6});
    std::cout << mat2 << '\n';

    frozenca::Matrix<int, 3> A (2, 3, 4);
    std::iota(std::begin(A), std::end(A), 0);
    std::cout << A << '\n';

    auto A2 = frozenca::transpose(A, {0, 2, 1});
    std::cout << A2 << '\n';

    A2 = frozenca::transpose(A, {1, 0, 2});
    std::cout << A2 << '\n';

    A2 = frozenca::transpose(A, {1, 2, 0});
    std::cout << A2 << '\n';

    A2 = frozenca::transpose(A, {2, 0, 1});
    std::cout << A2 << '\n';

    A2 = frozenca::transpose(A, {2, 1, 0});
    std::cout << A2 << '\n';


}
