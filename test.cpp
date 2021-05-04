#include "Matrix.h"
#include <iostream>
#include <numeric>

int main() {
    frozenca::Matrix<int, 4> mat (2, 2, 3, 3);
    std::iota(std::begin(mat), std::end(mat), 1);
    std::cout << mat << '\n';

    auto mat2 = frozenca::reshape<int, 2, 4>(std::move(mat), {6, 6});
    std::cout << mat2 << '\n';



}
