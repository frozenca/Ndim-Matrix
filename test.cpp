#include "Matrix.h"
#include <iostream>
#include <numeric>

int main() {
    frozenca::Matrix<float, 2> m {{1, 2, 3}, {4, 5, 6}};
    m = {{4, 6}, {1, 3}};
    std::cout << m(1, 1) << '\n';
    std::cout << m[{1, 1}] << '\n';

    // 3 x 4 x 5
    frozenca::Matrix<int, 3> m2 (3, 4, 5);
    std::iota(std::begin(m2), std::end(m2), 0);
    std::cout << m2 << '\n';

    auto sub = m2.submatrix({1, 1, 1}, {2, 3, 4});
    std::cout << sub << '\n';

    auto sub2 = m2.row(1);
    std::cout << sub2 << '\n';

    auto sub3 = m2.col(1);
    std::cout << sub3 << '\n';

    auto sub4 = sub.row(0);
    std::cout << sub4 << '\n';

    auto sub5 = sub.col(1);
    std::cout << sub5 << '\n';

}