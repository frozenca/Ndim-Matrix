#include "Matrix.h"
#include <iostream>
#include <numeric>

namespace fc = frozenca;

int main() {
    fc::Mat<float> A {{2, 0, 2, 0.6},
                      {3, 3, 4, -2},
                      {5, 5, 4, 2},
                      {-1, -2, 3.4, -1}};

    auto B = fc::inv(A);

    std::cout << B << '\n';

    fc::Mat<float> C {{-4, -3, -2},
                      {-1, 0, 1},
                      {2, 3, 4}};

    std::cout << fc::norm(C) << '\n';

}
