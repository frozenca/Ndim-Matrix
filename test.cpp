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

    fc::Mat<int> D {{12, -51, 4},
                    {6, 167, -68},
                    {-4, 24, -41}};

    auto [Q, R] = fc::QR(D);

    std::cout << Q << '\n';
    std::cout << R << '\n';

    auto D_3 = fc::pow(D, 3);

    std::cout << D_3 << '\n';

    A.swapRows(2, 3);

    std::cout << A << '\n';


}
