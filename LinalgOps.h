#ifndef FROZENCA_LINALGOPS_H
#define FROZENCA_LINALGOPS_H

#include <bit>
#include <ranges>
#include "MatrixImpl.h"

namespace frozenca {

namespace {

template <std::semiregular U, std::semiregular V, std::semiregular T>
requires DotProductableTo<U, V, T>
void DotTo(T& m,
           const MatrixView<U, 1>& m1,
           const MatrixView<V, 1>& m2) {
    m += std::inner_product(std::begin(m1), std::end(m1), std::begin(m2), T{0});
}

template <std::semiregular U, std::semiregular V, std::semiregular T>
requires DotProductableTo<U, V, T>
void DotTo(MatrixView<T, 1>& m,
           const MatrixView<U, 1>& m1,
           const MatrixView<V, 2>& m2) {
    assert(m.dims(0) == m2.dims(1));
    std::size_t c = m2.dims(1);
    for (std::size_t j = 0; j < c; ++j) {
        auto col2 = m2.col(j);
        m[j] += std::inner_product(std::begin(m1), std::end(m1), std::begin(col2), T{0});
    }
}

template <std::semiregular U, std::semiregular V, std::semiregular T, std::size_t N2>
requires DotProductableTo<U, V, T> && (N2 > 2)
void DotTo(MatrixView<T, N2 - 1>& m,
           const MatrixView<U, 1>& m1,
           const MatrixView<V, N2>& m2) {
    assert(m.dims(0) == m2.dims(0));
    std::size_t r = m.dims(0);
    for (std::size_t i = 0; i < r; ++i) {
        auto row0 = m.row(i);
        auto row2 = m2.row(i);
        DotTo(row0, m1, row2);
    }
}

template <std::semiregular U, std::semiregular V, std::semiregular T,
        std::size_t N1, std::size_t N2>
requires DotProductableTo<U, V, T> && (N1 > 1)
void DotTo(MatrixView<T, N1 - 1>& m,
           const MatrixView<U, N1>& m1,
           const MatrixView<V, 1>& m2) {
    assert(m.dims(0) == m1.dims(0));
    std::size_t r = m.dims(0);
    for (std::size_t i = 0; i < r; ++i) {
        auto row0 = m.row(i);
        auto row1 = m1.row(i);
        DotTo(row0, row1, m2);
    }
}

template <std::semiregular U, std::semiregular V, std::semiregular T,
        std::size_t N1, std::size_t N2>
requires DotProductableTo<U, V, T> && (N1 > 1) && (N2 > 1)
void DotTo(MatrixView<T, N1 + N2 - 2>& m,
           const MatrixView<U, N1>& m1,
           const MatrixView<V, N2>& m2) {
    assert(m.dims(0) == m1.dims(0));
    std::size_t r = m.dims(0);
    for (std::size_t i = 0; i < r; ++i) {
        auto row0 = m.row(i);
        auto row1 = m1.row(i);
        DotTo(row0, row1, m2);
    }
}

template <typename Derived0, typename Derived1, typename Derived2,
        std::semiregular U, std::semiregular V, std::semiregular T,
        std::size_t N1, std::size_t N2>
requires DotProductableTo<U, V, T>
void DotTo(MatrixBase<Derived0, T, (N1 + N2 - 2)>& m,
           const MatrixBase<Derived1, U, N1>& m1,
           const MatrixBase<Derived2, V, N2>& m2) {
    MatrixView<T, (N1 + N2 - 2)> m_view (m);
    MatrixView<U, N1> m1_view (m1);
    MatrixView<V, N2> m2_view (m2);
    DotTo(m_view, m1_view, m2_view);
}

template <typename Derived1, typename Derived2,
        std::semiregular U, std::semiregular V, std::semiregular T>
requires DotProductableTo<U, V, T>
void DotTo(T& m,
           const MatrixBase<Derived1, U, 1>& m1,
           const MatrixBase<Derived2, V, 1>& m2) {
    MatrixView<U, 1> m1_view (m1);
    MatrixView<V, 1> m2_view (m2);
    DotTo(m, m1_view, m2_view);
}

template <std::semiregular U, std::semiregular V, std::semiregular T,
        std::size_t N1, std::size_t N2, std::size_t N>
requires DotProductableTo<U, V, T> && (std::max(N1, N2) == N)
void MatmulTo(MatrixView<T, N>& m,
           const MatrixView<U, N1>& m1,
           const MatrixView<V, N2>& m2) {
    if constexpr (N == 2) {
        DotTo(m, m1, m2);
    } else {
        m.applyFunctionWithBroadcast(m1, m2, MatmulTo<U, V, T, std::min(N1, N - 1), std::min(N2, N - 1), N - 1>);
    }
}

} // anonymous namespace

template <typename Derived1, typename Derived2,
        std::semiregular U, std::semiregular V,
        std::size_t M, std::size_t N,
        std::semiregular T = MulType<U, V>> requires DotProductableTo<U, V, T>
decltype(auto) dot(const MatrixBase<Derived1, U, M>& m1, const MatrixBase<Derived2, V, N>& m2) {
    auto dims = dotDims(m1.dims(), m2.dims());
    Matrix<T, (M + N - 2)> res = zeros<T, (M + N - 2)>(dims);
    DotTo(res, m1, m2);
    return res;
}

template <typename Derived1, typename Derived2,
        std::semiregular U, std::semiregular V,
        std::semiregular T = MulType<U, V>> requires DotProductableTo<U, V, T>
decltype(auto) dot(const MatrixBase<Derived1, U, 1>& m1, const MatrixBase<Derived2, V, 1>& m2) {
    if (m1.dims(0) != m2.dims(0)) {
        throw std::invalid_argument("Cannot do dot product, shape is not aligned");
    }
    T res {0};
    DotTo(res, m1, m2);
    return res;
}

template <typename Derived1, typename Derived2,
        std::semiregular U, std::semiregular V,
        std::size_t N1, std::size_t N2,
        std::semiregular T = MulType<U, V>> requires DotProductableTo<U, V, T>
decltype(auto) matmul(const MatrixBase<Derived1, U, N1>& m1, const MatrixBase<Derived2, V, N2>& m2) {
    constexpr std::size_t N = std::max({N1, N2, 2lu});
    auto dims = matmulDims(m1.dims(), m2.dims());
    Matrix<T, N> res = zeros<T, N>(dims);
    auto m1_view = [&](){
        if constexpr (N1 == 1) {
            return MatrixView<U, 2>({1, m1.dims(0)}, const_cast<U*>(m1.dataView()), {m1.dims(0), 1});
        } else {
            return MatrixView<U, N1>(m1);
        }
    };
    auto m2_view = [&](){
        if constexpr (N2 == 1) {
            return MatrixView<V, 2>({m2.dims(0), 1}, const_cast<V*>(m2.dataView()), {1, 1});
        } else {
            return MatrixView<V, N2>(m2);
        }
    };
    if constexpr (N == 2) {
        DotTo(res, m1_view(), m2_view());
    } else {
        res.applyFunctionWithBroadcast(m1_view(), m2_view(), MatmulTo<U, V, T,
                std::min(std::max(N1, 2lu), N - 1),
                std::min(std::max(N2, 2lu), N - 1), N - 1>);
    }
    return res;
}

template <typename Derived, isScalar A, isScalar B = RealTypeT<A>> requires RealTypeTo<A, B>
std::tuple<std::vector<std::size_t>, Matrix<B, 2>, Matrix<B, 2>> LUP(const MatrixBase<Derived, A, 2>& mat) {

    std::size_t n = mat.dims(0);
    std::size_t C = mat.dims(1);
    if (n != C) {
        throw std::invalid_argument("Not a square matrix, cannot do LUP decomposition");
    }
    std::vector<std::size_t> P (n);
    std::iota(std::begin(P), std::end(P), 0lu);

    Matrix<B, 2> A_ = mat;
    Matrix<B, 2> U = zeros_like(A_);
    Matrix<B, 2> L = identity<B>(n);

    for (std::size_t k = 0; k < n; ++k) {
        A p {0};
        std::size_t k_ = -1;
        for (std::size_t i = k; i < n; ++i) {
            auto val = std::fabs(A_(i, k));
            if (val > p) {
                p = val;
                k_ = i;
            }
        }
        if (p == A{0}) {
            throw std::invalid_argument("Singular matrix");
        }
        std::swap(P[k], P[k_]);
        for (std::size_t i = 0; i < n; ++i) {
            std::swap(A_(k, i), A_(k_, i));
        }
        for (std::size_t i = k + 1; i < n; ++i) {
            A_(i, k) /= A_(k, k);
            for (std::size_t j = k + 1; j < n; ++j) {
                A_(i, j) -= A_(i, k) * A_(k, j);
            }
        }
    }

    for (std::size_t i = 0; i < n; ++i) {
        for (std::size_t j = 0; j < n; ++j) {
            if (j < i) {
                L(i, j) = A_(i, j);
            } else {
                U(i, j) = A_(i, j);
            }
        }
    }

    return {P, L, U};
}

template <typename Derived, isScalar A, isScalar B = RealTypeT<A>> requires RealTypeTo<A, B>
std::pair<Matrix<B, 2>, Matrix<B, 2>> Cholesky(const MatrixBase<Derived, A, 2>& mat) {
    std::size_t n = mat.dims(0);
    std::size_t C = mat.dims(1);
    if (n != C) {
        throw std::invalid_argument("Not a square matrix, cannot do Cholesky decomposition");
    }
    Matrix<B, 2> A_ = mat;
    Matrix<B, 2> L = zeros_like(A_);

    for (std::size_t i = 0; i < n; ++i) {
        for (std::size_t j = 0; j < i + 1; ++j) {
            A sum {0};
            for (std::size_t k = 0; k < j; ++k) {
                if constexpr (isComplex<A>) {
                    sum += L(i, k) * conj(L(j, k));
                } else {
                    sum += L(i, k) * L(j, k);
                }
            }

            if (i == j) {
                L(i, j) = std::sqrt(A_(i, i) - sum);
            } else {
                L(i, j) = ((A{1.0} / L(j, j)) * (A_(i, j) - sum));
            }
        }
    }
    auto L_ = transpose(L);
    if constexpr (isComplex<A>) {
        L_.conj();
    }
    return {L, L_};
}

template <typename Derived, isScalar A>
bool isLowerTriangular(const MatrixBase<Derived, A, 2>& mat) {
    std::size_t R = mat.dims(0);
    std::size_t C = mat.dims(1);
    for (std::size_t i = 0; i < R; ++i) {
        for (std::size_t j = i + 1; j < C; ++j) {
            if (mat(i, j) != A{0}) {
                return false;
            }
        }
    }
    return true;
}

template <typename Derived, isScalar A>
bool isUpperTriangular(const MatrixBase<Derived, A, 2>& mat) {
    std::size_t R = mat.dims(0);
    std::size_t C = mat.dims(1);
    for (std::size_t i = 0; i < R; ++i) {
        for (std::size_t j = 0; j < std::min(i, C); ++j) {
            if (mat(i, j) != A{0}) {
                return false;
            }
        }
    }
    return true;
}

template <typename Derived, isScalar A>
bool isTriangular(const MatrixBase<Derived, A, 2>& mat) {
    return isLowerTriangular(mat) || isUpperTriangular(mat);
}

template <typename Derived, isScalar A>
A det(const MatrixBase<Derived, A, 2>& mat) {
    std::size_t n = mat.dims(0);
    std::size_t C = mat.dims(1);
    if (n != C) {
        throw std::invalid_argument("Not a square matrix, cannot compute determinant");
    }

    if (isTriangular(mat)) {
        A det_val {1};
        for (std::size_t i = 0; i < n; ++i) {
            det_val *= mat(i, i);
        }
        return det_val;
    }
    auto [P, L, U] = LUP(mat);
    return det(L) * det(U);
}

namespace {

template <typename Derived, isScalar A, isScalar B = RealTypeT<A>> requires RealTypeTo<A, B>
Matrix<B, 2> inv_impl(const MatrixBase<Derived, A, 2>& mat) {
    std::size_t n = mat.dims(0);
    if (n == 1) {
        if (mat(0, 0) == A{0}) {
            throw std::invalid_argument("Singular matrix, cannot invertible");
        }
        Matrix<B, 2> Inv = full_like(mat, B{1} / mat(0, 0));
        return Inv;
    } else if (n == 2) {
        if (mat(0, 0) == A{0} || mat(1, 1) == A{0}) {
            throw std::invalid_argument("Singular matrix, cannot invertible");
        }
        auto det_val = mat(0, 0) * mat(1, 1) - mat(0, 1) * mat(1, 0);
        Matrix<B, 2> Inv {{mat(1, 1), -mat(0, 1)}, {-mat(1, 0), mat(0, 0)}};
        Inv /= det_val;
        return Inv;
    } else {
        auto half = std::bit_floor(n);
        if (std::has_single_bit(n)) {
            half >>= 1;
        }
        auto P = mat.submatrix({0, 0}, {half, half});
        auto Q = mat.submatrix({0, half}, {half, n});
        auto R = mat.submatrix({half, 0}, {n, half});
        auto S = mat.submatrix({half, half}, {n, n});

        Matrix<B, 2> Inv = empty_like(mat);
        auto InvP = Inv.submatrix({0, 0}, {half, half});
        auto InvQ = Inv.submatrix({0, half}, {half, n});
        auto InvR = Inv.submatrix({half, 0}, {n, half});
        auto InvS = Inv.submatrix({half, half}, {n, n});
        if (std::all_of(std::begin(Q), std::end(Q), [](const auto& q){return q == A{0};})) {
            // [ P^-1          0    ]
            // [ -S^-1RP^-1    S^-1 ]
            InvP = inv_impl(P);
            std::fill(std::begin(InvQ), std::end(InvQ), A{0});
            InvS = inv_impl(S);
            InvR = -dot(dot(InvS, R), InvP);
        } else {
            // [ P^-1 + TUV    TU ]
            // [ UV            U  ]
            // T = -P^-1Q
            // V = -RP^-1
            // InvS = U = (S + VQ)^-1

            InvP = inv_impl(P);
            auto T = -dot(InvP, Q);
            auto V = -dot(R, InvP);
            InvS = inv_impl(S + dot(V, Q));
            InvQ = dot(T, InvS);
            InvR = dot(InvS, V);
            InvP += dot(T, InvR);
        }
        return Inv;
    }
}

} // anonymous namespace

template <typename Derived, isScalar A, isScalar B = RealTypeT<A>> requires RealTypeTo<A, B>
Matrix<B, 2> inv(const MatrixBase<Derived, A, 2>& mat) {
    std::size_t n = mat.dims(0);
    std::size_t C = mat.dims(1);
    if (n != C) {
        throw std::invalid_argument("Not a square matrix, cannot invertible");
    }
    return inv_impl(mat);
}

} // namespace frozenca

#endif //FROZENCA_LINALGOPS_H
