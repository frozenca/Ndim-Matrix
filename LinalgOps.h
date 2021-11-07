#ifndef FROZENCA_LINALGOPS_H
#define FROZENCA_LINALGOPS_H

#include <bit>
#include <cmath>
#include <queue>
#include <vector>
#include "MatrixImpl.h"

namespace frozenca {

namespace {

static constexpr float tolerance_soft = 1e-6;
static constexpr float tolerance_hard = 1e-10;
static constexpr std::size_t max_iter = 100;
static constexpr std::size_t local_iter = 15;

template <std::semiregular U, std::semiregular V, std::semiregular T>
requires DotProductableTo<U, V, T>
void DotTo(T& m,
           const MatrixView<U, 1, true>& m1,
           const MatrixView<V, 1, true>& m2) {
    m += std::inner_product(std::begin(m1), std::end(m1), std::begin(m2), T{0});
}

template <std::semiregular U, std::semiregular V, std::semiregular T, bool isRowMajor2>
requires DotProductableTo<U, V, T>
void DotTo(MatrixView<T, 1, false>& m,
           const MatrixView<U, 1, true>& m1,
           const MatrixView<V, 2, true, isRowMajor2>& m2) {
    assert(m.dims(0) == m2.dims(1));
    std::size_t c = m.dims(0);
    for (std::size_t j = 0; j < c; ++j) {
        auto col2 = m2.col(j);
        m[j] += std::inner_product(std::begin(m1), std::end(m1), std::begin(col2), T{0});
    }
}

template <std::semiregular U, std::semiregular V, std::semiregular T>
requires DotProductableTo<U, V, T>
void DotTo(MatrixView<T, 1, false>& m,
           const MatrixView<U, 2, true>& m1,
           const MatrixView<V, 1, true>& m2) {
    assert(m.dims(0) == m1.dims(0));
    std::size_t r = m.dims(0);
    for (std::size_t j = 0; j < r; ++j) {
        auto row1 = m1.row(j);
        m[j] += std::inner_product(std::begin(row1), std::end(row1), std::begin(m2), T{0});
    }
}

template <std::semiregular U, std::semiregular V, std::semiregular T,
        std::size_t N2>
requires DotProductableTo<U, V, T> && (N2 > 2)
void DotTo(MatrixView<T, N2 - 1, false>& m,
           const MatrixView<U, 1, true>& m1,
           const MatrixView<V, N2, true>& m2) {
    assert(m.dims(0) == m2.dims(0));
    std::size_t r = m.dims(0);
    for (std::size_t i = 0; i < r; ++i) {
        auto row0 = m.row(i);
        auto row2 = m2.row(i);
        DotTo(row0, m1, row2);
    }
}

template <std::semiregular U, std::semiregular V, std::semiregular T,
        std::size_t N1>
requires DotProductableTo<U, V, T> && (N1 > 2)
void DotTo(MatrixView<T, N1 - 1, false>& m,
           const MatrixView<U, N1, true>& m1,
           const MatrixView<V, 1, true>& m2) {
    assert(m.dims(0) == m1.dims(0));
    std::size_t r = m.dims(0);
    for (std::size_t i = 0; i < r; ++i) {
        auto row0 = m.row(i);
        auto row1 = m1.row(i);
        DotTo(row0, row1, m2);
    }
}

template <std::semiregular U, std::semiregular V, std::semiregular T,
        std::size_t N1, std::size_t N2, bool isRowMajor2>
requires DotProductableTo<U, V, T> && (N1 + N2 > 2)
void DotTo(MatrixView<T, N1 + N2 - 2, false>& m,
           const MatrixView<U, N1, true>& m1,
           const MatrixView<V, N2, true, isRowMajor2>& m2) {
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
        std::size_t N1, std::size_t N2, bool isRowMajor2>
requires DotProductableTo<U, V, T> && (N1 + N2 > 2)
void DotTo(MatrixBase<Derived0, T, (N1 + N2 - 2)>& m,
           const MatrixBase<Derived1, U, N1>& m1,
           const MatrixBase<Derived2, V, N2, isRowMajor2>& m2) {
    MatrixView<T, (N1 + N2 - 2), false> m_view (m);
    MatrixView<U, N1, true> m1_view (m1);
    MatrixView<V, N2, true, isRowMajor2> m2_view (m2);
    DotTo(m_view, m1_view, m2_view);
}

template <typename Derived1, typename Derived2,
        std::semiregular U, std::semiregular V, std::semiregular T>
requires DotProductableTo<U, V, T>
void DotTo(T& m,
           const MatrixBase<Derived1, U, 1>& m1,
           const MatrixBase<Derived2, V, 1>& m2) {
    MatrixView<U, 1, true> m1_view (m1);
    MatrixView<V, 1, true> m2_view (m2);
    DotTo(m, m1_view, m2_view);
}

template <std::semiregular U, std::semiregular V, std::semiregular T,
        std::size_t N1, std::size_t N2, std::size_t N, bool isRowMajor2>
requires DotProductableTo<U, V, T> && (std::max(N1, N2) == N)
void MatmulTo(MatrixView<T, N, false>& m,
              const MatrixView<U, N1, true>& m1,
              const MatrixView<V, N2, true, isRowMajor2>& m2) {
    if constexpr (N == 2) {
        DotTo(m, m1, m2);
    } else {
        static_assert(isRowMajor2);
        m.applyFunctionWithBroadcast(m1, m2, MatmulTo<U, V, T, std::min(N1, N - 1), std::min(N2, N - 1), N - 1>);
    }
}

} // anonymous namespace

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
        std::size_t M, std::size_t N, bool isRowMajor2,
        std::semiregular T = MulType<U, V>> requires DotProductableTo<U, V, T>
decltype(auto) dot(const MatrixBase<Derived1, U, M, true>& m1, const MatrixBase<Derived2, V, N, isRowMajor2>& m2) {
    auto dims = dotDims(m1.dims(), m2.dims());
    Matrix<T, (M + N - 2)> res = zeros<T, (M + N - 2)>(dims);
    DotTo(res, m1, m2);
    return res;
}

// (m x n) x (n x 1)
template <typename Derived1, typename Derived2,
        std::semiregular U, std::semiregular V,
        std::semiregular T = MulType<U, V>> requires DotProductableTo<U, V, T>
decltype(auto) dot(const MatrixBase<Derived1, U, 2>& m1, const MatrixBase<Derived2, V, 1>& m2) {
    if (m1.dims(1) != m2.dims(0)) {
        throw std::invalid_argument("Cannot do dot product, shape is not aligned");
    }
    Vec<T> res (m1.dims(0));
    DotTo(res, m1, m2);
    return res;
}

// (1 x m) x (m x n)
template <typename Derived1, typename Derived2,
        std::semiregular U, std::semiregular V,
        std::semiregular T = MulType<U, V>> requires DotProductableTo<U, V, T>
decltype(auto) dot(const MatrixBase<Derived2, V, 1>& m1, const MatrixBase<Derived1, U, 2, false>& m2) {
    if (m1.dims(0) != m2.dims(0)) {
        throw std::invalid_argument("Cannot do dot product, shape is not aligned");
    }
    Vec<T> res (m2.dims(1));
    DotTo(res, m1, m2);
    return res;
}

// (m x 1) x (1 x n)
template <typename Derived1, typename Derived2,
        std::semiregular U, std::semiregular V,
        std::semiregular T = MulType<U, V>> requires DotProductableTo<U, V, T>
decltype(auto) outer(const MatrixBase<Derived2, V, 1>& m1, const MatrixBase<Derived1, U, 1>& m2) {
    std::size_t m = m1.dims(0);
    std::size_t n = m2.dims(0);
    Mat<T> res (m, n);
    for (std::size_t i = 0; i < m; ++i) {
        for (std::size_t j = 0; j < n; ++j) {
            res[{i, j}] = m1[i] * m2[j];
        }
    }
    return res;
}

template <typename Derived1, typename Derived2,
        std::semiregular U, std::semiregular V,
        std::size_t N1, std::size_t N2, bool isRowMajor1, bool isRowMajor2,
        std::semiregular T = MulType<U, V>> requires DotProductableTo<U, V, T>
decltype(auto) matmul(const MatrixBase<Derived1, U, N1, isRowMajor1>& m1, const MatrixBase<Derived2, V, N2, isRowMajor2>& m2) {
    constexpr std::size_t N = std::max({N1, N2, 2lu});
    auto dims = matmulDims(m1.dims(), m2.dims());
    Matrix<T, N> res = zeros<T, N>(dims);
    auto m1_view = [&](){
        if constexpr (N1 == 1) {
            return MatrixView<U, 2, true, true>({1, m1.dims(0)}, m1.dataView(), {m1.dims(0), 1});
        } else {
            return MatrixView<U, N1, true, isRowMajor1>(m1);
        }
    };
    auto m2_view = [&](){
        if constexpr (N2 == 1) {
            return MatrixView<V, 2, true, false>({m2.dims(0), 1}, m2.dataView(), {1, 1});
        } else {
            return MatrixView<V, N2, true, isRowMajor2>(m2);
        }
    };
    if constexpr (N == 2) {
        DotTo(res, m1_view(), m2_view());
    } else {
        static_assert(isRowMajor1 && isRowMajor2);
        res.applyFunctionWithBroadcast(m1_view(), m2_view(), MatmulTo<U, V, T,
                std::min(std::max(N1, 2lu), N - 1),
                std::min(std::max(N2, 2lu), N - 1), N - 1>);
    }
    return res;
}

template <typename Derived, isScalar S, isScalar T = ScalarTypeT<S>> requires ScalarTypeTo<S, T>
std::tuple<std::vector<std::size_t>, Mat<T>, Mat<T>> LUP(const MatrixBase<Derived, S, 2>& mat) {

    std::size_t n = mat.dims(0);
    std::size_t C = mat.dims(1);
    if (n != C) {
        throw std::invalid_argument("Not a square matrix, cannot do LUP decomposition");
    }
    std::vector<std::size_t> P (n);
    std::iota(std::begin(P), std::end(P), 0lu);

    Mat<T> M = mat;
    Mat<T> U = zeros_like(M);
    Mat<T> L = identity<T>(n);

    for (std::size_t k = 0; k < n; ++k) {
        S p {0};
        std::size_t k_ = -1;
        for (std::size_t i = k; i < n; ++i) {
            auto val = std::fabs(M[{i, k}]);
            if (val > p) {
                p = val;
                k_ = i;
            }
        }
        if (p == S{0}) {
            throw std::invalid_argument("Singular matrix");
        }
        std::swap(P[k], P[k_]);
        M.swapRows(k, k_);
        for (std::size_t i = k + 1; i < n; ++i) {
            M[{i, k}] /= M[{k, k}];
            for (std::size_t j = k + 1; j < n; ++j) {
                M[{i, j}] -= M[{i, k}] * M[{k, j}];
            }
        }
    }

    for (std::size_t i = 0; i < n; ++i) {
        for (std::size_t j = 0; j < n; ++j) {
            if (j < i) {
                L[{i, j}] = M[{i, j}];
            } else {
                U[{i, j}] = M[{i, j}];
            }
        }
    }

    return {P, L, U};
}

template <typename Derived, isScalar U, bool isRowMajor, isScalar T = ScalarTypeT<U>> requires ScalarTypeTo<U, T>
std::pair<Mat<T>, Mat<T>> Cholesky(const MatrixBase<Derived, U, 2, isRowMajor>& mat) {
    std::size_t n = mat.dims(0);
    std::size_t C = mat.dims(1);
    if (n != C) {
        throw std::invalid_argument("Not a square matrix, cannot do Cholesky decomposition");
    }
    Mat<T, isRowMajor> M = mat;
    Mat<T> L = zeros_like(M);

    for (std::size_t i = 0; i < n; ++i) {
        for (std::size_t j = 0; j < i + 1; ++j) {
            U sum {0};
            for (std::size_t k = 0; k < j; ++k) {
                if constexpr (isComplex<U>) {
                    sum += L[{i, k}] * conj(L[{j, k}]);
                } else {
                    sum += L[{i, k}] * L[{j, k}];
                }
            }

            if (i == j) {
                L[{i, j}] = std::sqrt(M[{i, i}] - sum);
            } else {
                L[{i, j}] = ((U{1.0f} / L[{j, j}]) * (M[{i, j}] - sum));
            }
        }
    }
    auto L_ = transpose(L);
    if constexpr (isComplex<U>) {
        L_.conj();
    }
    return {L, L_};
}

template <typename Derived, isScalar U, bool isRowMajor>
bool isLowerTriangular(const MatrixBase<Derived, U, 2, isRowMajor>& mat) {
    std::size_t R = mat.dims(0);
    std::size_t C = mat.dims(1);
    for (std::size_t i = 0; i < R; ++i) {
        for (std::size_t j = i + 1; j < C; ++j) {
            if (mat[{i, j}] != U{0}) {
                return false;
            }
        }
    }
    return true;
}

template <typename Derived, isScalar U, bool isRowMajor>
bool isUpperTriangular(const MatrixBase<Derived, U, 2, isRowMajor>& mat) {
    std::size_t R = mat.dims(0);
    std::size_t C = mat.dims(1);
    for (std::size_t i = 0; i < R; ++i) {
        for (std::size_t j = 0; j < std::min(i, C); ++j) {
            if (mat[{i, j}] != U{0}) {
                return false;
            }
        }
    }
    return true;
}

template <typename Derived, isScalar U, bool isRowMajor>
bool isTriangular(const MatrixBase<Derived, U, 2, isRowMajor>& mat) {
    return isLowerTriangular(mat) || isUpperTriangular(mat);
}

template <typename Derived, isScalar T, bool isRowMajor>
T tr(const MatrixBase<Derived, T, 2, isRowMajor>& mat) {
    std::size_t n = std::min(mat.dims(0), mat.dims(1));
    T val {0};
    for (std::size_t i = 0; i < n; ++i) {
        val += mat[{i, i}];
    }
    return val;
}

template <typename Derived, isScalar T, bool isRowMajor>
T det(const MatrixBase<Derived, T, 2, isRowMajor>& mat) {
    std::size_t n = mat.dims(0);
    std::size_t C = mat.dims(1);
    if (n != C) {
        throw std::invalid_argument("Not a square matrix, cannot compute determinant");
    }

    if (isTriangular(mat)) {
        T det_val {1};
        for (std::size_t i = 0; i < n; ++i) {
            det_val *= mat[{i, i}];
        }
        return det_val;
    }
    auto [P, L, U] = LUP(mat);
    return det(L) * det(U);
}

namespace {

template <typename Derived, isScalar U, bool isRowMajor, isScalar T = ScalarTypeT<U>> requires ScalarTypeTo<U, T>
Mat<T, isRowMajor> inv_impl(const MatrixBase<Derived, U, 2, isRowMajor>& mat) {
    std::size_t n = mat.dims(0);

    Mat<T, isRowMajor> M = mat;
    Mat<T, isRowMajor> Inv = identity<T, isRowMajor>(n);

    if constexpr (isRowMajor) {
        for (std::size_t i = 0; i < n; ++i) {
            // pivoting
            for (std::size_t k = i + 1; k < n; ++k) {
                if (M[{i, i}] == T{0} && std::fabs(M[{k, i}]) != T{0}) {
                    M.swapRows(i, k);
                    Inv.swapRows(i, k);
                    break;
                }
            }
            if (M[{i, i}] == T{0}) {
                throw std::invalid_argument("Singular matrix, cannot invertible");
            }
            for (std::size_t j = i + 1; j < n; ++j) {
                T coeff = M[{j, i}] / M[{i, i}];
                Inv.row(j) -= coeff * Inv.row(i);
                M.row(j) -= coeff * M.row(i);
            }
        }
        for (std::size_t i = n - 1; i < n; --i) {
            for (std::size_t j = 0; j < i; ++j) {
                T coeff = M[{j, i}] / M[{i, i}];
                Inv.row(j) -= coeff * Inv.row(i);
                M[{j, i}] = T{0};
            }
            Inv.row(i) /= M[{i, i}];
            M[{i, i}] = 1;
        }
    } else {
        for (std::size_t i = 0; i < n; ++i) {
            // pivoting
            for (std::size_t k = i + 1; k < n; ++k) {
                if (M[{i, i}] == T{0} && std::fabs(M[{i, k}]) != T{0}) {
                    M.swapCols(i, k);
                    Inv.swapCols(i, k);
                    break;
                }
            }
            if (M[{i, i}] == T{0}) {
                throw std::invalid_argument("Singular matrix, cannot invertible");
            }
            for (std::size_t j = i + 1; j < n; ++j) {
                T coeff = M[{i, j}] / M[{i, i}];
                Inv.col(j) -= coeff * Inv.col(i);
                M.col(j) -= coeff * M.col(i);
            }
        }
        for (std::size_t i = n - 1; i < n; --i) {
            for (std::size_t j = 0; j < i; ++j) {
                T coeff = M[{i, j}] / M[{i, i}];
                Inv.col(j) -= coeff * Inv.col(i);
                M[{i, j}] = T{0};
            }
            Inv.col(i) /= M[{i, i}];
            M[{i, i}] = 1;
        }
    }
    return Inv;
}

} // anonymous namespace

template <typename Derived, isScalar U, bool isRowMajor, isScalar T = ScalarTypeT < U>> requires ScalarTypeTo<U, T>
Mat<T, isRowMajor> inv(const MatrixBase<Derived, U, 2, isRowMajor>& mat) {
    std::size_t n = mat.dims(0);
    std::size_t C = mat.dims(1);
    if (n != C) {
        throw std::invalid_argument("Not a square matrix, cannot invertible");
    }
    return inv_impl(mat);
}

namespace {

template <typename Derived, isScalar U, isScalar T = ScalarTypeT<U>> requires ScalarTypeTo<U, T>
Mat<T> pow_impl(const MatrixBase<Derived, U, 2>& mat, int p) {
    assert(p >= 1);
    if (p == 1) {
        return mat;
    } else {
        auto P = pow_impl(mat, p / 2);
        if (p % 2) {
            return dot(dot(P, P), mat);
        } else {
            return dot(P, P);
        }
    }
}

} // anonymous namespace

template <typename Derived, isScalar U, isScalar T = ScalarTypeT < U>> requires ScalarTypeTo<U, T>
Mat<T> pow(const MatrixBase<Derived, U, 2>& mat, int p) {
    std::size_t n = mat.dims(0);
    std::size_t C = mat.dims(1);
    if (n != C) {
        throw std::invalid_argument("Not a square matrix, cannot invertible");
    }
    if (p == 0) {
        return identity<T>(n);
    } else if (p < 0) {
        return pow_impl(inv_impl(mat), -p);
    } else {
        return pow_impl(mat, p);
    }
}

template <typename Derived, isScalar U, isReal T = RealTypeT<U>> requires RealTypeTo<U, T>
T norm(const MatrixBase<Derived, U, 1>& vec, std::size_t p = 2) {
    if (p == 0) {
        throw std::invalid_argument("Norm is undefined");
    } else if (p == 1) {
        return std::abs(std::accumulate(std::begin(vec), std::end(vec),
                                        U{0}, [](U accu, U val) { return accu + std::abs(val); }));
    }
    T pow_sum = std::abs(std::accumulate(std::begin(vec), std::end(vec),
                                         U{0}, [&p](U accu, U val) { return accu + std::pow(std::abs(val), static_cast<float>(p)); }));
    return std::pow(pow_sum, 1.0f / static_cast<float>(p));
}

template <typename Derived, isScalar U, isReal T = RealTypeT < U>> requires RealTypeTo<U, T>
T norm(const MatrixBase<Derived, U, 2>& mat, std::size_t p = 2, std::size_t q = 2) {
    if (p == 2 && q == 2) { // Frobenius norm
        T pow_sum = std::abs(std::accumulate(std::begin(mat), std::end(mat),
                                             U{0}, [](U accu, U val) { return accu + std::pow(std::abs(val), 2.0f); }));
        return std::sqrt(pow_sum);
    }
    std::size_t R = mat.dims(0);
    std::size_t C = mat.dims(1);
    T pow_sum {0};
    for (std::size_t c = 0; c < C; ++c) {
        pow_sum += std::pow(norm(mat.col(c), p), q);
    }
    return std::pow(pow_sum, 1.0f / static_cast<float>(q));
}

namespace {

template <typename Derived, isScalar U, isScalar T = ScalarTypeT < U>> requires ScalarTypeTo<U, T>
Mat<T> getQ(const MatrixBase<Derived, U, 2>& V) {
    std::size_t R = V.dims(0);
    std::size_t C = V.dims(1);
    Mat<T> Q = zeros_like(V);
    auto curr_col = Q.col(0);
    curr_col = V.col(0);
    if (norm(curr_col, 1) == T{0}) {
        return Q;
    }
    curr_col /= norm(curr_col);
    for (std::size_t i = 1; i < C; ++i) {
        curr_col = Q.col(i);
        curr_col = V.col(i);
        for (std::size_t j = 0; j < i; ++j) {
            auto other_col = Q.col(j);
            auto proj_val = dot(other_col, curr_col) / dot(other_col, other_col);
            curr_col -= proj_val * other_col;
        }
        if (norm(curr_col, 1) == T{0}) {
            return Q;
        }
        curr_col /= norm(curr_col);
    }
    return Q;
}

} // anonymous namespace

template <typename Derived, isScalar U, isScalar T = ScalarTypeT < U>> requires ScalarTypeTo<U, T>
std::pair<Mat<T>, Mat<T>> QR(const MatrixBase<Derived, U, 2>& mat) {
    auto Q = getQ(mat);
    auto R = dot(transpose(Q), mat);
    return {Q, R};
}

template <typename Derived, isScalar S, isScalar T = ScalarTypeT < S>> requires ScalarTypeTo<S, T>
std::tuple<Mat<T>, Mat<T>, Mat<T>> SVD(const MatrixBase<Derived, S, 2>& mat, std::size_t trunc) {
    std::size_t iter = 0;
    std::size_t m = mat.dims(0);
    std::size_t n = mat.dims(1);
    Mat<T> U = mat;
    Mat<T> Sigma = zeros<T, 2>({n, n});
    Mat<T> V = identity<T>(n);
    auto dot_func = [&](auto& ri, auto& rj){
        if constexpr (isComplex<S>) {
            return dot(conj(ri), rj);
        } else {
            return dot(ri, rj);
        }
    };

    auto compute_zeta = [&](const auto& alpha, const auto& beta, const auto& gamma){
        if constexpr (isComplex<S>) {
            return std::real(beta - alpha) / std::real(gamma + std::conj(gamma));
        } else {
            return (beta - alpha) / (2.0f * gamma);
        }
    };

    auto frob_norm = norm(mat);
    float sum_inners = 0.0f;

    while (++iter < max_iter) {
        float ratio = 0.0f;
        for (std::size_t i = 0; i < m; ++i) {
            auto ri = U.col(i);
            for (std::size_t j = i + 1; j < m; ++j) {
                auto rj = U.col(j);
                auto alpha = dot_func(ri, ri);
                auto beta = dot_func(rj, rj);
                auto gamma = dot_func(ri, rj);
                sum_inners += gamma;
                float zeta = compute_zeta(alpha, beta, gamma);
                auto sign = std::signbit(zeta) ? -1.0f : +1.0f;
                auto t = sign / (std::abs(zeta) + std::sqrt(1.0f + std::pow(zeta, 2.0f)));
                float c = 1.0f / std::sqrt(1.0f + std::pow(t, 2.0f));
                float s = c * t;

                for (std::size_t k = 0; k < n; ++k) {
                    auto rot1 = U[{k, i}];
                    auto rot2 = U[{k, j}];
                    U[{k, i}] = c * rot1 - s * rot2;
                    U[{k, j}] = s * rot1 + c * rot2;
                }

                for (std::size_t k = 0; k < n; ++k) {
                    auto rot1 = V[{k, i}];
                    auto rot2 = V[{k, j}];
                    V[{k, i}] = c * rot1 - s * rot2;
                    V[{k, j}] = s * rot1 + c * rot2;
                }
                ratio = sum_inners / frob_norm;
            }
        }
        if (ratio < tolerance_soft) {
            break;
        }
    }

    auto comp = [](const auto& p1, const auto& p2) {
        return std::abs(p1.first) < std::abs(p2.first);
    };

    std::priority_queue<std::pair<T, std::size_t>,
            std::vector<std::pair<T, std::size_t>>,
            decltype(comp)> pq(comp);
    for (std::size_t j = 0; j < m; ++j) {
        auto rj = U.col(j);
        Sigma[{j, j}] = std::sqrt(dot_func(rj, rj));
        if (Sigma[{j, j}] != T{0}) {
            rj /= Sigma[{j, j}];
        }
        pq.emplace(Sigma[{j, j}], j);
    }

    if (trunc > std::min(m, n)) {
        trunc = std::min(m, n);
    }
    Mat <T> U_ = zeros<T, 2>({m, trunc});
    Mat <T> Sigma_ = zeros<T, 2>({trunc, trunc});
    Mat <T> V_ = zeros<T, 2>({n, trunc});
    for (std::size_t i = 0; i < trunc; ++i) {
        auto [val, idx] = pq.top();
        pq.pop();
        std::swap_ranges(U_.col(i).begin(), U_.col(i).end(), U.col(idx).begin());
        Sigma_[{i, i}] = Sigma[{idx, idx}];
        std::swap_ranges(V_.col(i).begin(), V_.col(i).end(), V.col(idx).begin());
    }
    return {U_, Sigma_, V_};
}

template <typename Derived, isScalar U, isScalar T = ScalarTypeT<U>> requires ScalarTypeTo<U, T>
std::tuple<Mat<T>, Mat<T>, Mat<T>> SVD(const MatrixBase<Derived, U, 2>& mat) {
    std::size_t m = mat.dims(0);
    std::size_t n = mat.dims(1);
    return SVD(mat, std::min(m, n));
}

namespace {

template <typename Derived, isScalar U, isScalar T = ScalarTypeT<U>> requires ScalarTypeTo<U, T>
Mat<T> pinv_diagonal(const MatrixBase<Derived, U, 2>& mat) {
    Mat<T> M = mat;
    std::size_t m = mat.dims(0);
    std::size_t n = mat.dims(1);
    std::size_t sz = std::min(m, n);
    for (std::size_t i = 0; i < sz; ++i) {
        if (M[{i, i}] != T{0}) {
            M[{i, i}] = T{1} / M[{i, i}];
        }
    };
    return transpose(M);
}

} // anonymous namespace

template <typename Derived, isScalar U, isScalar T = ScalarTypeT<U>> requires ScalarTypeTo<U, T>
Mat<T> pinv(const MatrixBase<Derived, U, 2>& mat) {
    auto [U_, S_, V_] = SVD(mat);
    auto conjif = [&](const auto& v) {
        if constexpr (isComplex<U>) {
            return conj(v);
        } else {
            return v;
        }
    };
    return dot(dot(V_, pinv_diagonal(S_)), transpose(conjif(U_)));
}

template <isScalar T>
decltype(auto) getSign (const T& t) {
    if constexpr (isComplex<T>) {
        if (std::abs(t) < tolerance_soft) {
            return static_cast<T>(1.0f);
        }
        return t / std::abs(t);
    } else {
        return std::signbit(t) ? -1.0f : +1.0f;
    }
};

template <typename Derived, isScalar T>
Vec<T> normalize(const MatrixBase<Derived, T, 1>& vec) {
    Vec<T> u = vec;
    auto u_norm = norm(u);
    if (u_norm < tolerance_hard) {
        return u;
    } else {
        u /= u_norm;
        return u;
    }
}

template <typename Derived, isScalar U, isScalar T = ScalarTypeT < U>> requires ScalarTypeTo<U, T>
Vec<T> Householder(const MatrixBase<Derived, U, 1>& vec) {
    Vec<T> u = vec;
    T v1 = vec[0];
    auto v_norm = norm(vec);
    auto sign = getSign(v1);
    u[0] += sign * v_norm;
    return normalize(u);
}

template <typename Derived, isScalar U, isScalar T = ScalarTypeT < U>> requires ScalarTypeTo<U, T>
Mat<T> Hessenberg(const MatrixBase<Derived, U, 2>& mat, bool both = false) {
    std::size_t n = mat.dims(0);
    std::size_t C = mat.dims(1);
    if (n != C) {
        throw std::invalid_argument("Not a square matrix, cannot transform to Hessenberg");
    }
    if (n < 3) {
        return mat;
    }

    auto conjif = [&](const auto& v) {
        if constexpr (isComplex<U>) {
            return conj(v);
        } else {
            return v;
        }
    };

    Mat<T> H = mat;

    if (both) { // we want both upper and lower Hessenberg! (tridiagonal)
        // Note: the original matrix should be symmetric in this case.
        for (std::size_t k = 0; k < n - 1; ++k) {
            // erase below subdiagonal
            auto ck1 = H.col(k).submatrix(k + 1);
            if (norm(ck1) > tolerance_soft) {
                auto vk1 = Householder(ck1);
                // apply Householder from the left
                auto Sub1 = H.submatrix({k + 1, k});
                Sub1 -= outer(vk1, dot(2.0f * conjif(vk1), Sub1));
            }

            // erase right of superdiagonal
            auto ck2 = H.row(k).submatrix(k + 1);
            if (norm(ck2) > tolerance_soft) {
                auto vk2 = Householder(ck2);
                // apply Householder from the right
                auto Sub2 = H.submatrix({k, k + 1});
                Sub2 -= outer(dot(Sub2, 2.0f * vk2), conjif(vk2));
            }
        }
    } else { // nope, upper Hessenberg is enough. (default)
        for (std::size_t k = 0; k < n - 2; ++k) {
            // (n - k - 1) x 1
            auto ck = H.col(k).submatrix(k + 1);
            if (norm(ck) < tolerance_soft) {
                continue;
            }
            auto vk = Householder(ck);

            // apply Householder from the left
            auto Sub1 = H.submatrix({k + 1, k});
            Sub1 -= outer(vk, dot(2.0f * conjif(vk), Sub1));

            // apply Householder from the right
            auto Sub2 = H.submatrix({0, k + 1});
            Sub2 -= outer(dot(Sub2, 2.0f * vk), conjif(vk));
        }
    }

    return H;
}

namespace {

template <typename Derived, isScalar U, isScalar T = ScalarTypeT<U>> requires ScalarTypeTo<U, T>
std::pair<Mat<T>, Mat<T>> HessenbergWithVec(const MatrixBase<Derived, U, 2>& mat) {
    std::size_t n = mat.dims(0);
    std::size_t C = mat.dims(1);
    Mat<T> V = identity<T>(n);
    if (n != C) {
        throw std::invalid_argument("Not a square matrix, cannot transform to Hessenberg");
    }
    if (n < 3) {
        return {mat, V};
    }

    auto conjif = [&](const auto& v) {
        if constexpr (isComplex<U>) {
            return conj(v);
        } else {
            return v;
        }
    };

    Mat<T> H = mat;


    for (std::size_t k = 0; k < n - 2; ++k) {
        // (n - k - 1) x 1
        auto ck = H.col(k).submatrix(k + 1);
        if (norm(ck) < tolerance_soft) {
            continue;
        }
        auto vk = Householder(ck);

        // apply Householder from the left
        auto Sub1 = H.submatrix({k + 1, k});
        Sub1 -= outer(vk, dot(2.0f * conjif(vk), Sub1));

        // apply Householder from the right
        auto Sub2 = H.submatrix({0, k + 1});
        Sub2 -= outer(dot(Sub2, 2.0f * vk), conjif(vk));
        auto VSub2 = V.submatrix({0, k + 1});
        VSub2 -= outer(dot(VSub2, 2.0f * vk), conjif(vk));
    }

    return {H, V};
}

template <typename Derived, isScalar U, isScalar T = CmpTypeT<U>> requires CmpTypeTo<U, T>
std::vector<T> eigenTwo(const MatrixBase<Derived, U, 2>& M) {
    auto tr = M[{0, 0}] + M[{1, 1}];
    auto dt = M[{0, 0}] * M[{1, 1}] - M[{0, 1}] * M[{1, 0}];
    auto sq = tr * tr - 4.0f * dt;
    auto getSq = [&]() {
        if constexpr (isComplex<T>) {
            return std::real(sq);
        } else {
            return sq;
        }
    };
    if (isComplex<T> || getSq() >= 0) { // real eigenvalues
        auto root1 = (tr + std::sqrt(sq)) / 2.0f;
        auto root2 = (tr - std::sqrt(sq)) / 2.0f;
        return {root1, root2};
    } else {
        return {};
    }
}

template <typename Derived, isScalar T>
void addNotNullColumn(std::vector<std::pair<T, Vec<T>>>& vec, const T& eigenv, const MatrixBase<Derived, T, 2>& emat) {
    for (std::size_t i = 0; i < emat.dims(0); ++i) {
        if (norm(emat.col(i)) > tolerance_soft) {
            vec.emplace_back(eigenv, normalize(emat.col(i)));
        return;
        }
    }
}

template <typename Derived, isScalar U, isScalar T = CmpTypeT<U>> requires CmpTypeTo<U, T>
std::vector<std::pair<T, Vec<T>>> eigenVecTwo(const MatrixBase<Derived, U, 2>& M) {
    std::vector<std::pair<T, Vec<T>>> res;
    auto eigenvals = eigenTwo(M);
    if (eigenvals.empty()) {
        return {};
    }
    auto root1 = eigenvals[0];
    auto root2 = eigenvals[1];
    auto Id = identity<T>(2);
    auto M1 = M - root2 * Id;
    auto M2 = M - root1 * Id;

    addNotNullColumn(res, root1, M1);
    addNotNullColumn(res, root2, M2);
    return res;
}

template <typename Derived, isScalar U, isScalar T = CmpTypeT<U>> requires CmpTypeTo<U, T>
std::vector<T> eigenThreeImpl(const MatrixBase<Derived, U, 2>& M) {
    using UC = CmpTypeT<U>;

    // [[a, b, c]
    //  [d, e, f]
    //  [g, h, i]]
    auto a = M[{0, 0}];
    auto e = M[{1, 1}];
    auto i = M[{2, 2}];
    if (isTriangular(M)) {
        return {a, e, i};
    }
    auto b = M[{0, 1}];
    auto c = M[{0, 2}];
    auto d = M[{1, 0}];
    auto f = M[{1, 2}];
    auto g = M[{2, 0}];
    auto h = M[{2, 1}];

    // char poly : x^3 + b_x^2 + c_x + d_
    auto trM = tr(M);
    auto trsq = tr(dot(M, M));
    auto detM = det(M);

    auto b_ = trM;
    auto c_ = (trM * trM - trsq) / 2.0f;
    auto d_ = detM;

    // b_^2 - 3c_
    auto l0 = std::pow(b_, 2.0f) - 3.0f * c_;
    // 2b_^3 - 9b_c_ + 27d_
    auto l1 = 2.0f * std::pow(b_, 3.0f) - 9.0f * b_ * c_ + 27.0f * d_;

    // sqrt(l1^2 - 4l0^3)
    auto l2 = std::sqrt(static_cast<UC>(std::pow(l1, 2.0f) - 4.0f * std::pow(l0, 3.0f)));
    auto C = std::pow((l1 + l2) / 2.0f, 1.0f / 3.0f);
    auto z = (-1.0f + std::sqrt(static_cast<UC>(-3.0f))) / 2.0f;

    auto root1 = -(b_ + C + l0 / C) / 3.0f;
    C *= z;
    auto root2 = -(b_ + C + l0 / C) / 3.0f;
    C *= z;
    auto root3 = -(b_ + C + l0 / C) / 3.0f;

    return {root1, root2, root3};
}

template <typename Derived, isScalar U, isScalar T = CmpTypeT<U>> requires CmpTypeTo<U, T>
std::vector<T> eigenThree(const MatrixBase<Derived, U, 2>& M) {
    if constexpr (isComplex<T>) {
        return eigenThreeImpl(M);
    } else {
        auto roots_raw = eigenThreeImpl(M);
        std::vector<T> roots;
        for (const auto& root_raw : roots_raw) {
            if (std::imag(root_raw) < tolerance_soft) {
                roots.push_back(std::real(root_raw));
            }
        }
        return roots;
    }
}

template <typename Derived, isScalar U, isScalar T = CmpTypeT<U>> requires CmpTypeTo<U, T>
std::vector<std::pair<T, Vec<T>>> eigenVecThree(const MatrixBase<Derived, U, 2>& M) {
    std::vector<std::pair<T, Vec<T>>> res;
    auto eigenvals = eigenThreeImpl(M);
    auto root1 = eigenvals[0];
    auto root2 = eigenvals[1];
    auto root3 = eigenvals[2];
    auto Id = identity<T>(3);
    auto C1 = M - root1 * Id;
    auto C2 = M - root2 * Id;
    auto C3 = M - root3 * Id;
    auto M1 = dot(C2, C3);
    auto M2 = dot(C1, C3);
    auto M3 = dot(C1, C2);

    addNotNullColumn(res, root1, M1);
    addNotNullColumn(res, root2, M2);
    addNotNullColumn(res, root3, M3);
    return res;
}

template <typename Derived, isScalar T>
bool isSubdiagonalNeglegible(MatrixBase<Derived, T, 2>& M, std::size_t idx) {
    if (std::abs(M[{idx + 1, idx}]) <
        tolerance_soft * (std::abs(M[{idx, idx}]) + std::abs(M[{idx + 1, idx + 1}]))) {
        M[{idx + 1, idx}] = T{0};
        return true;
    }
    return false;
}

template <isScalar T>
T computeShift(const T& a, const T& b, const T& c, const T& d) {
    auto tr = a + d;
    auto det = a * d - b * c;
    auto disc = std::sqrt(tr * tr - 4.0f * det);
    auto root1 = (tr + disc) / 2.0f;
    auto root2 = (tr - disc) / 2.0f;
    if (std::abs(root1 - d) < std::abs(root2 - d)) {
        return root1;
    } else {
        return root2;
    }
}

template <isScalar T, isReal U = RealTypeT<T>>
std::tuple<T, T, U> givensRotation(const T& a, const T& b) {
    if (b == T{0}) {
        return {getSign(a), 0, std::abs(a)};
    } else if (a == T{0}) {
        return {0, getSign(b), std::abs(b)};
    } else if (std::abs(a) > std::abs(b)) {
        auto t = b / a;
        auto u = getSign(a) * std::sqrt(1.0f + t * t);
        return {1.0f / u, t / u, std::abs(a * u)};
    } else {
        auto t = a / b;
        auto u = getSign(b) * std::sqrt(1.0f + t * t);
        return {t / u, 1.0f / u, std::abs(b * u)};
    }
}

// QR algorithm used in eigendecomposition.
// not to be confused with QR decomposition
template <typename Derived, isScalar U, isScalar T = CmpTypeT<U>> requires CmpTypeTo<U, T>
std::vector<T> QRIteration(const MatrixBase<Derived, U, 2>& mat) {
    std::size_t iter = 0;
    std::size_t total_iter = 0;
    std::size_t n = mat.dims(0);

    auto conjif = [&](const auto& v) {
        if constexpr (isComplex<U>) {
            return conj(v);
        } else {
            return v;
        }
    };

    Mat<T> M = mat;
    std::size_t p = n - 1;
    while (true) {
        while (p > 0) {
            if (!isSubdiagonalNeglegible(M, p - 1)) {
                break;
            }
            iter = 0;
            --p;
        }
        if (p == 0) {
            break;
        }
        if (++iter > local_iter) {
            break;
        }
        if (++total_iter > max_iter) {
            break;
        }
        std::size_t top = p - 1;
        while (top > 0 && !isSubdiagonalNeglegible(M, top - 1)) {
            --top;
        }
        auto shift = computeShift(M[{p - 1, p - 1}], M[{p - 1, p}],
                                  M[{p, p - 1}], M[{p, p}]);

        // initial Givens rotation
        auto x = M[{top, top}] - shift;
        auto y = M[{top + 1, top}];
        auto [c, s, r] = givensRotation(x, y);
        Mat<T> R {{c, -s},
                  {s, c}};
        Mat<T> RT{{c, s},
                  {-s, c}};
        if (r > tolerance_hard) {
            auto Sub1 = M.submatrix({top, top}, {top + 2, n});
            Sub1 = dot(conjif(RT), Sub1);
            std::size_t bottom = std::min(top + 3, p + 1);
            auto Sub2 = M.submatrix({0, top}, {bottom, top + 2});
            Sub2 = dot(Sub2, R);
        }
        for (std::size_t k = top + 1; k < p; ++k) {
            x = M[{k, k - 1}];
            y = M[{k + 1, k - 1}];
            std::tie(c, s, r) = givensRotation(x, y);
            if (r > tolerance_hard) {
                M[{k, k - 1}] = r;
                M[{k + 1, k - 1}] = T{0};
                R[{0, 0}] = RT[{0, 0}] = R[{1, 1}] = RT[{1, 1}] = c;
                R[{0, 1}] = RT[{1, 0}] = -s;
                R[{1, 0}] = RT[{0, 1}] = s;

                auto Sub1 = M.submatrix({k, k}, {k + 2, n});
                Sub1 = dot(conjif(RT), Sub1);
                std::size_t bottom = std::min(k + 3, p + 1);
                auto Sub2 = M.submatrix({0, k}, {bottom, k + 2});
                Sub2 = dot(Sub2, R);
            }
        }

    }
    std::vector<T> res;
    for (std::size_t k = 0; k < n; ++k) {
        res.push_back(M[{k, k}]);
    }
    return res;
}

template <typename Derived, typename Derived2, isScalar T>
Mat<T> computeEigenvectors(const MatrixBase<Derived, T, 2>& M,
                           const MatrixBase<Derived2, T, 2>& Q) {
    std::size_t n = M.dims(0);
    Mat<T> X = identity<T>(n);

    for (std::size_t k = n - 1; k < n; --k) {
        for (std::size_t i = k - 1; i < n; --i) {
            X[{i, k}] -= M[{i, k}];
            if (k - i > 1 && k - i - 1 < n) {
                auto row_vec = M.row(i).submatrix(i + 1, k);
                auto col_vec = X.col(k).submatrix(i + 1, k);
                X[{i, k}] -= dot(row_vec, col_vec);
            }
            auto z = M[{i, i}] - M[{k, k}];
            if (z == T{0}) {
                z = static_cast<T>(tolerance_hard);
            }
            X[{i, k}] /= z;
        }
    }
    return dot(Q, X);
}

// QR algorithm used in eigendecomposition.
// not to be confused with QR decomposition
template <typename Derived, typename Derived2,
        isScalar U, isScalar T = CmpTypeT<U>> requires CmpTypeTo<U, T>
std::vector<std::pair<T, Vec<T>>> QRIterationWithVec(const MatrixBase<Derived, U, 2>& mat,
                                                     const MatrixBase<Derived2, U, 2>& V) {
    std::size_t iter = 0;
    std::size_t total_iter = 0;
    std::size_t n = mat.dims(0);

    auto conjif = [&](const auto& v) {
        if constexpr (isComplex<U>) {
            return conj(v);
        } else {
            return v;
        }
    };

    Mat<T> M = mat;
    std::size_t p = n - 1;
    Mat<T> Q = V;
    while (true) {
        while (p > 0) {
            if (!isSubdiagonalNeglegible(M, p - 1)) {
                break;
            }
            iter = 0;
            --p;
        }
        if (p == 0) {
            break;
        }
        if (++iter > 20) {
            break;
        }
        if (++total_iter > max_iter) {
            break;
        }
        std::size_t top = p - 1;
        while (top > 0 && !isSubdiagonalNeglegible(M, top - 1)) {
            --top;
        }
        auto shift = computeShift(M[{p - 1, p - 1}], M[{p - 1, p}],
                                  M[{p, p - 1}], M[{p, p}]);

        // initial Givens rotation
        auto x = M[{top, top}] - shift;
        auto y = M[{top + 1, top}];
        auto [c, s, r] = givensRotation(x, y);
        Mat<T> R {{c, -s},
                  {s, c}};
        Mat<T> RT{{c, s},
                  {-s, c}};
        if (r > tolerance_hard) {
            auto Sub1 = M.submatrix({top, top}, {top + 2, n});
            Sub1 = dot(conjif(RT), Sub1);
            std::size_t bottom = std::min(top + 3, p + 1);
            auto Sub2 = M.submatrix({0, top}, {bottom, top + 2});
            Sub2 = dot(Sub2, R);
            auto QSub2 = Q.submatrix({0, top}, {n, top + 2});
            QSub2 = dot(QSub2, R);
        }
        for (std::size_t k = top + 1; k < p; ++k) {
            x = M[{k, k - 1}];
            y = M[{k + 1, k - 1}];
            std::tie(c, s, r) = givensRotation(x, y);
            if (r > tolerance_hard) {
                M[{k, k - 1}] = r;
                M[{k + 1, k - 1}] = T{0};
                R[{0, 0}] = RT[{0, 0}] = R[{1, 1}] = RT[{1, 1}] = c;
                R[{0, 1}] = RT[{1, 0}] = -s;
                R[{1, 0}] = RT[{0, 1}] = s;

                auto Sub1 = M.submatrix({k, k}, {k + 2, n});
                Sub1 = dot(conjif(RT), Sub1);
                std::size_t bottom = std::min(k + 3, p + 1);
                auto Sub2 = M.submatrix({0, k}, {bottom, k + 2});
                Sub2 = dot(Sub2, R);
                auto QSub2 = Q.submatrix({0, k}, {n, k + 2});
                QSub2 = dot(QSub2, R);
            }
        }
    }
    std::vector<std::pair<T, Vec<T>>> res;
    auto eV = computeEigenvectors(M, Q);
    for (std::size_t k = 0; k < n; ++k) {
        res.emplace_back(M[{k, k}], normalize(eV.col(k)));
    }
    return res;
}

} // anonymous namespace

template <typename Derived, isScalar U, isScalar T = CmpTypeT<U>> requires CmpTypeTo<U, T>
std::vector<T> eigenval(const MatrixBase<Derived, U, 2>& M) {
    std::size_t n = M.dims(0);
    std::size_t C = M.dims(1);
    if (n != C) {
        throw std::invalid_argument("Not a square Matrix, cannot compute eigenvalues");
    }

    if (n == 1) { // 1 x 1
        return {M[{0, 0}]};
    } else if (n == 2) { // 2 x 2
        return eigenTwo(M);
    } else if (n == 3) { // 3 x 3
        return eigenThree(M);
    } else { // for 4 x 4 we need advanced algorithm
        auto H = Hessenberg(M);
        return QRIteration(H);
    }
}

template <typename Derived, isScalar U, isScalar T = CmpTypeT<U>> requires CmpTypeTo<U, T>
std::vector<std::pair<T, Vec<T>>> eigenvec(const MatrixBase<Derived, U, 2>& M) {
    std::size_t n = M.dims(0);
    std::size_t C = M.dims(1);
    if (n != C) {
        throw std::invalid_argument("Not a square Matrix, cannot compute eigenvalues");
    }

    if (n == 1) { // 1 x 1
        auto val = M[{0, 0}];
        Vec<T> vec {T{1}};
        return {{val, vec}};
    } else if (n == 2) { // 2 x 2
        return eigenVecTwo(M);
    } else if (n == 3) { // 3 x 3
        return eigenVecThree(M);
    } else { // for 4 x 4 we need advanced algorithm
        auto [H, V] = HessenbergWithVec(M);
        return QRIterationWithVec(H, V);
    }
}

} // namespace frozenca

#endif //FROZENCA_LINALGOPS_H
