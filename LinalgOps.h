#ifndef FROZENCA_LINALGOPS_H
#define FROZENCA_LINALGOPS_H

#include <bit>
#include <cmath>
#include <queue>
#include <vector>
#include "MatrixImpl.h"

namespace frozenca {

namespace {

template <std::semiregular U, std::semiregular V, std::semiregular T>
requires DotProductableTo<U, V, T>
void DotTo(T& m,
           const MatrixView<U, 1>& m1,
           const MatrixView<V, 1>& m2) {
    m += std::transform_reduce(std::execution::par_unseq, std::begin(m1), std::end(m1), std::begin(m2), T{0});
}

template <std::semiregular U, std::semiregular V, std::semiregular T>
requires DotProductableTo<U, V, T>
void DotTo(MatrixView<T, 1>& m,
           const MatrixView<U, 1>& m1,
           const MatrixView<V, 2>& m2) {
    assert(m.dims(0) == m2.dims(1));
    std::size_t c = m.dims(0);
    for (std::size_t j = 0; j < c; ++j) {
        auto col2 = m2.col(j);
        m[j] += std::transform_reduce(std::execution::par_unseq,
                                      std::begin(m1), std::end(m1), std::begin(col2), T{0});
    }
}

template <std::semiregular U, std::semiregular V, std::semiregular T>
requires DotProductableTo<U, V, T>
void DotTo(MatrixView<T, 1>& m,
           const MatrixView<U, 2>& m1,
           const MatrixView<V, 1>& m2) {
    assert(m.dims(0) == m1.dims(0));
    std::size_t r = m.dims(0);
    for (std::size_t j = 0; j < r; ++j) {
        auto row1 = m1.row(j);
        m[j] += std::transform_reduce(std::execution::par_unseq,
                                      std::begin(row1), std::end(row1), std::begin(m2), T{0});
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
requires DotProductableTo<U, V, T> && (N1 + N2 > 2)
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
requires DotProductableTo<U, V, T> && (N1 + N2 > 2)
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
        std::size_t M, std::size_t N,
        std::semiregular T = MulType<U, V>> requires DotProductableTo<U, V, T>
decltype(auto) dot(const MatrixBase<Derived1, U, M>& m1, const MatrixBase<Derived2, V, N>& m2) {
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
decltype(auto) dot(const MatrixBase<Derived2, V, 1>& m1, const MatrixBase<Derived1, U, 2>& m2) {
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

template <typename Derived, isScalar S, isScalar T = RealTypeT<S>> requires RealTypeTo<S, T>
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

template <typename Derived, isScalar U, isScalar T = RealTypeT<U>> requires RealTypeTo<U, T>
std::pair<Mat<T>, Mat<T>> Cholesky(const MatrixBase<Derived, U, 2>& mat) {
    std::size_t n = mat.dims(0);
    std::size_t C = mat.dims(1);
    if (n != C) {
        throw std::invalid_argument("Not a square matrix, cannot do Cholesky decomposition");
    }
    Mat<T> M = mat;
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

template <typename Derived, isScalar U>
bool isLowerTriangular(const MatrixBase<Derived, U, 2>& mat) {
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

template <typename Derived, isScalar U>
bool isUpperTriangular(const MatrixBase<Derived, U, 2>& mat) {
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

template <typename Derived, isScalar U>
bool isTriangular(const MatrixBase<Derived, U, 2>& mat) {
    return isLowerTriangular(mat) || isUpperTriangular(mat);
}

template <typename Derived, isScalar T>
T tr(const MatrixBase<Derived, T, 2>& mat) {
    std::size_t n = std::min(mat.dims(0), mat.dims(1));
    T val {0};
    for (std::size_t i = 0; i < n; ++i) {
        val += mat[{i, i}];
    }
    return val;
}

template <typename Derived, isScalar T>
T det(const MatrixBase<Derived, T, 2>& mat) {
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

template <typename Derived, isScalar U, isScalar T = RealTypeT<U>> requires RealTypeTo<U, T>
Mat<T> inv_impl(const MatrixBase<Derived, U, 2>& mat) {
    std::size_t n = mat.dims(0);

    Mat<T> M = mat;
    Mat<T> Inv = identity<T>(n);

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
    return Inv;
}

} // anonymous namespace

template <typename Derived, isScalar U, isScalar T = RealTypeT<U>> requires RealTypeTo<U, T>
Mat<T> inv(const MatrixBase<Derived, U, 2>& mat) {
    std::size_t n = mat.dims(0);
    std::size_t C = mat.dims(1);
    if (n != C) {
        throw std::invalid_argument("Not a square matrix, cannot invertible");
    }
    return inv_impl(mat);
}

namespace {

template <typename Derived, isScalar U, isScalar T = RealTypeT<U>> requires RealTypeTo<U, T>
Mat <T> pow_impl(const MatrixBase<Derived, U, 2>& mat, int p) {
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

template <typename Derived, isScalar U, isScalar T = RealTypeT<U>> requires RealTypeTo<U, T>
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

template <typename Derived, isScalar U, isScalar T = RealTypeT<U>> requires RealTypeTo<U, T>
T norm(const MatrixBase<Derived, U, 1>& vec, std::size_t p = 2) {
    if (p == 0) {
        throw std::invalid_argument("Norm is undefined");
    } else if (p == 1) {
        return std::reduce(std::execution::par_unseq, std::begin(vec), std::end(vec),
                           T{0}, [](T accu, T val) { return accu + std::abs(val); });
    }
    T pow_sum = std::reduce(std::execution::par_unseq, std::begin(vec), std::end(vec),
                            T{0}, [&p](T accu, T val) { return accu + std::pow(std::abs(val), static_cast<float>(p)); });
    return std::pow(pow_sum, 1.0f / static_cast<float>(p));
}

template <typename Derived, isScalar U, isScalar T = RealTypeT<U>> requires RealTypeTo<U, T>
T norm(const MatrixBase<Derived, U, 2>& mat, std::size_t p = 2, std::size_t q = 2) {
    if (p == 2 && q == 2) { // Frobenius norm
        T pow_sum = std::reduce(std::execution::par_unseq, std::begin(mat), std::end(mat),
                                T{0}, [](T accu, U val) { return accu + std::pow(std::abs(val), 2.0f); });
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

template <typename Derived, isScalar U, isScalar T = RealTypeT<U>> requires RealTypeTo<U, T>
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

template <typename Derived, isScalar U, isScalar T = RealTypeT<U>> requires RealTypeTo<U, T>
std::pair<Mat<T>, Mat<T>> QR(const MatrixBase<Derived, U, 2>& mat) {
    auto Q = getQ(mat);
    auto R = dot(transpose(Q), mat);
    return {Q, R};
}

template <typename Derived, isScalar S, isScalar T = RealTypeT<S>> requires RealTypeTo<S, T>
std::tuple<Mat<T>, Mat<T>, Mat<T>> SVD(const MatrixBase<Derived, S, 2>& mat, std::size_t trunc) {
    constexpr float conv_criterion = 1e-6;
    constexpr std::size_t max_iter = 100;

    std::size_t iter = 0;
    std::size_t m = mat.dims(0);
    std::size_t n = mat.dims(1);
    Mat<T> U = mat;
    Mat<T> Sigma = zeros<T, 2>({n, n});
    Mat<T> V = identity<T>(n);
    auto dot_func = [&](auto& ri, auto& rj){
        if constexpr (isComplex<S>) {
            return compdot(ri, rj);
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

    while (++iter < max_iter) {
        float max_ratio = 0.0f;
        for (std::size_t i = 0; i < m; ++i) {
            auto ri = U.row(i);
            for (std::size_t j = i + 1; j < m; ++j) {
                auto rj = U.row(j);
                auto alpha = dot_func(ri, ri);
                auto beta = dot_func(rj, rj);
                auto gamma = dot_func(ri, rj);
                float zeta = compute_zeta(alpha, beta, gamma);
                auto sign = std::signbit(zeta) ? -1 : +1;
                auto t = sign / (std::abs(zeta) + std::sqrt(1.0f + std::pow(zeta, 2.0f)));
                float c = 1.0f / std::sqrt(1.0 + std::pow(t, 2.0f));
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
                float curr_ratio = std::abs(c) / std::sqrt(std::abs(beta * alpha));
                max_ratio = std::max(max_ratio, curr_ratio);
            }
        }
        if (max_ratio < conv_criterion) {
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
        auto rj = U.row(j);
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
        std::swap_ranges(std::execution::par_unseq, U_.col(i).begin(), U_.col(i).end(), U.col(idx).begin());
        Sigma_[{i, i}] = Sigma[{idx, idx}];
        std::swap_ranges(std::execution::par_unseq, V_.col(i).begin(), V_.col(i).end(), V.col(idx).begin());
    }
    return {U_, Sigma_, V_};
}

template <typename Derived, isScalar U, isScalar T = RealTypeT<U>> requires RealTypeTo<U, T>
std::tuple<Mat<T>, Mat<T>, Mat<T>> SVD(const MatrixBase<Derived, U, 2>& mat) {
    std::size_t m = mat.dims(0);
    std::size_t n = mat.dims(1);
    return SVD(mat, std::min(m, n));
}

template <typename Derived, isScalar U, isScalar T = RealTypeT<U>> requires RealTypeTo<U, T>
Vec<T> Householder(const MatrixBase<Derived, U, 1>& vec) {
    Vec<T> u = vec;
    T v1 = vec[0];
    auto getSign = [&]() {
        if constexpr (isComplex<T>) {
            return v1 / std::abs(v1);
        } else {
            return std::signbit(v1) ? -1.0f : +1.0f;
        }
    };

    auto sign = getSign();
    auto v_norm = norm(vec);
    u[0] += sign * v_norm;
    auto u_norm = norm(u);
    u /= u_norm;
    return u;
}

template <typename Derived, isScalar U, isScalar T = RealTypeT<U>> requires RealTypeTo<U, T>
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
            auto vk1 = Householder(ck1);

            // apply Householder from the left
            auto Sub1 = H.submatrix({k + 1, k});
            Sub1 -= outer(vk1, dot(2.0f * conjif(vk1), Sub1));

            // erase right of superdiagonal
            auto ck2 = H.row(k).submatrix(k + 1);
            auto vk2 = Householder(ck2);

            // apply Householder from the right
            auto Sub2 = H.submatrix({k, k + 1});
            Sub2 -= outer(dot(Sub2, 2.0f * vk2), conjif(vk2));
        }
    } else { // nope, upper Hessenberg is enough. (default)
        for (std::size_t k = 0; k < n - 2; ++k) {
            // (n - k - 1) x 1
            auto ck = H.col(k).submatrix(k + 1);
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

template <isReal T>
std::pair<T, T> Rayleigh(const T& a, const T& b, const T& c, const T& d) {
    auto tr = a + d;
    auto dt = a * d - b * c;
    auto sq = tr * tr - 4.0f * dt;
    if (sq >= 0) { // real eigenvalues or want complex eigenvalues
        auto root1 = (tr + std::sqrt(sq)) / 2.0f;
        auto root2 = (tr - std::sqrt(sq)) / 2.0f;

        // choose the one closer to d
        auto root = (std::abs(root1 - d) < std::abs(root2 - d)) ? root1 : root2;

        // z^2 + bz + c = (z-r)^2
        return {-2.0f * root, root * root};
    } else { // don't want complex eigenvalues, we want char poly directly.
        return {-tr, dt};
    }
}

template <isComplex T>
std::pair<T, T> Rayleigh(const T& a, const T& b, const T& c, const T& d) {
    auto tr = a + d;
    auto dt = a * d - b * c;
    auto sq = tr * tr - 4.0f * dt;
    auto root1 = (tr + std::sqrt(sq)) / 2.0f;
    auto root2 = (tr - std::sqrt(sq)) / 2.0f;
    // choose the one closer to d
    auto root = (std::abs(root1 - d) < std::abs(root2 - d)) ? root1 : root2;
    // z^2 + bz + c = (z-r)^2
    return {-2.0f * root, root * root};
}

template <typename Derived, isScalar U, isScalar T = RealTypeT<U>> requires RealTypeTo<U, T>
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

template <typename Derived, isScalar U, isScalar T = RealTypeT<U>> requires RealTypeTo<U, T>
std::vector<T> eigenThree(const MatrixBase<Derived, U, 2>& M) {
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

    constexpr double tolerance = 1e-6;
    if constexpr (isComplex<T>) {
        return {root1, root2, root3};
    } else {
        std::vector<T> roots;
        if (std::imag(root1) < tolerance) {
            roots.push_back(std::real(root1));
        }
        if (std::imag(root2) < tolerance) {
            roots.push_back(std::real(root2));
        }
        if (std::imag(root3) < tolerance) {
            roots.push_back(std::real(root3));
        }
        return roots;
    }
}

// QR algorithm used in eigendecomposition.
// not to be confused with QR decomposition
template <typename Derived, isScalar U, isScalar T = RealTypeT<U>> requires RealTypeTo<U, T>
std::vector<T> QRIteration(const MatrixBase<Derived, U, 2>& mat) {
    std::size_t n = mat.dims(0);
    // assumption : mat is (upper) Hessenberg
    constexpr float tolerance = 1e-6;
    assert(n > 3); // n <= 3 will be handled analytically

    Mat<T> H = mat;
    std::size_t p = n; // effective matrix size

    auto conjif = [&](const auto& v) {
        if constexpr (isComplex<U>) {
            return conj(v);
        } else {
            return v;
        }
    };

    while (p > 2) {
        auto [s, t] = Rayleigh(H[{p - 2, p - 2}], H[{p - 2, p - 1}],
                               H[{p - 1, p - 2}], H[{p - 1, p - 1}]);

        // compute first 3 elements of first column
        auto x = H[{0, 0}] * H[{0, 0}] + H[{0, 1}] * H[{1, 0}] - s * H[{0, 0}] + t;
        auto y = H[{1, 0}] * (H[{0, 0}] + H[{1, 1}] - s);
        auto z = H[{1, 0}] * H[{2, 1}];

        // repeatedly apply Householder
        for (std::size_t k = 0; k < p - 2; ++k) {
            Vec<T> v {x, y, z};
            auto vk = Householder(v);

            std::size_t r = (k == 0) ? 0 : (k - 1);
            auto Sub1 = H.submatrix({k, r}, {k + 3, n});
            Sub1 -= outer(vk, dot(2.0f * conjif(vk), Sub1));

            r = std::min(k + 4, p);
            auto Sub2 = H.submatrix({0, k}, {r, k + 3});
            Sub2 -= outer(dot(Sub2, 2.0f * vk), conjif(vk));

            x = H[{k + 1, k}];
            y = H[{k + 2, k}];
            if (k < p - 3) {
                z = H[{k + 3, k}];
            }
        }

        // for last x and y, find Givens rotation
        auto rad = std::sqrt(std::pow(x, 2.0f) + std::pow(y, 2.0f));
        auto cos_val = x / rad;
        auto sin_val = y / rad;
        Mat<T> R {{cos_val, sin_val}, {-sin_val, cos_val}};
        Mat<T> RT {{cos_val, -sin_val}, {sin_val, cos_val}};

        auto Sub1 = H.submatrix({p - 2, p - 3}, {p, n});
        Sub1 = dot(conjif(RT), Sub1);
        auto Sub2 = H.submatrix({0, p - 2}, {p, p});
        Sub2 = dot(Sub2, R);

        // check convergence, deflate if necessary
        if (std::abs(H[{p - 1, p - 2}]) < tolerance *
                                          (std::abs(H[{p - 2, p - 2}]) + std::abs(H[{p - 1, p - 1}]), 1.0f)) {
            H[{p - 1, p - 2}] = T{0};
            p -= 1;
        } else if (std::abs(H[{p - 2, p - 3}]) < tolerance *
                                                 (std::abs(H[{p - 3, p - 3}]) + std::abs(H[{p - 2, p - 2}]), 1.0f)) {
            H[{p - 2, p - 3}] = T{0};
            p -= 2;
        }
    }

    std::vector<T> res;
    for (std::size_t k = 0; k < n - 1; ) {
        if (std::abs(H[{k + 1, k}]) > tolerance) {
            auto H_four = H.submatrix({k, k}, {k + 2, k + 2});
            auto ev = eigenTwo(H_four);
            std::ranges::move(ev, std::back_inserter(res));
            k += 2;
        } else {
            res.push_back(H[{k, k}]);
            k += 1;
        }
    }
    return res;
}

} // anonymous namespace

template <typename Derived, isScalar U, isScalar T = RealTypeT<U>> requires RealTypeTo<U, T>
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
        std::cout << H << '\n';
        return QRIteration(H);
    }
}

} // namespace frozenca

#endif //FROZENCA_LINALGOPS_H
