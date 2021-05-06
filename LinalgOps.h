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

template <isComplex U, isComplex V, isComplex T>
void CompDotTo(T& m,
           const MatrixView<U, 1>& m1,
           const MatrixView<V, 1>& m2) {
    m += std::transform_reduce(std::execution::par_unseq, std::begin(m1), std::end(m1), std::begin(m2), T{0},
                               [](const auto& u, const auto& v){ return u + v;},
                               [](const auto& u, const auto& v){ return std::conj(u) * v;});
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
        m[j] += std::transform_reduce(std::execution::par_unseq,
                                      std::begin(m1), std::end(m1), std::begin(col2), T{0});
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


template <typename Derived1, typename Derived2,
        isComplex U, isComplex V, isComplex T>
void CompDotTo(T& m,
           const MatrixBase<Derived1, U, 1>& m1,
           const MatrixBase<Derived2, V, 1>& m2) {
    MatrixView<U, 1> m1_view (m1);
    MatrixView<V, 1> m2_view (m2);
    CompDotTo(m, m1_view, m2_view);
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
        isComplex U, isComplex V,
        isComplex T = MulType<U, V>>
decltype(auto) compdot(const MatrixBase<Derived1, U, 1>& m1, const MatrixBase<Derived2, V, 1>& m2) {
    if (m1.dims(0) != m2.dims(0)) {
        throw std::invalid_argument("Cannot do dot product, shape is not aligned");
    }
    T res {0};
    CompDotTo(res, m1, m2);
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
std::tuple<std::vector<std::size_t>, Mat<B>, Mat<B>> LUP(const MatrixBase<Derived, A, 2>& mat) {

    std::size_t n = mat.dims(0);
    std::size_t C = mat.dims(1);
    if (n != C) {
        throw std::invalid_argument("Not a square matrix, cannot do LUP decomposition");
    }
    std::vector<std::size_t> P (n);
    std::iota(std::begin(P), std::end(P), 0lu);

    Mat<B> A_ = mat;
    Mat<B> U = zeros_like(A_);
    Mat<B> L = identity<B>(n);

    for (std::size_t k = 0; k < n; ++k) {
        A p {0};
        std::size_t k_ = -1;
        for (std::size_t i = k; i < n; ++i) {
            auto val = std::fabs(A_[{i, k}]);
            if (val > p) {
                p = val;
                k_ = i;
            }
        }
        if (p == A{0}) {
            throw std::invalid_argument("Singular matrix");
        }
        std::swap(P[k], P[k_]);
        A_.swapRows(k, k_);
        for (std::size_t i = k + 1; i < n; ++i) {
            A_[{i, k}] /= A_[{k, k}];
            for (std::size_t j = k + 1; j < n; ++j) {
                A_[{i, j}] -= A_[{i, k}] * A_[{k, j}];
            }
        }
    }

    for (std::size_t i = 0; i < n; ++i) {
        for (std::size_t j = 0; j < n; ++j) {
            if (j < i) {
                L[{i, j}] = A_[{i, j}];
            } else {
                U[{i, j}] = A_[{i, j}];
            }
        }
    }

    return {P, L, U};
}

template <typename Derived, isScalar A, isScalar B = RealTypeT<A>> requires RealTypeTo<A, B>
std::pair<Mat<B>, Mat<B>> Cholesky(const MatrixBase<Derived, A, 2>& mat) {
    std::size_t n = mat.dims(0);
    std::size_t C = mat.dims(1);
    if (n != C) {
        throw std::invalid_argument("Not a square matrix, cannot do Cholesky decomposition");
    }
    Mat<B> A_ = mat;
    Mat<B> L = zeros_like(A_);

    for (std::size_t i = 0; i < n; ++i) {
        for (std::size_t j = 0; j < i + 1; ++j) {
            A sum {0};
            for (std::size_t k = 0; k < j; ++k) {
                if constexpr (isComplex<A>) {
                    sum += L[{i, k}] * conj(L[{j, k}]);
                } else {
                    sum += L[{i, k}] * L[{j, k}];
                }
            }

            if (i == j) {
                L[{i, j}] = std::sqrt(A_[{i, i}] - sum);
            } else {
                L[{i, j}] = ((A{1.0f} / L[{j, j}]) * (A_[{i, j}] - sum));
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
            if (mat[{i, j}] != A{0}) {
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
            if (mat[{i, j}] != A{0}) {
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
            det_val *= mat[{i, i}];
        }
        return det_val;
    }
    auto [P, L, U] = LUP(mat);
    return det(L) * det(U);
}

namespace {

template <typename Derived, isScalar A, isScalar B = RealTypeT<A>> requires RealTypeTo<A, B>
Mat<B> inv_impl(const MatrixBase<Derived, A, 2>& mat) {
    std::size_t n = mat.dims(0);

    Mat<B> M = mat;
    Mat<B> Inv = identity<B>(n);

    for (std::size_t i = 0; i < n; ++i) {
        // pivoting
        for (std::size_t k = i + 1; k < n; ++k) {
            if (M[{i, i}] == B{0} && std::fabs(M[{k, i}]) != B{0}) {
                M.swapRows(i, k);
                Inv.swapRows(i, k);
                break;
            }
        }
        if (M[{i, i}] == B{0}) {
            throw std::invalid_argument("Singular matrix, cannot invertible");
        }
        for (std::size_t j = i + 1; j < n; ++j) {
            B coeff = M[{j, i}] / M[{i, i}];
            Inv.row(j) -= coeff * Inv.row(i);
            M.row(j) -= coeff * M.row(i);
        }
    }
    for (std::size_t i = n - 1; i < n; --i) {
        for (std::size_t j = 0; j < i; ++j) {
            B coeff = M[{j, i}] / M[{i, i}];
            Inv.row(j) -= coeff * Inv.row(i);
            M[{j, i}] = B{0};
        }
        Inv.row(i) /= M[{i, i}];
        M[{i, i}] = 1;
    }
    return Inv;
}

} // anonymous namespace

template <typename Derived, isScalar A, isScalar B = RealTypeT<A>> requires RealTypeTo<A, B>
Mat<B> inv(const MatrixBase<Derived, A, 2>& mat) {
    std::size_t n = mat.dims(0);
    std::size_t C = mat.dims(1);
    if (n != C) {
        throw std::invalid_argument("Not a square matrix, cannot invertible");
    }
    return inv_impl(mat);
}

namespace {

template <typename Derived, isScalar A, isScalar B = RealTypeT<A>> requires RealTypeTo<A, B>
Mat <B> pow_impl(const MatrixBase<Derived, A, 2>& mat, int p) {
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

template <typename Derived, isScalar A, isScalar B = RealTypeT<A>> requires RealTypeTo<A, B>
Mat<B> pow(const MatrixBase<Derived, A, 2>& mat, int p) {
    std::size_t n = mat.dims(0);
    std::size_t C = mat.dims(1);
    if (n != C) {
        throw std::invalid_argument("Not a square matrix, cannot invertible");
    }
    if (p == 0) {
        return identity<B>(n);
    } else if (p < 0) {
        return pow_impl(inv_impl(mat), -p);
    } else {
        return pow_impl(mat, p);
    }
}

template <typename Derived, isScalar A, isScalar B = RealTypeT<A>> requires RealTypeTo<A, B>
B norm(const MatrixBase<Derived, A, 1>& vec, std::size_t p = 2) {
    if (p == 0) {
        throw std::invalid_argument("Norm is undefined");
    } else if (p == 1) {
        return std::reduce(std::execution::par_unseq, std::begin(vec), std::end(vec), B{0}, [](B accu, B val) {
            return accu + std::abs(val);
        });
    }
    B pow_sum = std::reduce(std::execution::par_unseq, std::begin(vec), std::end(vec), B{0}, [&p](B accu, B val) {
        return accu + std::pow(val, p);
    });
    return std::pow(pow_sum, 1.0f / static_cast<float>(p));
}

template <typename Derived, isScalar A, isScalar B = RealTypeT<A>> requires RealTypeTo<A, B>
B norm(const MatrixBase<Derived, A, 2>& mat, std::size_t p = 2, std::size_t q = 2) {
    if (p == 2 && q == 2) { // Frobenius norm
        B pow_sum = std::reduce(std::execution::par_unseq, std::begin(mat), std::end(mat), B{0}, [](B accu, A val) {
            return accu + std::pow(val, 2.0f);
        });
        return std::sqrt(pow_sum);
    }
    std::size_t R = mat.dims(0);
    std::size_t C = mat.dims(1);
    B pow_sum {0};
    for (std::size_t c = 0; c < C; ++c) {
        pow_sum += std::pow(norm(mat.col(c), p), q);
    }
    return std::pow(pow_sum, 1.0f / static_cast<float>(q));
}

namespace {

template <typename Derived, isScalar A, isScalar B = RealTypeT<A>> requires RealTypeTo<A, B>
Mat<B> getQ(const MatrixBase<Derived, A, 2>& V) {
    std::size_t R = V.dims(0);
    std::size_t C = V.dims(1);
    Mat<B> Q = zeros_like(V);
    auto curr_col = Q.col(0);
    curr_col = V.col(0);
    if (norm(curr_col, 1) == B{0}) {
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
        if (norm(curr_col, 1) == B{0}) {
            return Q;
        }
        curr_col /= norm(curr_col);
    }
    return Q;
}

} // anonymous namespace

template <typename Derived, isScalar A, isScalar B = RealTypeT<A>> requires RealTypeTo<A, B>
std::pair<Mat<B>, Mat<B>> QR(const MatrixBase<Derived, A, 2>& mat) {
    auto Q = getQ(mat);
    auto R = dot(transpose(Q), mat);
    return {Q, R};
}

template <typename Derived, isScalar A, isScalar B = RealTypeT<A>> requires RealTypeTo<A, B>
std::tuple<Mat<B>, Mat<B>, Mat<B>> SVD(const MatrixBase<Derived, A, 2>& mat, std::size_t trunc) {
    constexpr float conv_criterion = 1e-6;
    constexpr std::size_t max_iter = 1'000;

    std::size_t iter = 0;
    std::size_t m = mat.dims(0);
    std::size_t n = mat.dims(1);
    Mat<B> U = mat;
    Mat<B> Sigma = zeros<B, 2>({n, n});
    Mat<B> V = identity<B>(n);
    auto dot_func = [&](auto& ri, auto& rj){
        if constexpr (isComplex<A> || isComplex<B>) {
            return compdot(ri, rj);
        } else {
            return dot(ri, rj);
        }
    };

    auto compute_zeta = [&](const auto& alpha, const auto& beta, const auto& gamma){
        if constexpr (isComplex<A> || isComplex<B>) {
            return std::real(beta - alpha) / std::real(gamma + std::conj(gamma));
        } else {
            return (beta - alpha) / (2.0f * gamma);
        }
    };

    auto ratio_denom = [&](const auto& alpha, const auto& beta) {
        if constexpr (isComplex<A> || isComplex<B>) {
            return std::sqrt(std::real(beta * alpha));
        } else {
            return std::sqrt(beta * alpha);
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
                float curr_ratio = std::abs(c) / ratio_denom(alpha, beta);
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

    std::priority_queue<std::pair<B, std::size_t>,
            std::vector<std::pair<B, std::size_t>>,
            decltype(comp)> pq(comp);
    for (std::size_t j = 0; j < m; ++j) {
        auto rj = U.row(j);
        Sigma[{j, j}] = std::sqrt(dot_func(rj, rj));
        if (Sigma[{j, j}] != B{0}) {
            rj /= Sigma[{j, j}];
        }
        pq.emplace(Sigma[{j, j}], j);
    }
    if (trunc > std::min(m, n)) {
        trunc = std::min(m, n);
    }
    Mat <B> U_ = zeros<B, 2>({m, trunc});
    Mat <B> Sigma_ = zeros<B, 2>({trunc, trunc});
    Mat <B> V_ = zeros<B, 2>({n, trunc});
    for (std::size_t i = 0; i < trunc; ++i) {
        auto [val, idx] = pq.top();
        pq.pop();
        std::swap_ranges(std::execution::par_unseq, U_.col(i).begin(), U_.col(i).end(), U.col(idx).begin());
        Sigma_[{i, i}] = Sigma[{idx, idx}];
        std::swap_ranges(std::execution::par_unseq, V_.col(i).begin(), V_.col(i).end(), V.col(idx).begin());
    }
    return {U_, Sigma_, V_};
}

template <typename Derived, isScalar A, isScalar B = RealTypeT<A>> requires RealTypeTo<A, B>
std::tuple<Mat<B>, Mat<B>, Mat<B>> SVD(const MatrixBase<Derived, A, 2>& mat) {
    std::size_t m = mat.dims(0);
    std::size_t n = mat.dims(1);
    return SVD(mat, std::min(m, n));
}

} // namespace frozenca

#endif //FROZENCA_LINALGOPS_H
