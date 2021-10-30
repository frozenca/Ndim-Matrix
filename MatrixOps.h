#ifndef FROZENCA_MATRIXOPS_H
#define FROZENCA_MATRIXOPS_H

#include "MatrixImpl.h"

namespace frozenca {

// Matrix constructs

template <std::semiregular T, std::size_t N>
Matrix<T, N> empty(const std::array<std::size_t, N>& arr) {
    Matrix<T, N> mat (arr);
    return mat;
}

template <typename Derived, std::semiregular T, std::size_t N>
Matrix<T, N> empty_like(const MatrixBase<Derived, T, N>& base) {
    Matrix<T, N> mat (base.dims());
    return mat;
}

template <OneExists T>
Matrix<T, 2> eye(std::size_t n, std::size_t m) {
    Matrix<T, 2> mat (n, m);
    for (std::size_t i = 0; i < std::min(n, m); ++i) {
        mat[{i, i}] = T{1};
    }
    return mat;
}

template <OneExists T>
Matrix<T, 2> eye(std::size_t n) {
    return eye<T>(n, n);
}

template <OneExists T>
Matrix<T, 2> identity(std::size_t n) {
    return eye<T>(n, n);
}

template <OneExists T, std::size_t N>
Matrix<T, N> ones(const std::array<std::size_t, N>& arr) {
    Matrix<T, N> mat (arr);
    std::fill(std::begin(mat), std::end(mat), T{1});
    return mat;
}

template <typename Derived, OneExists T, std::size_t N>
Matrix<T, N> ones_like(const MatrixBase<Derived, T, N>& base) {
    Matrix<T, N> mat (base.dims());
    std::fill(std::begin(mat), std::end(mat), T{1});
    return mat;
}

template <std::semiregular T, std::size_t N>
Matrix<T, N> zeros(const std::array<std::size_t, N>& arr) {
    Matrix<T, N> mat (arr);
    std::fill(std::begin(mat), std::end(mat), T{0});
    return mat;
}

template <typename Derived, std::semiregular T, std::size_t N>
Matrix<T, N> zeros_like(const MatrixBase<Derived, T, N>& base) {
    Matrix<T, N> mat (base.dims());
    std::fill(std::begin(mat), std::end(mat), T{0});
    return mat;
}

template <std::semiregular T, std::size_t N>
Matrix<T, N> full(const std::array<std::size_t, N>& arr, const T& fill_value) {
    Matrix<T, N> mat (arr);
    std::fill(std::begin(mat), std::end(mat), fill_value);
    return mat;
}

template <typename Derived, std::semiregular T, std::size_t N>
Matrix<T, N> full_like(const MatrixBase<Derived, T, N>& base, const T& fill_value) {
    Matrix<T, N> mat (base.dims());
    std::fill(std::begin(mat), std::end(mat), fill_value);
    return mat;
}

// binary matrix operators

namespace {

template <std::semiregular U, std::semiregular V, std::semiregular T,
        std::size_t N1, std::size_t N2, std::size_t N>
requires AddableTo<U, V, T> && (std::max(N1, N2) == N)
void AddTo(MatrixView<T, N, false>& m,
           const MatrixView<U, N1, true>& m1,
           const MatrixView<V, N2, true>& m2) {
    if constexpr (N == 1) {
        m.applyFunctionWithBroadcast(m1, m2, PlusTo<U, V, T>);
    } else {
        m.applyFunctionWithBroadcast(m1, m2, AddTo<U, V, T, std::min(N1, N - 1), std::min(N2, N - 1), N - 1>);
    }
}

template <std::semiregular U, std::semiregular V, std::semiregular T,
        std::size_t N1, std::size_t N2, std::size_t N>
requires SubtractableTo<U, V, T> && (std::max(N1, N2) == N)
void SubtractTo(MatrixView<T, N, false>& m,
                const MatrixView<U, N1, true>& m1,
                const MatrixView<V, N2, true>& m2) {
    if constexpr (N == 1) {
        m.applyFunctionWithBroadcast(m1, m2, MinusTo<U, V, T>);
    } else {
        m.applyFunctionWithBroadcast(m1, m2, SubtractTo<U, V, T, std::min(N1, N - 1), std::min(N2, N - 1), N - 1>);
    }
}

template <std::semiregular U, std::semiregular V, std::semiregular T,
        std::size_t N1, std::size_t N2, std::size_t N>
requires MultipliableTo<U, V, T> && (std::max(N1, N2) == N)
void MultiplyTo(MatrixView<T, N, false>& m,
                const MatrixView<U, N1, true>& m1,
                const MatrixView<V, N2, true>& m2) {
    if constexpr (N == 1) {
        m.applyFunctionWithBroadcast(m1, m2, MultipliesTo<U, V, T>);
    } else {
        m.applyFunctionWithBroadcast(m1, m2, MultiplyTo<U, V, T, std::min(N1, N - 1), std::min(N2, N - 1), N - 1>);
    }
}

template <std::semiregular U, std::semiregular V, std::semiregular T,
        std::size_t N1, std::size_t N2, std::size_t N>
requires DividableTo<U, V, T> && (std::max(N1, N2) == N)
void DivideTo(MatrixView<T, N, false>& m,
              const MatrixView<U, N1, true>& m1,
              const MatrixView<V, N2, true>& m2) {
    if constexpr (N == 1) {
        m.applyFunctionWithBroadcast(m1, m2, DividesTo<U, V, T>);
    } else {
        m.applyFunctionWithBroadcast(m1, m2, DivideTo<U, V, T, std::min(N1, N - 1), std::min(N2, N - 1), N - 1>);
    }
}

template <std::semiregular U, std::semiregular V, std::semiregular T,
        std::size_t N1, std::size_t N2, std::size_t N>
requires RemaindableTo<U, V, T> && (std::max(N1, N2) == N)
void ModuloTo(MatrixView<T, N, false>& m,
              const MatrixView<U, N1, true>& m1,
              const MatrixView<V, N2, true>& m2) {
    if constexpr (N == 1) {
        m.applyFunctionWithBroadcast(m1, m2, ModulusTo<U, V, T>);
    } else {
        m.applyFunctionWithBroadcast(m1, m2, ModuloTo<U, V, T, std::min(N1, N - 1), std::min(N2, N - 1), N - 1>);
    }
}

} // anonymous namespace

template <typename Derived1, typename Derived2,
        std::semiregular U, std::semiregular V,
        std::size_t N1, std::size_t N2,
        std::semiregular T = AddType<U, V>> requires AddableTo<U, V, T>
decltype(auto) operator+ (const MatrixBase<Derived1, U, N1>& m1, const MatrixBase<Derived2, V, N2>& m2) {
    constexpr std::size_t N = std::max(N1, N2);
    auto dims = bidirBroadcastedDims(m1.dims(), m2.dims());
    Matrix<T, N> res = zeros<T, N>(dims);
    if constexpr (N == 1) {
        res.applyFunctionWithBroadcast(m1, m2, PlusTo<U, V, T>);
    } else {
        res.applyFunctionWithBroadcast(m1, m2, AddTo<U, V, T, std::min(N1, N - 1), std::min(N2, N - 1), N - 1>);
    }
    return res;
}

template <typename Derived1, typename Derived2,
        std::semiregular U, std::semiregular V,
        std::size_t N1, std::size_t N2,
        std::semiregular T = SubType<U, V>> requires SubtractableTo<U, V, T>
decltype(auto) operator- (const MatrixBase<Derived1, U, N1>& m1, const MatrixBase<Derived2, V, N2>& m2) {
    constexpr std::size_t N = std::max(N1, N2);
    auto dims = bidirBroadcastedDims(m1.dims(), m2.dims());
    Matrix<T, N> res = zeros<T, N>(dims);
    if constexpr (N == 1) {
        res.applyFunctionWithBroadcast(m1, m2, MinusTo<U, V, T>);
    } else {
        res.applyFunctionWithBroadcast(m1, m2, SubtractTo<U, V, T, std::min(N1, N - 1), std::min(N2, N - 1), N - 1>);
    }
    return res;
}

template <typename Derived1, typename Derived2,
        std::semiregular U, std::semiregular V,
        std::size_t N1, std::size_t N2,
        std::semiregular T = MulType<U, V>> requires MultipliableTo<U, V, T>
decltype(auto) operator* (const MatrixBase<Derived1, U, N1>& m1, const MatrixBase<Derived2, V, N2>& m2) {
    constexpr std::size_t N = std::max(N1, N2);
    auto dims = bidirBroadcastedDims(m1.dims(), m2.dims());
    Matrix<T, N> res = zeros<T, N>(dims);
    if constexpr (N == 1) {
        res.applyFunctionWithBroadcast(m1, m2, MultipliesTo<U, V, T>);
    } else {
        res.applyFunctionWithBroadcast(m1, m2, MultiplyTo<U, V, T, std::min(N1, N - 1), std::min(N2, N - 1), N - 1>);
    }
    return res;
}

template <typename Derived1, typename Derived2,
        std::semiregular U, std::semiregular V,
        std::size_t N1, std::size_t N2,
        std::semiregular T = DivType<U, V>> requires DividableTo<U, V, T>
decltype(auto) operator/ (const MatrixBase<Derived1, U, N1>& m1, const MatrixBase<Derived2, V, N2>& m2) {
    constexpr std::size_t N = std::max(N1, N2);
    auto dims = bidirBroadcastedDims(m1.dims(), m2.dims());
    Matrix<T, N> res = zeros<T, N>(dims);
    if constexpr (N == 1) {
        res.applyFunctionWithBroadcast(m1, m2, DividesTo<U, V, T>);
    } else {
        res.applyFunctionWithBroadcast(m1, m2, DivideTo<U, V, T, std::min(N1, N - 1), std::min(N2, N - 1), N - 1>);
    }
    return res;
}

template <typename Derived1, typename Derived2,
        std::semiregular U, std::semiregular V,
        std::size_t N1, std::size_t N2,
        std::semiregular T = ModType<U, V>> requires RemaindableTo<U, V, T>
decltype(auto) operator% (const MatrixBase<Derived1, U, N1>& m1, const MatrixBase<Derived2, V, N2>& m2) {
    constexpr std::size_t N = std::max(N1, N2);
    auto dims = bidirBroadcastedDims(m1.dims(), m2.dims());
    Matrix<T, N> res = zeros<T, N>(dims);
    if constexpr (N == 1) {
        res.applyFunctionWithBroadcast(m1, m2, ModulusTo<U, V, T>);
    } else {
        res.applyFunctionWithBroadcast(m1, m2, ModuloTo<U, V, T, std::min(N1, N - 1), std::min(N2, N - 1), N - 1>);
    }
    return res;
}

} // namespace frozenca

#endif //FROZENCA_MATRIXOPS_H
