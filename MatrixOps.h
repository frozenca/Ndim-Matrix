#ifndef FROZENCA_MATRIXOPS_H
#define FROZENCA_MATRIXOPS_H

#include "MatrixImpl.h"

namespace frozenca {

// Matrix constructs

template <std::semiregular T, std::size_t N>
Matrix<T, N> zeros(const std::array<std::size_t, N>& arr) {
    Matrix<T, N> mat (arr);
    std::ranges::fill(mat, T{0});
    return mat;
}

template <OneExists T, std::size_t N>
Matrix<T, N> ones(const std::array<std::size_t, N>& arr) {
    Matrix<T, N> mat (arr);
    std::ranges::fill(mat, T{1});
    return mat;
}

template <OneExists T>
Matrix<T, 2> eyes(std::size_t n, std::size_t m) {
    Matrix<T, 2> mat (n, m);
    for (std::size_t i = 0; i < std::min(n, m); ++i) {
        mat(i, i) = T{1};
    }
    return mat;
}

template <OneExists T>
Matrix<T, 2> eyes(std::size_t n) {
    return eyes<T>(n, n);
}

template <OneExists T>
Matrix<T, 2> identity(std::size_t n) {
    return eyes<T>(n, n);
}

template <std::semiregular T, std::size_t N>
Matrix<T, N> full(const std::array<std::size_t, N>& arr, const T& fill_value) {
    Matrix<T, N> mat (arr);
    std::ranges::fill(mat, fill_value);
    return mat;
}

// binary matrix operators

namespace {

template <std::semiregular U, std::semiregular V, std::semiregular T,
        std::size_t N1, std::size_t N2, std::size_t N>
requires AddableTo<U, V, T> && (std::max(N1, N2) == N)
void AddTo(MatrixView<T, N>& m,
           const MatrixView<U, N1>& m1,
           const MatrixView<V, N2>& m2) {
    if constexpr (N == 1) {
        m.applyFunctionWithBroadcast(m1, m2, PlusTo<U, V, T>);
    } else {
        m.applyFunctionWithBroadcast(m1, m2, AddTo<U, V, T, std::min(N1, N - 1), std::min(N2, N - 1), N - 1>);
    }
}

template <std::semiregular U, std::semiregular V, std::semiregular T,
        std::size_t N1, std::size_t N2, std::size_t N>
requires SubtractableTo<U, V, T> && (std::max(N1, N2) == N)
void SubtractTo(MatrixView<T, N>& m,
                const MatrixView<U, N1>& m1,
                const MatrixView<V, N2>& m2) {
    if constexpr (N == 1) {
        m.applyFunctionWithBroadcast(m1, m2, MinusTo<U, V, T>);
    } else {
        m.applyFunctionWithBroadcast(m1, m2, SubtractTo<U, V, T, std::min(N1, N - 1), std::min(N2, N - 1), N - 1>);
    }
}

template <std::semiregular U, std::semiregular V, std::semiregular T,
        std::size_t N1, std::size_t N2, std::size_t N>
requires MultipliableTo<U, V, T> && (std::max(N1, N2) == N)
void MultiplyTo(MatrixView<T, N>& m,
                const MatrixView<U, N1>& m1,
                const MatrixView<V, N2>& m2) {
    if constexpr (N == 1) {
        m.applyFunctionWithBroadcast(m1, m2, MultipliesTo<U, V, T>);
    } else {
        m.applyFunctionWithBroadcast(m1, m2, MultiplyTo<U, V, T, std::min(N1, N - 1), std::min(N2, N - 1), N - 1>);
    }
}

template <std::semiregular U, std::semiregular V, std::semiregular T,
        std::size_t N1, std::size_t N2, std::size_t N>
requires DividableTo<U, V, T> && (std::max(N1, N2) == N)
void DivideTo(MatrixView<T, N>& m,
              const MatrixView<U, N1>& m1,
              const MatrixView<V, N2>& m2) {
    if constexpr (N == 1) {
        m.applyFunctionWithBroadcast(m1, m2, DividesTo<U, V, T>);
    } else {
        m.applyFunctionWithBroadcast(m1, m2, DivideTo<U, V, T, std::min(N1, N - 1), std::min(N2, N - 1), N - 1>);
    }
}

template <std::semiregular U, std::semiregular V, std::semiregular T,
        std::size_t N1, std::size_t N2, std::size_t N>
requires RemaindableTo<U, V, T> && (std::max(N1, N2) == N)
void ModuloTo(MatrixView<T, N>& m,
              const MatrixView<U, N1>& m1,
              const MatrixView<V, N2>& m2) {
    if constexpr (N == 1) {
        m.applyFunctionWithBroadcast(m1, m2, ModulusTo<U, V, T>);
    } else {
        m.applyFunctionWithBroadcast(m1, m2, ModuloTo<U, V, T, std::min(N1, N - 1), std::min(N2, N - 1), N - 1>);
    }
}

} // anonymous namespace

template <typename Derived1, typename Derived2,
        std::semiregular U, std::semiregular V,
        std::size_t N1, std::size_t N2> requires WeakAddable<U, V>
decltype(auto) operator+ (const MatrixBase<Derived1, U, N1>& m1, const MatrixBase<Derived2, V, N2>& m2) {
    using T = std::invoke_result_t<decltype(Plus<U, V>), U, V>;
    constexpr std::size_t N = std::max(N1, N2);
    auto dims = bidirBroadcastedDims(m1.dims(), m2.dims());
    Matrix<T, N> res = zeros<T, N>(dims);
    res.applyFunctionWithBroadcast(m1, m2, AddTo<U, V, T, std::min(N1, N - 1), std::min(N2, N - 1), N - 1>);
    return res;
}

template <typename Derived1, typename Derived2,
        std::semiregular U, std::semiregular V,
        std::size_t N1, std::size_t N2> requires WeakSubtractable<U, V>
decltype(auto) operator- (const MatrixBase<Derived1, U, N1>& m1, const MatrixBase<Derived2, V, N2>& m2) {
    using T = std::invoke_result_t<decltype(Minus<U, V>), U, V>;
    constexpr std::size_t N = std::max(N1, N2);
    auto dims = bidirBroadcastedDims(m1.dims(), m2.dims());
    Matrix<T, N> res = zeros<T, N>(dims);
    res.applyFunctionWithBroadcast(m1, m2, SubtractTo<U, V, T, std::min(N1, N - 1), std::min(N2, N - 1), N - 1>);
    return res;
}

template <typename Derived1, typename Derived2,
        std::semiregular U, std::semiregular V,
        std::size_t N1, std::size_t N2> requires WeakMultipliable<U, V>
decltype(auto) operator* (const MatrixBase<Derived1, U, N1>& m1, const MatrixBase<Derived2, V, N2>& m2) {
    using T = std::invoke_result_t<decltype(Multiplies<U, V>), U, V>;
    constexpr std::size_t N = std::max(N1, N2);
    auto dims = bidirBroadcastedDims(m1.dims(), m2.dims());
    Matrix<T, N> res = zeros<T, N>(dims);
    res.applyFunctionWithBroadcast(m1, m2, MultiplyTo<U, V, T, std::min(N1, N - 1), std::min(N2, N - 1), N - 1>);
    return res;
}

template <typename Derived1, typename Derived2,
        std::semiregular U, std::semiregular V,
        std::size_t N1, std::size_t N2> requires WeakDividable<U, V>
decltype(auto) operator/ (const MatrixBase<Derived1, U, N1>& m1, const MatrixBase<Derived2, V, N2>& m2) {
    using T = std::invoke_result_t<decltype(Divides<U, V>), U, V>;
    constexpr std::size_t N = std::max(N1, N2);
    auto dims = bidirBroadcastedDims(m1.dims(), m2.dims());
    Matrix<T, N> res = zeros<T, N>(dims);
    res.applyFunctionWithBroadcast(m1, m2, DivideTo<U, V, T, std::min(N1, N - 1), std::min(N2, N - 1), N - 1>);
    return res;
}

template <typename Derived1, typename Derived2,
        std::semiregular U, std::semiregular V,
        std::size_t N1, std::size_t N2> requires WeakRemaindable<U, V>
decltype(auto) operator% (const MatrixBase<Derived1, U, N1>& m1, const MatrixBase<Derived2, V, N2>& m2) {
    using T = std::invoke_result_t<decltype(Modulus<U, V>), U, V>;
    constexpr std::size_t N = std::max(N1, N2);
    auto dims = bidirBroadcastedDims(m1.dims(), m2.dims());
    Matrix<T, N> res = zeros<T, N>(dims);
    res.applyFunctionWithBroadcast(m1, m2, ModuloTo<U, V, T, std::min(N1, N - 1), std::min(N2, N - 1), N - 1>);
    return res;
}



} // namespace frozenca

#endif //FROZENCA_MATRIXOPS_H