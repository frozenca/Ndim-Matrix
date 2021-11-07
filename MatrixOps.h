#ifndef FROZENCA_MATRIXOPS_H
#define FROZENCA_MATRIXOPS_H

#include "MatrixImpl.h"

namespace frozenca {

// Matrix constructs

template <std::semiregular T, std::size_t N, bool isRowMajor = true>
Matrix<T, N, isRowMajor> empty(const std::array<std::size_t, N>& arr) {
    Matrix<T, N, isRowMajor> mat (arr);
    return mat;
}

template <typename Derived, std::semiregular T, std::size_t N, bool isRowMajor1, bool isRowMajor2 = true>
Matrix<T, N, isRowMajor2> empty_like(const MatrixBase<Derived, T, N, isRowMajor1>& base) {
    Matrix<T, N, isRowMajor2> mat (base.dims());
    return mat;
}

template <OneExists T, bool isRowMajor = true>
Matrix<T, 2, isRowMajor> eye(std::size_t n, std::size_t m) {
    Matrix<T, 2, isRowMajor> mat (n, m);
    for (std::size_t i = 0; i < std::min(n, m); ++i) {
        mat[{i, i}] = T{1};
    }
    return mat;
}

template <OneExists T, bool isRowMajor = true>
Matrix<T, 2, isRowMajor> eye(std::size_t n) {
    return eye<T>(n, n);
}

template <OneExists T, bool isRowMajor = true>
Matrix<T, 2, isRowMajor> identity(std::size_t n) {
    return eye<T>(n, n);
}

template <OneExists T, std::size_t N, bool isRowMajor>
Matrix<T, N, isRowMajor> ones(const std::array<std::size_t, N>& arr) {
    Matrix<T, N> mat (arr);
    std::fill(std::begin(mat), std::end(mat), T{1});
    return mat;
}

template <typename Derived, OneExists T, std::size_t N, bool isRowMajor1, bool isRowMajor2 = true>
Matrix<T, N, isRowMajor2> ones_like(const MatrixBase<Derived, T, N, isRowMajor1>& base) {
    Matrix<T, N, isRowMajor2> mat (base.dims());
    std::fill(std::begin(mat), std::end(mat), T{1});
    return mat;
}

template <std::semiregular T, std::size_t N, bool isRowMajor = true>
Matrix<T, N, isRowMajor> zeros(const std::array<std::size_t, N>& arr) {
    Matrix<T, N, isRowMajor> mat (arr);
    std::fill(std::begin(mat), std::end(mat), T{0});
    return mat;
}

template <typename Derived, std::semiregular T, std::size_t N, bool isRowMajor1, bool isRowMajor2 = true>
Matrix<T, N, isRowMajor2> zeros_like(const MatrixBase<Derived, T, N, isRowMajor1>& base) {
    Matrix<T, N, isRowMajor2> mat (base.dims());
    std::fill(std::begin(mat), std::end(mat), T{0});
    return mat;
}

template <std::semiregular T, std::size_t N, bool isRowMajor = true>
Matrix<T, N, isRowMajor> full(const std::array<std::size_t, N>& arr, const T& fill_value) {
    Matrix<T, N> mat (arr);
    std::fill(std::begin(mat), std::end(mat), fill_value);
    return mat;
}

template <typename Derived, std::semiregular T, std::size_t N, bool isRowMajor>
Matrix<T, N, isRowMajor> full_like(const MatrixBase<Derived, T, N, isRowMajor>& base, const T& fill_value) {
    Matrix<T, N, isRowMajor> mat (base.dims());
    std::fill(std::begin(mat), std::end(mat), fill_value);
    return mat;
}

// binary matrix operators

namespace {

template <std::semiregular U, std::semiregular V, std::semiregular T,
        std::size_t N1, std::size_t N2, std::size_t N, bool isRowMajor>
requires AddableTo<U, V, T> && (std::max(N1, N2) == N)
void AddTo(MatrixView<T, N, false, isRowMajor>& m,
           const MatrixView<U, N1, true, isRowMajor>& m1,
           const MatrixView<V, N2, true, isRowMajor>& m2) {
    if constexpr (N == 1) {
        m.applyFunctionWithBroadcast(m1, m2, PlusTo<U, V, T>);
    } else {
        m.applyFunctionWithBroadcast(m1, m2, AddTo<U, V, T, std::min(N1, N - 1), std::min(N2, N - 1), N - 1, isRowMajor>);
    }
}

template <std::semiregular U, std::semiregular V, std::semiregular T,
        std::size_t N1, std::size_t N2, std::size_t N, bool isRowMajor>
requires SubtractableTo<U, V, T> && (std::max(N1, N2) == N)
void SubtractTo(MatrixView<T, N, false, isRowMajor>& m,
                const MatrixView<U, N1, true, isRowMajor>& m1,
                const MatrixView<V, N2, true, isRowMajor>& m2) {
    if constexpr (N == 1) {
        m.applyFunctionWithBroadcast(m1, m2, MinusTo<U, V, T>);
    } else {
        m.applyFunctionWithBroadcast(m1, m2, SubtractTo<U, V, T, std::min(N1, N - 1), std::min(N2, N - 1), N - 1, isRowMajor>);
    }
}

template <std::semiregular U, std::semiregular V, std::semiregular T,
        std::size_t N1, std::size_t N2, std::size_t N, bool isRowMajor>
requires MultipliableTo<U, V, T> && (std::max(N1, N2) == N)
void MultiplyTo(MatrixView<T, N, false, isRowMajor>& m,
                const MatrixView<U, N1, true, isRowMajor>& m1,
                const MatrixView<V, N2, true, isRowMajor>& m2) {
    if constexpr (N == 1) {
        m.applyFunctionWithBroadcast(m1, m2, MultipliesTo<U, V, T>);
    } else {
        m.applyFunctionWithBroadcast(m1, m2, MultiplyTo<U, V, T, std::min(N1, N - 1), std::min(N2, N - 1), N - 1, isRowMajor>);
    }
}

template <std::semiregular U, std::semiregular V, std::semiregular T,
        std::size_t N1, std::size_t N2, std::size_t N, bool isRowMajor>
requires DividableTo<U, V, T> && (std::max(N1, N2) == N)
void DivideTo(MatrixView<T, N, false, isRowMajor>& m,
              const MatrixView<U, N1, true, isRowMajor>& m1,
              const MatrixView<V, N2, true, isRowMajor>& m2) {
    if constexpr (N == 1) {
        m.applyFunctionWithBroadcast(m1, m2, DividesTo<U, V, T>);
    } else {
        m.applyFunctionWithBroadcast(m1, m2, DivideTo<U, V, T, std::min(N1, N - 1), std::min(N2, N - 1), N - 1, isRowMajor>);
    }
}

template <std::semiregular U, std::semiregular V, std::semiregular T,
        std::size_t N1, std::size_t N2, std::size_t N, bool isRowMajor>
requires RemaindableTo<U, V, T> && (std::max(N1, N2) == N)
void ModuloTo(MatrixView<T, N, false, isRowMajor>& m,
              const MatrixView<U, N1, true, isRowMajor>& m1,
              const MatrixView<V, N2, true, isRowMajor>& m2) {
    if constexpr (N == 1) {
        m.applyFunctionWithBroadcast(m1, m2, ModulusTo<U, V, T>);
    } else {
        m.applyFunctionWithBroadcast(m1, m2, ModuloTo<U, V, T, std::min(N1, N - 1), std::min(N2, N - 1), N - 1, isRowMajor>);
    }
}

} // anonymous namespace

template <typename Derived1, typename Derived2,
        std::semiregular U, std::semiregular V,
        std::size_t N1, std::size_t N2, bool isRowMajor,
        std::semiregular T = AddType<U, V>> requires AddableTo<U, V, T>
decltype(auto) operator+ (const MatrixBase<Derived1, U, N1, isRowMajor>& m1, const MatrixBase<Derived2, V, N2, isRowMajor>& m2) {
    constexpr std::size_t N = std::max(N1, N2);
    auto dims = bidirBroadcastedDims(m1.dims(), m2.dims());
    Matrix<T, N, isRowMajor> res = zeros<T, N, isRowMajor>(dims);
    if constexpr (N == 1) {
        res.applyFunctionWithBroadcast(m1, m2, PlusTo<U, V, T>);
    } else {
        res.applyFunctionWithBroadcast(m1, m2, AddTo<U, V, T, std::min(N1, N - 1), std::min(N2, N - 1), N - 1, isRowMajor>);
    }
    return res;
}

template <typename Derived1, typename Derived2,
        std::semiregular U, std::semiregular V,
        std::size_t N1, std::size_t N2, bool isRowMajor,
        std::semiregular T = SubType<U, V>> requires SubtractableTo<U, V, T>
decltype(auto) operator- (const MatrixBase<Derived1, U, N1, isRowMajor>& m1, const MatrixBase<Derived2, V, N2, isRowMajor>& m2) {
    constexpr std::size_t N = std::max(N1, N2);
    auto dims = bidirBroadcastedDims(m1.dims(), m2.dims());
    Matrix<T, N, isRowMajor> res = zeros<T, N, isRowMajor>(dims);
    if constexpr (N == 1) {
        res.applyFunctionWithBroadcast(m1, m2, MinusTo<U, V, T>);
    } else {
        res.applyFunctionWithBroadcast(m1, m2, SubtractTo<U, V, T, std::min(N1, N - 1), std::min(N2, N - 1), N - 1, isRowMajor>);
    }
    return res;
}

template <typename Derived1, typename Derived2,
        std::semiregular U, std::semiregular V,
        std::size_t N1, std::size_t N2, bool isRowMajor,
        std::semiregular T = MulType<U, V>> requires MultipliableTo<U, V, T>
decltype(auto) operator* (const MatrixBase<Derived1, U, N1, isRowMajor>& m1, const MatrixBase<Derived2, V, N2, isRowMajor>& m2) {
    constexpr std::size_t N = std::max(N1, N2);
    auto dims = bidirBroadcastedDims(m1.dims(), m2.dims());
    Matrix<T, N, isRowMajor> res = zeros<T, N, isRowMajor>(dims);
    if constexpr (N == 1) {
        res.applyFunctionWithBroadcast(m1, m2, MultipliesTo<U, V, T>);
    } else {
        res.applyFunctionWithBroadcast(m1, m2, MultiplyTo<U, V, T, std::min(N1, N - 1), std::min(N2, N - 1), N - 1, isRowMajor>);
    }
    return res;
}

template <typename Derived1, typename Derived2,
        std::semiregular U, std::semiregular V,
        std::size_t N1, std::size_t N2, bool isRowMajor,
        std::semiregular T = DivType<U, V>> requires DividableTo<U, V, T>
decltype(auto) operator/ (const MatrixBase<Derived1, U, N1, isRowMajor>& m1, const MatrixBase<Derived2, V, N2, isRowMajor>& m2) {
    constexpr std::size_t N = std::max(N1, N2);
    auto dims = bidirBroadcastedDims(m1.dims(), m2.dims());
    Matrix<T, N, isRowMajor> res = zeros<T, N, isRowMajor>(dims);
    if constexpr (N == 1) {
        res.applyFunctionWithBroadcast(m1, m2, DividesTo<U, V, T>);
    } else {
        res.applyFunctionWithBroadcast(m1, m2, DivideTo<U, V, T, std::min(N1, N - 1), std::min(N2, N - 1), N - 1, isRowMajor>);
    }
    return res;
}

template <typename Derived1, typename Derived2,
        std::semiregular U, std::semiregular V,
        std::size_t N1, std::size_t N2, bool isRowMajor,
        std::semiregular T = ModType<U, V>> requires RemaindableTo<U, V, T>
decltype(auto) operator% (const MatrixBase<Derived1, U, N1, isRowMajor>& m1, const MatrixBase<Derived2, V, N2, isRowMajor>& m2) {
    constexpr std::size_t N = std::max(N1, N2);
    auto dims = bidirBroadcastedDims(m1.dims(), m2.dims());
    Matrix<T, N, isRowMajor> res = zeros<T, N, isRowMajor>(dims);
    if constexpr (N == 1) {
        res.applyFunctionWithBroadcast(m1, m2, ModulusTo<U, V, T>);
    } else {
        res.applyFunctionWithBroadcast(m1, m2, ModuloTo<U, V, T, std::min(N1, N - 1), std::min(N2, N - 1), N - 1, isRowMajor>);
    }
    return res;
}

} // namespace frozenca

#endif //FROZENCA_MATRIXOPS_H
