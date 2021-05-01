#ifndef FROZENCA_MATRIXBASE_H
#define FROZENCA_MATRIXBASE_H

#include <algorithm>
#include <array>
#include <cassert>
#include <cstddef>
#include <concepts>
#include <functional>
#include <initializer_list>
#include <iostream>
#include <numeric>
#include <stdexcept>
#include <type_traits>
#include <utility>
#include "MatrixUtils.h"
#include "MatrixInitializer.h"

namespace frozenca {

template <std::semiregular T, std::size_t N>
class MatrixView;

template <typename Derived, std::semiregular T, std::size_t N>
class MatrixBase {
    static_assert(N > 1);
public:
    static constexpr std::size_t ndim = N;

private:
    std::array<std::size_t, N> dims_;
    std::size_t size_;
    std::array<std::size_t, N> strides_;

    Derived& self() { return static_cast<Derived&>(*this); }
    const Derived& self() const { return static_cast<const Derived&>(*this); }

public:
    MatrixBase() = delete;

protected:
    virtual ~MatrixBase() = default;

    MatrixBase(const std::array<std::size_t, N>& dims);

    template <typename... Dims>
    explicit MatrixBase(Dims... dims);

    template <typename DerivedOther, std::regular U> requires std::is_convertible_v<U, T>
    MatrixBase(const MatrixBase<DerivedOther, U, N>&);

    MatrixBase(typename MatrixInitializer<T, N>::type init);

public:
    template <typename U>
    MatrixBase(std::initializer_list<U>) = delete;

    template <typename U>
    MatrixBase& operator=(std::initializer_list<U>) = delete;

    using value_type = T;
    using reference = T&;
    using const_reference = const T&;
    using pointer = T*;

public:
    friend void swap(MatrixBase& a, MatrixBase& b) noexcept {
        std::swap(a.size_, b.size_);
        std::swap(a.dims_, b.dims_);
        std::swap(a.strides_, b.strides_);
    }

    auto begin() { return self().begin(); }
    auto cbegin() const { return self().cbegin(); }
    auto end() { return self().end(); }
    auto cend() const { return self().cend(); }
    auto rbegin() { return self().rbegin(); }
    auto crbegin() const { return self().crbegin(); }
    auto rend() { return self().rend(); }
    auto crend() const { return self().crend(); }

    template <IndexType... Args>
    reference operator()(Args... args);

    template <IndexType... Args>
    const_reference operator()(Args... args) const;

    reference operator[](const std::array<std::size_t, N>& pos);
    const_reference operator[](const std::array<std::size_t, N>& pos) const;

    [[nodiscard]] std::size_t size() const { return size_;}

    [[nodiscard]] const std::array<std::size_t, N>& dims() const {
        return dims_;
    }

    [[nodiscard]] std::size_t dims(std::size_t n) const {
        if (n >= N) {
            throw std::out_of_range("Out of range in dims");
        }
        return dims_[n];
    }

    [[nodiscard]] const std::array<std::size_t, N>& strides() const {
        return strides_;
    }

    [[nodiscard]] std::size_t strides(std::size_t n) const {
        if (n >= N) {
            throw std::out_of_range("Out of range in strides");
        }
        return strides_[n];
    }

    auto dataView() const {
        return self().dataView();
    }

    auto origStrides() const {
        return self().origStrides();
    }

    MatrixView<T, N> submatrix(const std::array<std::size_t, N>& pos_begin);
    MatrixView<T, N> submatrix(const std::array<std::size_t, N>& pos_begin, const std::array<std::size_t, N>& pos_end);
    MatrixView<T, N - 1> row(std::size_t n);
    MatrixView<T, N - 1> col(std::size_t n);
    MatrixView<T, N - 1> operator[](std::size_t n) { return row(n); }

    MatrixView<T, N> submatrix(const std::array<std::size_t, N>& pos_begin) const;
    MatrixView<T, N> submatrix(const std::array<std::size_t, N>& pos_begin, const std::array<std::size_t, N>& pos_end) const;
    MatrixView<T, N - 1> row(std::size_t n) const;
    MatrixView<T, N - 1> col(std::size_t n) const;
    MatrixView<T, N - 1> operator[](std::size_t n) const { return row(n); }

    friend std::ostream& operator<<(std::ostream& os, const MatrixBase& m) {
        os << '{';
        for (std::size_t i = 0; i != m.dims(0); ++i) {
            os << m[i];
            if (i + 1 != m.dims(0)) {
                os << ", ";
            }
        }
        return os << '}';
    }
};

template <typename Derived, std::semiregular T, std::size_t N>
MatrixBase<Derived, T, N>::MatrixBase(const std::array<std::size_t, N>& dims) : dims_ {dims} {
    if (std::ranges::find(dims_, 0lu) != std::end(dims_)) {
        throw std::invalid_argument("Zero dimension not allowed");
    }
    size_ = std::accumulate(std::begin(dims_), std::end(dims_), 1lu, std::multiplies<>{});
    strides_ = computeStrides(dims_);
}

template <typename Derived, std::semiregular T, std::size_t N>
template <typename... Dims>
MatrixBase<Derived, T, N>::MatrixBase(Dims... dims) : dims_{static_cast<std::size_t>(dims)...} {
    static_assert(sizeof...(Dims) == N);
    static_assert((std::is_integral_v<Dims> && ...));
    if (std::ranges::find(dims_, 0lu) != std::end(dims_)) {
        throw std::invalid_argument("Zero dimension not allowed");
    }
    size_ = std::accumulate(std::begin(dims_), std::end(dims_), 1lu, std::multiplies<>{});
    strides_ = computeStrides(dims_);
}

template <typename Derived, std::semiregular T, std::size_t N>
template <typename DerivedOther, std::regular U> requires std::is_convertible_v<U, T>
MatrixBase<Derived, T, N>::MatrixBase(const MatrixBase<DerivedOther, U, N>& other)
    : dims_(other.dims_), size_(other.size_), strides_(other.strides_) {}

template <typename Derived, std::semiregular T, std::size_t N>
MatrixBase<Derived, T, N>::MatrixBase(typename MatrixInitializer<T, N>::type init) : MatrixBase(deriveDims<N>(init)) {
}

template <typename Derived, std::semiregular T, std::size_t N>
template <IndexType... Args>
typename MatrixBase<Derived, T, N>::reference MatrixBase<Derived, T, N>::operator()(Args... args) {
    return const_cast<typename MatrixBase<Derived, T, N>::reference>(std::as_const(*this).operator()(args...));
}

template <typename Derived, std::semiregular T, std::size_t N>
template <IndexType... Args>
typename MatrixBase<Derived, T, N>::const_reference MatrixBase<Derived, T, N>::operator()(Args... args) const {
    static_assert(sizeof...(args) == N);
    std::array<std::size_t, N> pos {std::size_t(args)...};
    return this->operator[](pos);
}

template <typename Derived, std::semiregular T, std::size_t N>
typename MatrixBase<Derived, T, N>::reference MatrixBase<Derived, T, N>::operator[](const std::array<std::size_t, N>& pos) {
    return const_cast<typename MatrixBase<Derived, T, N>::reference>(std::as_const(*this).operator[](pos));
}

template <typename Derived, std::semiregular T, std::size_t N>
typename MatrixBase<Derived, T, N>::const_reference MatrixBase<Derived, T, N>::operator[](const std::array<std::size_t, N>& pos) const {
    if (!std::equal(std::cbegin(pos), std::cend(pos), std::cbegin(dims_), std::less<>{})) {
        throw std::out_of_range("Out of range in element access");
    }
    return *(cbegin() + std::inner_product(std::cbegin(pos), std::cend(pos), std::cbegin(strides_), 0lu));
}

template <typename Derived, std::semiregular T, std::size_t N>
MatrixView<T, N> MatrixBase<Derived, T, N>::submatrix(const std::array<std::size_t, N>& pos_begin) {
    return submatrix(pos_begin, dims_);
}

template <typename Derived, std::semiregular T, std::size_t N>
MatrixView<T, N> MatrixBase<Derived, T, N>::submatrix(const std::array<std::size_t, N>& pos_begin) const {
    return submatrix(pos_begin, dims_);
}

template <typename Derived, std::semiregular T, std::size_t N>
MatrixView<T, N> MatrixBase<Derived, T, N>::submatrix(const std::array<std::size_t, N>& pos_begin,
                                                const std::array<std::size_t, N>& pos_end) {
    return std::as_const(*this).submatrix(pos_begin, pos_end);
}

template <typename Derived, std::semiregular T, std::size_t N>
MatrixView<T, N> MatrixBase<Derived, T, N>::submatrix(const std::array<std::size_t, N>& pos_begin,
                                                      const std::array<std::size_t, N>& pos_end) const {
    if (!std::equal(std::cbegin(pos_begin), std::cend(pos_begin), std::cbegin(pos_end), std::less<>{})) {
        throw std::out_of_range("submatrix begin/end position error");
    }
    std::array<std::size_t, N> view_dims;
    std::transform(std::cbegin(pos_end), std::cend(pos_end), std::cbegin(pos_begin), std::begin(view_dims),
                   std::minus<>{});
    MatrixView<T, N> view(view_dims, const_cast<T*>(&this->operator[](pos_begin)), strides());
    return view;
}

template <typename Derived, std::semiregular T, std::size_t N>
MatrixView<T, N - 1> MatrixBase<Derived, T, N>::row(std::size_t n) {
    return std::as_const(*this).row(n);
}

template <typename Derived, std::semiregular T, std::size_t N>
MatrixView<T, N - 1> MatrixBase<Derived, T, N>::row(std::size_t n) const {
    const auto& orig_dims = dims();
    if (n >= orig_dims[0]) {
        throw std::out_of_range("row index error");
    }
    std::array<std::size_t, N - 1> row_dims;
    std::copy(std::cbegin(orig_dims) + 1, std::cend(orig_dims), std::begin(row_dims));
    std::array<std::size_t, N> pos_begin = {n, };
    std::array<std::size_t, N - 1> row_strides;

    std::array<std::size_t, N> orig_strides;
    if constexpr (std::is_same_v<Derived, MatrixView<T, N>>) {
        orig_strides = origStrides();
    } else {
        orig_strides = strides();
    }

    std::copy(std::cbegin(orig_strides) + 1, std::cend(orig_strides), std::begin(row_strides));
    MatrixView<T, N - 1> nth_row(row_dims, const_cast<T*>(&this->operator[](pos_begin)), row_strides);
    return nth_row;
}

template <typename Derived, std::semiregular T, std::size_t N>
MatrixView<T, N - 1> MatrixBase<Derived, T, N>::col(std::size_t n) {
    return std::as_const(*this).col(n);
}

template <typename Derived, std::semiregular T, std::size_t N>
MatrixView<T, N - 1> MatrixBase<Derived, T, N>::col(std::size_t n) const {
    const auto& orig_dims = dims();
    if (n >= orig_dims[N - 1]) {
        throw std::out_of_range("row index error");
    }
    std::array<std::size_t, N - 1> col_dims;
    std::copy(std::cbegin(orig_dims), std::cend(orig_dims) - 1, std::begin(col_dims));
    std::array<std::size_t, N> pos_begin = {0};
    pos_begin[N - 1] = n;
    std::array<std::size_t, N - 1> col_strides;

    std::array<std::size_t, N> orig_strides;
    if constexpr (std::is_same_v<Derived, MatrixView<T, N>>) {
        orig_strides = origStrides();
    } else {
        orig_strides = strides();
    }

    std::copy(std::cbegin(orig_strides), std::cend(orig_strides) - 1, std::begin(col_strides));
    MatrixView<T, N - 1> nth_col(col_dims, const_cast<T*>(&this->operator[](pos_begin)), col_strides);
    return nth_col;
}


template <typename Derived, std::semiregular T>
class MatrixBase<Derived, T, 1> {
public:
    static constexpr std::size_t ndim = 1;

private:
    std::size_t dims_;
    std::size_t size_;
    std::size_t strides_;

    Derived& self() { return static_cast<Derived&>(*this); }
    const Derived& self() const { return static_cast<const Derived&>(*this); }

public:
    MatrixBase() = delete;

protected:
    virtual ~MatrixBase() = default;

    template <typename Dim> requires std::is_integral_v<Dim>
    explicit MatrixBase(Dim dim) : dims_(dim), size_(dim), strides_(1) {};

    template <typename DerivedOther, std::regular U> requires std::is_convertible_v<U, T>
    MatrixBase(const MatrixBase<DerivedOther, U, 1>&);

    MatrixBase(typename MatrixInitializer<T, 1>::type init);

public:
    template <typename U>
    MatrixBase(std::initializer_list<U>) = delete;

    template <typename U>
    MatrixBase& operator=(std::initializer_list<U>) = delete;

    using value_type = T;
    using reference = T&;
    using const_reference = const T&;
    using pointer = T*;

public:
    friend void swap(MatrixBase& a, MatrixBase& b) noexcept {
        std::swap(a.size_, b.size_);
        std::swap(a.dims_, b.dims_);
        std::swap(a.strides_, b.strides_);
    }

    auto begin() { return self().begin(); }
    auto cbegin() const { return self().cbegin(); }
    auto end() { return self().end(); }
    auto cend() const { return self().cend(); }
    auto rbegin() { return self().rbegin(); }
    auto crbegin() const { return self().crbegin(); }
    auto rend() { return self().rend(); }
    auto crend() const { return self().crend(); }

    template <typename Dim> requires std::is_integral_v<Dim>
    reference operator()(Dim dim) {
        return operator[](dim);
    }

    template <typename Dim> requires std::is_integral_v<Dim>
    const_reference operator()(Dim dim) const {
        return operator[](dim);
    }

    [[nodiscard]] std::size_t size() const { return size_;}

    [[nodiscard]] std::size_t dims() const {
        return dims_;
    }

    [[nodiscard]] std::size_t strides() const {
        return strides_;
    }

    auto dataView() const {
        return self().dataView();
    }

    auto origStrides() const {
        return self().origStrides();
    }

    T& submatrix(const std::array<std::size_t, 1>& pos_begin);
    T& submatrix(const std::array<std::size_t, 1>& pos_begin, const std::array<std::size_t, 1>& pos_end);
    T& row(std::size_t n);
    T& col(std::size_t n);
    T& operator[](std::size_t n) { return *(begin() + n); }

    const T& submatrix(const std::array<std::size_t, 1>& pos_begin) const;
    const T& submatrix(const std::array<std::size_t, 1>& pos_begin, const std::array<std::size_t, 1>& pos_end) const;
    const T& row(std::size_t n) const;
    const T& col(std::size_t n) const;
    const T& operator[](std::size_t n) const { return *(cbegin() + n); }

    friend std::ostream& operator<<(std::ostream& os, const MatrixBase& m) {
        os << '{';
        for (std::size_t i = 0; i != m.dims_; ++i) {
            os << m[i];
            if (i + 1 != m.dims_) {
                os << ", ";
            }
        }
        return os << '}';
    }
};

template <typename Derived, std::semiregular T>
T& MatrixBase<Derived, T, 1>::submatrix(const std::array<std::size_t, 1>& pos_begin) {
    return submatrix(pos_begin, dims_);
}

template <typename Derived, std::semiregular T>
const T& MatrixBase<Derived, T, 1>::submatrix(const std::array<std::size_t, 1>& pos_begin) const {
    return submatrix(pos_begin, dims_);
}

template <typename Derived, std::semiregular T>
T& MatrixBase<Derived, T, 1>::submatrix(const std::array<std::size_t, 1>& pos_begin,
                                        const std::array<std::size_t, 1>& pos_end) {
    return const_cast<T&>(std::as_const(*this).submatrix(pos_begin, pos_end));
}

template <typename Derived, std::semiregular T>
const T& MatrixBase<Derived, T, 1>::submatrix(const std::array<std::size_t, 1>& pos_begin,
                                        const std::array<std::size_t, 1>& pos_end) const {
    if (pos_begin[0] >= pos_end[0]) {
        throw std::out_of_range("submatrix begin/end position error");
    }
    const T& view = this->operator[](pos_begin);
    return view;
}

template <typename Derived, std::semiregular T>
T& MatrixBase<Derived, T, 1>::row(std::size_t n) {
    return const_cast<T&>(std::as_const(*this).row(n));
}

template <typename Derived, std::semiregular T>
const T& MatrixBase<Derived, T, 1>::row(std::size_t n) const {
    if (n >= dims_) {
        throw std::out_of_range("row index error");
    }
    const T& view = this->operator[](n);
    return view;
}

template <typename Derived, std::semiregular T>
T& MatrixBase<Derived, T, 1>::col(std::size_t n) {
    return row(n);
}

template <typename Derived, std::semiregular T>
const T& MatrixBase<Derived, T, 1>::col(std::size_t n) const {
    return row(n);
}

} // namespace frozenca

#endif //FROZENCA_MATRIXBASE_H
