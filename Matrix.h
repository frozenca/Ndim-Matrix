#ifndef FROZENCA_MATRIX_H
#define FROZENCA_MATRIX_H

#include "MatrixBase.h"
#include "MatrixView.h"

namespace frozenca {

template <std::semiregular T, std::size_t N>
class Matrix final : public MatrixBase<Matrix<T, N>, T, N> {
private:
    std::unique_ptr<T[]> data_;

public:
    using Base = MatrixBase<Matrix<T, N>, T, N>;
    using Base::size;
    using Base::dims;
    using Base::strides;
    using Base::applyFunction;
    using Base::applyBroadcast;
    using Base::operator=;
    using Base::operator+=;
    using Base::operator-=;
    using Base::operator*=;
    using Base::operator/=;
    using Base::operator%=;

    using iterator = T*;
    using const_iterator = const T*;
    using reverse_iterator = std::reverse_iterator<iterator>;
    using const_reverse_iterator = std::reverse_iterator<const_iterator>;

    iterator begin() { return &data_[0];}
    const_iterator begin() const { return &data_[0]; }
    const_iterator cbegin() const { return &data_[0]; }
    iterator end() { return &data_[size()];}
    const_iterator end() const { return &data_[size()]; }
    const_iterator cend() const { return &data_[size()];}
    reverse_iterator rbegin() { return std::make_reverse_iterator(end());}
    const_reverse_iterator rbegin() const { return std::make_reverse_iterator(end()); }
    const_reverse_iterator crbegin() const { return std::make_reverse_iterator(cend());}
    reverse_iterator rend() { return std::make_reverse_iterator(begin());}
    const_reverse_iterator rend() const { return std::make_reverse_iterator(begin()); }
    const_reverse_iterator crend() const { return std::make_reverse_iterator(cbegin());}

    ~Matrix() override = default;
    Matrix(const Matrix& other) : Base(other.dims()), data_(std::make_unique<T[]>(size())) {
        std::size_t index = 0;
        for (auto it = std::cbegin(other); it != std::cend(other); ++it) {
            data_[index] = static_cast<T>(*it);
            index++;
        }
    }

    Matrix& operator=(const Matrix& other) {
        Matrix mat(other);
        swap(*this, mat);
        return *this;
    }

    Matrix(Matrix&& other) noexcept = default;
    Matrix& operator=(Matrix&& other) noexcept = default;

    Matrix(const std::array<std::size_t, N>& dims);

    template <IndexType... Dims>
    explicit Matrix(Dims... dims);

    explicit Matrix(typename MatrixInitializer<T, N>::type init);
    Matrix& operator=(typename MatrixInitializer<T, N>::type init);

    template <typename DerivedOther, std::semiregular U> requires std::is_convertible_v<U, T>
    explicit Matrix(const MatrixBase<DerivedOther, U, N>& other);

    template <typename DerivedOther, std::semiregular U> requires std::is_convertible_v<U, T>
    Matrix& operator=(const MatrixBase<DerivedOther, U, N>& other);

    friend void swap(Matrix& a, Matrix& b) noexcept {
        swap(static_cast<Base&>(a), static_cast<Base&>(b));
        std::swap(a.data_, b.data_);
    }

};

template <std::semiregular T, std::size_t N>
Matrix<T, N>::Matrix(const std::array<std::size_t, N>& dims) : Base(dims), data_(std::make_unique<T[]>(size())) {}

template <std::semiregular T, std::size_t N>
template <IndexType... Dims>
Matrix<T, N>::Matrix(Dims... dims) : Base(dims...), data_(std::make_unique<T[]>(size())) {}

template <std::semiregular T, std::size_t N>
Matrix<T, N>::Matrix(typename MatrixInitializer<T, N>::type init) : Base(init),
        data_(std::make_unique<T[]>(size())) {
    insertFlat(data_, init);
}

template <std::semiregular T, std::size_t N>
Matrix<T, N>& Matrix<T, N>::operator=(typename MatrixInitializer<T, N>::type init) {
    Matrix<T, N> mat(init);
    swap(*this, mat);
    return *this;
}

template <std::semiregular T, std::size_t N>
template <typename DerivedOther, std::semiregular U> requires std::is_convertible_v<U, T>
Matrix<T, N>::Matrix(const MatrixBase<DerivedOther, U, N>& other) : Base(other.dims()),
                                                                    data_(std::make_unique<T[]>(size())) {
    std::size_t index = 0;
    for (auto it = std::cbegin(other); it != std::cend(other); ++it) {
        data_[index] = static_cast<T>(*it);
        index++;
    }
}

template <std::semiregular T, std::size_t N>
template <typename DerivedOther, std::semiregular U> requires std::is_convertible_v<U, T>
Matrix<T, N>& Matrix<T, N>::operator=(const MatrixBase<DerivedOther, U, N>& other) {
    Matrix<T, N> mat(other);
    swap(*this, mat);
    return *this;
}

template <std::semiregular T>
class Matrix<T, 1> final : public MatrixBase<Matrix<T, 1>, T, 1> {
private:
    std::unique_ptr<T[]> data_;

public:
    using Base = MatrixBase<Matrix<T, 1>, T, 1>;
    using Base::dims;
    using Base::strides;
    using Base::applyFunction;
    using Base::applyBroadcast;
    using Base::operator=;
    using Base::operator+=;
    using Base::operator-=;
    using Base::operator*=;
    using Base::operator/=;
    using Base::operator%=;

    friend void swap(Matrix& a, Matrix& b) noexcept {
        swap(static_cast<Base&>(a), static_cast<Base&>(b));
        std::swap(a.data_, b.data_);
    }

    using iterator = T*;
    using const_iterator = const T*;
    using reverse_iterator = std::reverse_iterator<iterator>;
    using const_reverse_iterator = std::reverse_iterator<const_iterator>;

    iterator begin() { return &data_[0];}
    const_iterator begin() const { return &data_[0]; }
    const_iterator cbegin() const { return &data_[0]; }
    iterator end() { return &data_[dims(0)];}
    const_iterator end() const { return &data_[dims(0)]; }
    const_iterator cend() const { return &data_[dims(0)];}
    reverse_iterator rbegin() { return std::make_reverse_iterator(end());}
    const_reverse_iterator rbegin() const { return std::make_reverse_iterator(end()); }
    const_reverse_iterator crbegin() const { return std::make_reverse_iterator(cend());}
    reverse_iterator rend() { return std::make_reverse_iterator(begin());}
    const_reverse_iterator rend() const { return std::make_reverse_iterator(begin()); }
    const_reverse_iterator crend() const { return std::make_reverse_iterator(cbegin());}

    ~Matrix() override = default;
    Matrix(const Matrix& other) : Base(other.dims(0)), data_(std::make_unique<T[]>(dims(0))) {
        std::size_t index = 0;
        for (auto it = std::cbegin(other); it != std::cend(other); ++it) {
            data_[index] = static_cast<T>(*it);
            index++;
        }
    }

    Matrix& operator=(const Matrix& other) {
        Matrix mat(other);
        swap(*this, mat);
        return *this;
    }

    Matrix(Matrix&& other) noexcept = default;
    Matrix& operator=(Matrix&& other) noexcept = default;

    Matrix(const std::array<std::size_t, 1>& dims) : Base(dims[0]), data_(std::make_unique<T[]>(dims[0])) {}

    template <typename Dim> requires std::is_integral_v<Dim>
    explicit Matrix(Dim dim) : Base(dim), data_(std::make_unique<T[]>(dim)) {}

    explicit Matrix(typename MatrixInitializer<T, 1>::type init);
    Matrix& operator=(typename MatrixInitializer<T, 1>::type init);

    template <typename DerivedOther, std::semiregular U> requires std::is_convertible_v<U, T>
    explicit Matrix(const MatrixBase<DerivedOther, U, 1>& other);

    template <typename DerivedOther, std::semiregular U> requires std::is_convertible_v<U, T>
    Matrix& operator=(const MatrixBase<DerivedOther, U, 1>& other);

};


template <std::semiregular T>
Matrix<T, 1>::Matrix(typename MatrixInitializer<T, 1>::type init) : Base(init),
                                                                    data_(std::make_unique<T[]>(dims(0))) {
    insertFlat(data_, init);
}

template <std::semiregular T>
Matrix<T, 1>& Matrix<T, 1>::operator=(typename MatrixInitializer<T, 1>::type init) {
    Matrix<T, 1> mat(init);
    swap(*this, mat);
    return *this;
}

template <std::semiregular T>
template <typename DerivedOther, std::semiregular U> requires std::is_convertible_v<U, T>
Matrix<T, 1>::Matrix(const MatrixBase<DerivedOther, U, 1>& other) : Base(other),
                                                                    data_(std::make_unique<T[]>(dims(0))) {
    std::size_t index = 0;
    for (auto it = std::begin(other); it != std::end(other); ++it) {
        data_[index] = static_cast<T>(*it);
        index++;
    }
}

template <std::semiregular T>
template <typename DerivedOther, std::semiregular U> requires std::is_convertible_v<U, T>
Matrix<T, 1>& Matrix<T, 1>::operator=(const MatrixBase<DerivedOther, U, 1>& other) {
    Matrix<T, 1> mat(other);
    swap(*this, mat);
    return *this;
}

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

// binary matrix operators

namespace {

template <std::semiregular U, std::semiregular V, std::semiregular T,
        std::size_t N1, std::size_t N2, std::size_t N>
requires AddableTo<U, V, T> && (std::max(N1, N2) == N)
void AddTo(MatrixView<T, N>& m,
           const MatrixView<U, N1>& m1,
           const MatrixView<V, N2>& m2) {
    if constexpr (N == 1) {
        m.applyBroadcast(m1, m2, Plus<U, V, T>);
    } else {
        m.applyBroadcast(m1, m2, AddTo<U, V, T, std::min(N1, N - 1), std::min(N2, N - 1), N - 1>);
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
    res.applyBroadcast(m1, m2, AddTo<U, V, T, std::min(N1, N - 1), std::min(N2, N - 1), N - 1>);
    return res;
}




} // namespace frozenca

#endif //FROZENCA_MATRIX_H
