#ifndef FROZENCA_MATRIX_H
#define FROZENCA_MATRIX_H

#include <array>
#include <iterator>
#include <memory>
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

    ~Matrix() override = default;

    template <typename... Dims>
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

    using iterator = T*;
    using const_iterator = const T*;
    using reverse_iterator = std::reverse_iterator<iterator>;
    using const_reverse_iterator = std::reverse_iterator<const_iterator>;

    iterator begin() { return &data_[0];}
    const_iterator cbegin() const { return &data_[0]; }
    iterator end() { return &data_[size()];}
    const_iterator cend() const { return &data_[size()];}
    reverse_iterator rbegin() { return std::make_reverse_iterator(end());}
    const_reverse_iterator crbegin() const { return std::make_reverse_iterator(cend());}
    reverse_iterator rend() { return std::make_reverse_iterator(begin());}
    const_reverse_iterator crend() const { return std::make_reverse_iterator(cbegin());}

    template <std::semiregular U> requires std::is_convertible_v<U, T>
    Matrix& operator=(const U& val);

};

template <std::semiregular T, std::size_t N>
template <typename... Dims>
Matrix<T, N>::Matrix(Dims... dims) : Base(dims...), data_(std::make_unique<T[]>(size())) {
}

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
Matrix<T, N>::Matrix(const MatrixBase<DerivedOther, U, N>& other) : Base(other),
    data_(std::make_unique<T[]>(size())) {
    std::size_t index = 0;
    for (auto it = std::begin(other); it != std::end(other); ++it) {
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

template <std::semiregular T, std::size_t N>
template <std::semiregular U> requires std::is_convertible_v<U, T>
Matrix<T, N>& Matrix<T, N>::operator=(const U& val) {
    applyFunction([&val](auto& v) {v = val;});
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

    ~Matrix() override = default;

    template <typename Dim> requires std::is_integral_v<Dim>
    explicit Matrix(Dim dim) : Base(dim), data_(std::make_unique<T[]>(dim)) {};

    explicit Matrix(typename MatrixInitializer<T, 1>::type init);
    Matrix& operator=(typename MatrixInitializer<T, 1>::type init);

    template <typename DerivedOther, std::semiregular U> requires std::is_convertible_v<U, T>
    explicit Matrix(const MatrixBase<DerivedOther, U, 1>& other);

    template <typename DerivedOther, std::semiregular U> requires std::is_convertible_v<U, T>
    Matrix& operator=(const MatrixBase<DerivedOther, U, 1>& other);

    friend void swap(Matrix& a, Matrix& b) noexcept {
        swap(static_cast<Base&>(a), static_cast<Base&>(b));
        std::swap(a.data_, b.data_);
    }

    using iterator = T*;
    using const_iterator = const T*;
    using reverse_iterator = std::reverse_iterator<iterator>;
    using const_reverse_iterator = std::reverse_iterator<const_iterator>;

    iterator begin() { return &data_[0];}
    const_iterator cbegin() const { return &data_[0]; }
    iterator end() { return &data_[dims()];}
    const_iterator cend() const { return &data_[dims()];}
    reverse_iterator rbegin() { return std::make_reverse_iterator(end());}
    const_reverse_iterator crbegin() const { return std::make_reverse_iterator(cend());}
    reverse_iterator rend() { return std::make_reverse_iterator(begin());}
    const_reverse_iterator crend() const { return std::make_reverse_iterator(cbegin());}

    template <std::semiregular U> requires std::is_convertible_v<U, T>
    Matrix& operator=(const U& val);

};


template <std::semiregular T>
Matrix<T, 1>::Matrix(typename MatrixInitializer<T, 1>::type init) : Base(init),
                                                                    data_(std::make_unique<T[]>(dims())) {
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
                                                                    data_(std::make_unique<T[]>(dims())) {
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

} // namespace frozenca

#endif //FROZENCA_MATRIX_H
