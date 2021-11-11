#ifndef FROZENCA_MATRIXIMPL_H
#define FROZENCA_MATRIXIMPL_H

#include "MatrixBase.h"
#include "MatrixView.h"

namespace frozenca {

template <std::semiregular T, std::size_t N, bool isRowMajor>
class Matrix final : public MatrixBase<Matrix<T, N, isRowMajor>, T, N, isRowMajor> {
private:
    std::unique_ptr<T[]> data_;

public:
    using Base = MatrixBase<Matrix<T, N, isRowMajor>, T, N, isRowMajor>;
    using Base::size;
    using Base::dims;
    using Base::strides;
    using Base::applyFunction;
    using Base::applyFunctionWithBroadcast;
    using Base::operator=;
    using Base::operator+=;
    using Base::operator-=;
    using Base::operator*=;
    using Base::operator/=;
    using Base::operator%=;
    using value_type = T;
    using reference = T&;
    using const_reference = const T&;
    using pointer = T*;

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

    const T* dataView() const { return data_.get(); }

    ~Matrix() noexcept = default;
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

public:
    Matrix(const std::array<std::size_t, N>& dims, std::unique_ptr<T[]> buffer) noexcept
    : Base(dims), data_(std::move(buffer)) {}

    Matrix(const std::array<std::size_t, N>& new_dims, MatrixView<T, N, true, isRowMajor>& view) noexcept
            : Base(new_dims), data_(std::make_unique<T[]>(size())) {
        std::size_t index = 0;
        for (auto it = std::cbegin(view); it != std::cend(view); ++it) {
            data_[index] = static_cast<T>(*it);
            index++;
        }
    }

    template <IndexType... Dims>
    explicit Matrix(Dims... dims);

    explicit Matrix(typename MatrixInitializer<T, N>::type init) requires (isRowMajor);
    Matrix& operator=(typename MatrixInitializer<T, N>::type init) requires (isRowMajor);

    template <typename DerivedOther, std::semiregular U, bool isRowMajorOther> requires std::is_convertible_v<U, T>
    Matrix(const MatrixBase<DerivedOther, U, N, isRowMajorOther>& other);

    template <typename DerivedOther, std::semiregular U, bool isRowMajorOther> requires std::is_convertible_v<U, T>
    Matrix& operator=(const MatrixBase<DerivedOther, U, N, isRowMajorOther>& other);

    friend void swap(Matrix& a, Matrix& b) noexcept {
        swap(static_cast<Base&>(a), static_cast<Base&>(b));
        std::swap(a.data_, b.data_);
    }

    Matrix& transpose_();

    Matrix operator-() {
        return Base::Base::operator-();
    }
};

template <std::semiregular T, std::size_t N, bool isRowMajor>
Matrix<T, N, isRowMajor>::Matrix(const std::array<std::size_t, N>& dims) : Base(dims), data_(std::make_unique<T[]>(size())) {}

template <std::semiregular T, std::size_t N, bool isRowMajor>
template <IndexType... Dims>
Matrix<T, N, isRowMajor>::Matrix(Dims... dims) : Base(dims...), data_(std::make_unique<T[]>(size())) {}

template <std::semiregular T, std::size_t N, bool isRowMajor>
Matrix<T, N, isRowMajor>::Matrix(typename MatrixInitializer<T, N>::type init) requires (isRowMajor) : Base(init),
        data_(std::make_unique<T[]>(size())) {
    insertFlat(data_, init);
}

template <std::semiregular T, std::size_t N, bool isRowMajor>
Matrix<T, N, isRowMajor>& Matrix<T, N, isRowMajor>::operator=(typename MatrixInitializer<T, N>::type init) requires (isRowMajor) {
    Matrix<T, N, isRowMajor> mat(init);
    swap(*this, mat);
    return *this;
}

template <std::semiregular T, std::size_t N, bool isRowMajor>
template <typename DerivedOther, std::semiregular U, bool isRowMajorOther> requires std::is_convertible_v<U, T>
Matrix<T, N, isRowMajor>::Matrix(const MatrixBase<DerivedOther, U, N, isRowMajorOther>& other) : Base(other.dims()),
                                                                    data_(std::make_unique<T[]>(size())) {
    std::size_t index = 0;
    if constexpr (isRowMajor == isRowMajorOther) {
        for (auto it = std::cbegin(other); it != std::cend(other); ++it) {
            data_[index] = static_cast<T>(*it);
            index++;
        }
    } else {
        MatrixView<T, N, true, isRowMajor> other_view(other.dims(), other.dataView(), other.strides());
        for (auto it = std::cbegin(other_view); it != std::cend(other_view); ++it) {
            data_[index] = static_cast<T>(*it);
            index++;
        }
    }
}

template <std::semiregular T, std::size_t N, bool isRowMajor>
template <typename DerivedOther, std::semiregular U, bool isRowMajorOther> requires std::is_convertible_v<U, T>
Matrix<T, N, isRowMajor>& Matrix<T, N, isRowMajor>::operator=(const MatrixBase<DerivedOther, U, N, isRowMajorOther>& other) {
    Matrix<T, N, isRowMajor> mat(other);
    swap(*this, mat);
    return *this;
}

template <std::semiregular T, std::size_t M, std::size_t N, bool isRowMajor>
Matrix<T, M, isRowMajor> reshaped(Matrix<T, N, isRowMajor>&& orig, const std::array<std::size_t, M>& new_dims) noexcept {
    Matrix<T, M, isRowMajor> new_mat(new_dims, std::move(orig.data_));
    return new_mat;
}

template <std::semiregular T, std::size_t M, std::size_t N, bool isRowMajor>
Matrix<T, M, isRowMajor> reshape(Matrix<T, N, isRowMajor>&& orig, const std::array<std::size_t, M>& new_dims) {
    auto new_size = std::accumulate(std::begin(new_dims), std::end(new_dims), 1lu, std::multiplies<>{});
    if (new_size != orig.size()) {
        throw std::invalid_argument("Cannot reshape, size is different");
    }
    return reshaped(std::move(orig), new_dims);
}

template <typename Derived, std::semiregular T, std::size_t N, bool isRowMajor>
Matrix<T, N, isRowMajor> transpose(const MatrixBase<Derived, T, N, isRowMajor>& orig, const std::array<std::size_t, N>& perm) {
    std::array<std::size_t, N> identity_perm;
    std::iota(std::begin(identity_perm), std::end(identity_perm), 0lu);
    if (!std::ranges::is_permutation(identity_perm, perm)) {
        throw std::invalid_argument("Invalid permutation");
    }

    std::array<std::size_t, N> trans_dims;
    std::array<std::size_t, N> trans_strides;
    for (std::size_t i = 0; i < N; ++i) {
        trans_dims[i] = orig.dims(perm[i]);
        trans_strides[i] = orig.strides(perm[i]);
    }
    std::array<std::size_t, N> pos_begin = {0};
    MatrixView<T, N, true, isRowMajor> trans_view (trans_dims, orig.dataView(), trans_strides);
    Matrix<T, N, isRowMajor> transposed(trans_dims, trans_view);
    return transposed;
}

template <typename Derived, std::semiregular T, std::size_t N, bool isRowMajor>
Matrix<T, N, isRowMajor> transpose(const MatrixBase<Derived, T, N, isRowMajor>& orig) {
    std::array<std::size_t, N> reversed_perm;
    std::iota(std::rbegin(reversed_perm), std::rend(reversed_perm), 0lu);
    return transpose(orig, reversed_perm);
}

template <typename Derived, std::semiregular T, bool isRowMajor>
Matrix<T, 2, !isRowMajor> transpose_change_major(const MatrixBase<Derived, T, 2, isRowMajor>& orig) {
    auto dims = orig.dims();
    std::ranges::reverse(dims);
    std::unique_ptr<T[]> buffer = std::make_unique<T[]>(orig.size());
    std::size_t index = 0;
    for (auto it = std::cbegin(orig); it != std::cend(orig); ++it) {
        buffer[index] = static_cast<T>(*it);
        index++;
    }
    Matrix <T, 2, !isRowMajor> transposed(dims, std::move(buffer));
    return transposed;
}

template <std::semiregular T, std::size_t N, bool isRowMajor>
Matrix<T, N, isRowMajor>& Matrix<T, N, isRowMajor>::transpose_() {
    auto transposed = transpose(*this);
    std::swap(*this, transposed);
    return *this;
}

template <typename Derived, std::semiregular T, std::semiregular U, std::size_t N, bool isRowMajor> requires Multipliable<T, U>
Matrix<T, N, isRowMajor> operator* (const U& val, const MatrixBase<Derived, T, N, isRowMajor>& orig) {
    Matrix<T, N, isRowMajor> mat (orig);
    mat *= val;
    return mat;
}

} // namespace frozenca

#endif //FROZENCA_MATRIXIMPL_H
