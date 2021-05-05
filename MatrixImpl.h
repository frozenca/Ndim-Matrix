#ifndef FROZENCA_MATRIXIMPL_H
#define FROZENCA_MATRIXIMPL_H

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
    using Base::applyFunctionWithBroadcast;
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

private:
    // only invoked by reshape
    Matrix(const std::array<std::size_t, N>& dims, std::unique_ptr<T[]> buffer) noexcept
    : Base(dims), data_(std::move(buffer)) {}

    // only invoked by transpose
    Matrix(const std::array<std::size_t, N>& new_dims, const MatrixView<T, N>& view) noexcept
            : Base(new_dims), data_(std::make_unique<T[]>(size())) {
        std::size_t index = 0;
        for (auto it = std::cbegin(view); it != std::cend(view); ++it) {
            data_[index] = static_cast<T>(*it);
            index++;
        }
    }

public:
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

    template <std::semiregular T_, std::size_t M_, std::size_t N_>
    friend Matrix<T_, M_> reshaped(Matrix<T_, N_>&& orig, const std::array<std::size_t, M_>& new_dims) noexcept;

    template <std::semiregular T_, std::size_t N_>
    friend Matrix<T_, N_> transpose(const Matrix<T_, N_>& orig, const std::array<std::size_t, N_>& perm);
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
    using Base::applyFunctionWithBroadcast;
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

    const T* dataView() const { return data_.get(); }

    ~Matrix() noexcept = default;
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

private:
    // only invoked by reshape
    Matrix(const std::array<std::size_t, 1>& dims, std::unique_ptr<T[]> buffer) noexcept
            : Base(dims), data_(std::move(buffer)) {}

    // only invoked by transpose
    Matrix(const std::array<std::size_t, 1>& new_dims, const MatrixView<T, 1>& view) noexcept
            : Base(new_dims), data_(std::make_unique<T[]>(dims(0))) {
        std::size_t index = 0;
        for (auto it = std::cbegin(view); it != std::cend(view); ++it) {
            data_[index] = static_cast<T>(*it);
            index++;
        }
    }

public:
    template <typename Dim> requires std::is_integral_v<Dim>
    explicit Matrix(Dim dim) : Base(dim), data_(std::make_unique<T[]>(dim)) {}

    explicit Matrix(typename MatrixInitializer<T, 1>::type init);
    Matrix& operator=(typename MatrixInitializer<T, 1>::type init);

    template <typename DerivedOther, std::semiregular U> requires std::is_convertible_v<U, T>
    explicit Matrix(const MatrixBase<DerivedOther, U, 1>& other);

    template <typename DerivedOther, std::semiregular U> requires std::is_convertible_v<U, T>
    Matrix& operator=(const MatrixBase<DerivedOther, U, 1>& other);

    template <std::semiregular T_, std::size_t M_, std::size_t N_>
    friend Matrix<T_, M_> reshaped(Matrix<T_, N_>&& orig, const std::array<std::size_t, M_>& new_dims) noexcept;

    template <std::semiregular T_, std::size_t N_>
    friend Matrix<T_, N_> transpose(const Matrix<T_, N_>& orig, const std::array<std::size_t, N_>& perm);

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

template <std::semiregular T, std::size_t M, std::size_t N>
Matrix<T, M> reshaped(Matrix<T, N>&& orig, const std::array<std::size_t, M>& new_dims) noexcept {
    Matrix<T, M> new_mat(new_dims, std::move(orig.data_));
    return new_mat;
}

template <std::semiregular T, std::size_t M, std::size_t N>
Matrix<T, M> reshape(Matrix<T, N>&& orig, const std::array<std::size_t, M>& new_dims) {
    auto new_size = std::accumulate(std::begin(new_dims), std::end(new_dims), 1lu, std::multiplies<>{});
    if (new_size != orig.size()) {
        throw std::invalid_argument("Cannot reshape, size is different");
    }
    return reshaped(std::move(orig), new_dims);
}

template <std::semiregular T, std::size_t N>
Matrix<T, N> transpose(const Matrix<T, N>& orig, const std::array<std::size_t, N>& perm) {
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
    MatrixView<T, N> trans_view (trans_dims, const_cast<T*>(orig.dataView()), trans_strides);
    Matrix<T, N> transposed(trans_dims, trans_view);
    return transposed;
}

template <std::semiregular T, std::size_t N>
Matrix<T, N> transpose(const Matrix<T, N>& orig) {
    std::array<std::size_t, N> reversed_perm;
    std::iota(std::rbegin(reversed_perm), std::rend(reversed_perm), 0lu);
    return transpose(orig, reversed_perm);
}

} // namespace frozenca

#endif //FROZENCA_MATRIXIMPL_H
