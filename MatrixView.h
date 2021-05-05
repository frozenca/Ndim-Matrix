#ifndef FROZENCA_MATRIXVIEW_H
#define FROZENCA_MATRIXVIEW_H

#include <compare>
#include "MatrixBase.h"

namespace frozenca {

template <std::semiregular T, std::size_t N>
class MatrixView final : public MatrixBase<MatrixView<T, N>, T, N> {
private:
    T* data_view_;
    std::array<std::size_t, N> orig_strides_;

public:
    using Base = MatrixBase<MatrixView<T, N>, T, N>;
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

    ~MatrixView() noexcept = default;

    explicit MatrixView(const std::array<std::size_t, N>& dims,
                        T* data_view,
                        const std::array<std::size_t, N>& orig_strides);

    template <typename DerivedOther, std::semiregular U> requires std::is_convertible_v<U, T>
    MatrixView(const MatrixBase<DerivedOther, U, N>& other);

    template <typename DerivedOther, std::semiregular U> requires std::is_convertible_v<U, T>
    MatrixView& operator=(const MatrixBase<DerivedOther, U, N>& other);

    friend void swap(MatrixView& a, MatrixView& b) noexcept {
        swap(static_cast<Base&>(a), static_cast<Base&>(b));
        std::swap(a.data_view_, b.data_view_);
        std::swap(a.orig_strides_, b.orig_strides_);
    }

    template <typename T_>
    struct MVIterator {
        using difference_type = std::ptrdiff_t;
        using value_type = T_;
        using pointer = T_*;
        using reference = T_&;
        using iterator_category = std::random_access_iterator_tag;
        using MatViewType = std::conditional_t<std::is_const_v<T_>, const MatrixView*, MatrixView*>;

        MatViewType ptr_ = nullptr;
        std::array<std::size_t, N> pos_ = {0};
        std::size_t offset_ = 0;
        std::size_t index_ = 0;

        MVIterator(MatViewType ptr, std::array<std::size_t, N> pos = {0}) : ptr_ {ptr}, pos_ {pos} {
            ValidateOffset();
        }

        reference operator*() const {
            return ptr_->data_view_[offset_];
        }

        pointer operator->() const {
            return ptr_->data_view_ + offset_;
        }

        void ValidateOffset() {
            offset_ = std::inner_product(std::cbegin(pos_), std::cend(pos_), std::cbegin(ptr_->orig_strides_), 0lu);
            index_ = std::inner_product(std::cbegin(pos_), std::cend(pos_), std::cbegin(ptr_->strides()), 0lu);
            assert(index_ <= ptr_->size());
        }

        void Increment() {
            for (std::size_t i = N - 1; i < N; --i) {
                ++pos_[i];
                if (pos_[i] != ptr_->dims(i) || i == 0) {
                    break;
                } else {
                    pos_[i] = 0;
                }
            }
            ValidateOffset();
        }

        void Increment(std::ptrdiff_t n) {
            if (n < 0) {
                Decrement(-n);
                return;
            }
            auto carry = static_cast<std::size_t>(n);
            for (std::size_t i = N - 1; i < N; --i) {
                std::size_t curr_dim = ptr_->dims(i);
                pos_[i] += carry;
                if (pos_[i] < curr_dim || i == 0) {
                    break;
                } else {
                    carry = pos_[i] / curr_dim;
                    pos_[i] %= curr_dim;
                }
            }
            ValidateOffset();
        }

        void Decrement() {
            for (std::size_t i = N - 1; i < N; --i) {
                --pos_[i];
                if (pos_[i] != static_cast<std::size_t>(-1) || i == 0) {
                    break;
                } else {
                    pos_[i] = ptr_->dims(i) - 1;
                }
            }
            ValidateOffset();
        }

        void Decrement(std::ptrdiff_t n) {
            if (n < 0) {
                Increment(-n);
                return;
            }
            auto carry = static_cast<std::size_t>(n);
            for (std::size_t i = N - 1; i < N; --i) {
                std::size_t curr_dim = ptr_->dims(i);
                pos_[i] -= carry;
                if (pos_[i] < curr_dim || i == 0) {
                    break;
                } else {
                    carry = static_cast<std::size_t>(-quot(static_cast<long>(pos_[i]), static_cast<long>(curr_dim)));
                    pos_[i] = mod(static_cast<long>(pos_[i]), static_cast<long>(curr_dim));
                }
            }
            ValidateOffset();
        }

        MVIterator& operator++() {
            Increment();
            return *this;
        }

        MVIterator operator++(int) {
            MVIterator temp = *this;
            Increment();
            return temp;
        }

        MVIterator& operator--() {
            Decrement();
            return *this;
        }

        MVIterator operator--(int) {
            MVIterator temp = *this;
            Decrement();
            return temp;
        }

        MVIterator operator+(difference_type n) const {
            MVIterator temp = *this;
            temp.Increment(n);
            return temp;
        }

        MVIterator& operator+=(difference_type n) {
            Increment(n);
            return *this;
        }

        MVIterator operator-(difference_type n) const {
            MVIterator temp = *this;
            temp.Decrement(n);
            return temp;
        }

        MVIterator& operator-=(difference_type n) {
            Decrement(n);
            return *this;
        }

        reference operator[](difference_type n) const {
            return *(this + n);
        }

        template <typename T2> requires std::is_same_v<std::remove_cv_t<T_>, std::remove_cv_t<T2>>
        difference_type operator-(const MVIterator<T2>& other) const {
            return offset_ - other.offset_;
        }

    };

    // oh no.. *why* defining operator<=> doesn't work to automatically define these in gcc?
    template <typename T1, typename T2> requires std::is_same_v<std::remove_cv_t<T1>, std::remove_cv_t<T2>>
    friend bool operator==(const MVIterator<T1>& it1, const MVIterator<T2>& it2) {
        return it1.pos_ == it2.pos_;
    }

    template <typename T1, typename T2> requires std::is_same_v<std::remove_cv_t<T1>, std::remove_cv_t<T2>>
    friend bool operator!=(const MVIterator<T1>& it1, const MVIterator<T2>& it2) {
        return !(it1 == it2);
    }

    template <typename T1, typename T2> requires std::is_same_v<std::remove_cv_t<T1>, std::remove_cv_t<T2>>
    friend auto operator<=>(const MVIterator<T1>& it1, const MVIterator<T2>& it2) {
        return it1.pos_ <=> it2.pos_;
    }

    using iterator = MVIterator<T>;
    using const_iterator = MVIterator<const T>;
    using reverse_iterator = std::reverse_iterator<iterator>;
    using const_reverse_iterator = std::reverse_iterator<const_iterator>;

    iterator begin() { return iterator(this);}
    const_iterator begin() const { return const_iterator(this); }
    const_iterator cbegin() const { return const_iterator(this); }
    iterator end() { return iterator(this, {dims(0), });}
    const_iterator end() const { return const_iterator(this, {dims(0), });}
    const_iterator cend() const { return const_iterator(this, {dims(0), });}
    reverse_iterator rbegin() { return std::make_reverse_iterator(end());}
    const_reverse_iterator rbegin() const { return std::make_reverse_iterator(cend());}
    const_reverse_iterator crbegin() const { return std::make_reverse_iterator(cend());}
    reverse_iterator rend() { return std::make_reverse_iterator(begin());}
    const_reverse_iterator rend() const { return std::make_reverse_iterator(cbegin());}
    const_reverse_iterator crend() const { return std::make_reverse_iterator(cbegin());}

    [[nodiscard]] T* dataView() const {
        return data_view_;
    }

    [[nodiscard]] const std::array<std::size_t, N>& origStrides() const {
        return orig_strides_;
    }

    template <std::semiregular T_, std::size_t N_>
    friend Matrix<T_, N_> transpose(const Matrix<T_, N_>& orig, const std::array<std::size_t, N_>& perm);

    MatrixView& operator-() {
        Base::Base::operator-();
        return *this;
    }

};

template <std::semiregular T, std::size_t N>
MatrixView<T, N>::MatrixView(const std::array<std::size_t, N>& dims, T* data_view,
    const std::array<std::size_t, N>& orig_strides)
    : Base(dims), data_view_ {data_view}, orig_strides_ {orig_strides} {}

template <std::semiregular T, std::size_t N>
template <typename DerivedOther, std::semiregular U> requires std::is_convertible_v<U, T>
MatrixView<T, N>::MatrixView(const MatrixBase<DerivedOther, U, N>& other) : Base(other) {
    if constexpr (std::is_same_v<DerivedOther, MatrixView<U, N>>) {
        data_view_ = static_cast<T*>(other.dataView());
        orig_strides_ = other.origStrides();
    } else {
        data_view_ = const_cast<T*>(other.begin());
        orig_strides_ = other.strides();
    }
}

template <std::semiregular T, std::size_t N>
template <typename DerivedOther, std::semiregular U> requires std::is_convertible_v<U, T>
MatrixView<T, N>& MatrixView<T, N>::operator=(const MatrixBase<DerivedOther, U, N>& other) {
    for (auto it = begin(), it2 = other.begin(); it != end(); ++it, ++it2) {
        *it = static_cast<T>(*it2);
    }
    return *this;
}

template <std::semiregular T>
class MatrixView<T, 1> final : public MatrixBase<MatrixView<T, 1>, T, 1> {
private:
    T* data_view_;
    std::size_t orig_strides_;

public:
    using Base = MatrixBase<MatrixView<T, 1>, T, 1>;
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

    ~MatrixView() noexcept = default;

    explicit MatrixView(const std::array<std::size_t, 1>& dims,
                        T* data_view,
                        const std::array<std::size_t, 1>& orig_strides);

    template <typename DerivedOther, std::semiregular U> requires std::is_convertible_v<U, T>
    MatrixView(const MatrixBase<DerivedOther, U, 1>& other);

    template <typename DerivedOther, std::semiregular U> requires std::is_convertible_v<U, T>
    MatrixView& operator=(const MatrixBase<DerivedOther, U, 1>& other);

    friend void swap(MatrixView& a, MatrixView& b) noexcept {
        swap(static_cast<Base&>(a), static_cast<Base&>(b));
        std::swap(a.data_view_, b.data_view_);
        std::swap(a.orig_strides_, b.orig_strides_);
    }

    template <typename T_>
    struct MVIterator {
        using difference_type = std::ptrdiff_t;
        using value_type = T_;
        using pointer = T_*;
        using reference = T_&;
        using iterator_category = std::random_access_iterator_tag;
        using MatViewType = std::conditional_t<std::is_const_v<T_>, const MatrixView*, MatrixView*>;

        MatViewType ptr_ = nullptr;
        std::size_t pos_ = 0;
        std::size_t offset_ = 0;
        std::size_t index_ = 0;

        MVIterator(MatViewType ptr, std::size_t pos = 0) : ptr_ {ptr}, pos_ {pos} {
            ValidateOffset();
        }

        reference operator*() const {
            return ptr_->data_view_[offset_];
        }

        pointer operator->() const {
            return ptr_->data_view_ + offset_;
        }

        void ValidateOffset() {
            offset_ = pos_ * ptr_->orig_strides_;
            index_ = pos_ * ptr_->strides();
            assert(index_ <= ptr_->dims(0));
        }

        void Increment() {
            ++pos_;
            ValidateOffset();
        }

        void Increment(std::ptrdiff_t n) {
            if (n < 0) {
                Decrement(-n);
                return;
            }
            pos_ += n;
            ValidateOffset();
        }

        void Decrement() {
            --pos_;
            ValidateOffset();
        }

        void Decrement(std::ptrdiff_t n) {
            if (n < 0) {
                Increment(-n);
                return;
            }
            pos_ -= n;
            ValidateOffset();
        }

        MVIterator& operator++() {
            Increment();
            return *this;
        }

        MVIterator operator++(int) {
            MVIterator temp = *this;
            Increment();
            return temp;
        }

        MVIterator& operator--() {
            Decrement();
            return *this;
        }

        MVIterator operator--(int) {
            MVIterator temp = *this;
            Decrement();
            return temp;
        }

        MVIterator operator+(difference_type n) const {
            MVIterator temp = *this;
            temp.Increment(n);
            return temp;
        }

        MVIterator& operator+=(difference_type n) {
            Increment(n);
            return *this;
        }

        MVIterator operator-(difference_type n) const {
            MVIterator temp = *this;
            temp.Decrement(n);
            return temp;
        }

        MVIterator& operator-=(difference_type n) {
            Decrement(n);
            return *this;
        }

        reference operator[](difference_type n) const {
            return *(this + n);
        }

        template <typename T2> requires std::is_same_v<std::remove_cv_t<T_>, std::remove_cv_t<T2>>
        difference_type operator-(const MVIterator<T2>& other) const {
            return offset_ - other.offset_;
        }

    };

    // oh no.. *why* defining operator<=> doesn't work to automatically define these in gcc?
    template <typename T1, typename T2> requires std::is_same_v<std::remove_cv_t<T1>, std::remove_cv_t<T2>>
    friend bool operator==(const MVIterator<T1>& it1, const MVIterator<T2>& it2) {
        return it1.pos_ == it2.pos_;
    }

    template <typename T1, typename T2> requires std::is_same_v<std::remove_cv_t<T1>, std::remove_cv_t<T2>>
    friend bool operator!=(const MVIterator<T1>& it1, const MVIterator<T2>& it2) {
        return !(it1 == it2);
    }

    template <typename T1, typename T2> requires std::is_same_v<std::remove_cv_t<T1>, std::remove_cv_t<T2>>
    friend auto operator<=>(const MVIterator<T1>& it1, const MVIterator<T2>& it2) {
        return it1.pos_ <=> it2.pos_;
    }

    using iterator = MVIterator<T>;
    using const_iterator = MVIterator<const T>;
    using reverse_iterator = std::reverse_iterator<iterator>;
    using const_reverse_iterator = std::reverse_iterator<const_iterator>;

    iterator begin() { return iterator(this);}
    const_iterator begin() const { return const_iterator(this); }
    const_iterator cbegin() const { return const_iterator(this); }
    iterator end() { return iterator(this, {dims(0), });}
    const_iterator end() const { return const_iterator(this, {dims(0), });}
    const_iterator cend() const { return const_iterator(this, {dims(0), });}
    reverse_iterator rbegin() { return std::make_reverse_iterator(end());}
    const_reverse_iterator rbegin() const { return std::make_reverse_iterator(cend());}
    const_reverse_iterator crbegin() const { return std::make_reverse_iterator(cend());}
    reverse_iterator rend() { return std::make_reverse_iterator(begin());}
    const_reverse_iterator rend() const { return std::make_reverse_iterator(cbegin());}
    const_reverse_iterator crend() const { return std::make_reverse_iterator(cbegin());}

    [[nodiscard]] T* dataView() const {
        return data_view_;
    }

    [[nodiscard]] std::size_t origStrides() const {
        return orig_strides_;
    }

    template <std::semiregular T_, std::size_t N_>
    friend Matrix<T_, N_> transpose(const Matrix<T_, N_>& orig, const std::array<std::size_t, N_>& perm);

    MatrixView& operator-() {
        Base::Base::operator-();
        return *this;
    }

};

template <std::semiregular T>
MatrixView<T, 1>::MatrixView(const std::array<std::size_t, 1>& dims, T* data_view,
                             const std::array<std::size_t, 1>& orig_strides)
        : Base(dims[0]), data_view_ {data_view}, orig_strides_ {orig_strides[0]} {}

template <std::semiregular T>
template <typename DerivedOther, std::semiregular U> requires std::is_convertible_v<U, T>
MatrixView<T, 1>::MatrixView(const MatrixBase<DerivedOther, U, 1>& other) : Base(other.dims(0)) {
    data_view_ = const_cast<T*>(other.dataView());
    if constexpr (std::is_same_v<DerivedOther, MatrixView<U, 1>>) {
        orig_strides_ = other.origStrides();
    } else {
        orig_strides_ = other.strides();
    }
}

template <std::semiregular T>
template <typename DerivedOther, std::semiregular U> requires std::is_convertible_v<U, T>
MatrixView<T, 1>& MatrixView<T, 1>::operator=(const MatrixBase<DerivedOther, U, 1>& other) {
    for (auto it = begin(), it2 = other.begin(); it != end(); ++it, ++it2) {
        *it = static_cast<T>(*it2);
    }
    return *this;
}

} // namespace frozenca

#endif //FROZENCA_MATRIXVIEW_H
