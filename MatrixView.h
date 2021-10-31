#ifndef FROZENCA_MATRIXVIEW_H
#define FROZENCA_MATRIXVIEW_H

#include <compare>
#include "MatrixBase.h"

namespace frozenca {

template <std::semiregular T, std::size_t N, bool Const>
class MatrixView final : public MatrixBase<MatrixView<T, N, Const>, T, N> {
public:
    using Base = MatrixBase<MatrixView<T, N, Const>, T, N>;
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
    using reference = std::conditional_t<Const, const T&, T&>;
    using const_reference = const T&;
    using pointer = std::conditional_t<Const, const T*, T*>;
    using stride_type = std::array<std::size_t, N>;

private:
    pointer data_view_;
    stride_type orig_strides_;

public:
    ~MatrixView() noexcept = default;

    explicit MatrixView(const stride_type& dims, pointer data_view, const stride_type& orig_strides);

    template <typename DerivedOther>
    MatrixView(const MatrixBase <DerivedOther, T, N>& other);

    template <typename DerivedOther, std::semiregular U>
    requires std::is_convertible_v<U, T>
    MatrixView& operator=(const MatrixBase <DerivedOther, U, N>& other) requires (!Const);

    friend void swap(MatrixView& a, MatrixView& b) noexcept {
        swap(static_cast<Base&>(a), static_cast<Base&>(b));
        std::swap(a.data_view_, b.data_view_);
        std::swap(a.orig_strides_, b.orig_strides_);
    }


    template <bool IterConst>
    struct MVIterator {
        using difference_type = std::ptrdiff_t;
        using value_type = T;
        using pointer = std::conditional_t<(Const || IterConst), const T*, T*>;
        using reference = std::conditional_t<(Const || IterConst), const T&, T&>;
        using iterator_category = std::random_access_iterator_tag;
        using MatViewType = std::conditional_t<(Const || IterConst), const MatrixView*, MatrixView*>;

        MatViewType ptr_ = nullptr;
        std::array<std::size_t, N> pos_ = {0};
        std::size_t offset_ = 0;
        std::size_t index_ = 0;

        MVIterator() = default;

        MVIterator(MatViewType ptr, std::array<std::size_t, N> pos = {0}) : ptr_{ptr}, pos_{pos} {
            ValidateOffset();
        }

        reference operator*() const {
            return ptr_->data_view_[offset_];
        }

        pointer operator->() const {
            return ptr_->data_view_ + offset_;
        }

        void ValidateOffset() {
            offset_ = std::inner_product(std::cbegin(pos_), std::cend(pos_), std::cbegin(ptr_->origStrides()), 0lu);
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
            return *(*this + n);
        }

        template <bool IterConstOther>
        difference_type operator-(const MVIterator<IterConstOther>& other) const {
            return index_ - other.index_;
        }

    };

    template <bool IterConst>
    friend bool operator==(const MVIterator<IterConst>& it1, const MVIterator<IterConst>& it2) {
        return it1.ptr_->data_view_ + it1.index_ == it2.ptr_->data_view_ + it2.index_;
    }

    template <bool IterConst>
    friend bool operator!=(const MVIterator<IterConst>& it1, const MVIterator<IterConst>& it2) {
        return !(it1 == it2);
    }

    template <bool IterConst1, bool IterConst2>
    friend auto operator<=>(const MVIterator<IterConst1>& it1, const MVIterator<IterConst2>& it2) {
        return it1.ptr_->data_view_ + it1.index_ <=> it2.ptr_->data_view_ + it2.index_;
    }

    using iterator = MVIterator<Const>;
    using const_iterator = MVIterator<true>;
    using reverse_iterator = std::reverse_iterator<iterator>;
    using const_reverse_iterator = std::reverse_iterator<const_iterator>;

    iterator begin() { return iterator(this); }

    const_iterator begin() const { return const_iterator(this); }

    const_iterator cbegin() const { return const_iterator(this); }

    iterator end() { return iterator(this, {dims(0),}); }

    const_iterator end() const { return const_iterator(this, {dims(0),}); }

    const_iterator cend() const { return const_iterator(this, {dims(0),}); }

    reverse_iterator rbegin() { return std::make_reverse_iterator(end()); }

    const_reverse_iterator rbegin() const { return std::make_reverse_iterator(cend()); }

    const_reverse_iterator crbegin() const { return std::make_reverse_iterator(cend()); }

    reverse_iterator rend() { return std::make_reverse_iterator(begin()); }

    const_reverse_iterator rend() const { return std::make_reverse_iterator(cbegin()); }

    const_reverse_iterator crend() const { return std::make_reverse_iterator(cbegin()); }

    [[nodiscard]] pointer dataView() const {
        return data_view_;
    }

    [[nodiscard]] const stride_type& origStrides() const {
        return orig_strides_;
    }

    MatrixView& operator-() {
        Base::Base::operator-();
        return *this;
    }
};

template <std::semiregular T, std::size_t N, bool Const>
MatrixView<T, N, Const>::MatrixView(const stride_type& dims, pointer data_view, const stride_type& orig_strides)
    : Base(dims), data_view_ {data_view}, orig_strides_ {orig_strides} {}

template <std::semiregular T, std::size_t N, bool Const>
template <typename DerivedOther>
MatrixView<T, N, Const>::MatrixView(const MatrixBase<DerivedOther, T, N>& other) : Base(other) {
    if constexpr (std::is_same_v<DerivedOther, MatrixView<T, N, true>> ||
    std::is_same_v<DerivedOther, MatrixView<T, N, false>>) {
        data_view_ = const_cast<T*>(other.dataView());
        orig_strides_ = other.origStrides();
    } else {
        data_view_ = const_cast<T*>(other.begin());
        orig_strides_ = other.strides();
    }
}

template <std::semiregular T, std::size_t N, bool Const>
template <typename DerivedOther, std::semiregular U> requires std::is_convertible_v<U, T>
MatrixView<T, N, Const>& MatrixView<T, N, Const>::operator=(const MatrixBase<DerivedOther, U, N>& other) requires (!Const) {
    std::size_t len1 = end() - begin();
    std::size_t len2 = std::distance(std::begin(other), std::end(other));
    std::copy(std::begin(other), std::begin(other) + std::min(len1, len2), begin());
    return *this;
}

} // namespace frozenca

#endif //FROZENCA_MATRIXVIEW_H
