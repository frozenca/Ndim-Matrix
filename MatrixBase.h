#ifndef FROZENCA_MATRIXBASE_H
#define FROZENCA_MATRIXBASE_H

#include <execution>
#include <numeric>
#include "ObjectBase.h"
#include "MatrixInitializer.h"

namespace frozenca {

template <std::semiregular T, std::size_t N>
class MatrixView;

template <typename Derived, std::semiregular T, std::size_t N>
class MatrixBase : public ObjectBase<MatrixBase<Derived, T, N>> {
    static_assert(N > 1);
public:
    static constexpr std::size_t ndim = N;

private:
    std::array<std::size_t, N> dims_;
    std::size_t size_;
    std::array<std::size_t, N> strides_;

public:
    using Base = ObjectBase<MatrixBase<Derived, T, N>>;
    using Base::applyFunction;
    using Base::operator=;
    using Base::operator+=;
    using Base::operator-=;
    using Base::operator*=;
    using Base::operator/=;
    using Base::operator%=;

    Derived& self() { return static_cast<Derived&>(*this); }
    const Derived& self() const { return static_cast<const Derived&>(*this); }

protected:
    ~MatrixBase() noexcept = default;
    MatrixBase(const std::array<std::size_t, N>& dims);

    template <std::size_t M> requires (M < N)
    MatrixBase(const std::array<std::size_t, M>& dims);

    template <IndexType... Dims>
    explicit MatrixBase(Dims... dims);

    MatrixBase (const MatrixBase&) = default;
    MatrixBase& operator= (const MatrixBase&) = default;
    MatrixBase (MatrixBase&&) noexcept = default;
    MatrixBase& operator= (MatrixBase&&) noexcept = default;

    template <typename DerivedOther, std::semiregular U> requires std::is_convertible_v<U, T>
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
    auto begin() const { return self().begin(); }
    auto cbegin() const { return self().cbegin(); }
    auto end() { return self().end(); }
    auto end() const { return self().end(); }
    auto cend() const { return self().cend(); }
    auto rbegin() { return self().rbegin(); }
    auto rbegin() const { return self().rbegin(); }
    auto crbegin() const { return self().crbegin(); }
    auto rend() { return self().rend(); }
    auto rend() const { return self().rend(); }
    auto crend() const { return self().crend(); }

    template <IndexType... Args>
    reference operator()(Args... args) {
        return const_cast<reference>(std::as_const(*this).operator()(args...));
    }

    template <IndexType... Args>
    const_reference operator()(Args... args) const {
        static_assert(sizeof...(args) == N);
        std::array<std::size_t, N> pos {std::size_t(args)...};
        return operator[](pos);
    }

    reference operator[](const std::array<std::size_t, N>& pos) {
        return const_cast<reference>(std::as_const(*this).operator[](pos));
    }

    const_reference operator[](const std::array<std::size_t, N>& pos) const {
        if (!std::equal(std::cbegin(pos), std::cend(pos), std::cbegin(dims_), std::less<>{})) {
            throw std::out_of_range("Out of range in element access");
        }
        return *(cbegin() + std::transform_reduce(std::execution::par_unseq,
                                                  std::cbegin(pos), std::cend(pos), std::cbegin(strides_), 0lu));
    }

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
    MatrixView<T, N> submatrix(const std::array<std::size_t, N>& pos_begin,
                               const std::array<std::size_t, N>& pos_end);
    MatrixView<T, N - 1> row(std::size_t n);
    MatrixView<T, N - 1> col(std::size_t n);
    MatrixView<T, N - 1> operator[](std::size_t n) { return row(n); }

    MatrixView<T, N> submatrix(const std::array<std::size_t, N>& pos_begin) const;
    MatrixView<T, N> submatrix(const std::array<std::size_t, N>& pos_begin,
                               const std::array<std::size_t, N>& pos_end) const;
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

    template <typename DerivedOther1, typename DerivedOther2,
            std::semiregular U, std::semiregular V,
            std::size_t N1, std::size_t N2,
            std::invocable<MatrixView<T, N - 1>&,
                    const MatrixView<U, std::min(N1, N - 1)>&,
                    const MatrixView<V, std::min(N2, N - 1)>&> F>
    requires (std::max(N1, N2) == N)
    MatrixBase& applyFunctionWithBroadcast(const MatrixBase<DerivedOther1, U, N1>& m1,
                                           const MatrixBase<DerivedOther2, V, N2>& m2,
                                           F&& f);

    template <typename DerivedOther, std::semiregular U> requires Addable<T, U>
    MatrixBase& operator+=(const MatrixBase<DerivedOther, U, N>& other) {
        for (auto it = begin(), it2 = other.begin(); it != end(); ++it, ++it2) {
            *it += static_cast<T>(*it2);
        }
        return *this;
    }

    template <typename DerivedOther, std::semiregular U> requires Subtractable<T, U>
    MatrixBase& operator-=(const MatrixBase<DerivedOther, U, N>& other) {
        for (auto it = begin(), it2 = other.begin(); it != end(); ++it, ++it2) {
            *it -= static_cast<T>(*it2);
        }
        return *this;
    }

};

template <typename Derived, std::semiregular T, std::size_t N>
MatrixBase<Derived, T, N>::MatrixBase(const std::array<std::size_t, N>& dims) : dims_ {dims} {
    if (std::ranges::find(dims_, 0lu) != std::end(dims_)) {
        throw std::invalid_argument("Zero dimension not allowed");
    }
    size_ = std::reduce(std::execution::par_unseq, std::begin(dims_), std::end(dims_), 1lu, std::multiplies<>{});
    strides_ = computeStrides(dims_);
}

template <typename Derived, std::semiregular T, std::size_t N>
template <std::size_t M> requires (M < N)
MatrixBase<Derived, T, N>::MatrixBase(const std::array<std::size_t, M>& dims) : MatrixBase (prepend<N, M>(dims)) {}

template <typename Derived, std::semiregular T, std::size_t N>
template <IndexType... Dims>
MatrixBase<Derived, T, N>::MatrixBase(Dims... dims) : dims_{static_cast<std::size_t>(dims)...} {
    static_assert(sizeof...(Dims) == N);
    static_assert((std::is_integral_v<Dims> && ...));
    if (std::ranges::find(dims_, 0lu) != std::end(dims_)) {
        throw std::invalid_argument("Zero dimension not allowed");
    }
    size_ = std::reduce(std::execution::par_unseq, std::begin(dims_), std::end(dims_), 1lu, std::multiplies<>{});
    strides_ = computeStrides(dims_);
}

template <typename Derived, std::semiregular T, std::size_t N>
template <typename DerivedOther, std::semiregular U> requires std::is_convertible_v<U, T>
MatrixBase<Derived, T, N>::MatrixBase(const MatrixBase<DerivedOther, U, N>& other) : MatrixBase(other.dims()) {}

template <typename Derived, std::semiregular T, std::size_t N>
MatrixBase<Derived, T, N>::MatrixBase(typename MatrixInitializer<T, N>::type init) :
MatrixBase(deriveDims<N>(init)) {}

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
    std::ranges::transform(pos_end, pos_begin, std::begin(view_dims), std::minus<>{});
    MatrixView<T, N> view(view_dims, const_cast<T*>(&operator[](pos_begin)), strides());
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
    std::ranges::copy(orig_dims | std::views::drop(1), std::begin(row_dims));
    std::array<std::size_t, N> pos_begin = {n, };
    std::array<std::size_t, N - 1> row_strides;

    std::array<std::size_t, N> orig_strides;
    if constexpr (std::is_same_v<Derived, MatrixView<T, N>>) {
        orig_strides = origStrides();
    } else {
        orig_strides = strides();
    }

    std::ranges::copy(orig_strides | std::views::drop(1), std::begin(row_strides));
    MatrixView<T, N - 1> nth_row(row_dims, const_cast<T*>(&operator[](pos_begin)), row_strides);
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
    std::ranges::copy(orig_dims | std::views::take(N - 1), std::begin(col_dims));
    std::array<std::size_t, N> pos_begin = {0};
    pos_begin[N - 1] = n;
    std::array<std::size_t, N - 1> col_strides;

    std::array<std::size_t, N> orig_strides;
    if constexpr (std::is_same_v<Derived, MatrixView<T, N>>) {
        orig_strides = origStrides();
    } else {
        orig_strides = strides();
    }

    std::ranges::copy(orig_strides | std::views::take(N - 1), std::begin(col_strides));
    MatrixView<T, N - 1> nth_col(col_dims, const_cast<T*>(&operator[](pos_begin)), col_strides);
    return nth_col;
}

template <typename Derived, std::semiregular T, std::size_t N>
template <typename DerivedOther1, typename DerivedOther2,
        std::semiregular U, std::semiregular V,
        std::size_t N1, std::size_t N2,
        std::invocable<MatrixView<T, N - 1>&,
                const MatrixView<U, std::min(N1, N - 1)>&,
                const MatrixView<V, std::min(N2, N - 1)>&> F>
requires (std::max(N1, N2) == N)
MatrixBase<Derived, T, N>&
        MatrixBase<Derived, T, N>::applyFunctionWithBroadcast(const MatrixBase<DerivedOther1, U, N1>& m1,
                                                              const MatrixBase<DerivedOther2, V, N2>& m2,
                                                              F&& f) {
    if constexpr (N1 == N) {
        if constexpr (N2 == N) {
            auto r = dims(0);
            auto r1 = m1.dims(0);
            auto r2 = m2.dims(0);
            if (r1 == r) {
                if (r2 == r) {
                    for (std::size_t i = 0; i < r; ++i) {
                        auto row = this->row(i);
                        f(row, m1.row(i), m2.row(i));
                    }
                } else { // r2 < r == r1
                    auto row2 = m2.row(0);
                    for (std::size_t i = 0; i < r; ++i) {
                        auto row = this->row(i);
                        f(row, m1.row(i), row2);
                    }
                }
            } else if (r2 == r) { // r1 < r == r2
                auto row1 = m1.row(0);
                for (std::size_t i = 0; i < r; ++i) {
                    auto row = this->row(i);
                    f(row, row1, m2.row(i));
                }
            } else {
                assert(0); // cannot happen
            }
        } else { // N2 < N == N1
            auto r = dims(0);
            assert(r == m1.dims(0));
            MatrixView<V, N2> view2 (m2);
            for (std::size_t i = 0; i < r; ++i) {
                auto row = this->row(i);
                f(row, m1.row(i), view2);
            }
        }
    } else if constexpr (N2 == N) { // N1 < N == N2
        auto r = dims(0);
        assert(r == m2.dims(0));
        MatrixView<U, N1> view1 (m1);
        for (std::size_t i = 0; i < r; ++i) {
            auto row = this->row(i);
            f(row, view1, m2.row(i));
        }
    } else {
        assert(0); // cannot happen
    }
    return *this;
}

template <typename Derived, std::semiregular T>
class MatrixBase<Derived, T, 1> : public ObjectBase<MatrixBase<Derived, T, 1>> {
public:
    static constexpr std::size_t ndim = 1;

private:
    std::size_t dims_;
    std::size_t strides_;

    Derived& self() { return static_cast<Derived&>(*this); }
    const Derived& self() const { return static_cast<const Derived&>(*this); }

public:
    MatrixBase() = delete;
    using Base = ObjectBase<MatrixBase<Derived, T, 1>>;
    using Base::applyFunction;
    using Base::operator=;
    using Base::operator+=;
    using Base::operator-=;
    using Base::operator*=;
    using Base::operator/=;
    using Base::operator%=;
    using Base::operator-;

protected:
    ~MatrixBase() noexcept = default;
    template <typename Dim> requires std::is_integral_v<Dim>
    explicit MatrixBase(Dim dim) : dims_(dim), strides_(1) {};

    MatrixBase (const MatrixBase&) = default;
    MatrixBase& operator= (const MatrixBase&) = default;
    MatrixBase (MatrixBase&&) noexcept = default;
    MatrixBase& operator= (MatrixBase&&) noexcept = default;

    template <typename DerivedOther, std::semiregular U> requires std::is_convertible_v<U, T>
    MatrixBase(const MatrixBase<DerivedOther, U, 1>&);

    MatrixBase(typename MatrixInitializer<T, 1>::type init);

public:
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
    auto begin() const { return self().begin(); }
    auto cbegin() const { return self().cbegin(); }
    auto end() { return self().end(); }
    auto end() const { return self().end(); }
    auto cend() const { return self().cend(); }
    auto rbegin() { return self().rbegin(); }
    auto rbegin() const { return self().rbegin(); }
    auto crbegin() const { return self().crbegin(); }
    auto rend() { return self().rend(); }
    auto rend() const { return self().rend(); }
    auto crend() const { return self().crend(); }

    template <typename Dim> requires std::is_integral_v<Dim>
    reference operator()(Dim dim) {
        return operator[](dim);
    }

    template <typename Dim> requires std::is_integral_v<Dim>
    const_reference operator()(Dim dim) const {
        return operator[](dim);
    }

    [[nodiscard]] std::array<std::size_t, 1> dims() const {
        return {dims_};
    }

    [[nodiscard]] std::size_t dims(std::size_t n) const {
        if (n >= 1) {
            throw std::out_of_range("Out of range in dims");
        }
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

    MatrixView<T, 1> submatrix(std::size_t pos_begin);
    MatrixView<T, 1> submatrix(std::size_t pos_begin, std::size_t pos_end);
    T& row(std::size_t n);
    T& col(std::size_t n);
    T& operator[](std::size_t n) { return *(begin() + n); }

    MatrixView<T, 1> submatrix(std::size_t pos_begin) const;
    MatrixView<T, 1> submatrix(std::size_t pos_begin, std::size_t pos_end) const;
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

    template <typename DerivedOther1, typename DerivedOther2,
            std::semiregular U, std::semiregular V,
            std::invocable<T&, const U&, const V&> F>
    MatrixBase& applyFunctionWithBroadcast(const frozenca::MatrixBase<DerivedOther1, U, 1>& m1,
                                           const frozenca::MatrixBase<DerivedOther2, V, 1>& m2,
                                           F&& f);

    template <typename DerivedOther, std::semiregular U> requires Addable<T, U>
    MatrixBase& operator+=(const MatrixBase<DerivedOther, U, 1>& other) {
        for (auto it = begin(), it2 = other.begin(); it != end(); ++it, ++it2) {
            *it += static_cast<T>(*it2);
        }
        return *this;
    }

    template <typename DerivedOther, std::semiregular U> requires Subtractable<T, U>
    MatrixBase& operator-=(const MatrixBase<DerivedOther, U, 1>& other) {
        for (auto it = begin(), it2 = other.begin(); it != end(); ++it, ++it2) {
            *it -= static_cast<T>(*it2);
        }
        return *this;
    }

};

template <typename Derived, std::semiregular T>
template <typename DerivedOther, std::semiregular U> requires std::is_convertible_v<U, T>
MatrixBase<Derived, T, 1>::MatrixBase(const MatrixBase<DerivedOther, U, 1>& other) : MatrixBase(other.dims(0)) {}

template <typename Derived, std::semiregular T>
MatrixBase<Derived, T, 1>::MatrixBase(typename MatrixInitializer<T, 1>::type init)
: MatrixBase(deriveDims<1>(init)[0]) {}

template <typename Derived, std::semiregular T>
MatrixView<T, 1> MatrixBase<Derived, T, 1>::submatrix(std::size_t pos_begin) {
    return submatrix(pos_begin, dims_);
}

template <typename Derived, std::semiregular T>
MatrixView<T, 1> MatrixBase<Derived, T, 1>::submatrix(std::size_t pos_begin) const {
    return submatrix(pos_begin, dims_);
}

template <typename Derived, std::semiregular T>
MatrixView<T, 1> MatrixBase<Derived, T, 1>::submatrix(std::size_t pos_begin,
                                        std::size_t pos_end) {
    return std::as_const(*this).submatrix(pos_begin, pos_end);
}

template <typename Derived, std::semiregular T>
MatrixView<T, 1> MatrixBase<Derived, T, 1>::submatrix(std::size_t pos_begin,
                                              std::size_t pos_end) const {
    if (pos_begin >= pos_end) {
        throw std::out_of_range("submatrix begin/end position error");
    }
    MatrixView<T, 1> view ({pos_end - pos_begin}, const_cast<T*>(&operator[](pos_begin)), {strides_});
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
    const T& val = operator[](n);
    return val;
}

template <typename Derived, std::semiregular T>
T& MatrixBase<Derived, T, 1>::col(std::size_t n) {
    return row(n);
}

template <typename Derived, std::semiregular T>
const T& MatrixBase<Derived, T, 1>::col(std::size_t n) const {
    return row(n);
}

template <typename Derived, std::semiregular T>
template <typename DerivedOther1, typename DerivedOther2,
        std::semiregular U, std::semiregular V,
        std::invocable<T&, const U&, const V&> F>
MatrixBase<Derived, T, 1>& MatrixBase<Derived, T, 1>::applyFunctionWithBroadcast(
        const frozenca::MatrixBase<DerivedOther1, U, 1>& m1,
        const frozenca::MatrixBase<DerivedOther2, V, 1>& m2,
        F&& f) {
    // real update is done here by passing lvalue reference T&
    auto r = dims(0);
    auto r1 = m1.dims(0);
    auto r2 = m2.dims(0);

    if (r1 == r) {
        if (r2 == r) {
            for (std::size_t i = 0; i < r; ++i) {
                f(this->row(i), m1.row(i), m2.row(i));
            }
        } else { // r2 < r == r1
            auto row2 = m2.row(0);
            for (std::size_t i = 0; i < r; ++i) {
                f(this->row(i), m1.row(i), row2);
            }
        }
    } else if (r2 == r) { // r1 < r == r2
        auto row1 = m1.row(0);
        for (std::size_t i = 0; i < r; ++i) {
            f(this->row(i), row1, m2.row(i));
        }
    }
    return *this;
}

} // namespace frozenca

#endif //FROZENCA_MATRIXBASE_H
