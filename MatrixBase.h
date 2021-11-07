#ifndef FROZENCA_MATRIXBASE_H
#define FROZENCA_MATRIXBASE_H

#include <numeric>
#include "ObjectBase.h"
#include "MatrixInitializer.h"

namespace frozenca {

template <std::semiregular T, std::size_t N, bool Const, bool isRowMajor>
class MatrixView;

template <typename Derived, std::semiregular T, std::size_t N, bool isRowMajor>
class MatrixBase : public ObjectBase<MatrixBase<Derived, T, N, isRowMajor>> {
    static_assert(N > 1);
    static_assert(isRowMajor || N == 2);
public:
    static constexpr std::size_t ndim = N;
    using extent_type = std::array<std::size_t, N>;
    using value_type = T;
    using reference = T&;
    using const_reference = const T&;
    using pointer = T*;
    using view_type = MatrixView<T, N, false, isRowMajor>;
    using const_view_type = MatrixView<T, N, true, isRowMajor>;
    using row_type = MatrixView<T, N - 1, false, (N == 2) || isRowMajor>;
    using const_row_type = MatrixView<T, N - 1, true, (N == 2) ||isRowMajor>;

private:
    extent_type dims_;
    std::size_t size_;
    extent_type strides_;

public:
    using Base = ObjectBase<MatrixBase<Derived, T, N, isRowMajor>>;
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
    MatrixBase(const extent_type& dims);

    template <std::size_t M> requires (M < N)
    MatrixBase(const std::array<std::size_t, M>& dims);

    template <IndexType... Dims>
    explicit MatrixBase(Dims... dims);

    MatrixBase (const MatrixBase&) = default;
    MatrixBase& operator= (const MatrixBase&) = default;
    MatrixBase (MatrixBase&&) noexcept = default;
    MatrixBase& operator= (MatrixBase&&) noexcept = default;

    template <typename DerivedOther, std::semiregular U, bool isRowMajorOther> requires std::is_convertible_v<U, T>
    MatrixBase(const MatrixBase<DerivedOther, U, N, isRowMajorOther>&);

    MatrixBase(typename MatrixInitializer<T, N>::type init);

public:
    template <typename U>
    MatrixBase(std::initializer_list<U>) = delete;

    template <typename U>
    MatrixBase& operator=(std::initializer_list<U>) = delete;

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

    [[nodiscard]] std::size_t size() const { return size_;}
    [[nodiscard]] const extent_type& dims() const { return dims_; }
    [[nodiscard]] std::size_t dims(std::size_t n) const {
        if (n >= N) {
            throw std::out_of_range("Out of range in dims");
        }
        return dims_[n];
    }

    [[nodiscard]] const extent_type& strides() const { return strides_; }
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

    template <IndexType... Args>
    reference operator()(Args... args) {
        return const_cast<reference>(std::as_const(*this).operator()(args...));
    }

    template <IndexType... Args>
    const_reference operator()(Args... args) const {
        static_assert(sizeof...(args) == N);
        extent_type pos {std::size_t(args)...};
        return operator[](pos);
    }

    reference operator[](const extent_type& pos) {
        return const_cast<reference>(std::as_const(*this).operator[](pos));
    }

    const_reference operator[](const extent_type& pos) const {
        return *(cbegin() + std::inner_product(std::cbegin(pos), std::cend(pos), std::cbegin(strides()), 0lu));
    }

    view_type submatrix(const extent_type& pos_begin);
    view_type submatrix(const extent_type& pos_begin, const extent_type& pos_end);
    row_type row(std::size_t n);
    row_type col(std::size_t n);
    row_type operator[](std::size_t n) { return row(n); }

    const_view_type submatrix(const extent_type& pos_begin) const;
    const_view_type submatrix(const extent_type& pos_begin, const extent_type& pos_end) const;
    const_row_type row(std::size_t n) const;
    const_row_type col(std::size_t n) const;
    const_row_type operator[](std::size_t n) const { return row(n); }

    friend std::ostream& operator<<(std::ostream& os, const MatrixBase& m) {
        os << '{';
        for (std::size_t i = 0; i != m.dims_[0]; ++i) {
            os << m[i];
            if (i + 1 != m.dims_[0]) {
                os << ", ";
            }
        }
        return os << '}';
    }

    template <typename DerivedOther1, typename DerivedOther2,
            std::semiregular U, std::semiregular V,
            std::size_t N1, std::size_t N2,
            std::invocable<row_type&,
                    typename MatrixBase<DerivedOther1, U, N1, isRowMajor>::row_type&,
                    typename MatrixBase<DerivedOther2, V, N2, isRowMajor>::row_type&> F>
    requires (std::max(N1, N2) == N)
    MatrixBase& applyFunctionWithBroadcast(const MatrixBase<DerivedOther1, U, N1, isRowMajor>& m1,
                                           const MatrixBase<DerivedOther2, V, N2, isRowMajor>& m2,
                                           F&& f);

    template <typename DerivedOther> requires Addable<T, typename DerivedOther::value_type>
    MatrixBase& operator+=(const ObjectBase<DerivedOther>& other) {
        std::transform(begin(), end(), other.begin(), begin(), std::plus<>{});
        return *this;
    }

    template <typename DerivedOther> requires Subtractable<T, typename DerivedOther::value_type>
    MatrixBase& operator-=(const ObjectBase<DerivedOther>& other) {
        std::transform(begin(), end(), other.begin(), begin(), std::minus<>{});
        return *this;
    }

    MatrixBase& swapRows(std::size_t i, std::size_t j) {
        if (i >= dims(0) || j >= dims(0)) {
            throw std::out_of_range("Out of range in swapRows");
        }
        std::swap_ranges(row(i).begin(), row(i).end(), row(j).begin());
        return *this;
    }

    MatrixBase& swapCols(std::size_t i, std::size_t j) {
        if (i >= dims(N - 1) || j >= dims(N - 1)) {
            throw std::out_of_range("Out of range in swapCols");
        }
        std::swap_ranges(col(i).begin(), col(i).end(), col(j).begin());
        return *this;
    }

};

template <typename Derived, std::semiregular T, std::size_t N, bool isRowMajor>
MatrixBase<Derived, T, N, isRowMajor>::MatrixBase(const extent_type& dims) : dims_ {dims} {
    if (std::ranges::find(dims_, 0lu) != std::end(dims_)) {
        throw std::invalid_argument("Zero dimension not allowed");
    }
    size_ = std::accumulate(std::begin(dims_), std::end(dims_), 1lu, std::multiplies<>{});
    strides_ = computeStrides<N, isRowMajor>(dims_);
}

template <typename Derived, std::semiregular T, std::size_t N, bool isRowMajor>
template <std::size_t M> requires (M < N)
MatrixBase<Derived, T, N, isRowMajor>::MatrixBase(const std::array<std::size_t, M>& dims) : MatrixBase (prepend<N, M>(dims)) {}

template <typename Derived, std::semiregular T, std::size_t N, bool isRowMajor>
template <IndexType... Dims>
MatrixBase<Derived, T, N, isRowMajor>::MatrixBase(Dims... dims) : dims_{static_cast<std::size_t>(dims)...} {
    static_assert(sizeof...(Dims) == N);
    static_assert((std::is_integral_v<Dims> && ...));
    if (std::ranges::find(dims_, 0lu) != std::end(dims_)) {
        throw std::invalid_argument("Zero dimension not allowed");
    }
    size_ = std::accumulate(std::begin(dims_), std::end(dims_), 1lu, std::multiplies<>{});
    strides_ = computeStrides<N, isRowMajor>(dims_);
}

template <typename Derived, std::semiregular T, std::size_t N, bool isRowMajor>
template <typename DerivedOther, std::semiregular U, bool isRowMajorOther> requires std::is_convertible_v<U, T>
MatrixBase<Derived, T, N, isRowMajor>::MatrixBase(const MatrixBase<DerivedOther, U, N, isRowMajorOther>& other)
: MatrixBase(other.dims()) {}

template <typename Derived, std::semiregular T, std::size_t N, bool isRowMajor>
MatrixBase<Derived, T, N, isRowMajor>::MatrixBase(typename MatrixInitializer<T, N>::type init) :
MatrixBase(deriveDims<N>(init)) {}

template <typename Derived, std::semiregular T, std::size_t N, bool isRowMajor>
typename MatrixBase<Derived, T, N, isRowMajor>::view_type
        MatrixBase<Derived, T, N, isRowMajor>::submatrix(const extent_type& pos_begin) {
    return submatrix(pos_begin, dims_);
}

template <typename Derived, std::semiregular T, std::size_t N, bool isRowMajor>
typename MatrixBase<Derived, T, N, isRowMajor>::const_view_type
        MatrixBase<Derived, T, N, isRowMajor>::submatrix(const extent_type& pos_begin) const {
    return submatrix(pos_begin, dims_);
}

template <typename Derived, std::semiregular T, std::size_t N, bool isRowMajor>
typename MatrixBase<Derived, T, N, isRowMajor>::view_type
        MatrixBase<Derived, T, N, isRowMajor>::submatrix(const extent_type& pos_begin, const extent_type& pos_end) {
    return std::as_const(*this).submatrix(pos_begin, pos_end);
}

template <typename Derived, std::semiregular T, std::size_t N, bool isRowMajor>
typename MatrixBase<Derived, T, N, isRowMajor>::const_view_type
        MatrixBase<Derived, T, N, isRowMajor>::submatrix(const extent_type& pos_begin,
                                             const extent_type& pos_end) const {
    if (!std::equal(std::cbegin(pos_begin), std::cend(pos_begin), std::cbegin(pos_end), std::less<>{})) {
        throw std::out_of_range("submatrix begin/end position error");
    }
    extent_type view_dims;
    std::ranges::transform(pos_end, pos_begin, std::begin(view_dims), std::minus<>{});
    auto realStrides = [&](){
        if constexpr (std::is_same_v<Derived, view_type> || std::is_same_v<Derived, const_view_type>) {
            return origStrides();
        } else {
            return strides();
        }
    };
    const_view_type view(view_dims, &operator[](pos_begin), realStrides());
    return view;
}

template <typename Derived, std::semiregular T, std::size_t N, bool isRowMajor>
typename MatrixBase<Derived, T, N, isRowMajor>::row_type MatrixBase<Derived, T, N, isRowMajor>::row(std::size_t n) {
    return std::as_const(*this).row(n);
}

template <typename Derived, std::semiregular T, std::size_t N, bool isRowMajor>
typename MatrixBase<Derived, T, N, isRowMajor>::const_row_type MatrixBase<Derived, T, N, isRowMajor>::row(std::size_t n) const {
    const auto& orig_dims = dims();
    if (n >= orig_dims[0]) {
        throw std::out_of_range("row index error");
    }
    std::array<std::size_t, N - 1> row_dims;
    std::ranges::copy(orig_dims | std::views::drop(1), std::begin(row_dims));
    extent_type pos_begin = {n, };
    std::array<std::size_t, N - 1> row_strides;

    extent_type orig_strides;
    if constexpr (std::is_same_v<Derived, MatrixView<T, N, true>> || std::is_same_v<Derived, MatrixView<T, N, false>>) {
        orig_strides = origStrides();
    } else {
        orig_strides = strides();
    }

    std::ranges::copy(orig_strides | std::views::drop(1), std::begin(row_strides));
    const_row_type nth_row(row_dims, &operator[](pos_begin), row_strides);
    return nth_row;
}

template <typename Derived, std::semiregular T, std::size_t N, bool isRowMajor>
typename MatrixBase<Derived, T, N, isRowMajor>::row_type MatrixBase<Derived, T, N, isRowMajor>::col(std::size_t n) {
    return std::as_const(*this).col(n);
}

template <typename Derived, std::semiregular T, std::size_t N, bool isRowMajor>
typename MatrixBase<Derived, T, N, isRowMajor>::const_row_type MatrixBase<Derived, T, N, isRowMajor>::col(std::size_t n) const {
    const auto& orig_dims = dims();
    if (n >= orig_dims[N - 1]) {
        throw std::out_of_range("col index error");
    }
    std::array<std::size_t, N - 1> col_dims;
    std::ranges::copy(orig_dims | std::views::take(N - 1), std::begin(col_dims));
    extent_type pos_begin = {0};
    pos_begin[N - 1] = n;
    std::array<std::size_t, N - 1> col_strides;

    extent_type orig_strides;
    if constexpr (std::is_same_v<Derived, MatrixView<T, N, false>> || std::is_same_v<Derived, MatrixView<T, N, true>>) {
        orig_strides = origStrides();
    } else {
        orig_strides = strides();
    }

    std::ranges::copy(orig_strides | std::views::take(N - 1), std::begin(col_strides));
    const_row_type nth_col(col_dims, &operator[](pos_begin), col_strides);
    return nth_col;
}

template <typename Derived, std::semiregular T, std::size_t N, bool isRowMajor>
template <typename DerivedOther1, typename DerivedOther2,
        std::semiregular U, std::semiregular V,
        std::size_t N1, std::size_t N2,
        std::invocable<typename MatrixBase<Derived, T, N, isRowMajor>::row_type&,
                typename MatrixBase<DerivedOther1, U, N1, isRowMajor>::row_type&,
                typename MatrixBase<DerivedOther2, V, N2, isRowMajor>::row_type&> F>
requires (std::max(N1, N2) == N)
MatrixBase<Derived, T, N, isRowMajor>&
        MatrixBase<Derived, T, N, isRowMajor>::applyFunctionWithBroadcast(const MatrixBase<DerivedOther1, U, N1, isRowMajor>& m1,
                                                                          const MatrixBase<DerivedOther2, V, N2, isRowMajor>& m2,
                                                                          F&& f) {
    if constexpr (isRowMajor) {
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
                MatrixView<V, N2, true, isRowMajor> view2(m2);
                for (std::size_t i = 0; i < r; ++i) {
                    auto row = this->row(i);
                    f(row, m1.row(i), view2);
                }
            }
        } else if constexpr (N2 == N) { // N1 < N == N2
            auto r = dims(0);
            assert(r == m2.dims(0));
            MatrixView<U, N1, true, isRowMajor> view1(m1);
            for (std::size_t i = 0; i < r; ++i) {
                auto row = this->row(i);
                f(row, view1, m2.row(i));
            }
        } else {
            assert(0); // cannot happen
        }
    } else {
        if constexpr (N1 == N) {
            if constexpr (N2 == N) {
                auto c = dims(N - 1);
                auto c1 = m1.dims(N - 1);
                auto c2 = m2.dims(N - 1);
                if (c1 == c) {
                    if (c2 == c) {
                        for (std::size_t i = 0; i < c; ++i) {
                            auto col = this->col(i);
                            f(col, m1.col(i), m2.col(i));
                        }
                    } else { // c2 < c == c1
                        auto col2 = m2.col(c2 - 1);
                        for (std::size_t i = 0; i < c; ++i) {
                            auto col = this->col(i);
                            f(col, m1.col(i), col2);
                        }
                    }
                } else if (c2 == c) { // r1 < r == r2
                    auto col1 = m1.col(c1 - 1);
                    for (std::size_t i = 0; i < c; ++i) {
                        auto col = this->col(i);
                        f(col, col1, m2.col(i));
                    }
                } else {
                    assert(0); // cannot happen
                }
            } else { // N2 < N == N1
                auto c = dims(N - 1);
                assert(c == m1.dims(N1 - 1));
                MatrixView<V, N2, true, isRowMajor> view2(m2);
                for (std::size_t i = 0; i < c; ++i) {
                    auto col = this->col(i);
                    f(col, m1.col(i), view2);
                }
            }
        } else if constexpr (N2 == N) { // N1 < N == N2
            auto c = dims(N - 1);
            assert(c == m2.dims(N2 - 1));
            MatrixView<U, N1, true, isRowMajor> view1(m1);
            for (std::size_t i = 0; i < c; ++i) {
                auto col = this->col(i);
                f(col, view1, m2.col(i));
            }
        } else {
            assert(0); // cannot happen
        }
    }
    return *this;
}

template <typename Derived, std::semiregular T, bool isRowMajor>
class MatrixBase<Derived, T, 1, isRowMajor> : public ObjectBase<MatrixBase<Derived, T, 1, isRowMajor>> {
public:
    static_assert(isRowMajor);
    static constexpr std::size_t ndim = 1;
    using extent_type = std::array<std::size_t, 1>;
    using value_type = T;
    using reference = T&;
    using const_reference = const T&;
    using pointer = T*;
    using view_type = MatrixView<T, 1, false>;
    using const_view_type = MatrixView<T, 1, true>;
    using row_type = reference;
    using const_row_type = const_reference;

private:
    extent_type dims_;
    extent_type strides_;

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

protected:
    ~MatrixBase() noexcept = default;
    template <typename Dim> requires std::is_integral_v<Dim>
    explicit MatrixBase(Dim dim) : dims_{dim}, strides_{1} {};

    MatrixBase(const extent_type& dims) : MatrixBase(dims[0]) {};

    MatrixBase (const MatrixBase&) = default;
    MatrixBase& operator= (const MatrixBase&) = default;
    MatrixBase (MatrixBase&&) noexcept = default;
    MatrixBase& operator= (MatrixBase&&) noexcept = default;

    template <typename DerivedOther, std::semiregular U> requires std::is_convertible_v<U, T>
    MatrixBase(const MatrixBase<DerivedOther, U, 1>&);

    MatrixBase(typename MatrixInitializer<T, 1>::type init);

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

    [[nodiscard]] std::size_t size() const { return dims_[0];}
    [[nodiscard]] extent_type dims() const { return dims_;}
    [[nodiscard]] std::size_t dims(std::size_t n) const {
        if (n >= 1) {
            throw std::out_of_range("Out of range in dims");
        }
        return dims_[0];
    }

    [[nodiscard]] extent_type strides() const {
        return strides_;
    }

    auto dataView() const {
        return self().dataView();
    }

    auto origStrides() const {
        return self().origStrides();
    }

    template <IndexType Dim>
    reference operator()(Dim dim) {
        return const_cast<reference>(std::as_const(*this).operator()(dim));
    }

    template <IndexType Dim>
    const_reference operator()(Dim dim) const {
        return operator[](dim);
    }

    reference operator[](std::size_t pos) {
        return const_cast<reference>(std::as_const(*this).operator[](pos));
    }

    const_reference operator[](std::size_t pos) const {
        return *(cbegin() + pos);
    }

    view_type submatrix(std::size_t pos_begin);
    view_type submatrix(std::size_t pos_begin, std::size_t pos_end);
    row_type row(std::size_t n);
    row_type col(std::size_t n);

    const_view_type submatrix(std::size_t pos_begin) const;
    const_view_type submatrix(std::size_t pos_begin, std::size_t pos_end) const;
    const_row_type row(std::size_t n) const;
    const_row_type col(std::size_t n) const;

    friend std::ostream& operator<<(std::ostream& os, const MatrixBase& m) {
        os << '{';
        for (std::size_t i = 0; i != m.dims_[0]; ++i) {
            os << m[i];
            if (i + 1 != m.dims_[0]) {
                os << ", ";
            }
        }
        return os << '}';
    }

    template <typename DerivedOther1, typename DerivedOther2,
            std::semiregular U, std::semiregular V,
            std::invocable<T&, const U&, const V&> F>
    MatrixBase& applyFunctionWithBroadcast(const MatrixBase<DerivedOther1, U, 1>& m1,
                                           const MatrixBase<DerivedOther2, V, 1>& m2,
                                           F&& f);

    template <typename DerivedOther> requires Addable<T, typename DerivedOther::value_type>
    MatrixBase& operator+=(const ObjectBase<DerivedOther>& other) {
        std::transform(begin(), end(), other.begin(), begin(), std::plus<>{});
        return *this;
    }

    template <typename DerivedOther> requires Subtractable<T, typename DerivedOther::value_type>
    MatrixBase& operator-=(const ObjectBase<DerivedOther>& other) {
        std::transform(begin(), end(), other.begin(), begin(), std::minus<>{});
        return *this;
    }

    MatrixBase& swapRows(std::size_t i, std::size_t j) {
        if (i >= dims_[0] || j >= dims_[0]) {
            throw std::out_of_range("Out of range in swapRows");
        }
        std::swap(row(i), row(j));
        return *this;
    }

    MatrixBase& swapCols(std::size_t i, std::size_t j) {
        swapRows(i, j);
        return *this;
    }

};

template <typename Derived, std::semiregular T, bool isRowMajor>
template <typename DerivedOther, std::semiregular U> requires std::is_convertible_v<U, T>
MatrixBase<Derived, T, 1, isRowMajor>::MatrixBase(const MatrixBase<DerivedOther, U, 1>& other) : MatrixBase(other.dims(0)) {}

template <typename Derived, std::semiregular T, bool isRowMajor>
MatrixBase<Derived, T, 1, isRowMajor>::MatrixBase(typename MatrixInitializer<T, 1>::type init)
: MatrixBase(deriveDims<1>(init)[0]) {}

template <typename Derived, std::semiregular T, bool isRowMajor>
typename MatrixBase<Derived, T, 1, isRowMajor>::view_type MatrixBase<Derived, T, 1, isRowMajor>::submatrix(std::size_t pos_begin) {
    return submatrix(pos_begin, dims_[0]);
}

template <typename Derived, std::semiregular T, bool isRowMajor>
typename MatrixBase<Derived, T, 1, isRowMajor>::const_view_type
        MatrixBase<Derived, T, 1, isRowMajor>::submatrix(std::size_t pos_begin) const {
    return submatrix(pos_begin, dims_[0]);
}

template <typename Derived, std::semiregular T, bool isRowMajor>
typename MatrixBase<Derived, T, 1, isRowMajor>::view_type MatrixBase<Derived, T, 1, isRowMajor>::submatrix(std::size_t pos_begin,
                                        std::size_t pos_end) {
    return std::as_const(*this).submatrix(pos_begin, pos_end);
}

template <typename Derived, std::semiregular T, bool isRowMajor>
typename MatrixBase<Derived, T, 1, isRowMajor>::const_view_type MatrixBase<Derived, T, 1, isRowMajor>::submatrix(std::size_t pos_begin,
                                              std::size_t pos_end) const {
    if (pos_begin >= pos_end) {
        throw std::out_of_range("submatrix begin/end position error");
    }
    auto realStrides = [&](){
        if constexpr (std::is_same_v<Derived, MatrixView<T, 1>>) {
            return origStrides();
        } else {
            return strides();
        }
    };
    const_view_type view ({pos_end - pos_begin}, &operator[](pos_begin), realStrides());
    return view;
}

template <typename Derived, std::semiregular T, bool isRowMajor>
typename MatrixBase<Derived, T, 1, isRowMajor>::row_type MatrixBase<Derived, T, 1, isRowMajor>::row(std::size_t n) {
    return const_cast<T&>(std::as_const(*this).row(n));
}

template <typename Derived, std::semiregular T, bool isRowMajor>
typename MatrixBase<Derived, T, 1, isRowMajor>::const_row_type MatrixBase<Derived, T, 1, isRowMajor>::row(std::size_t n) const {
    if (n >= dims_[0]) {
        throw std::out_of_range("row index error");
    }
    return operator[](n);
}

template <typename Derived, std::semiregular T, bool isRowMajor>
typename MatrixBase<Derived, T, 1, isRowMajor>::row_type MatrixBase<Derived, T, 1, isRowMajor>::col(std::size_t n) {
    return row(n);
}

template <typename Derived, std::semiregular T, bool isRowMajor>
typename MatrixBase<Derived, T, 1, isRowMajor>::const_row_type MatrixBase<Derived, T, 1, isRowMajor>::col(std::size_t n) const {
    return row(n);
}

template <typename Derived, std::semiregular T, bool isRowMajor>
template <typename DerivedOther1, typename DerivedOther2,
        std::semiregular U, std::semiregular V,
        std::invocable<T&, const U&, const V&> F>
MatrixBase<Derived, T, 1, isRowMajor>& MatrixBase<Derived, T, 1, isRowMajor>::applyFunctionWithBroadcast(
        const MatrixBase<DerivedOther1, U, 1>& m1,
        const MatrixBase<DerivedOther2, V, 1>& m2,
        F&& f) {
    // real update is done here by passing lvalue reference T&
    auto r = dims_[0];
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
