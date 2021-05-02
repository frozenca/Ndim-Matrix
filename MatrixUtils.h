#ifndef FROZENCA_MATRIXUTILS_H
#define FROZENCA_MATRIXUTILS_H

#include <algorithm>
#include <array>
#include <cassert>
#include <concepts>
#include <cstddef>
#include <initializer_list>
#include <iostream>
#include <iterator>
#include <memory>
#include <stdexcept>
#include <type_traits>

namespace frozenca {

template <std::semiregular T, std::size_t N>
class Matrix;

template <std::semiregular T, std::size_t N>
class MatrixView;

template <typename Derived, std::semiregular T, std::size_t N>
class MatrixBase;

template <typename Derived>
class ObjectBase;

template <typename T>
constexpr bool NotMatrix = true;

template <std::semiregular T, std::size_t N>
constexpr bool NotMatrix<Matrix<T, N>> = false;

template <std::semiregular T, std::size_t N>
constexpr bool NotMatrix<MatrixView<T, N>> = false;

template <typename Derived, std::semiregular T, std::size_t N>
constexpr bool NotMatrix<MatrixBase<Derived, T, N>> = false;

template <typename Derived>
constexpr bool NotMatrix<ObjectBase<Derived>> = false;

template <typename T>
concept isNotMatrix = NotMatrix<T> && std::semiregular<T>;

template <typename T>
concept isMatrix = !NotMatrix<T>;

template <typename T>
concept OneExists = requires () {
    { T{1} } -> std::convertible_to<T>;
};

template <typename A, typename B>
concept WeakAddable = requires (A a, B b) {
    a + b;
};

template <typename A, typename B>
concept WeakSubtractable = requires (A a, B b) {
    a - b;
};

template <typename A, typename B>
concept WeakMultipliable = requires (A a, B b) {
    a * b;
};

template <typename A, typename B>
concept WeakDividable = requires (A a, B b) {
    a / b;
};

template <typename A, typename B>
concept WeakRemaindable = requires (A a, B b) {
    a / b;
    a % b;
};

template <typename A, typename B, typename C>
concept AddableTo = requires (A a, B b) {
    { a + b } -> std::convertible_to<C>;
};

template <typename A, typename B, typename C>
concept SubtractableTo = requires (A a, B b) {
    { a - b } -> std::convertible_to<C>;
};

template <typename A, typename B, typename C>
concept MultipliableTo = requires (A a, B b) {
    { a * b } -> std::convertible_to<C>;
};

template <typename A, typename B, typename C>
concept DividableTo = requires (A a, B b) {
    { a / b } -> std::convertible_to<C>;
};

template <typename A, typename B, typename C>
concept RemaindableTo = requires (A a, B b) {
    { a / b } -> std::convertible_to<C>;
    { a % b } -> std::convertible_to<C>;
};

template <typename A, typename B, typename C>
concept BitMaskableTo = requires (A a, B b) {
    { a & b } -> std::convertible_to<C>;
    { a | b } -> std::convertible_to<C>;
    { a ^ b } -> std::convertible_to<C>;
    { a << b } -> std::convertible_to<C>;
    { a >> b } -> std::convertible_to<C>;
};

template <typename A, typename B>
concept Addable = AddableTo<A, B, A>;

template <typename A, typename B>
concept Subtractable = SubtractableTo<A, B, A>;

template <typename A, typename B>
concept Multipliable = MultipliableTo<A, B, A>;

template <typename A, typename B>
concept Dividable = DividableTo<A, B, A>;

template <typename A, typename B>
concept Remaindable = RemaindableTo<A, B, A>;

template <typename A, typename B>
concept BitMaskable = BitMaskableTo<A, B, A>;

template <typename A, typename B> requires WeakAddable<A, B>
inline decltype(auto) Plus(A a, B b) {
    return a + b;
}

template <typename A, typename B> requires WeakSubtractable<A, B>
inline decltype(auto) Minus(A a, B b) {
    return a - b;
}

template <typename A, typename B> requires WeakMultipliable<A, B>
inline decltype(auto) Multiplies(A a, B b) {
    return a * b;
}

template <typename A, typename B> requires WeakDividable<A, B>
inline decltype(auto) Divides(A a, B b) {
    return a / b;
}

template <typename A, typename B> requires WeakRemaindable<A, B>
inline decltype(auto) Modulus(A a, B b) {
    return a % b;
}

template <typename A, typename B, typename C> requires AddableTo<A, B, C>
inline void Plus(C& c, const A& a, const B& b) {
    c = a + b;
}

template <typename A, typename B, typename C> requires SubtractableTo<A, B, C>
inline void Minus(C& c, const A& a, const B& b) {
    c = a - b;
}

template <typename A, typename B, typename C> requires MultipliableTo<A, B, C>
inline void Multiplies(C& c, const A& a, const B& b) {
    c = a * b;
}

template <typename A, typename B, typename C> requires DividableTo<A, B, C>
inline void Divides(C& c, const A& a, const B& b) {
    c = a / b;
}

template <typename A, typename B, typename C> requires RemaindableTo<A, B, C>
inline void Modulus(C& c, const A& a, const B& b) {
    c = a % b;
}

template <typename... Args>
inline constexpr bool All(Args... args) { return (... && args); };

template <typename... Args>
inline constexpr bool Some(Args... args) { return (... || args); };

template <std::size_t M, std::size_t N> requires (N < M)
std::array<std::size_t, M> prependDims(const std::array<std::size_t, N>& arr) {
    std::array<std::size_t, M> dims;
    std::ranges::fill(dims, 1u);
    std::ranges::copy(arr, std::begin(dims) + (M - N));
    return dims;
}

template <std::size_t M, std::size_t N>
bool bidirBroadcastable(const std::array<std::size_t, M>& sz1,
                        const std::array<std::size_t, N>& sz2) {
    if constexpr (M == N) {
        return (std::ranges::equal(sz1, sz2, [](const auto& d1, const auto& d2) {
            return (d1 == d2) || (d1 == 1) || (d2 == 1);}));
    } else if constexpr (M < N) {
        return bidirBroadcastable(prependDims<N, M>(sz1), sz2);
    } else {
        static_assert(M > N);
        return bidirBroadcastable(sz1, prependDims<M, N>(sz2));
    }
}

template <std::size_t M, std::size_t N>
std::array<std::size_t, std::max(M, N)> bidirBroadcastedDims(const std::array<std::size_t, M>& sz1,
                                                             const std::array<std::size_t, N>& sz2) {
    if constexpr (M == N) {
        if (!bidirBroadcastable(sz1, sz2)) {
            throw std::invalid_argument("Cannot broadcast");
        }
        std::array<std::size_t, M> sz;
        std::ranges::transform(sz1, sz2, std::begin(sz), [](const auto& d1, const auto& d2) {
            return std::max(d1, d2);
        });
        return sz;
    } else if constexpr (M < N) {
        return bidirBroadcastedDims(prependDims<N, M>(sz1), sz2);
    } else {
        static_assert(M > N);
        return bidirBroadcastedDims(sz1, prependDims<M, N>(sz2));
    }
}

template <typename... Args>
concept IndexType = All(std::is_integral_v<Args>...);

template <std::size_t N>
std::array<std::size_t, N> computeStrides(const std::array<std::size_t, N>& dims) {
    std::array<std::size_t, N> strides;
    std::size_t str = 1;
    for (std::size_t i = N - 1; i < N; --i) {
        strides[i] = str;
        str *= dims[i];
    }
    return strides;
}

template <std::size_t N, typename Initializer>
bool checkNonJagged(const Initializer& init) {
    auto i = std::cbegin(init);
    for (auto j = std::next(i); j != std::cend(init); ++j) {
        if (i->size() != j->size()) {
            return false;
        }
    }
    return true;
}

template <std::size_t N, typename Iter, typename Initializer>
void addDims(Iter& first, const Initializer& init) {
    if constexpr (N > 1) {
        if (!checkNonJagged<N>(init)) {
            throw std::invalid_argument("Jagged matrix initializer");
        }
    }
    *first = std::size(init);
    ++first;
    if constexpr (N > 1) {
        addDims<N - 1>(first, *std::begin(init));
    }
}

template <std::size_t N, typename Initializer>
std::array<std::size_t, N> deriveDims(const Initializer& init) {
    std::array<std::size_t, N> dims;
    auto f = std::begin(dims);
    addDims<N>(f, init);
    return dims;
}

template <std::semiregular T>
void addList(std::unique_ptr<T[]>& data,
             const T* first, const T* last,
             std::size_t& index) {
    for (; first != last; ++first) {
        data[index] = *first;
        ++index;
    }
}

template <std::semiregular T, typename I>
void addList(std::unique_ptr<T[]>& data,
             const std::initializer_list<I>* first, const std::initializer_list<I>* last,
             std::size_t& index) {
    for (; first != last; ++first) {
        addList(data, first->begin(), first->end(), index);
    }
}

template <std::semiregular T, typename I>
void insertFlat(std::unique_ptr<T[]>& data, std::initializer_list<I> list) {
    std::size_t index = 0;
    addList(data, std::begin(list), std::end(list), index);
}

inline long quot(long a, long b) {
    return (a / b) - (a % b < 0);
}

inline long mod(long a, long b) {
    return (a % b + b) % b;
}

} // namespace frozenca

#endif //FROZENCA_MATRIXUTILS_H
