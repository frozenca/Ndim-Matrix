#ifndef FROZENCA_MATRIXUTILS_H
#define FROZENCA_MATRIXUTILS_H

#include <array>
#include <cassert>
#include <complex>
#include <concepts>
#include <cstddef>
#include <initializer_list>
#include <iostream>
#include <iterator>
#include <memory>
#include <type_traits>

namespace frozenca {

template <typename T>
concept OneExists = requires () {
    { T{1} } -> std::convertible_to<T>;
};

template <typename A, typename B>
concept Addable = requires (A a, B b) {
    { a + b } -> std::convertible_to<A>;
};

template <typename A, typename B>
concept Subtractable = requires (A a, B b) {
    { a - b } -> std::convertible_to<A>;
};

template <typename A, typename B>
concept Multipliable = requires (A a, B b) {
    { a * b } -> std::convertible_to<A>;
};

template <typename A, typename B>
concept Dividable = requires (A a, B b) {
    { a / b } -> std::convertible_to<A>;
    { a % b } -> std::convertible_to<A>;
};

template <typename A, typename B>
concept BitMaskable = requires (A a, B b) {
    { a & b } -> std::convertible_to<A>;
    { a | b } -> std::convertible_to<A>;
    { a ^ b } -> std::convertible_to<A>;
    { a << b } -> std::convertible_to<A>;
    { a >> b } -> std::convertible_to<A>;
};

template <typename... Args>
inline constexpr bool All(Args... args) { return (... && args); };

template <typename... Args>
inline constexpr bool Some(Args... args) { return (... || args); };

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
