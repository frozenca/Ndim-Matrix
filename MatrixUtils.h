#ifndef FROZENCA_MATRIXUTILS_H
#define FROZENCA_MATRIXUTILS_H

#include <array>
#include <cassert>
#include <cstddef>
#include <iostream>
#include <iterator>
#include <memory>
#include <type_traits>

namespace frozenca {

template <typename, template <typename...> typename>
struct is_instance_impl : public std::false_type {};

template <template <typename...> typename U, typename... Ts>
struct is_instance_impl<U<Ts...>, U> : public std::true_type {};

template <typename T, template <typename ...> typename U>
inline constexpr bool is_instance_of = is_instance_impl<std::decay_t<T>, U>();

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
