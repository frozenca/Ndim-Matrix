#ifndef FROZENCA_MATRIXINITIALIZER_H
#define FROZENCA_MATRIXINITIALIZER_H

#include <concepts>
#include <initializer_list>

namespace frozenca {

template <std::semiregular T, std::size_t N>
struct MatrixInitializer {
    using type = std::initializer_list<typename MatrixInitializer<T, N - 1>::type>;
};

template <std::semiregular T>
struct MatrixInitializer<T, 1> {
    using type = std::initializer_list<T>;
};

template <std::semiregular T>
struct MatrixInitializer<T, 0>;

} // namespace frozenca

#endif //FROZENCA_MATRIXINITIALIZER_H
