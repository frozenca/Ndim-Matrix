#ifndef FROZENCA_OBJECTBASE_H
#define FROZENCA_OBJECTBASE_H

#include <cstddef>
#include <concepts>
#include <functional>
#include <utility>
#include "MatrixUtils.h"

namespace frozenca {

template <typename Derived>
class ObjectBase {
private:
    Derived& self() { return static_cast<Derived&>(*this); }
    const Derived& self() const { return static_cast<const Derived&>(*this); }

protected:
    ObjectBase() = default;
    virtual ~ObjectBase() = default;

public:
    auto begin() { return self().self().begin(); }
    auto cbegin() const { return self().self().cbegin(); }
    auto end() { return self().self().end(); }
    auto cend() const { return self().self().cend(); }
    auto rbegin() { return self().self().rbegin(); }
    auto crbegin() const { return self().self().crbegin(); }
    auto rend() { return self().self().rend(); }
    auto crend() const { return self().self().crend(); }

    template <typename F> requires std::invocable<F, typename Derived::reference>
    ObjectBase& applyFunction(F&& f);

    template <typename F, typename... Args> requires std::invocable<F, typename Derived::reference, Args...>
    ObjectBase& applyFunction(F&& f, Args&&... args);

    template <std::semiregular U> requires Addable<typename Derived::value_type, U>
    ObjectBase& operator+=(const U& val) {
        return applyFunction([&val](auto& v) {v += val;});
    }

    template <std::semiregular U> requires Subtractable<typename Derived::value_type, U>
    ObjectBase& operator-=(const U& val) {
        return applyFunction([&val](auto& v) {v -= val;});
    }

    template <std::semiregular U> requires Multipliable<typename Derived::value_type, U>
    ObjectBase& operator*=(const U& val) {
        return applyFunction([&val](auto& v) {v *= val;});
    }

    template <std::semiregular U> requires Dividable<typename Derived::value_type, U>
    ObjectBase& operator/=(const U& val) {
        return applyFunction([&val](auto& v) {v /= val;});
    }

    template <std::semiregular U> requires Dividable<typename Derived::value_type, U>
    ObjectBase& operator%=(const U& val) {
        return applyFunction([&val](auto& v) {v %= val;});
    }

    template <std::semiregular U> requires BitMaskable<typename Derived::value_type, U>
    ObjectBase& operator&=(const U& val) {
        return applyFunction([&val](auto& v) {v &= val;});
    }

    template <std::semiregular U> requires BitMaskable<typename Derived::value_type, U>
    ObjectBase& operator|=(const U& val) {
        return applyFunction([&val](auto& v) {v |= val;});
    }

    template <std::semiregular U> requires BitMaskable<typename Derived::value_type, U>
    ObjectBase& operator^=(const U& val) {
        return applyFunction([&val](auto& v) {v ^= val;});
    }

    template <std::semiregular U> requires BitMaskable<typename Derived::value_type, U>
    ObjectBase& operator<<=(const U& val) {
        return applyFunction([&val](auto& v) {v <<= val;});
    }

    template <std::semiregular U> requires BitMaskable<typename Derived::value_type, U>
    ObjectBase& operator>>=(const U& val) {
        return applyFunction([&val](auto& v) {v >>= val;});
    }
};

template <typename Derived>
template <typename F> requires std::invocable<F, typename Derived::reference>
ObjectBase<Derived>& ObjectBase<Derived>::applyFunction(F&& f) {
    for (auto it = begin(); it != end(); ++it) {
        f(*it);
    }
    return *this;
}

template <typename Derived>
template <typename F, typename... Args> requires std::invocable<F, typename Derived::reference, Args...>
ObjectBase<Derived>& ObjectBase<Derived>::applyFunction(F&& f, Args&&... args) {
    for (auto it = begin(); it != end(); ++it) {
        f(*it, std::forward<Args...>(args...));
    }
    return *this;
}

} // namespace frozenca

#endif //FROZENCA_OBJECTBASE_H
