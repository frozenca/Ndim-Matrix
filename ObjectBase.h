#ifndef FROZENCA_OBJECTBASE_H
#define FROZENCA_OBJECTBASE_H

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

public:
    virtual ~ObjectBase() = default;
    auto begin() { return self().self().begin(); }
    auto begin() const { return self().self().begin(); }
    auto cbegin() const { return self().self().cbegin(); }
    auto end() { return self().self().end(); }
    auto end() const { return self().self().end(); }
    auto cend() const { return self().self().cend(); }
    auto rbegin() { return self().self().rbegin(); }
    auto rbegin() const { return self().self().rbegin(); }
    auto crbegin() const { return self().self().crbegin(); }
    auto rend() { return self().self().rend(); }
    auto rend() const { return self().self().rend(); }
    auto crend() const { return self().self().crend(); }

    template <typename F> requires std::invocable<F, typename Derived::reference>
    ObjectBase& applyFunction(F&& f);

    template <typename F, typename... Args> requires std::invocable<F, typename Derived::reference, Args...>
    ObjectBase& applyFunction(F&& f, Args&&... args);

    template <typename DerivedOther, typename F> requires std::invocable<F, typename Derived::reference, typename DerivedOther::reference>
    ObjectBase& applyFunction(const ObjectBase<DerivedOther>& other, F&& f);

    template <typename DerivedOther, typename F, typename... Args> requires std::invocable<F, typename Derived::reference, typename DerivedOther::reference, Args...>
    ObjectBase& applyFunction(const ObjectBase<DerivedOther>& other, F&& f, Args&&... args);

    template <isNotMatrix U> requires Addable<typename Derived::value_type, U>
    ObjectBase& operator=(const U& val) {
        return applyFunction([&val](auto& v) {v = val;});
    }

    template <isNotMatrix U> requires Addable<typename Derived::value_type, U>
    ObjectBase& operator+=(const U& val) {
        return applyFunction([&val](auto& v) {v += val;});
    }

    template <isNotMatrix U> requires Subtractable<typename Derived::value_type, U>
    ObjectBase& operator-=(const U& val) {
        return applyFunction([&val](auto& v) {v -= val;});
    }

    template <isNotMatrix U> requires Multipliable<typename Derived::value_type, U>
    ObjectBase& operator*=(const U& val) {
        return applyFunction([&val](auto& v) {v *= val;});
    }

    template <isNotMatrix U> requires Dividable<typename Derived::value_type, U>
    ObjectBase& operator/=(const U& val) {
        return applyFunction([&val](auto& v) {v /= val;});
    }

    template <isNotMatrix U> requires Remaindable<typename Derived::value_type, U>
    ObjectBase& operator%=(const U& val) {
        return applyFunction([&val](auto& v) {v %= val;});
    }

    template <isNotMatrix U> requires BitMaskable<typename Derived::value_type, U>
    ObjectBase& operator&=(const U& val) {
        return applyFunction([&val](auto& v) {v &= val;});
    }

    template <isNotMatrix U> requires BitMaskable<typename Derived::value_type, U>
    ObjectBase& operator|=(const U& val) {
        return applyFunction([&val](auto& v) {v |= val;});
    }

    template <isNotMatrix U> requires BitMaskable<typename Derived::value_type, U>
    ObjectBase& operator^=(const U& val) {
        return applyFunction([&val](auto& v) {v ^= val;});
    }

    template <isNotMatrix U> requires BitMaskable<typename Derived::value_type, U>
    ObjectBase& operator<<=(const U& val) {
        return applyFunction([&val](auto& v) {v <<= val;});
    }

    template <isNotMatrix U> requires BitMaskable<typename Derived::value_type, U>
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

template <typename Derived>
template <typename DerivedOther, typename F> requires std::invocable<F, typename Derived::reference, typename DerivedOther::reference>
ObjectBase<Derived>& ObjectBase<Derived>::applyFunction(const ObjectBase<DerivedOther>& other, F&& f) {
    for (auto it = begin(), it2 = other.begin(); it != end(); ++it, ++it2) {
        f(*it, *it2);
    }
    return *this;
}

template <typename Derived>
template <typename DerivedOther, typename F, typename... Args> requires std::invocable<F, typename Derived::reference, typename DerivedOther::reference, Args...>
ObjectBase<Derived>& ObjectBase<Derived>::applyFunction(const ObjectBase<DerivedOther>& other, F&& f, Args&&... args) {
    for (auto it = begin(), it2 = other.begin(); it != end(); ++it, ++it2) {
        f(*it, *it2, std::forward<Args...>(args...));
    }
    return *this;
}

template <typename Derived, isNotMatrix U> requires Addable<typename Derived::value_type, U>
ObjectBase<Derived> operator+(const ObjectBase<Derived>& m, const U& val) {
    ObjectBase<Derived> res = m;
    res += val;
    return res;
}

template <typename Derived, isNotMatrix U> requires Subtractable<typename Derived::value_type, U>
ObjectBase<Derived> operator-(const ObjectBase<Derived>& m, const U& val) {
    ObjectBase<Derived> res = m;
    res -= val;
    return res;
}

template <typename Derived, isNotMatrix U> requires Multipliable<typename Derived::value_type, U>
ObjectBase<Derived> operator*(const ObjectBase<Derived>& m, const U& val) {
    ObjectBase<Derived> res = m;
    res *= val;
    return res;
}

template <typename Derived, isNotMatrix U> requires Dividable<typename Derived::value_type, U>
ObjectBase<Derived> operator/(const ObjectBase<Derived>& m, const U& val) {
    ObjectBase<Derived> res = m;
    res /= val;
    return res;
}

template <typename Derived, isNotMatrix U> requires Remaindable<typename Derived::value_type, U>
ObjectBase<Derived> operator%(const ObjectBase<Derived>& m, const U& val) {
    ObjectBase<Derived> res = m;
    res %= val;
    return res;
}

template <typename Derived, isNotMatrix U> requires BitMaskable<typename Derived::value_type, U>
ObjectBase<Derived> operator&(const ObjectBase<Derived>& m, const U& val) {
    ObjectBase<Derived> res = m;
    res &= val;
    return res;
}

template <typename Derived, isNotMatrix U> requires BitMaskable<typename Derived::value_type, U>
ObjectBase<Derived> operator^(const ObjectBase<Derived>& m, const U& val) {
    ObjectBase<Derived> res = m;
    res ^= val;
    return res;
}

template <typename Derived, isNotMatrix U> requires BitMaskable<typename Derived::value_type, U>
ObjectBase<Derived> operator|(const ObjectBase<Derived>& m, const U& val) {
    ObjectBase<Derived> res = m;
    res |= val;
    return res;
}

template <typename Derived, isNotMatrix U> requires BitMaskable<typename Derived::value_type, U>
ObjectBase<Derived> operator<<(const ObjectBase<Derived>& m, const U& val) {
    ObjectBase<Derived> res = m;
    res <<= val;
    return res;
}

template <typename Derived, isNotMatrix U> requires BitMaskable<typename Derived::value_type, U>
ObjectBase<Derived> operator>>(const ObjectBase<Derived>& m, const U& val) {
    ObjectBase<Derived> res = m;
    res >>= val;
    return res;
}

} // namespace frozenca

#endif //FROZENCA_OBJECTBASE_H
