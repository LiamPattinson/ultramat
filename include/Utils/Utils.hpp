#ifndef __ULTRA_UTILS_HPP
#define __ULTRA_UTILS_HPP

// Utils.hpp

#include <cstdlib>
#include <cmath>
#include <complex>
#include <utility>
#include <stdexcept>
#include <string>
#include <vector>
#include <array>
#include <tuple>
#include <functional>
#include <algorithm>
#include <numeric>
#include <limits>
#include <random>
#include <type_traits>
#include <concepts>
#include <ranges>
#include <compare>

#include <omp.h>

namespace ultra {

// Custom concepts and type traits

template<class T> struct is_complex { static constexpr bool value = false; };
template<std::floating_point T> struct is_complex<std::complex<T>> { static constexpr bool value = true; };

template<class T> concept arithmetic = std::is_arithmetic<T>::value;

template<class T> concept number = std::is_arithmetic<T>::value || is_complex<T>::value;

// Better complex overloading

#define ULTRA_COMPLEX_OVERLOAD(OP)\
template<std::floating_point T1, std::floating_point T2> requires (!std::is_same<T1,T2>::value)\
constexpr auto operator OP ( const std::complex<T1>& lhs, const std::complex<T2>& rhs){\
   return static_cast<std::complex<decltype(T1()*T2())>>(lhs) OP static_cast<std::complex<decltype(T1()*T2())>>(rhs);\
}\
\
template<std::floating_point T1, arithmetic T2> requires (!std::is_same<T1,T2>::value)\
constexpr auto operator OP ( const std::complex<T1>& lhs, const T2& rhs){\
   return static_cast<std::complex<decltype(T1()*T2())>>(lhs) OP static_cast<decltype(T1()*T2())>(rhs);\
}\
\
template<arithmetic T1, std::floating_point T2> requires (!std::is_same<T1,T2>::value)\
constexpr auto operator OP ( const T1& lhs, const std::complex<T2>& rhs){\
   return static_cast<decltype(T1()*T2())>(lhs) OP static_cast<std::complex<decltype(T1()*T2())>>(rhs);\
}

ULTRA_COMPLEX_OVERLOAD(+)
ULTRA_COMPLEX_OVERLOAD(-)
ULTRA_COMPLEX_OVERLOAD(*)
ULTRA_COMPLEX_OVERLOAD(/)
ULTRA_COMPLEX_OVERLOAD(==)

template<class T>
using complex_upcast = std::conditional_t< is_complex<T>::value, T,
    std::conditional_t< std::is_floating_point<T>::value, std::complex<T>, std::complex<double>>
>;

}

// include other utility files
#include "IteratorTuple.hpp"

#endif
