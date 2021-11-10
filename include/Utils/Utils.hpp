#ifndef __ULTRA_UTILS_HPP
#define __ULTRA_UTILS_HPP

/*! \file Utils.hpp
 *  \brief Defines general use utilities for use throughout ultramat.
 *
 *  Includes:
 *  - Numeric constants
 *  - Custom type traits
 *  - Improved complex numbers support
 *  - `Bool` class, for the avoidance of `std::vector<bool>`
 *  - `Shape` alias for `std::vector<std::size_t>`
 */

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

// ==============================================
// Constants

/*! @name Numeric Constants
 *  Useful constants defined with the `ultra::` namespace.
 */
///@{


constexpr const float inf = INFINITY;       //!< Infinity
constexpr const float Inf = INFINITY;       //!< An alias for `inf`
constexpr const float Infinity = INFINITY;  //!< An alias for `inf`
constexpr const float infty = INFINITY;     //!< An alias for `inf`
constexpr const float ninf = -inf;          //!< Negative infinity
constexpr const float Ninf = -inf;          //!< An alias for `ninf`
constexpr const float NInfinity = -inf;     //!< An alias for `ninf`
constexpr const float ninfty = -inf;        //!< An alias for `ninf`
constexpr const double NaN = NAN;           //!< NaN, Not-a-Number

//! \f$\pi\f$
constexpr const double pi = 3.1415926535897932384626433;

//! \f$e\f$, Euler's number, the base of the natural logarithm
constexpr const double e = 2.71828182845904523536028747135266249775724709369995;

//! \f$\gamma\f$, Euler's constant (not to be confused with \f$e\f$)
constexpr const double euler_gamma = 0.5772156649015328606065120900824024310421;

///@}

// ==============================================
// Custom concepts and type traits

//! Type trait for complex numbers
template<class T> struct is_complex { 
    //! Returns true for `std::complex<float>`, `std::complex<double>`, `std::complex<long double>`, and false otherwise
    static constexpr bool value = false; 
};

// Specialisation for `std::complex<float>`, `std::complex<double>`, and `std::complex<long double>`
template<std::floating_point T> struct is_complex<std::complex<T>> { static constexpr bool value = true; };

//! Concept for the `std::is_arithmetic` type trait
template<class T> concept arithmetic = std::is_arithmetic<T>::value;

//! Concept that encapsulates both arithmetic values and complex numbers
template<class T> concept number = std::is_arithmetic<T>::value || is_complex<T>::value;

//! Type trait that tests that at least one class in a type list matches the given class T
template<class T, class... Ts>
struct variadic_contains {
    static constexpr bool value = false; 
};

template<class T, class T2, class... Ts>
struct variadic_contains<T,T2,Ts...> {
    static constexpr bool value = std::is_same<T,T2>::value || variadic_contains<T,Ts...>::value; 
};

//! Type trait that returns the resulting type if all args are added together
template<class T, class... Ts>
struct upcast {
    using type = decltype( T{} + typename upcast<Ts...>::type{} );
};

template<class T>
struct upcast<T> {
    using type = T;
};

//! Shorthand for upcast<Ts...>::type
template<class... Ts> using upcast_t = upcast<Ts...>::type;

// ==============================================
// Better complex overloading

/*! \brief Macro defining more complete arithmetic overloads. Must be called for each operator individually.
 *
 *  Allows upcasting of `std::complex<float>` to `std::complex<double>` or `std::complex<long double>`, and
 *  the automatic conversion/upcasting of non-complex arithmetic types.
 */
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

//! Converts non-complex arithmetic types to an appropriate complex number.
template<class T>
using complex_upcast = std::conditional_t< is_complex<T>::value, T,
    std::conditional_t< std::is_floating_point<T>::value, std::complex<T>, std::complex<double>>
>;

// ==============================================
// Bool class

//! Looks like a bool, acts like a bool. Can be used in place of regular bool to avoid the horrors of std::vector<bool>
class Bool {
    bool _x;
    public:
    Bool() = default;
    Bool( const Bool& ) = default;
    Bool( Bool&& ) = default;
    Bool& operator=( const Bool& ) = default;
    Bool& operator=( Bool&& ) = default;
    inline Bool( bool x) : _x(x) {}
    inline operator bool() const { return _x; }
    inline operator bool&() { return _x; }
};

// ==============================================
// Shape alias

using Shape = std::vector<std::size_t>;

} // namespace ultra

// include other utility files
#include "IteratorTuple.hpp"

#endif
