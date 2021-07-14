#ifndef __ULTRA_DENSE_UTILS_HPP
#define __ULTRA_DENSE_UTILS_HPP

// DenseUtils.hpp
//
// Defines any utility functions/structs deemed useful.

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

namespace ultra {

// Define row/col order enum class

enum class DenseOrder { row_major, col_major, mixed };
const DenseOrder default_order = DenseOrder::row_major;

// get_common_order
// return row_major if all row_major, col_major if all col_major, and default otherwise

template<class... Ts> struct GetCommonOrderImpl;

template<class T1, class T2, class... Ts>
struct GetCommonOrderImpl<T1,T2,Ts...> {
    static constexpr DenseOrder Order = std::remove_cvref_t<T1>::order() == GetCommonOrderImpl<T2,Ts...>::order() ? std::remove_cvref_t<T1>::order() : DenseOrder::mixed;
    static constexpr DenseOrder order() { return Order; }
};

template<class T1>
struct GetCommonOrderImpl<T1> {
    static constexpr DenseOrder Order = std::remove_cvref_t<T1>::order();
    static constexpr DenseOrder order() { return Order; }
};

template<class... Ts>
struct get_common_order {
    static constexpr DenseOrder Order = GetCommonOrderImpl<Ts...>::order();
    static constexpr DenseOrder order(){ return Order; }
    static constexpr DenseOrder value = Order;
};

// Define 'DenseType' enum class.
// Used by Dense to determine whether to store shape/stride as std::vector or std::array, and whether to restrict reshaping

enum class DenseType : std::size_t { vec=1, mat=2, nd=0 };

// Define read-only enum class

enum class ReadWrite { writeable, read_only };

// Declare Array types and preferred interface

template<class, DenseType, DenseOrder> class Dense;
template<class, DenseOrder, std::size_t...> class DenseFixed;
template<class, ReadWrite = ReadWrite::writeable> class DenseView;
template<class, ReadWrite = ReadWrite::writeable> class DenseStripe;

template<class T,std::size_t... Dims>
using Array = std::conditional_t<sizeof...(Dims),DenseFixed<T,default_order,Dims...>,Dense<T,DenseType::nd,default_order>>;

template<class T,std::size_t... Dims> requires ( sizeof...(Dims) == 0 || sizeof...(Dims) == 1)
using Vector = std::conditional_t<sizeof...(Dims),DenseFixed<T,default_order,Dims...>,Dense<T,DenseType::vec,default_order>>;

template<class T,std::size_t... Dims> requires ( sizeof...(Dims) == 0 || sizeof...(Dims) == 2)
using Matrix = std::conditional_t<sizeof...(Dims),DenseFixed<T,default_order,Dims...>,Dense<T,DenseType::mat,default_order>>;

// is_dense type trait
template<class T> struct is_dense { static constexpr bool value = false; };
template<class T, DenseType Type, DenseOrder Order> struct is_dense<Dense<T,Type,Order>> { static constexpr bool value = true; };
template<class T, DenseOrder Order, std::size_t... dims> struct is_dense<DenseFixed<T,Order,dims...>> { static constexpr bool value = true; };
template<class T, ReadWrite RW> struct is_dense<DenseView<T,RW>> { static constexpr bool value = true; };

// Define slice : a tool for generating views of Arrays and related objects.
// Is an 'aggregate'/'pod' type, so should have a relatively intuitive interface by default.

struct Slice { 
    static constexpr std::ptrdiff_t all = std::numeric_limits<std::ptrdiff_t>::max();
    std::ptrdiff_t start;
    std::ptrdiff_t end;
    std::ptrdiff_t step=1;
};

// Bool class
// Looks like a bool, acts like a bool. Used in place of regular bool in dynamic dense objects to avoid the horrors of std::vector<bool>

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

template<class T> struct is_complex { static constexpr bool value = false; };
template<std::floating_point T> struct is_complex<std::complex<T>> { static constexpr bool value = true; };

// Custom concepts

template<class T> concept arithmetic = std::is_arithmetic<T>::value;

template<class T> concept number = std::is_arithmetic<T>::value || is_complex<T>::value;

template<class T> concept shapelike = std::ranges::sized_range<T> && std::integral<typename T::value_type> && !is_dense<T>::value;

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

// ==========================
// Fixed-Size Dense Utils
// Lots of compile-time template nonsense ahead.

// variadic_product

template<std::size_t... Ints> struct variadic_product;

template<> struct variadic_product<> { static constexpr std::size_t value = 1; };

template<std::size_t Int,std::size_t... Ints>
struct variadic_product<Int,Ints...> {
    static constexpr std::size_t value = Int*variadic_product<Ints...>::value;
};

// index_sequence_to_array

template<std::size_t... Ints>
constexpr auto index_sequence_to_array( std::index_sequence<Ints...> ) noexcept {
    return std::array<std::size_t,sizeof...(Ints)>{{Ints...}};
}

// reverse_index_sequence

template<class T1,class T2> struct reverse_index_sequence_impl;

template<std::size_t Int1, std::size_t... Ints1, std::size_t Int2, std::size_t... Ints2>
struct reverse_index_sequence_impl< std::index_sequence<Int1,Ints1...>, std::index_sequence<Int2,Ints2...>> {
    using type = 
        reverse_index_sequence_impl<std::index_sequence<Ints1...>,std::index_sequence<Int1,Int2,Ints2...>>::type;
};

template<std::size_t... Ints2>
struct reverse_index_sequence_impl< std::index_sequence<>, std::index_sequence<Ints2...>> {
    using type = std::index_sequence<Ints2...>;
};

template<std::size_t Int1, std::size_t... Ints1>
struct reverse_index_sequence_impl< std::index_sequence<Int1,Ints1...>, std::index_sequence<>> {
    using type = reverse_index_sequence_impl<std::index_sequence<Ints1...>,std::index_sequence<Int1>>::type;
};

template<class T> struct reverse_index_sequence;

template<std::size_t... Ints>
struct reverse_index_sequence<std::index_sequence<Ints...>> {
    using type = reverse_index_sequence_impl<std::index_sequence<Ints...>,std::index_sequence<>>::type;
};

// variadic stride

template<class T1,class T2> struct variadic_stride_impl;

template<std::size_t ShapeInt, std::size_t... ShapeInts, std::size_t StrideInt, std::size_t... StrideInts>
struct variadic_stride_impl< std::index_sequence<ShapeInt,ShapeInts...>, std::index_sequence<StrideInt,StrideInts...>> {
    using type = 
        variadic_stride_impl<std::index_sequence<ShapeInts...>,std::index_sequence<ShapeInt*StrideInt,StrideInt,StrideInts...>>::type;
};

template<std::size_t... StrideInts>
struct variadic_stride_impl< std::index_sequence<>, std::index_sequence<StrideInts...>> {
    using type = std::index_sequence<StrideInts...>;
};

template<std::size_t ShapeInt, std::size_t... ShapeInts>
struct variadic_stride_impl< std::index_sequence<ShapeInt,ShapeInts...>, std::index_sequence<>> {
    using type = variadic_stride_impl<std::index_sequence<ShapeInts...>,std::index_sequence<ShapeInt>>::type;
};

template<std::size_t... ShapeInts>
struct variadic_stride {
    static constexpr auto col_major = index_sequence_to_array(
        typename reverse_index_sequence<typename variadic_stride_impl<std::index_sequence<1,ShapeInts...>,std::index_sequence<>>::type>::type()
    );
    static constexpr auto row_major = index_sequence_to_array(
        typename variadic_stride_impl<typename reverse_index_sequence<std::index_sequence<ShapeInts...,1>>::type,std::index_sequence<>>::type()
    );
};

} // namespace
#endif
