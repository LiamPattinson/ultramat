#ifndef __ULTRA_UTILS_HPP
#define __ULTRA_UTILS_HPP

// Utils.hpp
//
// Defines any utility functions/structs deemed useful.
// Mostly a lot of template hacking.

#include <cstdlib>
#include <cmath>
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

// Type declarations

template<class T> class Array;

// Type determination

template<class T>
struct types {
    static const bool is_array = false;
};

template<class T>
struct types<Array<T>> {
    static const bool is_array = true;
};

// Define row/col order enum class

enum class RCOrder { row_major, col_major };
const RCOrder default_rc_order = RCOrder::row_major;

// Define read-only enum class

enum class ReadWriteStatus { writeable, read_only };

// Memory helpers
 
// variadic_memjump
// used with round-bracket indexing.
// requires stride of length dims+1, with the largest stride iterating all the way to the location of end().
// If row_major, the largest stride is stride[0]. If col_major, the largest stride is stride[dims].

// Base case
template<std::size_t N, std::ranges::range Stride, std::integral Int>
requires std::integral<typename Stride::value_type>
constexpr Stride::value_type variadic_memjump_impl(const Stride& stride, Int coord) noexcept {
    return stride[N] * coord; 
}

// Recursive step
template<std::size_t N, std::ranges::range Stride, std::integral Int, std::integral... Ints>
requires std::integral<typename Stride::value_type>
constexpr Stride::value_type variadic_memjump_impl( const Stride& stride, Int coord, Ints... coords) noexcept {
    return (stride[N] * coord) + variadic_memjump_impl<N+1,Stride,Ints...>(stride,coords...);
}

template<RCOrder Order, std::ranges::range Stride, std::integral... Ints>
requires std::integral<typename Stride::value_type>
constexpr Stride::value_type variadic_memjump( const Stride& stride, Ints... coords) noexcept {
    // if row major, must skip first element of stride
    return variadic_memjump_impl<(Order==RCOrder::row_major?1:0),Stride,Ints...>(stride,coords...);
}

// Define slice : a tool for generating views of Arrays and related objects.
// Is an 'aggregate'/'pod' type, so should have a relatively intuitive interface by default.

struct Slice { 
    static constexpr std::ptrdiff_t all = std::numeric_limits<std::ptrdiff_t>::max();
    std::ptrdiff_t start;
    std::ptrdiff_t end;
    std::ptrdiff_t step=1;
};


// iterator utils
// Kinda like std::begin and std::end, but for other aspects of the iterator interface.
// Implemented as functor objects to give more flexibility

struct Begin { template<class T> decltype(auto) operator()( T&& t) { return t.begin(); } };
struct Deref { template<class T> decltype(auto) operator()( T&& t) { return *t; } };
struct PrefixInc { template<class T> decltype(auto) operator()( T&& t) { return ++t; } };

// apply_to_each
// std::apply(f,tuple) calls a function f with args given by the tuple.
// apply_to_each returns a tuple given by (f(tuple_args[0]),f(tuple_args[1]),...) where f is unary.
// Similar to the possible implementation of std::apply from https://en.cppreference.com/w/cpp/utility/apply

template<class F,class Tuple, std::size_t... I>
constexpr decltype(auto) apply_to_each_impl( F&& f, Tuple&& t, std::index_sequence<I...>)
{
    return std::make_tuple( std::invoke(std::forward<F>(f), std::get<I>(std::forward<Tuple>(t)))...);
}

template<class F, class Tuple>
constexpr decltype(auto) apply_to_each( F&& f, Tuple&& t){
    return apply_to_each_impl( std::forward<F>(f), std::forward<Tuple>(t),
            std::make_index_sequence<std::tuple_size<std::remove_reference_t<Tuple>>::value>{});
}

} // namespace
#endif
