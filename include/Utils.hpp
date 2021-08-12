#ifndef __ULTRA_UTILS_HPP
#define __ULTRA_UTILS_HPP

// Utils.hpp
//
// Defines generic utilities, such as:
// - tuple manipulation
// - concepts and type traits
// - better complex support

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

// ==========================
// Iterator Tuple Utils
// Defines numerous functions to aid in manipulation of IteratorTuples and similar objects

template<class Tuple>
struct IteratorTupleHelper {

    template<std::size_t N> requires ( N != 0 )
    static void increment(Tuple& t) {
        ++std::get<N>(t);
        increment<N-1>(t);
    }

    template<std::size_t N> requires ( N == 0 )
    static void increment(Tuple& t) {
        ++std::get<N>(t);
    }

    template<std::size_t N> requires ( N != 0 )
    static void decrement(Tuple& t) {
        --std::get<N>(t);
        decrement<N-1>(t);
    }

    template<std::size_t N> requires ( N == 0 )
    static void decrement(Tuple& t) {
        --std::get<N>(t);
    }

    template<std::size_t N, std::integral I> requires ( N != 0 )
    static void add_in_place(Tuple& t, const I& ii) {
        std::get<N>(t) += ii;
        add_in_place<N-1>(t,ii);
    }

    template<std::size_t N, std::integral I> requires ( N == 0 )
    static void add_in_place(Tuple& t, const I& ii) {
        std::get<N>(t) += ii;
    }

    template<std::size_t N, std::integral I> requires ( N != 0 )
    static void sub_in_place(Tuple& t, const I& ii) {
        std::get<N>(t) -= ii;
        sub_in_place<N-1>(t,ii);
    }

    template<std::size_t N, std::integral I> requires ( N == 0 )
    static void sub_in_place(Tuple& t, const I& ii) {
        std::get<N>(t) -= ii;
    }

};

template<class Tuple>
void increment_tuple(Tuple& t){
    IteratorTupleHelper<Tuple>::template increment<std::tuple_size<Tuple>::value-1>(t);
}

template<class Tuple>
void decrement_tuple(Tuple& t){
    IteratorTupleHelper<Tuple>::template decrement<std::tuple_size<Tuple>::value-1>(t);
}

template<class Tuple, std::integral I>
void add_in_place_tuple(Tuple& t, const I& ii){
    IteratorTupleHelper<Tuple>::template add_in_place<std::tuple_size<Tuple>::value-1>(t,ii);
}

template<class Tuple, std::integral I>
void sub_in_place_tuple(Tuple& t, const I& ii){
    IteratorTupleHelper<Tuple>::template sub_in_place<std::tuple_size<Tuple>::value-1>(t,ii);
}

// begin_tuple/end_tuple
// Calls begin/end on all elements of a tuple, returns new tuple. This should be passed to IteratorTuple as follows:
// IteratorTuple its( begin_tuple(some_tuple));

template<class Tuple, std::size_t... I>
auto begin_tuple_impl( const Tuple& t, std::index_sequence<I...>){
    return std::make_tuple( std::get<I>(t).begin() ...);
}

template<class Tuple, std::size_t... I>
auto begin_tuple_impl( Tuple& t, std::index_sequence<I...>){
    return std::make_tuple( std::get<I>(t).begin() ...);
}

template<class Tuple, std::size_t... I>
auto end_tuple_impl( const Tuple& t, std::index_sequence<I...>){
    return std::make_tuple( std::get<I>(t).end() ...);
}

template<class Tuple, std::size_t... I>
auto end_tuple_impl( Tuple& t, std::index_sequence<I...>){
    return std::make_tuple( std::get<I>(t).end() ...);
}

template<class Tuple>
decltype(auto) begin_tuple(Tuple& t){
    return begin_tuple_impl(t,std::make_index_sequence<std::tuple_size<Tuple>::value>{});
}

template<class Tuple>
decltype(auto) begin_tuple(const Tuple& t){
    return begin_tuple_impl(t,std::make_index_sequence<std::tuple_size<Tuple>::value>{});
}

template<class Tuple>
decltype(auto) end_tuple(Tuple& t){
    return end_tuple_impl(t,std::make_index_sequence<std::tuple_size<Tuple>::value>{});
}

template<class Tuple>
decltype(auto) end_tuple(const Tuple& t){
    return end_tuple_impl(t,std::make_index_sequence<std::tuple_size<Tuple>::value>{});
}



// ==========================
// IteratorTuple
// Stick together multiple iterators into a single object.
// Equality/distance is checked on only the first iterator.
// All other operations, such as incrementing, random-access jumps, etc, are performed on all.
// Dereferencing directly will dereference the first iterator in the tuple.
// To dereference later elements, one must retrieve a reference to the required iterator by calling iterator_tuple.get<N>().

template<class... Its>
class IteratorTuple {
    
    std::tuple<Its...> _its;
    using first_it_t = std::tuple_element_t<0,std::tuple<Its...>>;
    
    public:
    
    IteratorTuple() = delete;
    IteratorTuple( const IteratorTuple& ) = default;
    IteratorTuple( IteratorTuple&& ) = default;
    IteratorTuple& operator=( const IteratorTuple& ) = default;
    IteratorTuple& operator=( IteratorTuple&& ) = default;

    IteratorTuple( std::tuple<Its...>&& its) : _its(std::move(its)) {}
    IteratorTuple( Its&&... its) : _its(std::make_tuple(std::move(its)...)) {}

    template<std::size_t idx>
    friend decltype(auto) get( IteratorTuple& tuple ){
        return std::get<idx>(tuple._its);
    }

    decltype(auto) operator*() { return *std::get<0>(_its); }
    decltype(auto) operator*() const { return *std::get<0>(_its); }
    IteratorTuple& operator++() { increment_tuple(_its); return *this; }
    IteratorTuple& operator--() { decrement_tuple(_its); return *this; }
    template<std::integral I> IteratorTuple& operator+=( const I& ii) { add_in_place_tuple(_its,ii); return *this; }
    template<std::integral I> IteratorTuple& operator-=( const I& ii) { sub_in_place_tuple(_its,ii); return *this; }
    template<std::integral I> IteratorTuple operator+( const I& ii) { auto result(*this); result+=ii; return result; }
    template<std::integral I> IteratorTuple operator-( const I& ii) { auto result(*this); result-=ii; return result; }
    bool operator==( const IteratorTuple& other) const { return std::get<0>(_its) == std::get<0>(other._its);}
    auto operator<=>( const IteratorTuple& other) const { return std::get<0>(_its) <=> std::get<0>(other._its);}
    std::ptrdiff_t operator-( const IteratorTuple& other) const { return std::get<0>(_its) - std::get<0>(other._its);}
    bool operator==( const first_it_t& other) const { return std::get<0>(_its) == other;}
    auto operator<=>( const first_it_t& other) const { return std::get<0>(_its) <=> other;}
    std::ptrdiff_t operator-( const first_it_t& other) const { return std::get<0>(_its) - other;}
    friend std::ptrdiff_t operator-( const first_it_t& l, const IteratorTuple& r) { return l - std::get<0>(r._its);}
};


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
#endif
