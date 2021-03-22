#ifndef __ULTRA_UTILS_HPP
#define __ULTRA_UTILS_HPP

// Utils.hpp
//
// Defines any utility functions/structs deemed useful.
// Mostly a lot of template hacking.

#include <tuple>
#include <type_traits>

namespace ultra {

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
