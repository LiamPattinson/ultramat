#ifndef __ULTRA_CUMULATIVE_HPP
#define __ULTRA_CUMULATIVE_HPP

// Cumulative
//
// Defines expressions for cumulative functions.

#include "Expression.hpp"
#include "Arithmetic.hpp"

namespace ultra {

// We can reuse functors defined in Arithmetic

// Expressions

template<class T> using  CumSumExpression = CumulativeExpression<Plus,T>;
template<class T> using CumProdExpression = CumulativeExpression<Multiplies,T>;

// Functions

template<class T, class StartT=int>
decltype(auto) cumsum( const Expression<T>& t, const StartT& start = 0){
    return CumSumExpression(static_cast<const T&>(t),start);
}

template<class T, class StartT=int>
decltype(auto) cumsum( Expression<T>&& t, const StartT& start = 0){
    return CumSumExpression(static_cast<T&&>(t),start);
}

template<class T, class StartT=int>
decltype(auto) cumprod( const Expression<T>& t, const StartT& start = 1){
    return CumProdExpression(static_cast<const T&>(t),start);
}

template<class T, class StartT=int>
decltype(auto) cumprod( const Expression<T>&& t, const StartT& start = 1){
    return CumProdExpression(static_cast<T&&>(t),start);
}

} // namespace
#endif
