#ifndef __ULTRA_DENSE_CUMULATIVE_HPP
#define __ULTRA_DENSE_CUMULATIVE_HPP

#include "ultramat/include/Dense/Expressions/DenseCumulativeExpression.hpp"

namespace ultra {

template<class T>
decltype(auto) cumsum( const DenseExpression<T>& t, std::size_t dim){
    return eval(DenseCumulativeExpression(Plus{},static_cast<const T&>(t),dim));
}

template<class T>
decltype(auto) cumsum( DenseExpression<T>&& t, std::size_t dim){
    return eval(DenseCumulativeExpression(Plus{},static_cast<T&&>(t),dim));
}

template<class T>
decltype(auto) cumprod( const DenseExpression<T>& t, std::size_t dim){
    return eval(DenseCumulativeExpression(Multiplies{},static_cast<const T&>(t),dim));
}

template<class T>
decltype(auto) cumprod( DenseExpression<T>&& t, std::size_t dim){
    return eval(DenseCumulativeExpression(Multiplies{},static_cast<T&&>(t),dim));
}

} // namespace ultra
#endif
