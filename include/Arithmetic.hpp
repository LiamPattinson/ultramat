#ifndef __ULTRA_ARITHMETIC_HPP
#define __ULTRA_ARITHMETIC_HPP

// Arithmetic
//
// Defines simple operator overloads for Expressions.

#include "Expression.hpp"

namespace ultra {

// Define classes for generalised arithmetic operations
// (Really have to wonder why these aren't in the standard...)

struct Plus       { template<class L,class R> operator()( const L& l, const R& r) const { return l + r; }};
struct Minus      { template<class L,class R> operator()( const L& l, const R& r) const { return l - r; }};
struct Multiplies { template<class L,class R> operator()( const L& l, const R& r) const { return l * r; }};
struct Divides    { template<class L,class R> operator()( const L& l, const R& r) const { return l / r; }};
struct Negate     { template<class T> operator()( const T& t) const { return -t; }};

} // namespace

// Define operator overloads

template<class L, class R>
ultra::ElementWiseExpression<ultra::Plus,L,R> operator+( const ultra::Expression<L>& l, const ultra::Expression<R>& r){
    return ultra::ElementWiseExpression<ultra::Plus,L,R>(static_cast<const L&>(l),static_cast<const R&>(r));
}

template<class L, class R>
ultra::ElementWiseExpression<ultra::Minus,L,R> operator-( const ultra::Expression<L>& l, const ultra::Expression<R>& r){
    return ultra::ElementWiseExpression<ultra::Minus,L,R>(static_cast<const L&>(l),static_cast<const R&>(r));
}

template<class L, class R>
ultra::ElementWiseExpression<ultra::Multiplies,L,R> operator*( const ultra::Expression<L>& l, const ultra::Expression<R>& r){
    return ultra::ElementWiseExpression<ultra::Multiplies,L,R>(static_cast<const L&>(l),static_cast<const R&>(r));
}

template<class L, class R>
ultra::ElementWiseExpression<ultra::Divides,L,R> operator/( const ultra::Expression<L>& l, const ultra::Expression<R>& r){
    return ultra::ElementWiseExpression<ultra::Divides,L,R>(static_cast<const L&>(l),static_cast<const R&>(r));
}

template<class T>
ultra::ElementWiseExpression<ultra::Negate,T> operator-( const ultra::Expression<T>& t){
    return ultra::ElementWiseExpression<ultra::Negate,T>(static_cast<const T&>(t));
}

#endif
