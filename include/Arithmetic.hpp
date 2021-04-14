#ifndef __ULTRA_ARITHMETIC_HPP
#define __ULTRA_ARITHMETIC_HPP

// Arithmetic
//
// Defines expressions for simple arithmetic.

#include "Expression.hpp"

namespace ultra {

// =========================
// Arithmetic

// Functors

struct Negate     { template<class T> decltype(auto) operator()( const T& t) const { return -t; }};
struct Plus       { template<class L,class R> decltype(auto) operator()( const L& l, const R& r) const { return l + r; }};
struct Minus      { template<class L,class R> decltype(auto) operator()( const L& l, const R& r) const { return l - r; }};
struct Multiplies { template<class L,class R> decltype(auto) operator()( const L& l, const R& r) const { return l * r; }};
struct Divides    { template<class L,class R> decltype(auto) operator()( const L& l, const R& r) const { return l / r; }};

// Expressions

template<class T>         using NegateExpression     = ElementWiseExpression<Negate,T>;
template<class L,class R> using PlusExpression       = ElementWiseExpression<Plus,L,R>;
template<class L,class R> using MinusExpression      = ElementWiseExpression<Minus,L,R>;
template<class L,class R> using MultipliesExpression = ElementWiseExpression<Multiplies,L,R>;
template<class L,class R> using DividesExpression    = ElementWiseExpression<Divides,L,R>;

// Operator overloads

// Negation 

template<class T>
decltype(auto) operator-( const Expression<T>& t){
    return NegateExpression(static_cast<const T&>(t));
}


template<class T>
decltype(auto) operator-( Expression<T>&& t){
    return NegateExpression(static_cast<T&&>(t));
}

// Addition

template<class L, class R>
decltype(auto) operator+( const Expression<L>& l, const Expression<R>& r){
    return PlusExpression(static_cast<const L&>(l),static_cast<const R&>(r));
}

template<class L, class R>
decltype(auto) operator+( const Expression<L>& l, Expression<R>&& r){
    return PlusExpression(static_cast<const L&>(l),static_cast<R&&>(r));
}

template<class L, class R>
decltype(auto) operator+( Expression<L>&& l, const Expression<R>& r){
    return PlusExpression(static_cast<L&&>(l),static_cast<const R&>(r));
}

template<class L, class R>
decltype(auto) operator+( Expression<L>&& l, Expression<R>&& r){
    return PlusExpression(static_cast<L&&>(l),static_cast<R&&>(r));
}

// Subtraction

template<class L, class R>
decltype(auto) operator-( const Expression<L>& l, const Expression<R>& r){
    return MinusExpression(static_cast<const L&>(l),static_cast<const R&>(r));
}

template<class L, class R>
decltype(auto) operator-( Expression<L>&& l, const Expression<R>& r){
    return MinusExpression(static_cast<L&&>(l),static_cast<const R&>(r));
}

template<class L, class R>
decltype(auto) operator-( const Expression<L>& l, Expression<R>&& r){
    return MinusExpression(static_cast<const L&>(l),static_cast<R&&>(r));
}

template<class L, class R>
decltype(auto) operator-( Expression<L>&& l, Expression<R>&& r){
    return MinusExpression(static_cast<L&&>(l),static_cast<R&&>(r));
}

// Multiplication

template<class L, class R>
decltype(auto) operator*( const Expression<L>& l, const Expression<R>& r){
    return MultipliesExpression(static_cast<const L&>(l),static_cast<const R&>(r));
}

template<class L, class R>
decltype(auto) operator*( Expression<L>&& l, const Expression<R>& r){
    return MultipliesExpression(static_cast<L&&>(l),static_cast<const R&>(r));
}

template<class L, class R>
decltype(auto) operator*( const Expression<L>& l, Expression<R>&& r){
    return MultipliesExpression(static_cast<const L&>(l),static_cast<R&&>(r));
}

template<class L, class R>
decltype(auto) operator*( Expression<L>&& l, Expression<R>&& r){
    return MultipliesExpression(static_cast<L&&>(l),static_cast<R&&>(r));
}

// Division

template<class L, class R>
decltype(auto) operator/( const Expression<L>& l, const Expression<R>& r){
    return DividesExpression(static_cast<const L&>(l),static_cast<const R&>(r));
}

template<class L, class R>
decltype(auto) operator/( Expression<L>&& l, const Expression<R>& r){
    return DividesExpression(static_cast<L&&>(l),static_cast<const R&>(r));
}

template<class L, class R>
decltype(auto) operator/( const Expression<L>& l, Expression<R>&& r){
    return DividesExpression(static_cast<const L&>(l),static_cast<R&&>(r));
}

template<class L, class R>
decltype(auto) operator/( Expression<L>&& l, Expression<R>&& r){
    return DividesExpression(static_cast<L&&>(l),static_cast<R&&>(r));
}

} // namespace

#endif
