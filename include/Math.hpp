#ifndef __ULTRA_MATH_HPP
#define __ULTRA_MATH_HPP

// Math
//
// Defines math functions for expressions.

#include <cstdlib>
#include <cmath>

#include "Expression.hpp"

namespace ultra {

// Define function objects for generalised arithmetic operations
// (Really have to wonder why these aren't in the standard...)

struct Negate     { template<class T> decltype(auto) operator()( const T& t) const { return -t; }};

struct Plus       { template<class L,class R> decltype(auto) operator()( const L& l, const R& r) const { return l + r; }};
struct Minus      { template<class L,class R> decltype(auto) operator()( const L& l, const R& r) const { return l - r; }};
struct Multiplies { template<class L,class R> decltype(auto) operator()( const L& l, const R& r) const { return l * r; }};
struct Divides    { template<class L,class R> decltype(auto) operator()( const L& l, const R& r) const { return l / r; }};

} // namespace

// Define operator overloads

template<class T>
decltype(auto) operator-( const ultra::Expression<T>& t){
    return ultra::ElementWiseExpression<ultra::Negate,T>(static_cast<const T&>(t));
}

template<class L, class R>
decltype(auto) operator+( const ultra::Expression<L>& l, const ultra::Expression<R>& r){
    return ultra::ElementWiseExpression<ultra::Plus,L,R>(static_cast<const L&>(l),static_cast<const R&>(r));
}

template<class L, class R>
decltype(auto) operator-( const ultra::Expression<L>& l, const ultra::Expression<R>& r){
    return ultra::ElementWiseExpression<ultra::Minus,L,R>(static_cast<const L&>(l),static_cast<const R&>(r));
}

template<class L, class R>
decltype(auto) operator*( const ultra::Expression<L>& l, const ultra::Expression<R>& r){
    return ultra::ElementWiseExpression<ultra::Multiplies,L,R>(static_cast<const L&>(l),static_cast<const R&>(r));
}

template<class L, class R>
decltype(auto) operator/( const ultra::Expression<L>& l, const ultra::Expression<R>& r){
    return ultra::ElementWiseExpression<ultra::Divides,L,R>(static_cast<const L&>(l),static_cast<const R&>(r));
}

namespace ultra {

// Define function objects for most cmath function.

// Unary functions
struct Abs    { template<class T> decltype(auto) operator()( const T& t) const { return std::abs(t);}};
struct Sin    { template<class T> decltype(auto) operator()( const T& t) const { return std::sin(t);}};
struct Cos    { template<class T> decltype(auto) operator()( const T& t) const { return std::cos(t);}};
struct Tan    { template<class T> decltype(auto) operator()( const T& t) const { return std::tan(t);}};
struct Asin   { template<class T> decltype(auto) operator()( const T& t) const { return std::asin(t);}};
struct Acos   { template<class T> decltype(auto) operator()( const T& t) const { return std::acos(t);}};
struct Atan   { template<class T> decltype(auto) operator()( const T& t) const { return std::atan(t);}};
struct Sinh   { template<class T> decltype(auto) operator()( const T& t) const { return std::sinh(t);}};
struct Cosh   { template<class T> decltype(auto) operator()( const T& t) const { return std::cosh(t);}};
struct Tanh   { template<class T> decltype(auto) operator()( const T& t) const { return std::tanh(t);}};
struct Asinh  { template<class T> decltype(auto) operator()( const T& t) const { return std::asinh(t);}};
struct Acosh  { template<class T> decltype(auto) operator()( const T& t) const { return std::acosh(t);}};
struct Atanh  { template<class T> decltype(auto) operator()( const T& t) const { return std::atanh(t);}};
struct Sqrt   { template<class T> decltype(auto) operator()( const T& t) const { return std::sqrt(t);}};
struct Cbrt   { template<class T> decltype(auto) operator()( const T& t) const { return std::cbrt(t);}};
struct Exp    { template<class T> decltype(auto) operator()( const T& t) const { return std::exp(t);}};
struct Exp2   { template<class T> decltype(auto) operator()( const T& t) const { return std::exp2(t);}};
struct Expm1  { template<class T> decltype(auto) operator()( const T& t) const { return std::expm1(t);}};
struct Log    { template<class T> decltype(auto) operator()( const T& t) const { return std::log(t);}};
struct Log2   { template<class T> decltype(auto) operator()( const T& t) const { return std::log2(t);}};
struct Log10  { template<class T> decltype(auto) operator()( const T& t) const { return std::log10(t);}};
struct Log1p  { template<class T> decltype(auto) operator()( const T& t) const { return std::log1p(t);}};
struct Ceil   { template<class T> decltype(auto) operator()( const T& t) const { return std::ceil(t);}};
struct Floor  { template<class T> decltype(auto) operator()( const T& t) const { return std::floor(t);}};
struct Round  { template<class T> decltype(auto) operator()( const T& t) const { return std::round(t);}};
struct Erf    { template<class T> decltype(auto) operator()( const T& t) const { return std::erf(t);}};
struct Erfc   { template<class T> decltype(auto) operator()( const T& t) const { return std::erfc(t);}};
struct Tgamma { template<class T> decltype(auto) operator()( const T& t) const { return std::tgamma(t);}};
struct Lgamma { template<class T> decltype(auto) operator()( const T& t) const { return std::lgamma(t);}};

// Binary functions (and ternary)
struct Pow    { template<class X,class Y> decltype(auto) operator()( const X& x, const Y& y) const { return std::pow(x,y);}};
struct Atan2  { template<class X,class Y> decltype(auto) operator()( const X& x, const Y& y) const { return std::atan2(x,y);}};
struct Hypot  { template<class X,class Y> decltype(auto) operator()( const X& x, const Y& y) const { return std::hypot(x,y);}
                template<class X,class Y,class Z> decltype(auto) operator()( const X& x, const Y& y, const Z& z) const { return std::hypot(x,y,z);}};

// Define functions

// Unary functions

template<class T> decltype(auto) abs( const Expression<T>& t){ return ElementWiseExpression<Abs,T>(static_cast<const T&>(t));}
template<class T> decltype(auto) sin( const Expression<T>& t){ return ElementWiseExpression<Sin,T>(static_cast<const T&>(t));}
template<class T> decltype(auto) cos( const Expression<T>& t){ return ElementWiseExpression<Cos,T>(static_cast<const T&>(t));}
template<class T> decltype(auto) tan( const Expression<T>& t){ return ElementWiseExpression<Tan,T>(static_cast<const T&>(t));}
template<class T> decltype(auto) asin( const Expression<T>& t){ return ElementWiseExpression<Asin,T>(static_cast<const T&>(t));}
template<class T> decltype(auto) acos( const Expression<T>& t){ return ElementWiseExpression<Acos,T>(static_cast<const T&>(t));}
template<class T> decltype(auto) atan( const Expression<T>& t){ return ElementWiseExpression<Atan,T>(static_cast<const T&>(t));}
template<class T> decltype(auto) sinh( const Expression<T>& t){ return ElementWiseExpression<Sinh,T>(static_cast<const T&>(t));}
template<class T> decltype(auto) cosh( const Expression<T>& t){ return ElementWiseExpression<Cosh,T>(static_cast<const T&>(t));}
template<class T> decltype(auto) tanh( const Expression<T>& t){ return ElementWiseExpression<Tanh,T>(static_cast<const T&>(t));}
template<class T> decltype(auto) asinh( const Expression<T>& t){ return ElementWiseExpression<Asinh,T>(static_cast<const T&>(t));}
template<class T> decltype(auto) acosh( const Expression<T>& t){ return ElementWiseExpression<Acosh,T>(static_cast<const T&>(t));}
template<class T> decltype(auto) atanh( const Expression<T>& t){ return ElementWiseExpression<Atanh,T>(static_cast<const T&>(t));}
template<class T> decltype(auto) sqrt( const Expression<T>& t){ return ElementWiseExpression<Sqrt,T>(static_cast<const T&>(t));}
template<class T> decltype(auto) cbrt( const Expression<T>& t){ return ElementWiseExpression<Cbrt,T>(static_cast<const T&>(t));}
template<class T> decltype(auto) exp( const Expression<T>& t){ return ElementWiseExpression<Exp,T>(static_cast<const T&>(t));}
template<class T> decltype(auto) exp2( const Expression<T>& t){ return ElementWiseExpression<Exp2,T>(static_cast<const T&>(t));}
template<class T> decltype(auto) expm1( const Expression<T>& t){ return ElementWiseExpression<Expm1,T>(static_cast<const T&>(t));}
template<class T> decltype(auto) log( const Expression<T>& t){ return ElementWiseExpression<Log,T>(static_cast<const T&>(t));}
template<class T> decltype(auto) log2( const Expression<T>& t){ return ElementWiseExpression<Log2,T>(static_cast<const T&>(t));}
template<class T> decltype(auto) log10( const Expression<T>& t){ return ElementWiseExpression<Log10,T>(static_cast<const T&>(t));}
template<class T> decltype(auto) log1p( const Expression<T>& t){ return ElementWiseExpression<Log1p,T>(static_cast<const T&>(t));}
template<class T> decltype(auto) ceil( const Expression<T>& t){ return ElementWiseExpression<Ceil,T>(static_cast<const T&>(t));}
template<class T> decltype(auto) floor( const Expression<T>& t){ return ElementWiseExpression<Floor,T>(static_cast<const T&>(t));}
template<class T> decltype(auto) round( const Expression<T>& t){ return ElementWiseExpression<Round,T>(static_cast<const T&>(t));}
template<class T> decltype(auto) erf( const Expression<T>& t){ return ElementWiseExpression<Erf,T>(static_cast<const T&>(t));}
template<class T> decltype(auto) erfc( const Expression<T>& t){ return ElementWiseExpression<Erfc,T>(static_cast<const T&>(t));}
template<class T> decltype(auto) tgamma( const Expression<T>& t){ return ElementWiseExpression<Tgamma,T>(static_cast<const T&>(t));}
template<class T> decltype(auto) lgamma( const Expression<T>& t){ return ElementWiseExpression<Lgamma,T>(static_cast<const T&>(t));}

template<class X,class Y>
decltype(auto) pow( const Expression<X>& x, const Expression<Y>& y){
    return ElementWiseExpression<Pow,X,Y>(static_cast<const X&>(x),static_cast<const Y&>(y));
}

template<class X,class Y>
decltype(auto) atan2( const Expression<X>& x, const Expression<Y>& y){
    return ElementWiseExpression<Atan2,X,Y>(static_cast<const X&>(x),static_cast<const Y&>(y));
}

template<class X,class Y>
decltype(auto) hypot( const Expression<X>& x, const Expression<Y>& y){
    return ElementWiseExpression<Hypot,X,Y>(static_cast<const X&>(x),static_cast<const Y&>(y));
}

template<class X,class Y,class Z>
decltype(auto) hypot( const Expression<X>& x, const Expression<Y>& y, const Expression<Z>& z){
    return ElementWiseExpression<Hypot,X,Y,Z>(static_cast<const X&>(x),static_cast<const Y&>(y),static_cast<const Z&>(z));
}

// Define cumulative functions

template<class T, class StartT=int>
decltype(auto) cumsum( const Expression<T>& t, const StartT& start = 0){
    return CumulativeExpression<Plus,T>(static_cast<const T&>(t),start);
}

template<class T, class StartT=int>
decltype(auto) cumprod( const Expression<T>& t, const StartT& start = 1){
    return CumulativeExpression<Multiplies,T>(static_cast<const T&>(t),start);
}

} // namespace
#endif
