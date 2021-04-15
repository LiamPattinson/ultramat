#ifndef __ULTRA_DENSE_MATH_HPP
#define __ULTRA_DENSE_MATH_HPP

// DenseMath
//
// Defines expressions for standard math functions.

#include "DenseExpression.hpp"

namespace ultra {

// =========================
// Arithmetic

// Functors

struct Negate     { template<class T> decltype(auto) operator()( const T& t) const { return -t; }};
struct Plus       { template<class L,class R> decltype(auto) operator()( const L& l, const R& r) const { return l + r; }};
struct Minus      { template<class L,class R> decltype(auto) operator()( const L& l, const R& r) const { return l - r; }};
struct Multiplies { template<class L,class R> decltype(auto) operator()( const L& l, const R& r) const { return l * r; }};
struct Divides    { template<class L,class R> decltype(auto) operator()( const L& l, const R& r) const { return l / r; }};

// DenseExpressions

template<class T>         using NegateDenseExpression     = ElementWiseDenseExpression<Negate,T>;
template<class L,class R> using PlusDenseExpression       = ElementWiseDenseExpression<Plus,L,R>;
template<class L,class R> using MinusDenseExpression      = ElementWiseDenseExpression<Minus,L,R>;
template<class L,class R> using MultipliesDenseExpression = ElementWiseDenseExpression<Multiplies,L,R>;
template<class L,class R> using DividesDenseExpression    = ElementWiseDenseExpression<Divides,L,R>;

// Operator overloads

// Negation 

template<class T>
decltype(auto) operator-( const DenseExpression<T>& t){
    return NegateDenseExpression(static_cast<const T&>(t));
}


template<class T>
decltype(auto) operator-( DenseExpression<T>&& t){
    return NegateDenseExpression(static_cast<T&&>(t));
}

// Addition

template<class L, class R>
decltype(auto) operator+( const DenseExpression<L>& l, const DenseExpression<R>& r){
    return PlusDenseExpression(static_cast<const L&>(l),static_cast<const R&>(r));
}

template<class L, class R>
decltype(auto) operator+( const DenseExpression<L>& l, DenseExpression<R>&& r){
    return PlusDenseExpression(static_cast<const L&>(l),static_cast<R&&>(r));
}

template<class L, class R>
decltype(auto) operator+( DenseExpression<L>&& l, const DenseExpression<R>& r){
    return PlusDenseExpression(static_cast<L&&>(l),static_cast<const R&>(r));
}

template<class L, class R>
decltype(auto) operator+( DenseExpression<L>&& l, DenseExpression<R>&& r){
    return PlusDenseExpression(static_cast<L&&>(l),static_cast<R&&>(r));
}

// Subtraction

template<class L, class R>
decltype(auto) operator-( const DenseExpression<L>& l, const DenseExpression<R>& r){
    return MinusDenseExpression(static_cast<const L&>(l),static_cast<const R&>(r));
}

template<class L, class R>
decltype(auto) operator-( DenseExpression<L>&& l, const DenseExpression<R>& r){
    return MinusDenseExpression(static_cast<L&&>(l),static_cast<const R&>(r));
}

template<class L, class R>
decltype(auto) operator-( const DenseExpression<L>& l, DenseExpression<R>&& r){
    return MinusDenseExpression(static_cast<const L&>(l),static_cast<R&&>(r));
}

template<class L, class R>
decltype(auto) operator-( DenseExpression<L>&& l, DenseExpression<R>&& r){
    return MinusDenseExpression(static_cast<L&&>(l),static_cast<R&&>(r));
}

// Multiplication

template<class L, class R>
decltype(auto) operator*( const DenseExpression<L>& l, const DenseExpression<R>& r){
    return MultipliesDenseExpression(static_cast<const L&>(l),static_cast<const R&>(r));
}

template<class L, class R>
decltype(auto) operator*( DenseExpression<L>&& l, const DenseExpression<R>& r){
    return MultipliesDenseExpression(static_cast<L&&>(l),static_cast<const R&>(r));
}

template<class L, class R>
decltype(auto) operator*( const DenseExpression<L>& l, DenseExpression<R>&& r){
    return MultipliesDenseExpression(static_cast<const L&>(l),static_cast<R&&>(r));
}

template<class L, class R>
decltype(auto) operator*( DenseExpression<L>&& l, DenseExpression<R>&& r){
    return MultipliesDenseExpression(static_cast<L&&>(l),static_cast<R&&>(r));
}

// Division

template<class L, class R>
decltype(auto) operator/( const DenseExpression<L>& l, const DenseExpression<R>& r){
    return DividesDenseExpression(static_cast<const L&>(l),static_cast<const R&>(r));
}

template<class L, class R>
decltype(auto) operator/( DenseExpression<L>&& l, const DenseExpression<R>& r){
    return DividesDenseExpression(static_cast<L&&>(l),static_cast<const R&>(r));
}

template<class L, class R>
decltype(auto) operator/( const DenseExpression<L>& l, DenseExpression<R>&& r){
    return DividesDenseExpression(static_cast<const L&>(l),static_cast<R&&>(r));
}

template<class L, class R>
decltype(auto) operator/( DenseExpression<L>&& l, DenseExpression<R>&& r){
    return DividesDenseExpression(static_cast<L&&>(l),static_cast<R&&>(r));
}

// =========================
// Unary Functions

// Functors

struct    Abs { template<class T> decltype(auto) operator()( const T& t) const { return std::abs(t);}};
struct    Sin { template<class T> decltype(auto) operator()( const T& t) const { return std::sin(t);}};
struct    Cos { template<class T> decltype(auto) operator()( const T& t) const { return std::cos(t);}};
struct    Tan { template<class T> decltype(auto) operator()( const T& t) const { return std::tan(t);}};
struct   Asin { template<class T> decltype(auto) operator()( const T& t) const { return std::asin(t);}};
struct   Acos { template<class T> decltype(auto) operator()( const T& t) const { return std::acos(t);}};
struct   Atan { template<class T> decltype(auto) operator()( const T& t) const { return std::atan(t);}};
struct   Sinh { template<class T> decltype(auto) operator()( const T& t) const { return std::sinh(t);}};
struct   Cosh { template<class T> decltype(auto) operator()( const T& t) const { return std::cosh(t);}};
struct   Tanh { template<class T> decltype(auto) operator()( const T& t) const { return std::tanh(t);}};
struct  Asinh { template<class T> decltype(auto) operator()( const T& t) const { return std::asinh(t);}};
struct  Acosh { template<class T> decltype(auto) operator()( const T& t) const { return std::acosh(t);}};
struct  Atanh { template<class T> decltype(auto) operator()( const T& t) const { return std::atanh(t);}};
struct   Sqrt { template<class T> decltype(auto) operator()( const T& t) const { return std::sqrt(t);}};
struct   Cbrt { template<class T> decltype(auto) operator()( const T& t) const { return std::cbrt(t);}};
struct    Exp { template<class T> decltype(auto) operator()( const T& t) const { return std::exp(t);}};
struct   Exp2 { template<class T> decltype(auto) operator()( const T& t) const { return std::exp2(t);}};
struct  Expm1 { template<class T> decltype(auto) operator()( const T& t) const { return std::expm1(t);}};
struct    Log { template<class T> decltype(auto) operator()( const T& t) const { return std::log(t);}};
struct   Log2 { template<class T> decltype(auto) operator()( const T& t) const { return std::log2(t);}};
struct  Log10 { template<class T> decltype(auto) operator()( const T& t) const { return std::log10(t);}};
struct  Log1p { template<class T> decltype(auto) operator()( const T& t) const { return std::log1p(t);}};
struct   Ceil { template<class T> decltype(auto) operator()( const T& t) const { return std::ceil(t);}};
struct  Floor { template<class T> decltype(auto) operator()( const T& t) const { return std::floor(t);}};
struct  Round { template<class T> decltype(auto) operator()( const T& t) const { return std::round(t);}};
struct    Erf { template<class T> decltype(auto) operator()( const T& t) const { return std::erf(t);}};
struct   Erfc { template<class T> decltype(auto) operator()( const T& t) const { return std::erfc(t);}};
struct Tgamma { template<class T> decltype(auto) operator()( const T& t) const { return std::tgamma(t);}};
struct Lgamma { template<class T> decltype(auto) operator()( const T& t) const { return std::lgamma(t);}};

// DenseExpressions

template<class T> using    AbsDenseExpression = ElementWiseDenseExpression<   Abs,T>;
template<class T> using    SinDenseExpression = ElementWiseDenseExpression<   Sin,T>;
template<class T> using    CosDenseExpression = ElementWiseDenseExpression<   Cos,T>;
template<class T> using    TanDenseExpression = ElementWiseDenseExpression<   Tan,T>;
template<class T> using   AsinDenseExpression = ElementWiseDenseExpression<  Asin,T>;
template<class T> using   AcosDenseExpression = ElementWiseDenseExpression<  Acos,T>;
template<class T> using   AtanDenseExpression = ElementWiseDenseExpression<  Atan,T>;
template<class T> using   SinhDenseExpression = ElementWiseDenseExpression<  Sinh,T>;
template<class T> using   CoshDenseExpression = ElementWiseDenseExpression<  Cosh,T>;
template<class T> using   TanhDenseExpression = ElementWiseDenseExpression<  Tanh,T>;
template<class T> using  AsinhDenseExpression = ElementWiseDenseExpression< Asinh,T>;
template<class T> using  AcoshDenseExpression = ElementWiseDenseExpression< Acosh,T>;
template<class T> using  AtanhDenseExpression = ElementWiseDenseExpression< Atanh,T>;
template<class T> using   SqrtDenseExpression = ElementWiseDenseExpression<  Sqrt,T>;
template<class T> using   CbrtDenseExpression = ElementWiseDenseExpression<  Cbrt,T>;
template<class T> using    ExpDenseExpression = ElementWiseDenseExpression<   Exp,T>;
template<class T> using   Exp2DenseExpression = ElementWiseDenseExpression<  Exp2,T>;
template<class T> using  Expm1DenseExpression = ElementWiseDenseExpression< Expm1,T>;
template<class T> using    LogDenseExpression = ElementWiseDenseExpression<   Log,T>;
template<class T> using   Log2DenseExpression = ElementWiseDenseExpression<  Log2,T>;
template<class T> using  Log10DenseExpression = ElementWiseDenseExpression< Log10,T>;
template<class T> using  Log1pDenseExpression = ElementWiseDenseExpression< Log1p,T>;
template<class T> using   CeilDenseExpression = ElementWiseDenseExpression<  Ceil,T>;
template<class T> using  FloorDenseExpression = ElementWiseDenseExpression< Floor,T>;
template<class T> using  RoundDenseExpression = ElementWiseDenseExpression< Round,T>;
template<class T> using    ErfDenseExpression = ElementWiseDenseExpression<   Erf,T>;
template<class T> using   ErfcDenseExpression = ElementWiseDenseExpression<  Erfc,T>;
template<class T> using TgammaDenseExpression = ElementWiseDenseExpression<Tgamma,T>;
template<class T> using LgammaDenseExpression = ElementWiseDenseExpression<Lgamma,T>;

// Functions

template<class T> decltype(auto)    abs( const DenseExpression<T>& t){ return    AbsDenseExpression(static_cast<const T&>(t));}
template<class T> decltype(auto)    sin( const DenseExpression<T>& t){ return    SinDenseExpression(static_cast<const T&>(t));}
template<class T> decltype(auto)    cos( const DenseExpression<T>& t){ return    CosDenseExpression(static_cast<const T&>(t));}
template<class T> decltype(auto)    tan( const DenseExpression<T>& t){ return    TanDenseExpression(static_cast<const T&>(t));}
template<class T> decltype(auto)   asin( const DenseExpression<T>& t){ return   AsinDenseExpression(static_cast<const T&>(t));}
template<class T> decltype(auto)   acos( const DenseExpression<T>& t){ return   AcosDenseExpression(static_cast<const T&>(t));}
template<class T> decltype(auto)   atan( const DenseExpression<T>& t){ return   AtanDenseExpression(static_cast<const T&>(t));}
template<class T> decltype(auto)   sinh( const DenseExpression<T>& t){ return   SinhDenseExpression(static_cast<const T&>(t));}
template<class T> decltype(auto)   cosh( const DenseExpression<T>& t){ return   CoshDenseExpression(static_cast<const T&>(t));}
template<class T> decltype(auto)   tanh( const DenseExpression<T>& t){ return   TanhDenseExpression(static_cast<const T&>(t));}
template<class T> decltype(auto)  asinh( const DenseExpression<T>& t){ return  AsinhDenseExpression(static_cast<const T&>(t));}
template<class T> decltype(auto)  acosh( const DenseExpression<T>& t){ return  AcoshDenseExpression(static_cast<const T&>(t));}
template<class T> decltype(auto)  atanh( const DenseExpression<T>& t){ return  AtanhDenseExpression(static_cast<const T&>(t));}
template<class T> decltype(auto)   sqrt( const DenseExpression<T>& t){ return   SqrtDenseExpression(static_cast<const T&>(t));}
template<class T> decltype(auto)   cbrt( const DenseExpression<T>& t){ return   CbrtDenseExpression(static_cast<const T&>(t));}
template<class T> decltype(auto)    exp( const DenseExpression<T>& t){ return    ExpDenseExpression(static_cast<const T&>(t));}
template<class T> decltype(auto)   exp2( const DenseExpression<T>& t){ return   Exp2DenseExpression(static_cast<const T&>(t));}
template<class T> decltype(auto)  expm1( const DenseExpression<T>& t){ return  Expm1DenseExpression(static_cast<const T&>(t));}
template<class T> decltype(auto)    log( const DenseExpression<T>& t){ return    LogDenseExpression(static_cast<const T&>(t));}
template<class T> decltype(auto)   log2( const DenseExpression<T>& t){ return   Log2DenseExpression(static_cast<const T&>(t));}
template<class T> decltype(auto)  log10( const DenseExpression<T>& t){ return  Log10DenseExpression(static_cast<const T&>(t));}
template<class T> decltype(auto)  log1p( const DenseExpression<T>& t){ return  Log1pDenseExpression(static_cast<const T&>(t));}
template<class T> decltype(auto)   ceil( const DenseExpression<T>& t){ return   CeilDenseExpression(static_cast<const T&>(t));}
template<class T> decltype(auto)  floor( const DenseExpression<T>& t){ return  FloorDenseExpression(static_cast<const T&>(t));}
template<class T> decltype(auto)  round( const DenseExpression<T>& t){ return  RoundDenseExpression(static_cast<const T&>(t));}
template<class T> decltype(auto)    erf( const DenseExpression<T>& t){ return    ErfDenseExpression(static_cast<const T&>(t));}
template<class T> decltype(auto)   erfc( const DenseExpression<T>& t){ return   ErfcDenseExpression(static_cast<const T&>(t));}
template<class T> decltype(auto) tgamma( const DenseExpression<T>& t){ return TgammaDenseExpression(static_cast<const T&>(t));}
template<class T> decltype(auto) lgamma( const DenseExpression<T>& t){ return LgammaDenseExpression(static_cast<const T&>(t));}

template<class T> decltype(auto)    abs( DenseExpression<T>&& t){ return    AbsDenseExpression(static_cast<T&&>(t));}
template<class T> decltype(auto)    sin( DenseExpression<T>&& t){ return    SinDenseExpression(static_cast<T&&>(t));}
template<class T> decltype(auto)    cos( DenseExpression<T>&& t){ return    CosDenseExpression(static_cast<T&&>(t));}
template<class T> decltype(auto)    tan( DenseExpression<T>&& t){ return    TanDenseExpression(static_cast<T&&>(t));}
template<class T> decltype(auto)   asin( DenseExpression<T>&& t){ return   AsinDenseExpression(static_cast<T&&>(t));}
template<class T> decltype(auto)   acos( DenseExpression<T>&& t){ return   AcosDenseExpression(static_cast<T&&>(t));}
template<class T> decltype(auto)   atan( DenseExpression<T>&& t){ return   AtanDenseExpression(static_cast<T&&>(t));}
template<class T> decltype(auto)   sinh( DenseExpression<T>&& t){ return   SinhDenseExpression(static_cast<T&&>(t));}
template<class T> decltype(auto)   cosh( DenseExpression<T>&& t){ return   CoshDenseExpression(static_cast<T&&>(t));}
template<class T> decltype(auto)   tanh( DenseExpression<T>&& t){ return   TanhDenseExpression(static_cast<T&&>(t));}
template<class T> decltype(auto)  asinh( DenseExpression<T>&& t){ return  AsinhDenseExpression(static_cast<T&&>(t));}
template<class T> decltype(auto)  acosh( DenseExpression<T>&& t){ return  AcoshDenseExpression(static_cast<T&&>(t));}
template<class T> decltype(auto)  atanh( DenseExpression<T>&& t){ return  AtanhDenseExpression(static_cast<T&&>(t));}
template<class T> decltype(auto)   sqrt( DenseExpression<T>&& t){ return   SqrtDenseExpression(static_cast<T&&>(t));}
template<class T> decltype(auto)   cbrt( DenseExpression<T>&& t){ return   CbrtDenseExpression(static_cast<T&&>(t));}
template<class T> decltype(auto)    exp( DenseExpression<T>&& t){ return    ExpDenseExpression(static_cast<T&&>(t));}
template<class T> decltype(auto)   exp2( DenseExpression<T>&& t){ return   Exp2DenseExpression(static_cast<T&&>(t));}
template<class T> decltype(auto)  expm1( DenseExpression<T>&& t){ return  Expm1DenseExpression(static_cast<T&&>(t));}
template<class T> decltype(auto)    log( DenseExpression<T>&& t){ return    LogDenseExpression(static_cast<T&&>(t));}
template<class T> decltype(auto)   log2( DenseExpression<T>&& t){ return   Log2DenseExpression(static_cast<T&&>(t));}
template<class T> decltype(auto)  log10( DenseExpression<T>&& t){ return  Log10DenseExpression(static_cast<T&&>(t));}
template<class T> decltype(auto)  log1p( DenseExpression<T>&& t){ return  Log1pDenseExpression(static_cast<T&&>(t));}
template<class T> decltype(auto)   ceil( DenseExpression<T>&& t){ return   CeilDenseExpression(static_cast<T&&>(t));}
template<class T> decltype(auto)  floor( DenseExpression<T>&& t){ return  FloorDenseExpression(static_cast<T&&>(t));}
template<class T> decltype(auto)  round( DenseExpression<T>&& t){ return  RoundDenseExpression(static_cast<T&&>(t));}
template<class T> decltype(auto)    erf( DenseExpression<T>&& t){ return    ErfDenseExpression(static_cast<T&&>(t));}
template<class T> decltype(auto)   erfc( DenseExpression<T>&& t){ return   ErfcDenseExpression(static_cast<T&&>(t));}
template<class T> decltype(auto) tgamma( DenseExpression<T>&& t){ return TgammaDenseExpression(static_cast<T&&>(t));}
template<class T> decltype(auto) lgamma( DenseExpression<T>&& t){ return LgammaDenseExpression(static_cast<T&&>(t));}

// =========================
// Binary/Ternary Functions

// Functors

struct Pow    { template<class X,class Y> decltype(auto) operator()( const X& x, const Y& y) const { return std::pow(x,y);}};
struct Atan2  { template<class X,class Y> decltype(auto) operator()( const X& x, const Y& y) const { return std::atan2(x,y);}};
struct Hypot  { template<class X,class Y> decltype(auto) operator()( const X& x, const Y& y) const { return std::hypot(x,y);}
                template<class X,class Y,class Z> decltype(auto) operator()( const X& x, const Y& y, const Z& z) const { return std::hypot(x,y,z);}};

// DenseExpressions

template<class X,class Y> using   PowDenseExpression = ElementWiseDenseExpression<  Pow,X,Y>;
template<class X,class Y> using Atan2DenseExpression = ElementWiseDenseExpression<Atan2,X,Y>;
template<class... Xs>     using HypotDenseExpression = ElementWiseDenseExpression<Hypot,Xs...>;

// DenseExpressions

template<class X,class Y>
decltype(auto) pow( const DenseExpression<X>& x, const DenseExpression<Y>& y){
    return PowDenseExpression(static_cast<const X&>(x),static_cast<const Y&>(y));
}

template<class X,class Y>
decltype(auto) pow( DenseExpression<X>&& x, const DenseExpression<Y>& y){
    return PowDenseExpression(static_cast<X&&>(x),static_cast<const Y&>(y));
}

template<class X,class Y>
decltype(auto) pow( const DenseExpression<X>& x, DenseExpression<Y>&& y){
    return PowDenseExpression(static_cast<const X&>(x),static_cast<Y&&>(y));
}

template<class X,class Y>
decltype(auto) pow( DenseExpression<X>&& x, DenseExpression<Y>&& y){
    return PowDenseExpression(static_cast<X&&>(x),static_cast<Y&&>(y));
}

template<class X,class Y>
decltype(auto) atan2( const DenseExpression<X>& x, const DenseExpression<Y>& y){
    return Atan2DenseExpression(static_cast<const X&>(x),static_cast<const Y&>(y));
}

template<class X,class Y>
decltype(auto) atan2( DenseExpression<X>&& x, const DenseExpression<Y>& y){
    return Atan2DenseExpression(static_cast<X&&>(x),static_cast<const Y&>(y));
}

template<class X,class Y>
decltype(auto) atan2( const DenseExpression<X>& x, DenseExpression<Y>&& y){
    return Atan2DenseExpression(static_cast<const X&>(x),static_cast<Y&&>(y));
}

template<class X,class Y>
decltype(auto) atan2( DenseExpression<X>&& x, DenseExpression<Y>&& y){
    return Atan2DenseExpression(static_cast<X&&>(x),static_cast<Y&&>(y));
}

template<class X,class Y>
decltype(auto) hypot( const DenseExpression<X>& x, const DenseExpression<Y>& y){
    return HypotDenseExpression(static_cast<const X&>(x),static_cast<const Y&>(y));
}

template<class X,class Y>
decltype(auto) hypot( DenseExpression<X>&& x, const DenseExpression<Y>& y){
    return HypotDenseExpression(static_cast<X&&>(x),static_cast<const Y&>(y));
}

template<class X,class Y>
decltype(auto) hypot( const DenseExpression<X>& x, DenseExpression<Y>&& y){
    return HypotDenseExpression(static_cast<const X&>(x),static_cast<Y&&>(y));
}

template<class X,class Y>
decltype(auto) hypot( DenseExpression<X>&& x, DenseExpression<Y>&& y){
    return HypotDenseExpression(static_cast<X&&>(x),static_cast<Y&&>(y));
}

template<class X,class Y,class Z>
decltype(auto) hypot( const DenseExpression<X>& x, const DenseExpression<Y>& y, const DenseExpression<Z>& z){
    return HypotDenseExpression(static_cast<const X&>(x),static_cast<const Y&>(y),static_cast<const Z&>(z));
}

template<class X,class Y,class Z>
decltype(auto) hypot( DenseExpression<X>&& x, const DenseExpression<Y>& y, const DenseExpression<Z>& z){
    return HypotDenseExpression(static_cast<X&&>(x),static_cast<const Y&>(y),static_cast<const Z&>(z));
}

template<class X,class Y,class Z>
decltype(auto) hypot( const DenseExpression<X>& x, DenseExpression<Y>&& y, const DenseExpression<Z>& z){
    return HypotDenseExpression(static_cast<const X&>(x),static_cast<Y&&>(y),static_cast<const Z&>(z));
}

template<class X,class Y,class Z>
decltype(auto) hypot( DenseExpression<X>&& x, DenseExpression<Y>&& y, const DenseExpression<Z>& z){
    return HypotDenseExpression(static_cast<X&&>(x),static_cast<Y&&>(y),static_cast<const Z&>(z));
}

template<class X,class Y,class Z>
decltype(auto) hypot( const DenseExpression<X>& x, const DenseExpression<Y>& y, DenseExpression<Z>&& z){
    return HypotDenseExpression(static_cast<const X&>(x),static_cast<const Y&>(y),static_cast<Z&&>(z));
}

template<class X,class Y,class Z>
decltype(auto) hypot( DenseExpression<X>&& x, const DenseExpression<Y>& y, DenseExpression<Z>&& z){
    return HypotDenseExpression(static_cast<X&&>(x),static_cast<const Y&>(y),static_cast<Z&&>(z));
}

template<class X,class Y,class Z>
decltype(auto) hypot( const DenseExpression<X>& x, DenseExpression<Y>&& y, DenseExpression<Z>&& z){
    return HypotDenseExpression(static_cast<const X&>(x),static_cast<Y&&>(y),static_cast<Z&&>(z));
}

template<class X,class Y,class Z>
decltype(auto) hypot( DenseExpression<X>&& x, DenseExpression<Y>&& y, DenseExpression<Z>&& z){
    return HypotDenseExpression(static_cast<X&&>(x),static_cast<Y&&>(y),static_cast<Z&&>(z));
}

// =========================
// Cumulative

// Expressions

template<class T> using  CumSumDenseExpression = CumulativeDenseExpression<Plus,T>;
template<class T> using CumProdDenseExpression = CumulativeDenseExpression<Multiplies,T>;

// Functions

template<class T, class StartT=int>
decltype(auto) cumsum( const DenseExpression<T>& t, const StartT& start = 0){
    return CumSumDenseExpression(static_cast<const T&>(t),start);
}

template<class T, class StartT=int>
decltype(auto) cumsum( DenseExpression<T>&& t, const StartT& start = 0){
    return CumSumDenseExpression(static_cast<T&&>(t),start);
}

template<class T, class StartT=int>
decltype(auto) cumprod( const DenseExpression<T>& t, const StartT& start = 1){
    return CumProdDenseExpression(static_cast<const T&>(t),start);
}

template<class T, class StartT=int>
decltype(auto) cumprod( const DenseExpression<T>&& t, const StartT& start = 1){
    return CumProdDenseExpression(static_cast<T&&>(t),start);
}

} // namespace
#endif
