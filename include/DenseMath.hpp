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

template<class T>
decltype(auto) operator-( const DenseExpression<T>& t){
    return NegateDenseExpression(static_cast<const T&>(t));
}

template<class T>
decltype(auto) operator-( DenseExpression<T>&& t){
    return NegateDenseExpression(static_cast<T&&>(t));
}

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
// Boolean Logic

// Functors

struct LogicalNot { template<class T> decltype(auto) operator()( const T& t) const { return !t; }};
struct LogicalAnd { template<class L,class R> decltype(auto) operator()( const L& l, const R& r) const { return l && r; }};
struct LogicalOr  { template<class L,class R> decltype(auto) operator()( const L& l, const R& r) const { return l || r; }};
struct LogicalEq  { template<class L,class R> decltype(auto) operator()( const L& l, const R& r) const { return l == r; }};
struct LogicalNeq { template<class L,class R> decltype(auto) operator()( const L& l, const R& r) const { return l != r; }};
struct LogicalGt  { template<class L,class R> decltype(auto) operator()( const L& l, const R& r) const { return l > r;  }};
struct LogicalLt  { template<class L,class R> decltype(auto) operator()( const L& l, const R& r) const { return l < r;  }};
struct LogicalGe  { template<class L,class R> decltype(auto) operator()( const L& l, const R& r) const { return l >= r; }};
struct LogicalLe  { template<class L,class R> decltype(auto) operator()( const L& l, const R& r) const { return l <= r; }};

// DenseExpressions

template<class T>         using LogicalNotDenseExpression = ElementWiseDenseExpression<LogicalNot,T>;
template<class L,class R> using LogicalAndDenseExpression = ElementWiseDenseExpression<LogicalAnd,L,R>;
template<class L,class R> using LogicalOrDenseExpression  = ElementWiseDenseExpression<LogicalOr,L,R>;
template<class L,class R> using LogicalEqDenseExpression  = ElementWiseDenseExpression<LogicalEq,L,R>;
template<class L,class R> using LogicalNeqDenseExpression = ElementWiseDenseExpression<LogicalNeq,L,R>;
template<class L,class R> using LogicalGtDenseExpression  = ElementWiseDenseExpression<LogicalGt,L,R>;
template<class L,class R> using LogicalLtDenseExpression  = ElementWiseDenseExpression<LogicalLt,L,R>;
template<class L,class R> using LogicalGeDenseExpression  = ElementWiseDenseExpression<LogicalGe,L,R>;
template<class L,class R> using LogicalLeDenseExpression  = ElementWiseDenseExpression<LogicalLe,L,R>;

// Operator overloads

template<class T>
decltype(auto) operator!( const DenseExpression<T>& t){
    return LogicalNotDenseExpression(static_cast<const T&>(t));
}

template<class T>
decltype(auto) operator!( DenseExpression<T>&& t){
    return LogicalNotDenseExpression(static_cast<T&&>(t));
}

template<class L, class R>
decltype(auto) operator&&( const DenseExpression<L>& l, const DenseExpression<R>& r){
    return LogicalAndDenseExpression(static_cast<const L&>(l),static_cast<const R&>(r));
}

template<class L, class R>
decltype(auto) operator&&( const DenseExpression<L>& l, DenseExpression<R>&& r){
    return LogicalAndDenseExpression(static_cast<const L&>(l),static_cast<R&&>(r));
}

template<class L, class R>
decltype(auto) operator&&( DenseExpression<L>&& l, const DenseExpression<R>& r){
    return LogicalAndDenseExpression(static_cast<L&&>(l),static_cast<const R&>(r));
}

template<class L, class R>
decltype(auto) operator&&( DenseExpression<L>&& l, DenseExpression<R>&& r){
    return LogicalAndDenseExpression(static_cast<L&&>(l),static_cast<R&&>(r));
}

template<class L, class R>
decltype(auto) operator||( const DenseExpression<L>& l, const DenseExpression<R>& r){
    return LogicalOrDenseExpression(static_cast<const L&>(l),static_cast<const R&>(r));
}

template<class L, class R>
decltype(auto) operator||( const DenseExpression<L>& l, DenseExpression<R>&& r){
    return LogicalOrDenseExpression(static_cast<const L&>(l),static_cast<R&&>(r));
}

template<class L, class R>
decltype(auto) operator||( DenseExpression<L>&& l, const DenseExpression<R>& r){
    return LogicalOrDenseExpression(static_cast<L&&>(l),static_cast<const R&>(r));
}

template<class L, class R>
decltype(auto) operator||( DenseExpression<L>&& l, DenseExpression<R>&& r){
    return LogicalOrDenseExpression(static_cast<L&&>(l),static_cast<R&&>(r));
}

template<class L, class R>
decltype(auto) operator==( const DenseExpression<L>& l, const DenseExpression<R>& r){
    return LogicalEqDenseExpression(static_cast<const L&>(l),static_cast<const R&>(r));
}

template<class L, class R>
decltype(auto) operator==( const DenseExpression<L>& l, DenseExpression<R>&& r){
    return LogicalEqDenseExpression(static_cast<const L&>(l),static_cast<R&&>(r));
}

template<class L, class R>
decltype(auto) operator==( DenseExpression<L>&& l, const DenseExpression<R>& r){
    return LogicalEqDenseExpression(static_cast<L&&>(l),static_cast<const R&>(r));
}

template<class L, class R>
decltype(auto) operator==( DenseExpression<L>&& l, DenseExpression<R>&& r){
    return LogicalEqDenseExpression(static_cast<L&&>(l),static_cast<R&&>(r));
}

template<class L, class R>
decltype(auto) operator!=( const DenseExpression<L>& l, const DenseExpression<R>& r){
    return LogicalNeqDenseExpression(static_cast<const L&>(l),static_cast<const R&>(r));
}

template<class L, class R>
decltype(auto) operator!=( const DenseExpression<L>& l, DenseExpression<R>&& r){
    return LogicalNeqDenseExpression(static_cast<const L&>(l),static_cast<R&&>(r));
}

template<class L, class R>
decltype(auto) operator!=( DenseExpression<L>&& l, const DenseExpression<R>& r){
    return LogicalNeqDenseExpression(static_cast<L&&>(l),static_cast<const R&>(r));
}

template<class L, class R>
decltype(auto) operator!=( DenseExpression<L>&& l, DenseExpression<R>&& r){
    return LogicalNeqDenseExpression(static_cast<L&&>(l),static_cast<R&&>(r));
}

template<class L, class R>
decltype(auto) operator>( const DenseExpression<L>& l, const DenseExpression<R>& r){
    return LogicalGtDenseExpression(static_cast<const L&>(l),static_cast<const R&>(r));
}

template<class L, class R>
decltype(auto) operator>( const DenseExpression<L>& l, DenseExpression<R>&& r){
    return LogicalGtDenseExpression(static_cast<const L&>(l),static_cast<R&&>(r));
}

template<class L, class R>
decltype(auto) operator>( DenseExpression<L>&& l, const DenseExpression<R>& r){
    return LogicalGtDenseExpression(static_cast<L&&>(l),static_cast<const R&>(r));
}

template<class L, class R>
decltype(auto) operator>( DenseExpression<L>&& l, DenseExpression<R>&& r){
    return LogicalGtDenseExpression(static_cast<L&&>(l),static_cast<R&&>(r));
}

template<class L, class R>
decltype(auto) operator<( const DenseExpression<L>& l, const DenseExpression<R>& r){
    return LogicalLtDenseExpression(static_cast<const L&>(l),static_cast<const R&>(r));
}

template<class L, class R>
decltype(auto) operator<( const DenseExpression<L>& l, DenseExpression<R>&& r){
    return LogicalLtDenseExpression(static_cast<const L&>(l),static_cast<R&&>(r));
}

template<class L, class R>
decltype(auto) operator<( DenseExpression<L>&& l, const DenseExpression<R>& r){
    return LogicalLtDenseExpression(static_cast<L&&>(l),static_cast<const R&>(r));
}

template<class L, class R>
decltype(auto) operator<( DenseExpression<L>&& l, DenseExpression<R>&& r){
    return LogicalLtDenseExpression(static_cast<L&&>(l),static_cast<R&&>(r));
}

template<class L, class R>
decltype(auto) operator>=( const DenseExpression<L>& l, const DenseExpression<R>& r){
    return LogicalGeDenseExpression(static_cast<const L&>(l),static_cast<const R&>(r));
}

template<class L, class R>
decltype(auto) operator>=( const DenseExpression<L>& l, DenseExpression<R>&& r){
    return LogicalGeDenseExpression(static_cast<const L&>(l),static_cast<R&&>(r));
}

template<class L, class R>
decltype(auto) operator>=( DenseExpression<L>&& l, const DenseExpression<R>& r){
    return LogicalGeDenseExpression(static_cast<L&&>(l),static_cast<const R&>(r));
}

template<class L, class R>
decltype(auto) operator>=( DenseExpression<L>&& l, DenseExpression<R>&& r){
    return LogicalGeDenseExpression(static_cast<L&&>(l),static_cast<R&&>(r));
}

template<class L, class R>
decltype(auto) operator<=( const DenseExpression<L>& l, const DenseExpression<R>& r){
    return LogicalLeDenseExpression(static_cast<const L&>(l),static_cast<const R&>(r));
}

template<class L, class R>
decltype(auto) operator<=( const DenseExpression<L>& l, DenseExpression<R>&& r){
    return LogicalLeDenseExpression(static_cast<const L&>(l),static_cast<R&&>(r));
}

template<class L, class R>
decltype(auto) operator<=( DenseExpression<L>&& l, const DenseExpression<R>& r){
    return LogicalLeDenseExpression(static_cast<L&&>(l),static_cast<const R&>(r));
}

template<class L, class R>
decltype(auto) operator<=( DenseExpression<L>&& l, DenseExpression<R>&& r){
    return LogicalLeDenseExpression(static_cast<L&&>(l),static_cast<R&&>(r));
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
// Reductions / Folds / Accumulations

// functors

struct Min { template<class T> decltype(auto) operator()( const T& x, const T& y) const { return (x < y ? x : y);}};
struct Max { template<class T> decltype(auto) operator()( const T& x, const T& y) const { return (x > y ? x : y);}};

// Functions

template<class F, class T, class ValueType>
decltype(auto) fold( const F& f, const DenseExpression<T>& t, const ValueType& start_val, std::size_t dim){
    static_assert(std::is_convertible< decltype(f(start_val,std::declval<typename std::remove_cvref_t<T>::value_type>())), std::remove_cvref_t<ValueType>>::value );
    return FoldDenseExpression(f,static_cast<const T&>(t),start_val,dim);
}

template<class F, class T, class ValueType>
decltype(auto) fold( const F& f, DenseExpression<T>&& t, const ValueType& start_val, std::size_t dim){
    static_assert(std::is_convertible< decltype(f(start_val,std::declval<typename std::remove_cvref_t<T>::value_type>())), std::remove_cvref_t<ValueType>>::value);
    return FoldDenseExpression(f,static_cast<T&&>(t),start_val,dim);
}

template<class F, class T>
decltype(auto) accumulate( const F& f, const DenseExpression<T>& t, std::size_t dim){
    return AccumulateDenseExpression(f,static_cast<const T&>(t),dim);
}

template<class F, class T>
decltype(auto) accumulate( const F& f, DenseExpression<T>&& t, std::size_t dim){
    return AccumulateDenseExpression(f,static_cast<const T&&>(t),dim);
}

template<class T>
decltype(auto) sum( const DenseExpression<T>& t, std::size_t dim){
    return accumulate(Plus{},static_cast<const T&>(t),dim);
}

template<class T>
decltype(auto) sum( DenseExpression<T>&& t, std::size_t dim){
    return accumulate(Plus{},static_cast<T&&>(t),dim);
}

template<class T>
decltype(auto) prod( const DenseExpression<T>& t, std::size_t dim){
    return accumulate(Multiplies{},static_cast<const T&>(t),dim);
}

template<class T>
decltype(auto) prod( DenseExpression<T>&& t, std::size_t dim){
    return accumulate(Multiplies{},static_cast<T&&>(t),dim);
}

template<class T>
decltype(auto) min( const DenseExpression<T>& t, std::size_t dim){
    return accumulate(Min{},static_cast<const T&>(t), dim);
}

template<class T>
decltype(auto) min( DenseExpression<T>&& t, std::size_t dim){
    return accumulate(Min{},static_cast<T&&>(t), dim);
}

template<class T>
decltype(auto) max( const DenseExpression<T>& t, std::size_t dim){
    return accumulate(Max{},static_cast<const T&>(t),dim);
}

template<class T>
decltype(auto) max( DenseExpression<T>&& t, std::size_t dim){
    return accumulate(Max{},static_cast<T&&>(t),dim);
}

// =========================
// BooleanFolds

struct AllOf {
    template<class Y>
    bool operator()( bool x, const Y& y) const {
        return x && y;
    }
    static constexpr bool start_bool = true;
    static constexpr bool early_exit_bool = false;
};

struct AnyOf {
    template<class Y>
    bool operator()( bool x, const Y& y) const {
        return x || y;
    }
    static constexpr bool start_bool = false;
    static constexpr bool early_exit_bool = true;
};

struct NoneOf {
    template<class Y>
    bool operator()( bool x, const Y& y) const {
        return x && !y;
    }
    static constexpr bool start_bool = true;
    static constexpr bool early_exit_bool = true;
};

template<class T>
decltype(auto) all_of( const DenseExpression<T>& t, std::size_t dim){
    return BooleanFoldDenseExpression(AllOf{},static_cast<const T&>(t),dim);
}

template<class T>
decltype(auto) all_of( DenseExpression<T>&& t, std::size_t dim){
    return BooleanFoldDenseExpression(AllOf{},static_cast<T&&>(t),dim);
}

template<class T>
decltype(auto) any_of( const DenseExpression<T>& t, std::size_t dim){
    return BooleanFoldDenseExpression(AnyOf{},static_cast<const T&>(t),dim);
}

template<class T>
decltype(auto) any_of( DenseExpression<T>&& t, std::size_t dim){
    return BooleanFoldDenseExpression(AnyOf{},static_cast<T&&>(t),dim);
}

template<class T>
decltype(auto) none_of( const DenseExpression<T>& t, std::size_t dim){
    return BooleanFoldDenseExpression(NoneOf{},static_cast<const T&>(t),dim);
}

template<class T>
decltype(auto) none_of( DenseExpression<T>&& t, std::size_t dim){
    return BooleanFoldDenseExpression(NoneOf{},static_cast<T&&>(t),dim);
}

// =========================
// Cumulative

template<class T>
decltype(auto) cumsum( const DenseExpression<T>& t, std::size_t dim){
    return eval(CumulativeDenseExpression(Plus{},static_cast<const T&>(t),dim));
}

template<class T>
decltype(auto) cumsum( DenseExpression<T>&& t, std::size_t dim){
    return eval(CumulativeDenseExpression(Plus{},static_cast<T&&>(t),dim));
}

template<class T>
decltype(auto) cumprod( const DenseExpression<T>& t, std::size_t dim){
    return eval(CumulativeDenseExpression(Multiplies{},static_cast<const T&>(t),dim));
}

template<class T>
decltype(auto) cumprod( DenseExpression<T>&& t, std::size_t dim){
    return eval(CumulativeDenseExpression(Multiplies{},static_cast<T&&>(t),dim));
}

// =========================
// Generators

// Zeros/Ones

template<class T> struct ZerosFunctor { constexpr T operator()( std::size_t ) const { return 0; } };
template<class T> struct OnesFunctor  { constexpr T operator()( std::size_t ) const { return 1; } };

template<std::ranges::sized_range Shape, class T=double>
decltype(auto) zeros( const Shape& shape) {
    return GeneratorExpression<ZerosFunctor<T>,ConstantGeneratorPolicy>( ZerosFunctor<T>(), shape);
}

template<class T=double>
decltype(auto) zeros( std::size_t N) {
    return zeros(std::array<std::size_t,1>{N});
}

template<std::ranges::sized_range Shape, class T=double>
decltype(auto) ones( const Shape& shape) {
    return GeneratorExpression<OnesFunctor<T>,ConstantGeneratorPolicy>( OnesFunctor<T>(), shape);
}

template<class T=double>
decltype(auto) ones( std::size_t N) {
    return ones(std::array<std::size_t,1>{N});
}

// linspace and logspace

template<class T>
class LinspaceFunctor {
    T _start;
    T _stop;
    std::size_t _N_minus_1;
public:
    LinspaceFunctor( const T& start, const T& stop, std::size_t N ) : _start(start), _stop(stop), _N_minus_1(N-1) {}
    T operator()( std::size_t idx) {
        // Not the most efficient, but should be robust.
        return _start*(static_cast<T>(_N_minus_1-idx)/_N_minus_1) + _stop*(static_cast<T>(idx)/_N_minus_1);
    }
};

template<class T> requires ( !std::is_integral<T>::value )
decltype(auto) linspace( const T& start, const T& stop, std::size_t N) {
    return GeneratorExpression<LinspaceFunctor<T>,LinearGeneratorPolicy>( LinspaceFunctor<T>(start,stop,N), std::array<std::size_t,1>{N});
}

template<class T> requires ( std::is_integral<T>::value )
decltype(auto) linspace( const T& start, const T& stop, std::size_t N) {
    return linspace<double>(start,stop,N);
}

template<class T>
class LogspaceFunctor {
    LinspaceFunctor<T> lin;
public:
    LogspaceFunctor( const T& start, const T& stop, std::size_t N ) : lin(start,stop,N) {}
    T operator()( std::size_t idx) {
        return std::pow(10,lin(idx));
    }
};

template<class T> requires ( !std::is_integral<T>::value )
decltype(auto) logspace( const T& start, const T& stop, std::size_t N) {
    return GeneratorExpression<LogspaceFunctor<T>,LinearGeneratorPolicy>( LogspaceFunctor<T>(start,stop,N), std::array<std::size_t,1>{N});
}

template<class T> requires ( std::is_integral<T>::value )
decltype(auto) logspace( const T& start, const T& stop, std::size_t N) {
    return logspace<double>(start,stop,N);
}

// Random number generation
// 
// Takes as first template parameter a random number distribution. See <random>.
// There is no standard concept for this, but at a minimum it must define a
// result_type, and must have the function `result_type operator()(std::size_t)`.
// It must also be copyable.
//
// A random number generator may be provided as a second argument. By default, we use
// the slow but high-quality mt19937_64. 
//
// RandomFunctors must be initialised with a distribution functor and a seed (typically
// std::size_t or unsigned). If no seed is provided, it will make use of the random
// number generator's default seed, plus 1 for each instance.

template<class Dist, std::uniform_random_bit_generator RNG = std::mt19937_64>
class RandomFunctor {

public:

    using dist_result_type = Dist::result_type;
    using rng_result_type = RNG::result_type;

private:

    Dist            _dist;
    RNG             _rng;
    static rng_result_type _static_seed;

public:
    
    RandomFunctor( const Dist& dist ) : 
        _dist(dist),
        _rng(_static_seed++)
    {}

    RandomFunctor( const RandomFunctor& other ) :
        _dist(other._dist),
        _rng(_static_seed++)
    {}

    dist_result_type operator()( std::size_t ) {
        return _dist(_rng);
    }
};

template<class Dist, std::uniform_random_bit_generator RNG>
RandomFunctor<Dist,RNG>::rng_result_type RandomFunctor<Dist,RNG>::_static_seed = RNG::default_seed;

template<class Dist, std::uniform_random_bit_generator RNG = std::mt19937_64, std::ranges::sized_range Range>
decltype(auto) random( const Dist& dist, const Range& range) {
    return GeneratorExpression<RandomFunctor<Dist,RNG>,ConstantGeneratorPolicy>( RandomFunctor<Dist,RNG>(dist), range);
}

template<class Dist, std::uniform_random_bit_generator RNG = std::mt19937_64>
decltype(auto) random( const Dist& dist, std::size_t N) {
    return random(dist,std::array<std::size_t,1>{N});
}

} // namespace
#endif
