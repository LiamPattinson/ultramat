#ifndef __ULTRA_MATH_HPP
#define __ULTRA_MATH_HPP

// Math
//
// Defines expressions for standard math functions.

#include "Expression.hpp"

namespace ultra {

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

// Expressions

template<class T> using    AbsExpression = ElementWiseExpression<   Abs,T>;
template<class T> using    SinExpression = ElementWiseExpression<   Sin,T>;
template<class T> using    CosExpression = ElementWiseExpression<   Cos,T>;
template<class T> using    TanExpression = ElementWiseExpression<   Tan,T>;
template<class T> using   AsinExpression = ElementWiseExpression<  Asin,T>;
template<class T> using   AcosExpression = ElementWiseExpression<  Acos,T>;
template<class T> using   AtanExpression = ElementWiseExpression<  Atan,T>;
template<class T> using   SinhExpression = ElementWiseExpression<  Sinh,T>;
template<class T> using   CoshExpression = ElementWiseExpression<  Cosh,T>;
template<class T> using   TanhExpression = ElementWiseExpression<  Tanh,T>;
template<class T> using  AsinhExpression = ElementWiseExpression< Asinh,T>;
template<class T> using  AcoshExpression = ElementWiseExpression< Acosh,T>;
template<class T> using  AtanhExpression = ElementWiseExpression< Atanh,T>;
template<class T> using   SqrtExpression = ElementWiseExpression<  Sqrt,T>;
template<class T> using   CbrtExpression = ElementWiseExpression<  Cbrt,T>;
template<class T> using    ExpExpression = ElementWiseExpression<   Exp,T>;
template<class T> using   Exp2Expression = ElementWiseExpression<  Exp2,T>;
template<class T> using  Expm1Expression = ElementWiseExpression< Expm1,T>;
template<class T> using    LogExpression = ElementWiseExpression<   Log,T>;
template<class T> using   Log2Expression = ElementWiseExpression<  Log2,T>;
template<class T> using  Log10Expression = ElementWiseExpression< Log10,T>;
template<class T> using  Log1pExpression = ElementWiseExpression< Log1p,T>;
template<class T> using   CeilExpression = ElementWiseExpression<  Ceil,T>;
template<class T> using  FloorExpression = ElementWiseExpression< Floor,T>;
template<class T> using  RoundExpression = ElementWiseExpression< Round,T>;
template<class T> using    ErfExpression = ElementWiseExpression<   Erf,T>;
template<class T> using   ErfcExpression = ElementWiseExpression<  Erfc,T>;
template<class T> using TgammaExpression = ElementWiseExpression<Tgamma,T>;
template<class T> using LgammaExpression = ElementWiseExpression<Lgamma,T>;

// Functions

template<class T> decltype(auto)    abs( const Expression<T>& t){ return    AbsExpression(static_cast<const T&>(t));}
template<class T> decltype(auto)    sin( const Expression<T>& t){ return    SinExpression(static_cast<const T&>(t));}
template<class T> decltype(auto)    cos( const Expression<T>& t){ return    CosExpression(static_cast<const T&>(t));}
template<class T> decltype(auto)    tan( const Expression<T>& t){ return    TanExpression(static_cast<const T&>(t));}
template<class T> decltype(auto)   asin( const Expression<T>& t){ return   AsinExpression(static_cast<const T&>(t));}
template<class T> decltype(auto)   acos( const Expression<T>& t){ return   AcosExpression(static_cast<const T&>(t));}
template<class T> decltype(auto)   atan( const Expression<T>& t){ return   AtanExpression(static_cast<const T&>(t));}
template<class T> decltype(auto)   sinh( const Expression<T>& t){ return   SinhExpression(static_cast<const T&>(t));}
template<class T> decltype(auto)   cosh( const Expression<T>& t){ return   CoshExpression(static_cast<const T&>(t));}
template<class T> decltype(auto)   tanh( const Expression<T>& t){ return   TanhExpression(static_cast<const T&>(t));}
template<class T> decltype(auto)  asinh( const Expression<T>& t){ return  AsinhExpression(static_cast<const T&>(t));}
template<class T> decltype(auto)  acosh( const Expression<T>& t){ return  AcoshExpression(static_cast<const T&>(t));}
template<class T> decltype(auto)  atanh( const Expression<T>& t){ return  AtanhExpression(static_cast<const T&>(t));}
template<class T> decltype(auto)   sqrt( const Expression<T>& t){ return   SqrtExpression(static_cast<const T&>(t));}
template<class T> decltype(auto)   cbrt( const Expression<T>& t){ return   CbrtExpression(static_cast<const T&>(t));}
template<class T> decltype(auto)    exp( const Expression<T>& t){ return    ExpExpression(static_cast<const T&>(t));}
template<class T> decltype(auto)   exp2( const Expression<T>& t){ return   Exp2Expression(static_cast<const T&>(t));}
template<class T> decltype(auto)  expm1( const Expression<T>& t){ return  Expm1Expression(static_cast<const T&>(t));}
template<class T> decltype(auto)    log( const Expression<T>& t){ return    LogExpression(static_cast<const T&>(t));}
template<class T> decltype(auto)   log2( const Expression<T>& t){ return   Log2Expression(static_cast<const T&>(t));}
template<class T> decltype(auto)  log10( const Expression<T>& t){ return  Log10Expression(static_cast<const T&>(t));}
template<class T> decltype(auto)  log1p( const Expression<T>& t){ return  Log1pExpression(static_cast<const T&>(t));}
template<class T> decltype(auto)   ceil( const Expression<T>& t){ return   CeilExpression(static_cast<const T&>(t));}
template<class T> decltype(auto)  floor( const Expression<T>& t){ return  FloorExpression(static_cast<const T&>(t));}
template<class T> decltype(auto)  round( const Expression<T>& t){ return  RoundExpression(static_cast<const T&>(t));}
template<class T> decltype(auto)    erf( const Expression<T>& t){ return    ErfExpression(static_cast<const T&>(t));}
template<class T> decltype(auto)   erfc( const Expression<T>& t){ return   ErfcExpression(static_cast<const T&>(t));}
template<class T> decltype(auto) tgamma( const Expression<T>& t){ return TgammaExpression(static_cast<const T&>(t));}
template<class T> decltype(auto) lgamma( const Expression<T>& t){ return LgammaExpression(static_cast<const T&>(t));}

template<class T> decltype(auto)    abs( Expression<T>&& t){ return    AbsExpression(static_cast<T&&>(t));}
template<class T> decltype(auto)    sin( Expression<T>&& t){ return    SinExpression(static_cast<T&&>(t));}
template<class T> decltype(auto)    cos( Expression<T>&& t){ return    CosExpression(static_cast<T&&>(t));}
template<class T> decltype(auto)    tan( Expression<T>&& t){ return    TanExpression(static_cast<T&&>(t));}
template<class T> decltype(auto)   asin( Expression<T>&& t){ return   AsinExpression(static_cast<T&&>(t));}
template<class T> decltype(auto)   acos( Expression<T>&& t){ return   AcosExpression(static_cast<T&&>(t));}
template<class T> decltype(auto)   atan( Expression<T>&& t){ return   AtanExpression(static_cast<T&&>(t));}
template<class T> decltype(auto)   sinh( Expression<T>&& t){ return   SinhExpression(static_cast<T&&>(t));}
template<class T> decltype(auto)   cosh( Expression<T>&& t){ return   CoshExpression(static_cast<T&&>(t));}
template<class T> decltype(auto)   tanh( Expression<T>&& t){ return   TanhExpression(static_cast<T&&>(t));}
template<class T> decltype(auto)  asinh( Expression<T>&& t){ return  AsinhExpression(static_cast<T&&>(t));}
template<class T> decltype(auto)  acosh( Expression<T>&& t){ return  AcoshExpression(static_cast<T&&>(t));}
template<class T> decltype(auto)  atanh( Expression<T>&& t){ return  AtanhExpression(static_cast<T&&>(t));}
template<class T> decltype(auto)   sqrt( Expression<T>&& t){ return   SqrtExpression(static_cast<T&&>(t));}
template<class T> decltype(auto)   cbrt( Expression<T>&& t){ return   CbrtExpression(static_cast<T&&>(t));}
template<class T> decltype(auto)    exp( Expression<T>&& t){ return    ExpExpression(static_cast<T&&>(t));}
template<class T> decltype(auto)   exp2( Expression<T>&& t){ return   Exp2Expression(static_cast<T&&>(t));}
template<class T> decltype(auto)  expm1( Expression<T>&& t){ return  Expm1Expression(static_cast<T&&>(t));}
template<class T> decltype(auto)    log( Expression<T>&& t){ return    LogExpression(static_cast<T&&>(t));}
template<class T> decltype(auto)   log2( Expression<T>&& t){ return   Log2Expression(static_cast<T&&>(t));}
template<class T> decltype(auto)  log10( Expression<T>&& t){ return  Log10Expression(static_cast<T&&>(t));}
template<class T> decltype(auto)  log1p( Expression<T>&& t){ return  Log1pExpression(static_cast<T&&>(t));}
template<class T> decltype(auto)   ceil( Expression<T>&& t){ return   CeilExpression(static_cast<T&&>(t));}
template<class T> decltype(auto)  floor( Expression<T>&& t){ return  FloorExpression(static_cast<T&&>(t));}
template<class T> decltype(auto)  round( Expression<T>&& t){ return  RoundExpression(static_cast<T&&>(t));}
template<class T> decltype(auto)    erf( Expression<T>&& t){ return    ErfExpression(static_cast<T&&>(t));}
template<class T> decltype(auto)   erfc( Expression<T>&& t){ return   ErfcExpression(static_cast<T&&>(t));}
template<class T> decltype(auto) tgamma( Expression<T>&& t){ return TgammaExpression(static_cast<T&&>(t));}
template<class T> decltype(auto) lgamma( Expression<T>&& t){ return LgammaExpression(static_cast<T&&>(t));}

// =========================
// Binary/Ternary Functions

// Functors

struct Pow    { template<class X,class Y> decltype(auto) operator()( const X& x, const Y& y) const { return std::pow(x,y);}};
struct Atan2  { template<class X,class Y> decltype(auto) operator()( const X& x, const Y& y) const { return std::atan2(x,y);}};
struct Hypot  { template<class X,class Y> decltype(auto) operator()( const X& x, const Y& y) const { return std::hypot(x,y);}
                template<class X,class Y,class Z> decltype(auto) operator()( const X& x, const Y& y, const Z& z) const { return std::hypot(x,y,z);}};

// Expressions

template<class X,class Y> using   PowExpression = ElementWiseExpression<  Pow,X,Y>;
template<class X,class Y> using Atan2Expression = ElementWiseExpression<Atan2,X,Y>;
template<class... Xs>     using HypotExpression = ElementWiseExpression<Hypot,Xs...>;

// Expressions

template<class X,class Y>
decltype(auto) pow( const Expression<X>& x, const Expression<Y>& y){
    return PowExpression(static_cast<const X&>(x),static_cast<const Y&>(y));
}

template<class X,class Y>
decltype(auto) pow( Expression<X>&& x, const Expression<Y>& y){
    return PowExpression(static_cast<X&&>(x),static_cast<const Y&>(y));
}

template<class X,class Y>
decltype(auto) pow( const Expression<X>& x, Expression<Y>&& y){
    return PowExpression(static_cast<const X&>(x),static_cast<Y&&>(y));
}

template<class X,class Y>
decltype(auto) pow( Expression<X>&& x, Expression<Y>&& y){
    return PowExpression(static_cast<X&&>(x),static_cast<Y&&>(y));
}

template<class X,class Y>
decltype(auto) atan2( const Expression<X>& x, const Expression<Y>& y){
    return Atan2Expression(static_cast<const X&>(x),static_cast<const Y&>(y));
}

template<class X,class Y>
decltype(auto) atan2( Expression<X>&& x, const Expression<Y>& y){
    return Atan2Expression(static_cast<X&&>(x),static_cast<const Y&>(y));
}

template<class X,class Y>
decltype(auto) atan2( const Expression<X>& x, Expression<Y>&& y){
    return Atan2Expression(static_cast<const X&>(x),static_cast<Y&&>(y));
}

template<class X,class Y>
decltype(auto) atan2( Expression<X>&& x, Expression<Y>&& y){
    return Atan2Expression(static_cast<X&&>(x),static_cast<Y&&>(y));
}

template<class X,class Y>
decltype(auto) hypot( const Expression<X>& x, const Expression<Y>& y){
    return HypotExpression(static_cast<const X&>(x),static_cast<const Y&>(y));
}

template<class X,class Y>
decltype(auto) hypot( Expression<X>&& x, const Expression<Y>& y){
    return HypotExpression(static_cast<X&&>(x),static_cast<const Y&>(y));
}

template<class X,class Y>
decltype(auto) hypot( const Expression<X>& x, Expression<Y>&& y){
    return HypotExpression(static_cast<const X&>(x),static_cast<Y&&>(y));
}

template<class X,class Y>
decltype(auto) hypot( Expression<X>&& x, Expression<Y>&& y){
    return HypotExpression(static_cast<X&&>(x),static_cast<Y&&>(y));
}

template<class X,class Y,class Z>
decltype(auto) hypot( const Expression<X>& x, const Expression<Y>& y, const Expression<Z>& z){
    return HypotExpression(static_cast<const X&>(x),static_cast<const Y&>(y),static_cast<const Z&>(z));
}

template<class X,class Y,class Z>
decltype(auto) hypot( Expression<X>&& x, const Expression<Y>& y, const Expression<Z>& z){
    return HypotExpression(static_cast<X&&>(x),static_cast<const Y&>(y),static_cast<const Z&>(z));
}

template<class X,class Y,class Z>
decltype(auto) hypot( const Expression<X>& x, Expression<Y>&& y, const Expression<Z>& z){
    return HypotExpression(static_cast<const X&>(x),static_cast<Y&&>(y),static_cast<const Z&>(z));
}

template<class X,class Y,class Z>
decltype(auto) hypot( Expression<X>&& x, Expression<Y>&& y, const Expression<Z>& z){
    return HypotExpression(static_cast<X&&>(x),static_cast<Y&&>(y),static_cast<const Z&>(z));
}

template<class X,class Y,class Z>
decltype(auto) hypot( const Expression<X>& x, const Expression<Y>& y, Expression<Z>&& z){
    return HypotExpression(static_cast<const X&>(x),static_cast<const Y&>(y),static_cast<Z&&>(z));
}

template<class X,class Y,class Z>
decltype(auto) hypot( Expression<X>&& x, const Expression<Y>& y, Expression<Z>&& z){
    return HypotExpression(static_cast<X&&>(x),static_cast<const Y&>(y),static_cast<Z&&>(z));
}

template<class X,class Y,class Z>
decltype(auto) hypot( const Expression<X>& x, Expression<Y>&& y, Expression<Z>&& z){
    return HypotExpression(static_cast<const X&>(x),static_cast<Y&&>(y),static_cast<Z&&>(z));
}

template<class X,class Y,class Z>
decltype(auto) hypot( Expression<X>&& x, Expression<Y>&& y, Expression<Z>&& z){
    return HypotExpression(static_cast<X&&>(x),static_cast<Y&&>(y),static_cast<Z&&>(z));
}


} // namespace
#endif
