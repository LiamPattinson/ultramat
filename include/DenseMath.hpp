#ifndef __ULTRA_DENSE_MATH_HPP
#define __ULTRA_DENSE_MATH_HPP

// DenseMath
//
// Defines expressions for standard math functions.

#include "DenseExpression.hpp"

namespace ultra {

// =========================
// Constants

constexpr const float inf = INFINITY;
constexpr const float Inf = INFINITY;
constexpr const float Infinity = INFINITY;
constexpr const float infty = INFINITY;
constexpr const float ninf = -inf;
constexpr const float Ninf = -inf;
constexpr const float NInfinity = -inf;
constexpr const double NaN = NAN;
constexpr const double pi = 3.1415926535897932384626433;
constexpr const double e = 2.71828182845904523536028747135266249775724709369995;
constexpr const double euler_gamma = 0.5772156649015328606065120900824024310421;

// =========================
// Arithmetic

// Functors

#define DENSE_MATH_UNARY_OPERATOR( Name, Op)\
\
    struct Name {\
        template<class T>\
        decltype(auto) operator()( const T& t) const {\
            return Op t;\
        }\
    };\
\
    template<class T>\
    using Name##DenseExpression = ElementWiseDenseExpression< Name, T>;\
\
    template<class T>\
    decltype(auto) operator Op ( const DenseExpression<T>& t){\
        return Name##DenseExpression(static_cast<const T&>(t));\
    }\
\
    template<class T>\
    decltype(auto) operator Op ( DenseExpression<T>&& t){\
        return Name##DenseExpression(static_cast<const T&>(t));\
    }

DENSE_MATH_UNARY_OPERATOR( Negate, -)
DENSE_MATH_UNARY_OPERATOR( LogicalNot, !)

#define DENSE_MATH_BINARY_OPERATOR( Name, Op)\
\
    struct Name {\
        template<class L, class R>\
        decltype(auto) operator()( const L& l, const R& r) const {\
            return l Op r;\
        }\
    };\
\
    template<class L,class R>\
    using Name##DenseExpression = ElementWiseDenseExpression< Name, L, R>;\
\
    template<class L, class R>\
    decltype(auto) operator Op ( const DenseExpression<L>& l, const DenseExpression<R>& r){\
        return Name##DenseExpression(static_cast<const L&>(l),static_cast<const R&>(r));\
    }\
\
    template<class L, class R>\
    decltype(auto) operator Op ( const DenseExpression<L>& l, DenseExpression<R>&& r){\
        return Name##DenseExpression(static_cast<const L&>(l),static_cast<R&&>(r));\
    }\
\
    template<class L, class R>\
    decltype(auto) operator Op ( DenseExpression<L>&& l, const DenseExpression<R>& r){\
        return Name##DenseExpression(static_cast<L&&>(l),static_cast<const R&>(r));\
    }\
\
    template<class L, class R>\
    decltype(auto) operator Op ( DenseExpression<L>&& l, DenseExpression<R>&& r){\
        return Name##DenseExpression(static_cast<L&&>(l),static_cast<R&&>(r));\
    }\
\
    template<class L, class R> requires number<R> \
    decltype(auto) operator Op ( const DenseExpression<L>& l, R r){\
        return static_cast<const L&>(l) Op DenseFixed<R,L::order(),1>(r);\
    }\
\
    template<class L, class R> requires number<R> \
    decltype(auto) operator Op ( DenseExpression<L>&& l, R r){\
        return static_cast<L&&>(l) Op DenseFixed<R,L::order(),1>(r);\
    }\
\
    template<class L, class R> requires number<L> \
    decltype(auto) operator Op ( L l, const DenseExpression<R>& r){\
        return DenseFixed<L,R::order(),1>(l) Op static_cast<const R&>(r);\
    }\
\
    template<class L, class R> requires number<L> \
    decltype(auto) operator Op ( L l, DenseExpression<R>&& r){\
        return DenseFixed<L,R::order(),1>(l) Op static_cast<R&&>(r);\
    }

DENSE_MATH_BINARY_OPERATOR( Plus, +)
DENSE_MATH_BINARY_OPERATOR( Minus, -)
DENSE_MATH_BINARY_OPERATOR( Multiplies, *)
DENSE_MATH_BINARY_OPERATOR( Divides, /)
DENSE_MATH_BINARY_OPERATOR( Modulus, %)
DENSE_MATH_BINARY_OPERATOR( LogicalAnd, &&)
DENSE_MATH_BINARY_OPERATOR( LogicalOr, ||)
DENSE_MATH_BINARY_OPERATOR( LogicalEq, ==)
DENSE_MATH_BINARY_OPERATOR( LogicalNeq, !=)
DENSE_MATH_BINARY_OPERATOR( LogicalGt, >)
DENSE_MATH_BINARY_OPERATOR( LogicalLt, <)
DENSE_MATH_BINARY_OPERATOR( LogicalGe, >=)
DENSE_MATH_BINARY_OPERATOR( LogicalLe, <=)

// =========================
// Unary Functions

#define DENSE_MATH_UNARY_FUNCTION( Name, FuncName, InternalFunc )\
\
    struct Name {\
        template<class T>\
        decltype(auto) operator()( const T& t) const {\
            return InternalFunc(t);\
        }\
    };\
\
    template<class T>\
    using Name##DenseExpression = ElementWiseDenseExpression< Name, T>;\
\
    template<class T>\
    decltype(auto) FuncName ( const DenseExpression<T>& t){\
        return Name##DenseExpression(static_cast<const T&>(t));\
    }\
\
    template<class T>\
    decltype(auto) FuncName ( DenseExpression<T>&& t){\
        return Name##DenseExpression(static_cast<T&&>(t));\
    }\
\
    template<class T> requires number<T> \
    decltype(auto) FuncName ( T t) {\
        return InternalFunc(t);\
    }

DENSE_MATH_UNARY_FUNCTION(  Abs, abs, std::abs )
DENSE_MATH_UNARY_FUNCTION(  Sin, sin, std::sin )
DENSE_MATH_UNARY_FUNCTION(  Cos, cos, std::cos )
DENSE_MATH_UNARY_FUNCTION(  Tan, tan, std::tan )
DENSE_MATH_UNARY_FUNCTION(  Asin, asin , std::asin )
DENSE_MATH_UNARY_FUNCTION(  Acos, acos , std::acos )
DENSE_MATH_UNARY_FUNCTION(  Atan, atan , std::atan )
DENSE_MATH_UNARY_FUNCTION(  Sinh, sinh , std::sinh )
DENSE_MATH_UNARY_FUNCTION(  Cosh, cosh , std::cosh )
DENSE_MATH_UNARY_FUNCTION(  Tanh, tanh , std::tanh )
DENSE_MATH_UNARY_FUNCTION(  Asinh, asinh , std::asinh )
DENSE_MATH_UNARY_FUNCTION(  Acosh, acosh , std::acosh )
DENSE_MATH_UNARY_FUNCTION(  Atanh, atanh , std::atanh )
DENSE_MATH_UNARY_FUNCTION(  Sqrt, sqrt , std::sqrt )
DENSE_MATH_UNARY_FUNCTION(  Cbrt, cbrt , std::cbrt )
DENSE_MATH_UNARY_FUNCTION(  Exp, exp , std::exp )
DENSE_MATH_UNARY_FUNCTION(  Exp2, exp2 , std::exp2 )
DENSE_MATH_UNARY_FUNCTION(  Expm1, expm1 , std::expm1 )
DENSE_MATH_UNARY_FUNCTION(  Log, log , std::log )
DENSE_MATH_UNARY_FUNCTION(  Log2, log2 , std::log2 )
DENSE_MATH_UNARY_FUNCTION(  Log10, log10 , std::log10 )
DENSE_MATH_UNARY_FUNCTION(  Log1p, log1p , std::log1p )
DENSE_MATH_UNARY_FUNCTION(  Ceil, ceil , std::ceil )
DENSE_MATH_UNARY_FUNCTION(  Floor, floor , std::floor )
DENSE_MATH_UNARY_FUNCTION(  Trunc, trunc , std::trunc )
DENSE_MATH_UNARY_FUNCTION(  Round, round , std::round )
DENSE_MATH_UNARY_FUNCTION(  Erf, erf , std::erf )
DENSE_MATH_UNARY_FUNCTION(  Erfc, erfc , std::erfc )
DENSE_MATH_UNARY_FUNCTION(  Tgamma, tgamma , std::tgamma )
DENSE_MATH_UNARY_FUNCTION(  Lgamma, lgamma , std::lgamma )
DENSE_MATH_UNARY_FUNCTION(  SignBit, signbit , std::signbit )
DENSE_MATH_UNARY_FUNCTION(  IsFinite, isfinite , std::isfinite )
DENSE_MATH_UNARY_FUNCTION(  IsInf, isinf , std::isinf )
DENSE_MATH_UNARY_FUNCTION(  IsNaN, isnan , std::isnan )
DENSE_MATH_UNARY_FUNCTION(  IsNormal, isnormal , std::isnormal )
DENSE_MATH_UNARY_FUNCTION(  Real, real , std::real )
DENSE_MATH_UNARY_FUNCTION(  Imag, imag , std::imag )
DENSE_MATH_UNARY_FUNCTION(  Arg, arg , std::arg )
DENSE_MATH_UNARY_FUNCTION(  Norm, norm , std::norm )
DENSE_MATH_UNARY_FUNCTION(  Conj, conj , std::conj )

template<std::floating_point T>
T _to_radians( const T& t){
    constexpr const double factor = pi/180;
    return t*factor;
}

template<std::floating_point T>
T _to_degrees( const T& t){
    constexpr const double factor = 180/pi;
    return t*factor;
}

template<class T>
T _square( const T& t){
    return t*t;
}

template<class T>
T _cube( const T& t){
    return t*t*t;
}

DENSE_MATH_UNARY_FUNCTION(  ToRadians, to_radians , _to_radians )
DENSE_MATH_UNARY_FUNCTION(  ToDegrees, to_degress , _to_degrees )
DENSE_MATH_UNARY_FUNCTION(  Square, square , _square )
DENSE_MATH_UNARY_FUNCTION(  Cube, cube , _cube )

#define DENSE_MATH_UNARY_COMPLEX_FUNC(NAME,FUNC)\
template<class T> auto _complex_##FUNC ( const T& t){ return std::FUNC(static_cast<complex_upcast<T>>(t)); }\
DENSE_MATH_UNARY_FUNCTION(  Complex##NAME, complex_##FUNC , _complex_##FUNC )

DENSE_MATH_UNARY_COMPLEX_FUNC(Sqrt,sqrt)
DENSE_MATH_UNARY_COMPLEX_FUNC(Log,log)
DENSE_MATH_UNARY_COMPLEX_FUNC(Log10,log10)
DENSE_MATH_UNARY_COMPLEX_FUNC(Acos,acos)
DENSE_MATH_UNARY_COMPLEX_FUNC(Asin,asin)
DENSE_MATH_UNARY_COMPLEX_FUNC(Atanh,atanh)

// =========================
// Binary Functions

#define DENSE_MATH_BINARY_FUNCTION( Name, FuncName, InternalFunc)\
\
    struct Name {\
        template<class L, class R>\
        decltype(auto) operator()( const L& l, const R& r) const {\
            return InternalFunc(l,r);\
        }\
    };\
\
    template<class L,class R>\
    using Name##DenseExpression = ElementWiseDenseExpression< Name, L, R>;\
\
    template<class L, class R>\
    decltype(auto) FuncName ( const DenseExpression<L>& l, const DenseExpression<R>& r){\
        return Name##DenseExpression(static_cast<const L&>(l),static_cast<const R&>(r));\
    }\
\
    template<class L, class R>\
    decltype(auto) FuncName ( const DenseExpression<L>& l, DenseExpression<R>&& r){\
        return Name##DenseExpression(static_cast<const L&>(l),static_cast<R&&>(r));\
    }\
\
    template<class L, class R>\
    decltype(auto) FuncName ( DenseExpression<L>&& l, const DenseExpression<R>& r){\
        return Name##DenseExpression(static_cast<L&&>(l),static_cast<const R&>(r));\
    }\
\
    template<class L, class R>\
    decltype(auto) FuncName ( DenseExpression<L>&& l, DenseExpression<R>&& r){\
        return Name##DenseExpression(static_cast<L&&>(l),static_cast<R&&>(r));\
    }\
\
    template<class L, class R> requires number<R> \
    decltype(auto) FuncName ( const DenseExpression<L>& l, R r){\
        return FuncName(static_cast<const L&>(l),DenseFixed<R,L::order(),1>(r));\
    }\
\
    template<class L, class R> requires number<R> \
    decltype(auto) FuncName ( DenseExpression<L>&& l, R r){\
        return FuncName(static_cast<L&&>(l),DenseFixed<R,L::order(),1>(r));\
    }\
\
    template<class L, class R> requires number<L> \
    decltype(auto) FuncName ( L l, const DenseExpression<R>& r){\
        return FuncName(DenseFixed<L,R::order(),1>(l),static_cast<const R&>(r));\
    }\
\
    template<class L, class R> requires number<L> \
    decltype(auto) FuncName ( L l, DenseExpression<R>&& r){\
        return FuncName(DenseFixed<L,R::order(),1>(l),static_cast<R&&>(r));\
    }\
\
    template<class L, class R> requires number<L> && number<R>\
    decltype(auto) FuncName ( L l, R r){\
        return InternalFunc(l,r);\
    }\

DENSE_MATH_BINARY_FUNCTION(Pow,pow,std::pow)
DENSE_MATH_BINARY_FUNCTION(Atan2,atan2,std::atan2)
DENSE_MATH_BINARY_FUNCTION(Hypot2,hypot,std::hypot)
DENSE_MATH_BINARY_FUNCTION(CopySign,copysign,std::copysign)
DENSE_MATH_BINARY_FUNCTION(Polar,polar,std::polar)

template<class T, class U>
auto _complex_pow ( const T& t, const U& u){
    return std::pow(static_cast<complex_upcast<T>>(t),static_cast<complex_upcast<U>>(u));
}

DENSE_MATH_BINARY_FUNCTION(ComplexPow,complex_pow,_complex_pow)

// =========================
// Ternary Functions

#define DENSE_MATH_TERNARY_FUNCTION_PREAMBLE( Name, FuncName, InternalFunc)\
\
    struct Name {\
        template<class X, class Y, class Z>\
        decltype(auto) operator()( const X& x, const Y& y, const Z& z) const {\
            return InternalFunc(x,y,z);\
        }\
    };\
\
    template<class X,class Y,class Z>\
    using Name##DenseExpression = ElementWiseDenseExpression< Name, X, Y, Z>;                                                                   

#define DENSE_MATH_TERNARY_FUNCTION_DEFINITIONS( Name, FuncName, InternalFunc)\
    template<class X, class Y,class Z>\
    decltype(auto) FuncName ( const DenseExpression<X>& x, const DenseExpression<Y>& y, const DenseExpression<Z>& z){\
        return Name##DenseExpression(static_cast<const X&>(x),static_cast<const Y&>(y),static_cast<const Z&>(z));\
    }\
\
    template<class X, class Y,class Z>\
    decltype(auto) FuncName ( DenseExpression<X>&& x, const DenseExpression<Y>& y, const DenseExpression<Z>& z){\
        return Name##DenseExpression(static_cast<X&&>(x),static_cast<const Y&>(y),static_cast<const Z&>(z));\
    }\
\
    template<class X, class Y,class Z>\
    decltype(auto) FuncName ( const DenseExpression<X>& x, DenseExpression<Y>&& y, const DenseExpression<Z>& z){\
        return Name##DenseExpression(static_cast<const X&>(x),static_cast<Y&&>(y),static_cast<const Z&>(z));\
    }\
\
    template<class X, class Y,class Z>\
    decltype(auto) FuncName ( const DenseExpression<X>& x, const DenseExpression<Y>& y, DenseExpression<Z>&& z){\
        return Name##DenseExpression(static_cast<const X&>(x),static_cast<const Y&>(y),static_cast<Z&&>(z));\
    }\
\
    template<class X, class Y,class Z>\
    decltype(auto) FuncName ( DenseExpression<X>&& x, DenseExpression<Y>&& y, const DenseExpression<Z>& z){\
        return Name##DenseExpression(static_cast<X&&>(x),static_cast<Y&&>(y),static_cast<const Z&>(z));\
    }\
\
    template<class X, class Y,class Z>\
    decltype(auto) FuncName ( DenseExpression<X>&& x, const DenseExpression<Y>& y, DenseExpression<Z>&& z){\
        return Name##DenseExpression(static_cast<X&&>(x),static_cast<const Y&>(y),static_cast<Z&&>(z));\
    }\
\
    template<class X, class Y,class Z>\
    decltype(auto) FuncName ( const DenseExpression<X>& x, DenseExpression<Y>&& y, DenseExpression<Z>&& z){\
        return Name##DenseExpression(static_cast<const X&>(x),static_cast<Y&&>(y),static_cast<Z&&>(z));\
    }\
\
    template<class X, class Y,class Z>\
    decltype(auto) FuncName ( DenseExpression<X>&& x, DenseExpression<Y>&& y, DenseExpression<Z>&& z){\
        return Name##DenseExpression(static_cast<X&&>(x),static_cast<Y&&>(y),static_cast<Z&&>(z));\
    }\
\
    template<class X, class Y,class Z> requires number<X> \
    decltype(auto) FuncName ( X x, const DenseExpression<Y>& y, const DenseExpression<Z>& z){\
        return FuncName(DenseFixed<X,Y::order(),1>(x),static_cast<const Y&>(y),static_cast<const Z&>(z));\
    }\
\
    template<class X, class Y,class Z> requires number<X> \
    decltype(auto) FuncName ( X x, DenseExpression<Y>&& y, const DenseExpression<Z>& z){\
        return FuncName(DenseFixed<X,Y::order(),1>(x),static_cast<Y&&>(y),static_cast<const Z&>(z));\
    }\
\
    template<class X, class Y,class Z> requires number<X> \
    decltype(auto) FuncName ( X x, const DenseExpression<Y>& y, DenseExpression<Z>&& z){\
        return FuncName(DenseFixed<X,Y::order(),1>(x),static_cast<const Y&>(y),static_cast<Z&&>(z));\
    }\
\
    template<class X, class Y,class Z> requires number<X> \
    decltype(auto) FuncName ( X x, DenseExpression<Y>&& y, DenseExpression<Z>&& z){\
        return FuncName(DenseFixed<X,Y::order(),1>(x),static_cast<Y&&>(y),static_cast<Z&&>(z));\
    }\
\
    template<class X, class Y,class Z> requires number<Y> \
    decltype(auto) FuncName ( const DenseExpression<X>& x, Y y, const DenseExpression<Z>& z){\
        return FuncName(static_cast<const X&>(x),DenseFixed<Y,X::order(),1>(y),static_cast<const Z&>(z));\
    }\
\
    template<class X, class Y,class Z> requires number<Y> \
    decltype(auto) FuncName ( DenseExpression<X>&& x, Y y, const DenseExpression<Z>& z){\
        return FuncName(static_cast<X&&>(x),DenseFixed<Y,X::order(),1>(y),static_cast<const Z&>(z));\
    }\
\
    template<class X, class Y,class Z> requires number<Y> \
    decltype(auto) FuncName ( const DenseExpression<X>& x, Y y, DenseExpression<Z>&& z){\
        return FuncName(static_cast<const X&>(x),DenseFixed<Y,X::order(),1>(y),static_cast<Z&&>(z));\
    }\
\
    template<class X, class Y,class Z> requires number<Y> \
    decltype(auto) FuncName ( DenseExpression<X>&& x, Y y, DenseExpression<Z>&& z){\
        return FuncName(static_cast<X&&>(x),DenseFixed<Y,X::order(),1>(y),static_cast<Z&&>(z));\
    }\
\
    template<class X, class Y,class Z> requires number<Z> \
    decltype(auto) FuncName ( const DenseExpression<X>& x, const DenseExpression<Y>& y, Z z){\
        return FuncName(static_cast<const X&>(x),static_cast<const Y&>(y),DenseFixed<Z,X::order(),1>(z));\
    }\
\
    template<class X, class Y,class Z> requires number<Z> \
    decltype(auto) FuncName ( DenseExpression<X>&& x, const DenseExpression<Y>&& y, Z z){\
        return FuncName(static_cast<X&&>(x),static_cast<const Y&>(y),DenseFixed<Z,X::order(),1>(z));\
    }\
\
    template<class X, class Y,class Z> requires number<Z> \
    decltype(auto) FuncName ( const DenseExpression<X>& x, DenseExpression<Y>&& y, Z z){\
        return FuncName(static_cast<const X&>(x),static_cast<Y&&>(y),DenseFixed<Z,X::order(),1>(z));\
    }\
\
    template<class X, class Y,class Z> requires number<Z> \
    decltype(auto) FuncName ( DenseExpression<X>&& x, DenseExpression<Y>&& y, Z z){\
        return FuncName(static_cast<X&&>(x),static_cast<Y&&>(y),DenseFixed<Z,X::order(),1>(z));\
    }\
\
    template<class X, class Y,class Z> requires number<X> && number<Y>\
    decltype(auto) FuncName ( X x, Y y, const DenseExpression<Z>& z){\
        return FuncName(DenseFixed<X,Z::order(),1>(x),DenseFixed<Y,Z::order(),1>(y),static_cast<const Z&>(z));\
    }\
\
    template<class X, class Y,class Z> requires number<X> && number<Y>\
    decltype(auto) FuncName ( X x, Y y, DenseExpression<Z>&& z){\
        return FuncName(DenseFixed<X,Z::order(),1>(x),DenseFixed<Y,Z::order(),1>(y),static_cast<Z&&>(z));\
    }\
\
    template<class X, class Y,class Z> requires number<X> && number<Z>\
    decltype(auto) FuncName ( X x, const DenseExpression<Y>& y, Z z){\
        return FuncName(DenseFixed<X,Y::order(),1>(x),static_cast<const Y&>(y),DenseFixed<Z,Y::order(),1>(z));\
    }\
\
    template<class X, class Y,class Z> requires number<X> && number<Z>\
    decltype(auto) FuncName ( X x, DenseExpression<Y>&& y, Z z){\
        return FuncName(DenseFixed<X,Y::order(),1>(x),static_cast<Y&&>(y),DenseFixed<Z,Y::order(),1>(z));\
    }\
\
    template<class X, class Y,class Z> requires number<Y> && number<Z>\
    decltype(auto) FuncName ( const DenseExpression<X>& x, Y y, Z z){\
        return FuncName(static_cast<const X&>(x),DenseFixed<Y,X::order(),1>(y),DenseFixed<Z,X::order(),1>(z));\
    }\
\
    template<class X, class Y,class Z> requires number<Y> && number<Z>\
    decltype(auto) FuncName ( DenseExpression<X>&& x, Y y, Z z){\
        return FuncName(static_cast<X&&>(x),DenseFixed<Y,X::order(),1>(y),DenseFixed<Z,X::order(),1>(z));\
    }\
\
    template<class X, class Y,class Z> requires number<X> && number<Y> && number<Z>\
    decltype(auto) FuncName ( X x, Y y, Z z){\
        return InternalFunc(x,y,z);\
    }

#define DENSE_MATH_TERNARY_FUNCTION( Name, FuncName, InternalFunc)\
    DENSE_MATH_TERNARY_FUNCTION_PREAMBLE( Name, FuncName, InternalFunc)\
    DENSE_MATH_TERNARY_FUNCTION_DEFINITIONS( Name, FuncName, InternalFunc)\

DENSE_MATH_TERNARY_FUNCTION(Hypot3,hypot,std::hypot)
DENSE_MATH_TERNARY_FUNCTION(MultiplyAdd,multiply_add,std::fma)

template<class Cond, class L, class R>
decltype(L()+R()) _where( const Cond& c, const L& l, const R& r){
    return c ? l : r;
}

DENSE_MATH_TERNARY_FUNCTION_DEFINITIONS(Where,where,_where)

// =========================
// Reductions / Folds / Accumulations

// functors

struct Min { template<class T> T operator()( const T& x, const T& y) const { return x < y ? x : y;}};
struct Max { template<class T> T operator()( const T& x, const T& y) const { return x > y ? x : y;}};

#ifndef ULTRA_PAIRWISE_SUM_BASE_CASE 
#define ULTRA_PAIRWISE_SUM_BASE_CASE 10
#endif

struct PairwiseSum {
    template<class Begin, class End, class T>
    T operator()( Begin it, End end, const T& start){
        std::ptrdiff_t size = end - it;
        T result = start;
        if( size < ULTRA_PAIRWISE_SUM_BASE_CASE ){
            for(;it != end; ++it) result += *it;
        } else {
            result += operator()(it,it+size/2,T(0)) + operator()(it+size/2,end,T(0));
        }
        return result;
    }
};

struct KahanSum {
    // Computes Kahan Summation, technically the Neumaier sum (improved Kahan-Babuska)
    // Very precise, though much slower than the naive implementatin or PairwiseSum.
    template<class Begin, class End, class T>
    T operator()( Begin it, End end, const T& start){
        T sum = start;
        T c = 0;
        for(; it != end; ++it){
            T x = *it;
            volatile T t = sum + x;
            if( std::fabs(sum) >= std::fabs(x) ){
                volatile T z = sum - t;
                c += z + x;
            } else {
                volatile T z = x - t;
                c += z + sum;
            }
            sum = t;
        }
        return sum + c;
    }
};

// General Functions

template<class F, class T, class ValueType>
requires (std::is_convertible< decltype(std::declval<F>()(std::declval<std::remove_cvref_t<T>>().begin(), std::declval<std::remove_cvref_t<T>>().end(), std::declval<ValueType>())), std::remove_cvref_t<ValueType>>::value )
decltype(auto) fold( const F& f, const DenseExpression<T>& t, const ValueType& start_val, std::size_t dim=0){
    return GeneralFoldDenseExpression(f,static_cast<const T&>(t),start_val,dim);
}

template<class F, class T, class ValueType>
requires (std::is_convertible< decltype(std::declval<F>()(std::declval<std::remove_cvref_t<T>>().begin(), std::declval<std::remove_cvref_t<T>>().end(), std::declval<ValueType>())), std::remove_cvref_t<ValueType>>::value )
decltype(auto) fold( const F& f, DenseExpression<T>&& t, const ValueType& start_val, std::size_t dim=0){
    return GeneralFoldDenseExpression(f,static_cast<T&&>(t),start_val,dim);
}

template<class F, class T, class ValueType>
requires (std::is_convertible< decltype(std::declval<F>()(std::declval<ValueType>(),std::declval<typename std::remove_cvref_t<T>::value_type>())), std::remove_cvref_t<ValueType>>::value )
decltype(auto) fold( const F& f, const DenseExpression<T>& t, const ValueType& start_val, std::size_t dim=0){
    return LinearFoldDenseExpression(f,static_cast<const T&>(t),start_val,dim);
}

template<class F, class T, class ValueType>
requires (std::is_convertible< decltype(std::declval<F>()(std::declval<ValueType>(),std::declval<typename std::remove_cvref_t<T>::value_type>())), std::remove_cvref_t<ValueType>>::value )
decltype(auto) fold( const F& f, DenseExpression<T>&& t, const ValueType& start_val, std::size_t dim=0){
    return LinearFoldDenseExpression(f,static_cast<T&&>(t),start_val,dim);
}

template<class F, class T>
decltype(auto) accumulate( const F& f, const DenseExpression<T>& t, std::size_t dim=0){
    return AccumulateDenseExpression(f,static_cast<const T&>(t),dim);
}

template<class F, class T>
decltype(auto) accumulate( const F& f, DenseExpression<T>&& t, std::size_t dim=0){
    return AccumulateDenseExpression(f,static_cast<T&&>(t),dim);
}

// sum
// * fast_sum -> Naively accumulate into a single variable. Fastest, but most susceptible to numerical errors.
// * pairwise_sum -> Sum recursively. Almost as fast, much less error.
// * precise_sum -> Kahan summation. Slowest sum, but most precise
// Pairwise sum is the default.

template<class T>
decltype(auto) fast_sum( const DenseExpression<T>& t, std::size_t dim=0){
    return accumulate(Plus{},static_cast<const T&>(t),dim);
}

template<class T>
decltype(auto) fast_sum( DenseExpression<T>&& t, std::size_t dim=0){
    return accumulate(Plus{},static_cast<T&&>(t),dim);
}

template<class T>
decltype(auto) pairwise_sum( const DenseExpression<T>& t, std::size_t dim=0){
    return fold( PairwiseSum{}, static_cast<const T&>(t), (typename T::value_type)0, dim);
}

template<class T>
decltype(auto) pairwise_sum( DenseExpression<T>&& t, std::size_t dim=0){
    return fold( PairwiseSum{}, static_cast<T&&>(t), (typename T::value_type)0, dim);
}

template<class T>
decltype(auto) precise_sum( const DenseExpression<T>& t, std::size_t dim=0){
    return fold( KahanSum{}, static_cast<const T&>(t), (typename T::value_type)0, dim);
}

template<class T>
decltype(auto) precise_sum( DenseExpression<T>&& t, std::size_t dim=0){
    return fold( KahanSum{}, static_cast<T&&>(t), (typename T::value_type)0, dim);
}

template<class T>
decltype(auto) sum( const DenseExpression<T>& t, std::size_t dim=0){
    return pairwise_sum(static_cast<const T&>(t),dim);
}

template<class T>
decltype(auto) sum( DenseExpression<T>&& t, std::size_t dim=0){
    return pairwise_sum(static_cast<T&&>(t),dim);
}

// Product/min/max -- all simple accumulations

template<class T>
decltype(auto) prod( const DenseExpression<T>& t, std::size_t dim=0){
    return accumulate(Multiplies{},static_cast<const T&>(t),dim);
}

template<class T>
decltype(auto) prod( DenseExpression<T>&& t, std::size_t dim=0){
    return accumulate(Multiplies{},static_cast<T&&>(t),dim);
}

template<class T>
decltype(auto) min( const DenseExpression<T>& t, std::size_t dim=0){
    return accumulate(Min{},static_cast<const T&>(t), dim);
}

template<class T>
decltype(auto) min( DenseExpression<T>&& t, std::size_t dim=0){
    return accumulate(Min{},static_cast<T&&>(t), dim);
}

template<class T>
decltype(auto) max( const DenseExpression<T>& t, std::size_t dim=0){
    return accumulate(Max{},static_cast<const T&>(t),dim);
}

template<class T>
decltype(auto) max( DenseExpression<T>&& t, std::size_t dim=0){
    return accumulate(Max{},static_cast<T&&>(t),dim);
}

// mean/var/stddev
// Similar to sum; includes fast, pairwise, and precise, with pairwise as the default.
// Lots of repetition here, so implemented using macros

#define ULTRA_VAR_MEAN_STDDEV(PREFIX)\
\
template<class T>\
decltype(auto) PREFIX##mean( const DenseExpression<T>& t, std::size_t dim=0){\
    return PREFIX##sum(static_cast<const T&>(t),dim) / t.shape(dim);\
}\
\
template<class T>\
decltype(auto) PREFIX##mean( DenseExpression<T>&& t, std::size_t dim=0){\
    return PREFIX##sum(static_cast<T&&>(t),dim) / t.shape(dim);\
}\
\
template<class T>\
decltype(auto) PREFIX##average( const DenseExpression<T>& t, std::size_t dim=0){\
    return PREFIX##mean(static_cast<const T&>(t),dim);\
}\
\
template<class T>\
decltype(auto) PREFIX##average( DenseExpression<T>&& t, std::size_t dim=0){\
    return PREFIX##mean(static_cast<T&&>(t),dim);\
}\
\
template<class T>\
decltype(auto) PREFIX##var( const DenseExpression<T>& t, std::size_t dim=0, std::size_t ddof=0){\
    /* As the input expression must be used twice, it must be evaluated here to avoid performing each calculation twice.*/\
    auto x = eval(static_cast<const T&>(t));\
    auto shape = x.shape(); shape[dim] = 1;\
    auto mu = eval(PREFIX##mean(x,dim)).reshape(shape);\
    if( ddof >= x.shape(dim) ) throw std::runtime_error("Ultra var/stddev: choice of ddof would result in negative/zero denominator");\
    std::size_t denom = x.shape(dim) - ddof;\
    return eval(PREFIX##sum(norm(x - mu)) / denom);\
}\
\
template<class T>\
decltype(auto) PREFIX##var( DenseExpression<T>&& t, std::size_t dim=0, std::size_t ddof=0){\
    auto x = eval(static_cast<T&&>(t));\
    auto shape = x.shape(); shape[dim] = 1;\
    auto mu = eval(PREFIX##mean(x,dim)).reshape(shape);\
    if( ddof >= x.shape(dim) ) throw std::runtime_error("Ultra var/stddev: choice of ddof would result in negative/zero denominator");\
    std::size_t denom = x.shape(dim) - ddof;\
    return eval(PREFIX##sum(norm(x - mu)) / denom);\
}\
\
template<class T>\
decltype(auto) PREFIX##stddev( const DenseExpression<T>& t, std::size_t dim=0, std::size_t ddof=0){\
    return sqrt(PREFIX##var(static_cast<const T&>(t),dim,ddof));\
}\
\
template<class T>\
decltype(auto) PREFIX##stddev( DenseExpression<T>&& t, std::size_t dim=0, std::size_t ddof=0){\
    return sqrt(PREFIX##var(static_cast<T&&>(t),dim,ddof));\
}\

ULTRA_VAR_MEAN_STDDEV(fast_)
ULTRA_VAR_MEAN_STDDEV(pairwise_)
ULTRA_VAR_MEAN_STDDEV(precise_)
ULTRA_VAR_MEAN_STDDEV()

// Boolean Folds: all_of, any_of, none_of.
// Optimised to stop early where appropriate.

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
// (Actually uses ScalarDenseExpression as opposed to Generators)

template<shapelike Shape, class T=double>
decltype(auto) zeros( const Shape& shape) {
    return ScalarDenseExpression<T,default_order>(0,shape);
}

template<class T=double>
decltype(auto) zeros( std::size_t N) {
    return zeros(std::array<std::size_t,1>{N});
}

template<shapelike Shape, class T=double>
decltype(auto) ones( const Shape& shape) {
    return ScalarDenseExpression<T,default_order>(1,shape);
}

template<class T=double>
decltype(auto) ones( std::size_t N) {
    return ones(std::array<std::size_t,1>{N});
}

// linspace and logspace
// Generate an evenly distributed set of values in the range [start,stop]
// logspace calculates pow(10,linspace(start,stop,N))

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

template<class T1, class T2> requires (!std::is_integral<decltype(T1()*T2())>::value)
decltype(auto) linspace( const T1& start, const T2& stop, std::size_t N) {
    return GeneratorExpression( LinspaceFunctor<decltype(T1()*T2())>(start,stop,N), std::array<std::size_t,1>{N});
}

template<class T1, class T2> requires std::integral<decltype(T1()*T2())>
decltype(auto) linspace( const T1& start, const T2& stop, std::size_t N) {
    return linspace<double>(start,stop,N);
}

template<class T1,class T2>
decltype(auto) logspace( const T1& start, const T2& stop, std::size_t N) {
    return pow(10,linspace(start,stop,N));
}

// arange/regspace
// Generate points distributed set of values in the range [start,stop) with a given step size

template<class T>
class ArangeFunctor {
    T _start;
    T _step;
public:
    ArangeFunctor( const T& start, const T& step ) : _start(start), _step(step) {}
    T operator()( std::size_t idx) {
        return _start + idx*_step;
    }
};

template<class T>
decltype(auto) arange( const T& start, const T& stop, const T& step) {
    double size = (0.+stop-start)/step;
    std::size_t num_vals = ( std::fabs(size - std::round(size)) < 1e-5*std::fabs(stop)  ? std::round(size) : std::ceil(size));
    if( num_vals <= 0 ){
        throw std::runtime_error("Ultra: arange, stop must be greater than start for positive step, or less than start for negative step");
    }
    return GeneratorExpression( ArangeFunctor<T>(start,step), std::array<std::size_t,1>{num_vals});
}

template<class T1,class T2,class T3>
decltype(auto) arange( const T1& start, const T2& stop, const T3& step) {
    using common_t = decltype(T1()*T2()*T3());
    return arange( (common_t)start, (common_t)stop, (common_t)step);
}

template<class T1, class T2, class T3>
decltype(auto) regspace( const T1& start, const T2& stop, const T3& step) {
    return arange(start,stop,step);
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

    RandomFunctor( RandomFunctor&& other ) :
        _dist(std::move(other._dist)),
        _rng(other._rng)
    {}

    dist_result_type operator()( std::size_t ) {
        return _dist(_rng);
    }
};

template<class Dist, std::uniform_random_bit_generator RNG>
RandomFunctor<Dist,RNG>::rng_result_type RandomFunctor<Dist,RNG>::_static_seed = RNG::default_seed;

template<class Dist, std::uniform_random_bit_generator RNG = std::mt19937_64, shapelike Range>
decltype(auto) random( const Dist& dist, const Range& range) {
    return GeneratorExpression( RandomFunctor<Dist,RNG>(dist), range);
}

template<class Dist, std::uniform_random_bit_generator RNG = std::mt19937_64>
decltype(auto) random( const Dist& dist, std::size_t N) {
    return GeneratorExpression( RandomFunctor<Dist,RNG>(dist), std::array<std::size_t,1>{N});
}

// random_uniform
// Floating point: produces values in the range [min,max)
// Integrer: produces values in the range [min,max] (inclusive of max)

template<class T1, class T2, shapelike Range>
requires std::floating_point<decltype(T1()*T2())>
decltype(auto) random_uniform( T1 min, T2 max, const Range& range) {
    return random( std::uniform_real_distribution<decltype(T1()*T2())>(min,max), range);
}

template<class T1, class T2, shapelike Range>
requires std::integral<T1> && std::integral<T2>
decltype(auto) random_uniform( T1 min, T2 max, const Range& range) {
    return random( std::uniform_int_distribution<decltype(T1()*T2())>(min,max), range);
}

template<class T1, class T2> requires number<T1> && number<T2>
decltype(auto) random_uniform( T1 min, T2 max, std::size_t N) {
    return random_uniform( min, max, std::array<std::size_t,1>{N});
}

// random_normal/random_gaussian (identical functions)
// Produces random numbers according to:
// f(x;\mu,\sigma) = \frac{1}{2\pi\sigma}\exp\left(-\frac{(x-\mu)^2}{2\sigma^2}\right)
// where \mu is mean and \sigma is stddev

template<class T1, class T2, shapelike Range>
requires number<T1> && number<T2>
decltype(auto) random_normal( T1 mean, T2 stddev, const Range& range) {
    using real_t = std::conditional_t< std::is_floating_point<decltype(T1()*T2())>::value, decltype(T1()*T2()), double>;
    return random( std::normal_distribution<real_t>(mean,stddev), range);
}

template<class T1, class T2, shapelike Range>
requires number<T1> && number<T2>
decltype(auto) random_gaussian( T1 mean, T2 stddev, const Range& range) {
    return random_normal( mean, stddev, range);
}

template<class T1, class T2> requires number<T1> && number<T2>
decltype(auto) random_normal( T1 mean, T2 stddev, std::size_t N) {
    return random_normal( mean, stddev, std::array<std::size_t,1>{N});
}

template<class T1, class T2> requires number<T1> && number<T2>
decltype(auto) random_gaussian( T1 mean, T2 stddev, std::size_t N) {
    return random_gaussian( mean, stddev, std::array<std::size_t,1>{N});
}

} // namespace
#endif
