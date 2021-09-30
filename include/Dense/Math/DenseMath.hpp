#ifndef __ULTRA_DENSE_MATH_HPP
#define __ULTRA_DENSE_MATH_HPP

/*! \file DenseMath.hpp
 *  \brief Defines expression templates for functions in the `cmath` and `complex` libraries, plus extras.
 *
 *  Defines the following functions:
 *  - Absolute value: `abs`
 *  - Trigonometric functions: `sin`, `cos`, `tan`, `asin`, `acos`, `atan`, `atan2`
 *  - Hyperbolic functions: `sinh`, `cosh`, `tanh`, `asinh`, `acosh`, `atanh`
 *  - Powers and roots: `pow`, `sqrt`, `cbrt`, `square`, `cube`, `hypot`
 *  - Exponentials: `exp`, `exp2`, `expm1`
 *  - Logarithms: `log`, `log2`, `log10`, `log1p`
 *  - Complex numbers: `real`, `imag`, `arg`, `norm`, `conj`, `polar`
 *  - Complex-safe functions: `complex_sqrt`, `complex_log`, `complex_log10`, `complex_acos`, `complex_asin`, `complex_atanh`, `complex_pow`
 *  - Special functions: `erf`, `erfc`, `tgamma`, `lgamma`
 *  - Integer rounding: `ceil`, `floor, `trunc`, `round`
 *  - Working with angles: `to_radians`, `to_degrees`
 *  - Utilities: `where`, `fma`, `copysign`, `signbit`, `isfinite`, `isinf`, `isnan`, `isnormal`
 *
 *  These are defined using `DenseElementWiseExpression` in all cases, with liberal use of macros to avoid code duplication.
 *  The `hypot` function is defined for both binary and ternary expressions. The `where` function is optimised to only
 *  evaluate the necessary path depending on whether the 'condition' element evaluates to `true` or `false`.
 */

#include "ultramat/include/Dense/Expressions/DenseElementWiseExpression.hpp"
#include "ultramat/include/Dense/Expressions/DenseWhereExpression.hpp"

namespace ultra {

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
    using Dense##Name##Expression = DenseElementWiseExpression< Name, T>;\
\
    template<class T>\
    decltype(auto) FuncName ( const DenseExpression<T>& t){\
        return Dense##Name##Expression(static_cast<const T&>(t));\
    }\
\
    template<class T>\
    decltype(auto) FuncName ( DenseExpression<T>&& t){\
        return Dense##Name##Expression(static_cast<T&&>(t));\
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
    using Dense##Name##Expression = DenseElementWiseExpression< Name, L, R>;\
\
    template<class L, class R>\
    decltype(auto) FuncName ( const DenseExpression<L>& l, const DenseExpression<R>& r){\
        return Dense##Name##Expression(static_cast<const L&>(l),static_cast<const R&>(r));\
    }\
\
    template<class L, class R>\
    decltype(auto) FuncName ( const DenseExpression<L>& l, DenseExpression<R>&& r){\
        return Dense##Name##Expression(static_cast<const L&>(l),static_cast<R&&>(r));\
    }\
\
    template<class L, class R>\
    decltype(auto) FuncName ( DenseExpression<L>&& l, const DenseExpression<R>& r){\
        return Dense##Name##Expression(static_cast<L&&>(l),static_cast<const R&>(r));\
    }\
\
    template<class L, class R>\
    decltype(auto) FuncName ( DenseExpression<L>&& l, DenseExpression<R>&& r){\
        return Dense##Name##Expression(static_cast<L&&>(l),static_cast<R&&>(r));\
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
// 
// Note that the macro for ternary functions is split into 'preamble' and 'definitions'. This is to
// aid DenseWhereExpression, which is optimised to only execute one of two paths.

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
    using Dense##Name##Expression = DenseElementWiseExpression< Name, X, Y, Z>;                                                                   

#define DENSE_MATH_TERNARY_FUNCTION_DEFINITIONS( Name, FuncName, InternalFunc)\
    template<class X, class Y,class Z>\
    decltype(auto) FuncName ( const DenseExpression<X>& x, const DenseExpression<Y>& y, const DenseExpression<Z>& z){\
        return Dense##Name##Expression(static_cast<const X&>(x),static_cast<const Y&>(y),static_cast<const Z&>(z));\
    }\
\
    template<class X, class Y,class Z>\
    decltype(auto) FuncName ( DenseExpression<X>&& x, const DenseExpression<Y>& y, const DenseExpression<Z>& z){\
        return Dense##Name##Expression(static_cast<X&&>(x),static_cast<const Y&>(y),static_cast<const Z&>(z));\
    }\
\
    template<class X, class Y,class Z>\
    decltype(auto) FuncName ( const DenseExpression<X>& x, DenseExpression<Y>&& y, const DenseExpression<Z>& z){\
        return Dense##Name##Expression(static_cast<const X&>(x),static_cast<Y&&>(y),static_cast<const Z&>(z));\
    }\
\
    template<class X, class Y,class Z>\
    decltype(auto) FuncName ( const DenseExpression<X>& x, const DenseExpression<Y>& y, DenseExpression<Z>&& z){\
        return Dense##Name##Expression(static_cast<const X&>(x),static_cast<const Y&>(y),static_cast<Z&&>(z));\
    }\
\
    template<class X, class Y,class Z>\
    decltype(auto) FuncName ( DenseExpression<X>&& x, DenseExpression<Y>&& y, const DenseExpression<Z>& z){\
        return Dense##Name##Expression(static_cast<X&&>(x),static_cast<Y&&>(y),static_cast<const Z&>(z));\
    }\
\
    template<class X, class Y,class Z>\
    decltype(auto) FuncName ( DenseExpression<X>&& x, const DenseExpression<Y>& y, DenseExpression<Z>&& z){\
        return Dense##Name##Expression(static_cast<X&&>(x),static_cast<const Y&>(y),static_cast<Z&&>(z));\
    }\
\
    template<class X, class Y,class Z>\
    decltype(auto) FuncName ( const DenseExpression<X>& x, DenseExpression<Y>&& y, DenseExpression<Z>&& z){\
        return Dense##Name##Expression(static_cast<const X&>(x),static_cast<Y&&>(y),static_cast<Z&&>(z));\
    }\
\
    template<class X, class Y,class Z>\
    decltype(auto) FuncName ( DenseExpression<X>&& x, DenseExpression<Y>&& y, DenseExpression<Z>&& z){\
        return Dense##Name##Expression(static_cast<X&&>(x),static_cast<Y&&>(y),static_cast<Z&&>(z));\
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

// Define a function to be used when calling ultra::where(c,l,r) on only scalars
template<class Cond, class L, class R>
decltype(L()+R()) _where( const Cond& c, const L& l, const R& r){
    return c ? l : r;
}

DENSE_MATH_TERNARY_FUNCTION_DEFINITIONS(Where,where,_where)

} // namespace ultra
#endif
