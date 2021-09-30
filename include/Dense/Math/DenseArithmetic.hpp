#ifndef __ULTRA_DENSE_ARITHMETIC_HPP
#define __ULTRA_DENSE_ARITHMETIC_HPP

/*! \file DenseArithmetic.hpp
 *  \brief Defines expression templates for simple arithmetic operations.
 *
 *  Defines the following arithmetic operators for use with `Dense` objects and expressions:
 *  - Negation (unary, `-`)
 *  - Unary plus (unary, `+`)
 *  - Addition (binary, `+`)
 *  - Subtraction (binary, `-`)
 *  - Multiplication (binary, `*`)
 *  - Division (binary, `/`)
 *  - Modulo (binary, `%`)
 *
 *  Additionally defines the boolean operators:
 *  - Not (unary, `!`)
 *  - And (binary, `&&`)
 *  - Or (binary, `||`)
 *  - Equal to (binary, `==`)
 *  - Not equal to (binary, `!=`)
 *  - Greater than (binary, `>`)
 *  - Less than (binary, `<`)
 *  - Greater than or equal to (binary, `>=`)
 *  - Less than or equal to (binary, `<=`)
 *
 *  These are defined using `DenseElementWiseExpression` in all cases, with liberal use of macros to avoid code duplication.
 *  
 *  Note that multiplication is always an element-wise operation within Ultramat, and special linear algebra operations such
 *  as the dot product, matrix-vector multiplication, or matrix-matrix multiplication are called using their own functions.
 */

#include "ultramat/include/Dense/Expressions/DenseElementWiseExpression.hpp"

namespace ultra {

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
    using Dense##Name##Expression = DenseElementWiseExpression< Name, T>;\
\
    template<class T>\
    decltype(auto) operator Op ( const DenseExpression<T>& t){\
        return Dense##Name##Expression(static_cast<const T&>(t));\
    }\
\
    template<class T>\
    decltype(auto) operator Op ( DenseExpression<T>&& t){\
        return Dense##Name##Expression(static_cast<const T&>(t));\
    }

DENSE_MATH_UNARY_OPERATOR( UnaryPlus, +)
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
    using Dense##Name##Expression = DenseElementWiseExpression< Name, L, R>;\
\
    template<class L, class R>\
    decltype(auto) operator Op ( const DenseExpression<L>& l, const DenseExpression<R>& r){\
        return Dense##Name##Expression(static_cast<const L&>(l),static_cast<const R&>(r));\
    }\
\
    template<class L, class R>\
    decltype(auto) operator Op ( const DenseExpression<L>& l, DenseExpression<R>&& r){\
        return Dense##Name##Expression(static_cast<const L&>(l),static_cast<R&&>(r));\
    }\
\
    template<class L, class R>\
    decltype(auto) operator Op ( DenseExpression<L>&& l, const DenseExpression<R>& r){\
        return Dense##Name##Expression(static_cast<L&&>(l),static_cast<const R&>(r));\
    }\
\
    template<class L, class R>\
    decltype(auto) operator Op ( DenseExpression<L>&& l, DenseExpression<R>&& r){\
        return Dense##Name##Expression(static_cast<L&&>(l),static_cast<R&&>(r));\
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

} // namespace ultra
#endif
