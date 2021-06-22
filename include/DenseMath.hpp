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

#define DENSE_MATH_UNARY_OPERATOR( Name, Op) \
                                                                                                    \
    struct Name { template<class T> decltype(auto) operator()( const T& t) const { return Op t; }}; \
                                                                                                    \
    template<class T> using Name##DenseExpression = ElementWiseDenseExpression< Name, T>;           \
                                                                                                    \
    template<class T>                                                                               \
    decltype(auto) operator Op ( const DenseExpression<T>& t){                                     \
        return Name##DenseExpression(static_cast<const T&>(t));                                     \
    }                                                                                               \
                                                                                                    \
    template<class T>                                                                               \
    decltype(auto) operator Op ( DenseExpression<T>&& t){                                          \
        return Name##DenseExpression(static_cast<T&&>(t));                                          \
    }                                                                                               

DENSE_MATH_UNARY_OPERATOR( Negate, -)
DENSE_MATH_UNARY_OPERATOR( LogicalNot, !)

#define DENSE_MATH_BINARY_OPERATOR( Name, Op) \
                                                                                                                           \
    struct Name { template<class L, class R> decltype(auto) operator()( const L& l, const R& r) const { return l Op r; }}; \
                                                                                                                           \
    template<class L,class R> using Name##DenseExpression = ElementWiseDenseExpression< Name, L, R>;                       \
                                                                                                                           \
    template<class L, class R>                                                                                             \
    decltype(auto) operator Op ( const DenseExpression<L>& l, const DenseExpression<R>& r){                               \
        return Name##DenseExpression(static_cast<const L&>(l),static_cast<const R&>(r));                                   \
    }                                                                                                                      \
                                                                                                                           \
    template<class L, class R>                                                                                             \
    decltype(auto) operator Op ( const DenseExpression<L>& l, DenseExpression<R>&& r){                                    \
        return Name##DenseExpression(static_cast<const L&>(l),static_cast<R&&>(r));                                        \
    }                                                                                                                      \
                                                                                                                           \
    template<class L, class R>                                                                                             \
    decltype(auto) operator Op ( DenseExpression<L>&& l, const DenseExpression<R>& r){                                    \
        return Name##DenseExpression(static_cast<L&&>(l),static_cast<const R&>(r));                                        \
    }                                                                                                                      \
                                                                                                                           \
    template<class L, class R>                                                                                             \
    decltype(auto) operator Op ( DenseExpression<L>&& l, DenseExpression<R>&& r){                                         \
        return Name##DenseExpression(static_cast<L&&>(l),static_cast<R&&>(r));                                             \
    }

DENSE_MATH_BINARY_OPERATOR( Plus, +)
DENSE_MATH_BINARY_OPERATOR( Minus, -)
DENSE_MATH_BINARY_OPERATOR( Multiplies, *)
DENSE_MATH_BINARY_OPERATOR( Divides, /)
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

#define DENSE_MATH_UNARY_FUNCTION( Name, Func ) \
                                                                                                            \
    struct Name { template<class T> decltype(auto) operator()( const T& t) const { return std::Func(t); }}; \
                                                                                                            \
    template<class T> using Name##DenseExpression = ElementWiseDenseExpression< Name, T>;                   \
                                                                                                            \
    template<class T>                                                                                       \
    decltype(auto) Func ( const DenseExpression<T>& t){                                                     \
        return Name##DenseExpression(static_cast<const T&>(t));                                             \
    }                                                                                                       \
                                                                                                            \
    template<class T>                                                                                       \
    decltype(auto) Func ( DenseExpression<T>&& t){                                                          \
        return Name##DenseExpression(static_cast<T&&>(t));                                                  \
    }                                                                                               

DENSE_MATH_UNARY_FUNCTION( Abs, abs )
DENSE_MATH_UNARY_FUNCTION( Sin, sin )
DENSE_MATH_UNARY_FUNCTION( Cos, cos )
DENSE_MATH_UNARY_FUNCTION( Tan, tan )
DENSE_MATH_UNARY_FUNCTION( Asin, asin )
DENSE_MATH_UNARY_FUNCTION( Acos, acos )
DENSE_MATH_UNARY_FUNCTION( Atan, atan )
DENSE_MATH_UNARY_FUNCTION( Sinh, sinh )
DENSE_MATH_UNARY_FUNCTION( Cosh, cosh )
DENSE_MATH_UNARY_FUNCTION( Tanh, tanh )
DENSE_MATH_UNARY_FUNCTION( Asinh, asinh )
DENSE_MATH_UNARY_FUNCTION( Acosh, acosh )
DENSE_MATH_UNARY_FUNCTION( Atanh, atanh )
DENSE_MATH_UNARY_FUNCTION( Sqrt, sqrt )
DENSE_MATH_UNARY_FUNCTION( Cbrt, cbrt )
DENSE_MATH_UNARY_FUNCTION( Exp, exp )
DENSE_MATH_UNARY_FUNCTION( Exp2, exp2 )
DENSE_MATH_UNARY_FUNCTION( Expm1, expm1 )
DENSE_MATH_UNARY_FUNCTION( Log, log )
DENSE_MATH_UNARY_FUNCTION( Log2, log2 )
DENSE_MATH_UNARY_FUNCTION( Log10, log10 )
DENSE_MATH_UNARY_FUNCTION( Log1p, log1p )
DENSE_MATH_UNARY_FUNCTION( Ceil, ceil )
DENSE_MATH_UNARY_FUNCTION( Floor, floor )
DENSE_MATH_UNARY_FUNCTION( Round, round )
DENSE_MATH_UNARY_FUNCTION( Erf, erf )
DENSE_MATH_UNARY_FUNCTION( Erfc, erfc )
DENSE_MATH_UNARY_FUNCTION( Tgamma, tgamma )
DENSE_MATH_UNARY_FUNCTION( Lgamma, lgamma )

// =========================
// Binary Functions

#define DENSE_MATH_BINARY_FUNCTION( Name, Func) \
                                                                                                                                   \
    struct Name { template<class L, class R> decltype(auto) operator()( const L& l, const R& r) const { return std::Func(l,r); }}; \
                                                                                                                                   \
    template<class L,class R> using Name##DenseExpression = ElementWiseDenseExpression< Name, L, R>;                               \
                                                                                                                                   \
    template<class L, class R>                                                                                                     \
    decltype(auto) Func ( const DenseExpression<L>& l, const DenseExpression<R>& r){                                               \
        return Name##DenseExpression(static_cast<const L&>(l),static_cast<const R&>(r));                                           \
    }                                                                                                                              \
                                                                                                                                   \
    template<class L, class R>                                                                                                     \
    decltype(auto) Func ( const DenseExpression<L>& l, DenseExpression<R>&& r){                                                    \
        return Name##DenseExpression(static_cast<const L&>(l),static_cast<R&&>(r));                                                \
    }                                                                                                                              \
                                                                                                                                   \
    template<class L, class R>                                                                                                     \
    decltype(auto) Func ( DenseExpression<L>&& l, const DenseExpression<R>& r){                                                    \
        return Name##DenseExpression(static_cast<L&&>(l),static_cast<const R&>(r));                                                \
    }                                                                                                                              \
                                                                                                                                   \
    template<class L, class R>                                                                                                     \
    decltype(auto) Func ( DenseExpression<L>&& l, DenseExpression<R>&& r){                                                         \
        return Name##DenseExpression(static_cast<L&&>(l),static_cast<R&&>(r));                                                     \
    }

DENSE_MATH_BINARY_FUNCTION(Pow,pow)
DENSE_MATH_BINARY_FUNCTION(Atan2,atan2)
DENSE_MATH_BINARY_FUNCTION(Hypot2,hypot)

// =========================
// Ternary Function(s)

#define DENSE_MATH_TERNARY_FUNCTION( Name, Func) \
                                                                                                                                                          \
    struct Name { template<class X, class Y, class Z> decltype(auto) operator()( const X& x, const Y& y, const Z& z) const { return std::Func(x,y,z); }}; \
                                                                                                                                                          \
    template<class X,class Y,class Z> using Name##DenseExpression = ElementWiseDenseExpression< Name, X, Y, Z>;                                           \
                                                                                                                                                          \
    template<class X, class Y,class Z>                                                                                                                    \
    decltype(auto) Func ( const DenseExpression<X>& x, const DenseExpression<Y>& y, const DenseExpression<Z>& z){                                         \
        return Name##DenseExpression(static_cast<const X&>(x),static_cast<const Y&>(y),static_cast<const Z&>(z));                                         \
    }                                                                                                                                                     \
                                                                                                                                                          \
    template<class X, class Y,class Z>                                                                                                                    \
    decltype(auto) Func ( DenseExpression<X>&& x, const DenseExpression<Y>& y, const DenseExpression<Z>& z){                                              \
        return Name##DenseExpression(static_cast<X&&>(x),static_cast<const Y&>(y),static_cast<const Z&>(z));                                              \
    }                                                                                                                                                     \
                                                                                                                                                          \
    template<class X, class Y,class Z>                                                                                                                    \
    decltype(auto) Func ( const DenseExpression<X>& x, DenseExpression<Y>&& y, const DenseExpression<Z>& z){                                              \
        return Name##DenseExpression(static_cast<const X&>(x),static_cast<Y&&>(y),static_cast<const Z&>(z));                                              \
    }                                                                                                                                                     \
                                                                                                                                                          \
    template<class X, class Y,class Z>                                                                                                                    \
    decltype(auto) Func ( const DenseExpression<X>& x, const DenseExpression<Y>& y, DenseExpression<Z>&& z){                                              \
        return Name##DenseExpression(static_cast<const X&>(x),static_cast<const Y&>(y),static_cast<Z&&>(z));                                              \
    }                                                                                                                                                     \
                                                                                                                                                          \
    template<class X, class Y,class Z>                                                                                                                    \
    decltype(auto) Func ( DenseExpression<X>&& x, DenseExpression<Y>&& y, const DenseExpression<Z>& z){                                                   \
        return Name##DenseExpression(static_cast<X&>(x),static_cast<Y&&>(y),static_cast<const Z&>(z));                                                    \
    }                                                                                                                                                     \
                                                                                                                                                          \
    template<class X, class Y,class Z>                                                                                                                    \
    decltype(auto) Func ( DenseExpression<X>&& x, const DenseExpression<Y>& y, DenseExpression<Z>&& z){                                                   \
        return Name##DenseExpression(static_cast<X&&>(x),static_cast<const Y&>(y),static_cast<Z&&>(z));                                                   \
    }                                                                                                                                                     \
                                                                                                                                                          \
    template<class X, class Y,class Z>                                                                                                                    \
    decltype(auto) Func ( const DenseExpression<X>& x, DenseExpression<Y>&& y, DenseExpression<Z>&& z){                                                   \
        return Name##DenseExpression(static_cast<const X&>(x),static_cast<Y&&>(y),static_cast<Z&&>(z));                                                   \
    }                                                                                                                                                     \
                                                                                                                                                          \
    template<class X, class Y,class Z>                                                                                                                    \
    decltype(auto) Func ( DenseExpression<X>&& x, DenseExpression<Y>&& y, DenseExpression<Z>&& z){                                                        \
        return Name##DenseExpression(static_cast<X&&>(x),static_cast<Y&&>(y),static_cast<Z&&>(z));                                                        \
    }                                                                                                                                                     \

DENSE_MATH_TERNARY_FUNCTION(Hypot3,hypot)

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
