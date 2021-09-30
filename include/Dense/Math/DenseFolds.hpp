#ifndef __ULTRA_DENSE_FOLDS_HPP
#define __ULTRA_DENSE_FOLDS_HPP

#include "DenseArithmetic.hpp"
#include "DenseMath.hpp"
#include "ultramat/include/Dense/Expressions/DenseFoldExpression.hpp"

namespace ultra {

// =========================
// General Functions

template<class F, class T, class ValueType>
requires (std::is_convertible< decltype(std::declval<F>()(std::declval<std::remove_cvref_t<T>>().begin(), std::declval<std::remove_cvref_t<T>>().end(), std::declval<ValueType>())), std::remove_cvref_t<ValueType>>::value )
decltype(auto) fold( const F& f, const DenseExpression<T>& t, const ValueType& start_val, std::size_t dim=0){
    return DenseGeneralFoldExpression(f,static_cast<const T&>(t),start_val,dim);
}

template<class F, class T, class ValueType>
requires (std::is_convertible< decltype(std::declval<F>()(std::declval<std::remove_cvref_t<T>>().begin(), std::declval<std::remove_cvref_t<T>>().end(), std::declval<ValueType>())), std::remove_cvref_t<ValueType>>::value )
decltype(auto) fold( const F& f, DenseExpression<T>&& t, const ValueType& start_val, std::size_t dim=0){
    return DenseGeneralFoldExpression(f,static_cast<T&&>(t),start_val,dim);
}

template<class F, class T, class ValueType>
requires (std::is_convertible< decltype(std::declval<F>()(std::declval<ValueType>(),std::declval<typename std::remove_cvref_t<T>::value_type>())), std::remove_cvref_t<ValueType>>::value )
decltype(auto) fold( const F& f, const DenseExpression<T>& t, const ValueType& start_val, std::size_t dim=0){
    return DenseLinearFoldExpression(f,static_cast<const T&>(t),start_val,dim);
}

template<class F, class T, class ValueType>
requires (std::is_convertible< decltype(std::declval<F>()(std::declval<ValueType>(),std::declval<typename std::remove_cvref_t<T>::value_type>())), std::remove_cvref_t<ValueType>>::value )
decltype(auto) fold( const F& f, DenseExpression<T>&& t, const ValueType& start_val, std::size_t dim=0){
    return DenseLinearFoldExpression(f,static_cast<T&&>(t),start_val,dim);
}

template<class F, class T>
decltype(auto) accumulate( const F& f, const DenseExpression<T>& t, std::size_t dim=0){
    return DenseAccumulateExpression(f,static_cast<const T&>(t),dim);
}

template<class F, class T>
decltype(auto) accumulate( const F& f, DenseExpression<T>&& t, std::size_t dim=0){
    return DenseAccumulateExpression(f,static_cast<T&&>(t),dim);
}

// =========================
// Min/Max

struct _Min { template<class T> T operator()( const T& x, const T& y) const { return x < y ? x : y;}};
struct _Max { template<class T> T operator()( const T& x, const T& y) const { return x > y ? x : y;}};

template<class T>
decltype(auto) min( const DenseExpression<T>& t, std::size_t dim=0){
    return accumulate(_Min{},static_cast<const T&>(t), dim);
}

template<class T>
decltype(auto) min( DenseExpression<T>&& t, std::size_t dim=0){
    return accumulate(_Min{},static_cast<T&&>(t), dim);
}

template<class T>
decltype(auto) max( const DenseExpression<T>& t, std::size_t dim=0){
    return accumulate(_Max{},static_cast<const T&>(t),dim);
}

template<class T>
decltype(auto) max( DenseExpression<T>&& t, std::size_t dim=0){
    return accumulate(_Max{},static_cast<T&&>(t),dim);
}

// =========================
// Summation

// * fast_sum -> Naively accumulate into a single variable. Fastest, but most susceptible to numerical errors.
// * pairwise_sum -> Sum recursively. Almost as fast, much less error.
// * precise_sum -> Kahan summation. Slowest sum, but most precise

// Pairwise sum is the default.

// Fast sum

template<class T>
decltype(auto) fast_sum( const DenseExpression<T>& t, std::size_t dim=0){
    return accumulate(Plus{},static_cast<const T&>(t),dim);
}

template<class T>
decltype(auto) fast_sum( DenseExpression<T>&& t, std::size_t dim=0){
    return accumulate(Plus{},static_cast<T&&>(t),dim);
}

// Pairwise sum

#ifndef ULTRA_PAIRWISE_SUM_BASE_CASE 
#define ULTRA_PAIRWISE_SUM_BASE_CASE 10
#endif

struct _PairwiseSum {
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

template<class T>
decltype(auto) pairwise_sum( const DenseExpression<T>& t, std::size_t dim=0){
    return fold( _PairwiseSum{}, static_cast<const T&>(t), (typename T::value_type)0, dim);
}

template<class T>
decltype(auto) pairwise_sum( DenseExpression<T>&& t, std::size_t dim=0){
    return fold( _PairwiseSum{}, static_cast<T&&>(t), (typename T::value_type)0, dim);
}

// Kahan summation

struct _KahanSum {
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

template<class T>
decltype(auto) kahan_sum( const DenseExpression<T>& t, std::size_t dim=0){
    return fold( _KahanSum{}, static_cast<const T&>(t), (typename T::value_type)0, dim);
}

template<class T>
decltype(auto) kahan_sum( DenseExpression<T>&& t, std::size_t dim=0){
    return fold( _KahanSum{}, static_cast<T&&>(t), (typename T::value_type)0, dim);
}

template<class T>
decltype(auto) precise_sum( const DenseExpression<T>& t, std::size_t dim=0){
    return kahan_sum(static_cast<const T&>(t),dim);
}

template<class T>
decltype(auto) precise_sum( DenseExpression<T>&& t, std::size_t dim=0){
    return kahan_sum(static_cast<T&&>(t),dim);
}

// Default

template<class T>
decltype(auto) sum( const DenseExpression<T>& t, std::size_t dim=0){
    return pairwise_sum(static_cast<const T&>(t),dim);
}

template<class T>
decltype(auto) sum( DenseExpression<T>&& t, std::size_t dim=0){
    return pairwise_sum(static_cast<T&&>(t),dim);
}

// =========================
// Product

template<class T>
decltype(auto) prod( const DenseExpression<T>& t, std::size_t dim=0){
    return accumulate(Multiplies{},static_cast<const T&>(t),dim);
}

template<class T>
decltype(auto) prod( DenseExpression<T>&& t, std::size_t dim=0){
    return accumulate(Multiplies{},static_cast<T&&>(t),dim);
}

// =========================
// Mean, Variance, Standard Deviation
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

// =========================
// All of, Any of, None of

// Each of these begins with value `start_bool`, and along each fold searchs for the first instance of `early_exit_bool`. On finding it,
// it returns `!start_bool`. If `early_exit_bool` is not found, simply return `start_bool`. The functions shown are not actually called.

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
    return DenseBooleanFoldExpression(AllOf{},static_cast<const T&>(t),dim);
}

template<class T>
decltype(auto) all_of( DenseExpression<T>&& t, std::size_t dim){
    return DenseBooleanFoldExpression(AllOf{},static_cast<T&&>(t),dim);
}

template<class T>
decltype(auto) any_of( const DenseExpression<T>& t, std::size_t dim){
    return DenseBooleanFoldExpression(AnyOf{},static_cast<const T&>(t),dim);
}

template<class T>
decltype(auto) any_of( DenseExpression<T>&& t, std::size_t dim){
    return DenseBooleanFoldExpression(AnyOf{},static_cast<T&&>(t),dim);
}

template<class T>
decltype(auto) none_of( const DenseExpression<T>& t, std::size_t dim){
    return DenseBooleanFoldExpression(NoneOf{},static_cast<const T&>(t),dim);
}

template<class T>
decltype(auto) none_of( DenseExpression<T>&& t, std::size_t dim){
    return DenseBooleanFoldExpression(NoneOf{},static_cast<T&&>(t),dim);
}

} // namespace ultra
#endif
