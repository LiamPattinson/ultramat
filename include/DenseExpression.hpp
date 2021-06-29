#ifndef __ULTRA_DENSE_EXPRESSION_HPP
#define __ULTRA_DENSE_EXPRESSION_HPP

// DenseExpression.hpp
//
// See https://en.wikipedia.org/wiki/Expression_templates
// 
// Expressions are used to represent complex computations and to
// avoid unnecessary allocations and copies.

#include "Utils.hpp"

namespace ultra {

// Base DenseExpression
// Requires that class T has a size, dims, shape, and stride.
// Additionally requires that T is iterable and 'stripeable'.

template<class T>
struct DenseExpression {

    constexpr T& derived() noexcept { return static_cast<T&>(*this); }
    constexpr const T& derived() const noexcept { return static_cast<const T&>(*this); }

    decltype(auto) size() const { return derived().size(); }
    decltype(auto) dims() const { return derived().dims(); }
    decltype(auto) shape() const { return derived().shape(); }
    decltype(auto) shape(std::size_t ii) const { return derived().shape(ii); }
    decltype(auto) order() const { return derived().order(); }

    constexpr bool is_contiguous() const noexcept { return derived().is_contiguous(); }
    constexpr bool is_omp_parallelisable() const noexcept { return derived().is_omp_parallelisable(); }

    decltype(auto) begin() { return derived().begin(); }
    decltype(auto) begin() const { return derived().begin(); }

    decltype(auto) get_stripe(std::size_t stripe, std::size_t dim, RCOrder order) { return derived().get_stripe(stripe,dim,order); }
    decltype(auto) get_stripe(std::size_t stripe, std::size_t dim, RCOrder order) const { return derived().get_stripe(stripe,dim,order); }
    decltype(auto) num_stripes(std::size_t dim) const { return derived().num_stripes(dim); }
    decltype(auto) required_stripe_dim() const { return derived().required_stripe_dim(); } // simply return dims() if no preference.
};

// Generic exception for when expressions go wrong.

class ExpressionException : public std::runtime_error {
    using std::runtime_error::runtime_error;
};

// eval
// Forces evaluation of an expession to a temporary, and returns it.
// When applied to a container such as an array, it does not store a temporary, but rather provides
// perfect forwarding of that container.

template<class T>
using eval_result = std::conditional_t< 
    types<std::remove_cvref_t<T>>::is_dense,
    decltype(std::forward<T>(std::declval<T>())),
    Array<typename std::remove_cvref_t<T>::value_type>
>;

template<class T>
decltype(auto) eval( const DenseExpression<T>& t){
    return eval_result<T>(static_cast<const T&>(t));
}

template<class T>
decltype(auto) eval( DenseExpression<T>&& t){
    return eval_result<T>(static_cast<T&&>(t));
}

// reshape
// Performs eval, then transforms.
// Note: Other transforms, such as view/slice/permute/transpose must be performed on an lvalue, so cannot be used here.

template<class T, std::ranges::range Shape> requires std::integral<typename Shape::value_type>
decltype(auto) reshape( const DenseExpression<T>& t, const Shape& shape){
    return eval(static_cast<const T&>(t)).reshape(shape);
}

template<class T, std::ranges::range Shape> requires std::integral<typename Shape::value_type>
decltype(auto) reshape( DenseExpression<T>&& t, const Shape& shape){
    return eval(static_cast<T&&>(t)).reshape(shape);
}

// Expressions utils

struct Begin { template<class T> decltype(auto) operator()( T&& t) { return t.begin(); }};
struct End { template<class T> decltype(auto) operator()( T&& t) { return t.end(); }};
struct Deref { template<class T> decltype(auto) operator()( T&& t) { return *t; }};
struct PrefixInc { template<class T> decltype(auto) operator()( T&& t) { return ++t; }};
struct Order{ template<class T> bool operator()( T&& t) { return t.order(); }};
struct IsContiguous { template<class T> bool operator()( T&& t) { return t.is_contiguous(); }};
struct IsOmpParallelisable { template<class T> bool operator()( T&& t) { return t.is_omp_parallelisable(); }};

struct GetStripe {
    std::size_t _stripe, _dim;
    RCOrder _order;
    template<class T>
    decltype(auto) operator()( T&& t) { return t.get_stripe(_stripe,_dim,_order); }
};

// apply_to_each
// std::apply(f,tuple) calls a function f with args given by the tuple.
// apply_to_each returns a tuple given by (f(tuple_args[0]),f(tuple_args[1]),...) where f is unary.
// Similar to the possible implementation of std::apply from https://en.cppreference.com/w/cpp/utility/apply

template<class F,class Tuple, std::size_t... I>
constexpr decltype(auto) apply_to_each_impl( F&& f, Tuple&& t, std::index_sequence<I...>){
    return std::make_tuple( std::invoke(std::forward<F>(f), std::get<I>(std::forward<Tuple>(t)))...);
}

template<class F, class Tuple>
constexpr decltype(auto) apply_to_each( F&& f, Tuple&& t){
    return apply_to_each_impl( std::forward<F>(f), std::forward<Tuple>(t),
            std::make_index_sequence<std::tuple_size<std::remove_reference_t<Tuple>>::value>{});
}

// all_of_tuple/any_of_tuple
// For a tuple of bools, determine all_of/any_of

template<class Tuple, std::size_t... I>
constexpr bool all_of_tuple_impl(Tuple&& t, std::index_sequence<I...>){
    return (std::get<I>(std::forward<Tuple>(t)) & ...);
}

template<class Tuple, std::size_t... I>
constexpr bool any_of_tuple_impl(Tuple&& t, std::index_sequence<I...>){
    return (std::get<I>(std::forward<Tuple>(t)) | ...);
}

template<class Tuple>
constexpr bool all_of_tuple(Tuple&& t){
    return all_of_tuple_impl(std::forward<Tuple>(t), std::make_index_sequence<std::tuple_size<std::remove_reference_t<Tuple>>::value>{});
}

template<class Tuple>
constexpr bool any_of_tuple(Tuple&& t){
    return any_of_tuple_impl(std::forward<Tuple>(t), std::make_index_sequence<std::tuple_size<std::remove_reference_t<Tuple>>::value>{});
}

// get_common_order
// for a tuple of RCOrder, return row_major if all row_major, col_major if all col_major, and mixed_order otherwise.

template<std::size_t N, class Tuple>
struct GetCommonOrderImpl {
    RCOrder operator()( Tuple&& t) const noexcept {
        RCOrder order = std::get<N>(std::forward<Tuple>(t)).order();
        return (order  == GetCommonOrderImpl<N-1,Tuple>{}(std::forward<Tuple>(t)) ? order : RCOrder::mixed_order);
    }
};

template<class Tuple> struct GetCommonOrderImpl<0,Tuple>{
    RCOrder operator()( Tuple&& t) const noexcept{ return std::get<0>(std::forward<Tuple>(t)).order();}
};

template<class Tuple>
RCOrder get_common_order(Tuple&& t){
    return GetCommonOrderImpl<std::tuple_size<std::remove_cvref_t<Tuple>>::value-1,Tuple>{}(std::forward<Tuple>(t));
}

// ElementWiseDenseExpression
// Performs an operation taking an arbitray number of args. Operation acts independently over each arg.
// All inputs and outputs must be of the same shape.

template<class F, class... Args>
class ElementWiseDenseExpression : public DenseExpression<ElementWiseDenseExpression<F,Args...>> {

public:

    using value_type = decltype( std::apply( F{}, std::tuple<typename std::remove_cvref_t<Args>::value_type...>()));

private:
    
    // tuple type: If receiving 'Args&' or 'const Args&' as lvalue references, use this. Otherwise, if receiving rvalue reference,
    //             store Args 'by-value'. This avoids dangling references when the user chooses to return an expression from a function.
    using tuple_t = std::tuple< std::conditional_t< std::is_lvalue_reference<Args>::value, Args, std::remove_cvref_t<Args>>...>;
    tuple_t _args;

    template<std::size_t... I>
    decltype(auto) test_dims_impl(std::index_sequence<I...>) const {
        return std::array<bool,sizeof...(Args)>{{std::get<0>(_args).dims() != std::get<I>(_args).dims()...}};
    }
    decltype(auto) test_dims() const {
        return test_dims_impl(std::make_index_sequence<sizeof...(Args)>{});
    }

    template<std::size_t... I>
    decltype(auto) test_shape_impl(std::size_t ii,std::index_sequence<I...>) const {
        return std::array<bool,sizeof...(Args)>{{std::get<0>(_args).shape(ii) != std::get<I>(_args).shape(ii)...}};
    }
    decltype(auto) test_shape(std::size_t ii) const {
        return test_shape_impl(ii,std::make_index_sequence<sizeof...(Args)>{});
    }


public:

    ElementWiseDenseExpression() = delete;
    ElementWiseDenseExpression( const ElementWiseDenseExpression& ) = delete;
    ElementWiseDenseExpression( ElementWiseDenseExpression&& ) = default;
    ElementWiseDenseExpression& operator=( const ElementWiseDenseExpression& ) = delete;
    ElementWiseDenseExpression& operator=( ElementWiseDenseExpression&& ) = default;

    ElementWiseDenseExpression( Args&&... args) : _args(std::forward<Args>(args)...) {
        if( std::ranges::any_of(test_dims(),[](bool b){return b;}) ){
            throw std::runtime_error("ElementWiseDenseExpression: args have incompatible dimensions.");
        }
        for( std::size_t ii=0, end=std::get<0>(_args).dims(); ii<end; ++ii){
            if( std::ranges::any_of(test_shape(ii),[](bool b){return b;}) ){
                throw std::runtime_error("ElementWiseDenseExpression: args have incompatible shapes.");
            }
        }
        // No need to test size, this requirement is satisfied implicitly.
    }

    decltype(auto) size() const { return std::get<0>(_args).size(); }
    decltype(auto) dims() const { return std::get<0>(_args).dims(); }
    decltype(auto) shape() const { return std::get<0>(_args).shape(); }
    decltype(auto) shape(std::size_t ii) const { return std::get<0>(_args).shape(ii); }
    decltype(auto) order() const noexcept { return get_common_order(_args); }
    decltype(auto) num_stripes(std::size_t dim) const { return std::get<0>(_args).num_stripes(dim); }
    decltype(auto) required_stripe_dim() const { return dims(); }

    constexpr bool is_contiguous() const noexcept { return all_of_tuple(apply_to_each(IsContiguous{},_args)); }
    constexpr bool is_omp_parallelisable() const noexcept { return all_of_tuple(apply_to_each(IsOmpParallelisable{},_args)); }

    // Define const_iterator class
 
    // Notes:
    // begin() should return a compound iterator over Args, containing _its.
    // Dereferencing this will return F(*_its[0],*_its[1],...).
    // Incrementing this will increment _its.

    class const_iterator {
        using ItTuple = std::tuple< typename std::remove_cvref_t<Args>::const_iterator ...>;
        
        F f;
        ItTuple _its;
        
        public:
        
        const_iterator( ItTuple&& its) : f{}, _its(std::move(its)) {}
        decltype(auto) operator*() { return std::apply(f,apply_to_each(Deref{},_its)); }
        const_iterator& operator++() { apply_to_each(PrefixInc{},_its); return *this; }
    };

    const_iterator begin() const { return const_iterator(apply_to_each(Begin{},_args)); }

    // Define stripe class
    // As element-wise operations are strictly non-modifying, only read_only stripes are permitted.
 
    class Stripe {
        
        using StripeTuple = std::tuple< decltype(std::declval<std::add_const_t<std::remove_cvref_t<Args>>>().get_stripe(0,0,RCOrder::col_major)) ...>;

        StripeTuple _stripes;

        public:

        Stripe( StripeTuple&& stripes) : _stripes(std::move(stripes)) {}
        
        // Define iterator type
        // Notes:
        // begin() should return a compound iterator over Args, containing _its.
        // Dereferencing this will return F(*_its[0],*_its[1],...).
        // Incrementing this will increment _its.

        class Iterator {
            
            using ItTuple = std::tuple<typename decltype(std::declval<std::add_const_t<std::remove_cvref_t<Args>>>().get_stripe(0,0,RCOrder::col_major))::Iterator ...>;
            
            F       _f;
            ItTuple _its;
            
            public:
            
            Iterator( ItTuple&& its) : _f{}, _its(std::move(its)) {}
            decltype(auto) operator*() { return std::apply(_f,apply_to_each(Deref{},_its)); }
            Iterator& operator++() { apply_to_each(PrefixInc{},_its); return *this; }
        };
        
        Iterator begin() const { return Iterator(apply_to_each(Begin{},_stripes)); }
    };

    // Get stripes from each Arg
    decltype(auto) get_stripe( std::size_t stripe_num, std::size_t dim, RCOrder order) const {
        return Stripe(std::move(apply_to_each(GetStripe{stripe_num,dim,order},_args)));
    }
};

// ScalarDenseExpression
// Broadcast a scalar to a given shape, allowing it to take part in DenseExpressions.

template<class T>
class ScalarDenseExpression : public DenseExpression<ScalarDenseExpression<T>> {

public:

    using value_type = std::remove_cvref_t<T>;
    static_assert( std::is_arithmetic<T>::value, "Ultramat: Cannot broadcast given type, must be arithmetic type");

private:
    
    value_type               _scalar;
    std::vector<std::size_t> _shape;
    RCOrder                  _rc_order;

public:

    ScalarDenseExpression() = delete;
    ScalarDenseExpression( const ScalarDenseExpression& ) = delete;
    ScalarDenseExpression( ScalarDenseExpression&& ) = default;
    ScalarDenseExpression& operator=( const ScalarDenseExpression& ) = delete;
    ScalarDenseExpression& operator=( ScalarDenseExpression&& ) = default;

    template<std::ranges::range Shape> requires ( std::is_integral<typename Shape::value_type>::value )
    ScalarDenseExpression( T t, const Shape& shape, RCOrder rc_order) : _scalar(t), _shape(shape.size()), _rc_order(rc_order) {
        std::ranges::copy(shape,_shape.begin());
    }

    decltype(auto) size() const { return std::accumulate(_shape.begin(),_shape.end(),1,std::multiplies<std::size_t>{}); }
    decltype(auto) dims() const { return _shape.size(); }
    decltype(auto) shape() const { return _shape; }
    decltype(auto) shape(std::size_t ii) const { return _shape[ii]; }
    decltype(auto) order() const noexcept { return _rc_order; }
    decltype(auto) num_stripes(std::size_t dim) const { return size()/shape(dim); }
    decltype(auto) required_stripe_dim() const { return dims(); }

    constexpr bool is_contiguous() const noexcept { return true; }
    constexpr bool is_omp_parallelisable() const noexcept { return true; }

    // Define const_iterator class

    class const_iterator {
        
        value_type _t;
        
        public:
        
        const_iterator( value_type t) : _t(t) {}
        decltype(auto) operator*() { return _t; }
        const_iterator& operator++() { /* Do nothing! */ return *this; }
    };

    const_iterator begin() const { return const_iterator(_scalar); }

    // Define stripe class

    class Stripe {
        
        value_type _t;

        public:

        Stripe( value_type t) : _t(t) {}
        
        // Borrow iterator type
        using Iterator = const_iterator;
        Iterator begin() const { return Iterator(_t); }
    };

    decltype(auto) get_stripe( std::size_t, std::size_t, RCOrder ) const {
        return Stripe(_scalar);
    }
};

// FoldDenseExpression
// Binary operation over a single arg. Performs a 'fold' over a single dimension.
// Full reductions are performed by iteratively folding until only 1 dimension is left.

// Inherits from one of three 'policy' classes
// - GeneralFold: start_val is provided of type StartT, takes function of form `StartT f(StartT,T) `
// - Accumulate: start_val is not provided, takes function of form `T f(T,T)`. Includes min, max, sum, prod
// - BooleanFold: all_of, any_of, none_of, all may stop early under the right conditions

template<class F, class StartType, class T>
class GeneralFoldPolicy {
protected:
    using input_value_type = typename std::remove_cvref_t<T>::value_type;
    using start_type = StartType;
    using value_type = decltype(std::declval<F>()(std::declval<start_type>(),std::declval<input_value_type>()));
    static_assert( std::is_convertible<start_type,std::remove_cvref_t<value_type>>::value );
    static constexpr bool is_general_fold = true;
    static constexpr bool is_accumulate = false;
    static constexpr bool is_boolean_fold = false;
    start_type _start_val;
public:
    GeneralFoldPolicy( const start_type& start_val) : _start_val(start_val) {}
    start_type get() const { return _start_val; }
};

template<class F, class ValueType, class T>
class AccumulatePolicy {
protected:
    using input_value_type = typename std::remove_cvref_t<T>::value_type;
    using value_type = decltype(std::declval<F>()(std::declval<input_value_type>(),std::declval<input_value_type>()));
    static_assert( std::is_convertible<ValueType,input_value_type>::value );
    static constexpr bool is_general_fold = false;
    static constexpr bool is_accumulate = true;
    static constexpr bool is_boolean_fold = false;
};

template<class F, class ValueType, class T>
class BooleanFoldPolicy {
    static_assert( std::is_same<ValueType,bool>::value );
protected:
    using value_type = ValueType;
    using input_value_type = typename std::remove_cvref_t<T>::value_type;
    using result_type = decltype(std::declval<F>()(std::declval<ValueType>(),std::declval<input_value_type>()));
    static_assert( std::is_same<result_type,value_type>::value );
    static constexpr bool is_general_fold = false;
    static constexpr bool is_accumulate = false;
    static constexpr bool is_boolean_fold = true;
};

template<class F, class ValueType, class T, template<class,class,class> class FoldPolicy>
class FoldDenseExpressionImpl : public DenseExpression<FoldDenseExpressionImpl<F,ValueType,T,FoldPolicy>>, public FoldPolicy<F,ValueType,T> {
    
public:

    using input_value_type = FoldPolicy<F,ValueType,T>::input_value_type;
    using value_type = FoldPolicy<F,ValueType,T>::value_type;

    static constexpr bool is_general_fold = FoldPolicy<F,ValueType,T>::is_general_fold;
    static constexpr bool is_accumulate = FoldPolicy<F,ValueType,T>::is_accumulate;
    static constexpr bool is_boolean_fold = FoldPolicy<F,ValueType,T>::is_boolean_fold;

private:

    using arg_t = std::conditional_t< std::is_lvalue_reference<T>::value, T, std::remove_cvref_t<T>>;
    using ref_t = const std::remove_cvref_t<arg_t>&;
    using Stripe_t = decltype(std::declval<std::add_const_t<std::remove_cvref_t<T>>>().get_stripe(0,0,RCOrder::col_major));

    F           _f;
    arg_t       _t;
    std::size_t _fold_dim;
    std::size_t _fold_size;

public:

    FoldDenseExpressionImpl() = delete;
    FoldDenseExpressionImpl( const FoldDenseExpressionImpl& ) = delete;
    FoldDenseExpressionImpl( FoldDenseExpressionImpl&& ) = default;
    FoldDenseExpressionImpl& operator=( const FoldDenseExpressionImpl& ) = delete;
    FoldDenseExpressionImpl& operator=( FoldDenseExpressionImpl&& ) = default;

    FoldDenseExpressionImpl( const F& f, T&& t, const ValueType& start_val, std::size_t fold_dim ) : 
        FoldPolicy<F,ValueType,T>(start_val),
        _f(f),
        _t(std::forward<T>(t)),
        _fold_dim(fold_dim),
        _fold_size(t.shape(fold_dim))
    {
        if( _fold_dim >= _t.dims() ) throw ExpressionException("Ultramat: Fold dimension must be smaller than dims()");
    }

    FoldDenseExpressionImpl( const F& f, T&& t, std::size_t fold_dim ) : 
        _f(f),
        _t(std::forward<T>(t)),
        _fold_dim(fold_dim),
        _fold_size(t.shape(fold_dim))
    {
        if( _fold_dim >= _t.dims() ) throw ExpressionException("Ultramat: Fold dimension must be smaller than dims()");
    }

    std::size_t size() const { 
        std::size_t result=1;
        for(std::size_t ii=0; ii<dims(); ++ii) result*=shape(ii);
        return result;
    }

    std::size_t dims() const { return std::max((std::size_t)1,_t.dims()-1); }
    decltype(auto) shape(std::size_t ii) const { return (_t.dims()==1 ? 1 : _t.shape(ii < _fold_dim ? ii : ii+1)); }
    decltype(auto) order() const noexcept { return _t.order(); }
    decltype(auto) num_stripes(std::size_t dim) const { return size()/shape(dim); }
    constexpr bool is_contiguous() const noexcept { return _t.is_contiguous(); }
    constexpr bool is_omp_parallelisable() const noexcept { return _t.is_omp_parallelisable(); }
    decltype(auto) required_stripe_dim() const { return dims(); }

    // Define const_iterator class
 
    // Notes:
    // begin() should return a const_iterator containing a stripe generator over _t, passing on the stripe dim
    // Dereferencing this will generate a stripe, perform the fold operation, and return the result.
    // Incrementing this will increment the stripe generator.

    class const_iterator : public FoldPolicy<F,ValueType,T> {

        F           _f;
        ref_t       _t;
        std::size_t _stripe_dim;
        RCOrder     _order;
        std::size_t _stripe_num;
        std::size_t _stripe_inc;
        
        public:
        
        const_iterator( const F& f, ref_t t, std::size_t stripe_dim, RCOrder order, std::size_t stripe_num, std::size_t stripe_inc, value_type start_val) :
            FoldPolicy<F,ValueType,T>(start_val),
            _f(f),
            _t(t),
            _stripe_dim(stripe_dim),
            _order(order),
            _stripe_num(stripe_num),
            _stripe_inc(stripe_inc)
        {}

        const_iterator( const F& f, ref_t t, std::size_t stripe_dim, RCOrder order, std::size_t stripe_num, std::size_t stripe_inc ) :
            _f(f),
            _t(t),
            _stripe_dim(stripe_dim),
            _order(order),
            _stripe_num(stripe_num),
            _stripe_inc(stripe_inc)
        {}

        const_iterator( const F& f, ref_t t, std::size_t stripe_dim, std::size_t stripe_num, std::size_t stripe_inc, value_type start_val) :
            const_iterator(f,t,stripe_dim,t.order(),stripe_num,stripe_inc,start_val)
        {}

        const_iterator( const F& f, ref_t t, std::size_t stripe_dim, std::size_t stripe_num, std::size_t stripe_inc ) :
            const_iterator(f,t,stripe_dim,t.order(),stripe_num,stripe_inc)
        {}

        decltype(auto) operator*() requires (is_general_fold) {
            F f = _f;
            value_type val = FoldPolicy<F,ValueType,T>::get();
            auto stripe = _t.get_stripe(_stripe_num,_stripe_dim,_order);
            for( auto&& x : stripe ) val = f(val,x);
            return val;
        }

        decltype(auto) operator*() requires (is_accumulate) {
            F f = _f;
            auto stripe = _t.get_stripe(_stripe_num,_stripe_dim,_order);
            auto it = stripe.begin();
            auto end = stripe.end();
            std::remove_cvref_t<value_type> val = *it;
            ++it;
            for(; it != end; ++it ) val = f(val,*it);
            return val;
        }

        decltype(auto) operator*() requires (is_boolean_fold) {
            // Rather than using F directly, we will instead make use of start_bool and early_exit_bool 
            bool result = F::start_bool;
            auto stripe = _t.get_stripe(_stripe_num,_stripe_dim,_order);
            for( auto&& x : stripe){
                if( x == F::early_exit_bool ) return !result;
            }
            return result;
        }

        const_iterator& operator++() { _stripe_num+=_stripe_inc; return *this; }
        bool operator==(const const_iterator& it) { return _stripe_num == it._stripe_num; }
    };

    const_iterator begin() const requires (is_general_fold) {
        return const_iterator(_f,_t,_fold_dim,0,1,FoldPolicy<F,ValueType,T>::get());
    }
    
    const_iterator end() const requires (is_general_fold) {
        return const_iterator(_f,_t,_fold_dim,num_stripes(),1,FoldPolicy<F,ValueType,T>::get());
    }

    const_iterator begin() const requires (!is_general_fold) {
        return const_iterator(_f,_t,_fold_dim,0,1);
    }

    const_iterator end() const requires (!is_general_fold) {
        return const_iterator(_f,_t,_fold_dim,num_stripes(),1);
    }

    // Define stripe class

    class Stripe : public FoldPolicy<F,ValueType,T> {

        F           _f;
        ref_t       _t;
        std::size_t _fold_dim;
        RCOrder     _order;

        std::size_t _start_stripe_num;
        std::size_t _end_stripe_num;
        std::size_t _stripe_num_inc;

        public:
        
        Stripe( const F& f, ref_t t, std::size_t fold_dim, RCOrder order,  const value_type& val,
                std::size_t start_stripe_num, std::size_t end_stripe_num, std::size_t stripe_num_inc):
            FoldPolicy<F,ValueType,T>(val),
            _f(f),
            _t(t),
            _fold_dim(fold_dim),
            _order(order),
            _start_stripe_num(start_stripe_num),
            _end_stripe_num(end_stripe_num),
            _stripe_num_inc(stripe_num_inc)
        {}

        Stripe( const F& f, ref_t t, std::size_t fold_dim, RCOrder order,
                std::size_t start_stripe_num, std::size_t end_stripe_num, std::size_t stripe_num_inc):
            _f(f),
            _t(t),
            _fold_dim(fold_dim),
            _order(order),
            _start_stripe_num(start_stripe_num),
            _end_stripe_num(end_stripe_num),
            _stripe_num_inc(stripe_num_inc)
        {}
 
        // Define iterator type
        using Iterator = const_iterator;

        Iterator begin() const requires (is_general_fold) {
            return Iterator( _f, _t, _fold_dim, _order, _start_stripe_num, _stripe_num_inc, FoldPolicy<F,ValueType,T>::get());
        }

        Iterator begin() const requires (!is_general_fold) {
            return Iterator( _f, _t, _fold_dim, _order, _start_stripe_num, _stripe_num_inc);
        }

        Iterator end() const requires (is_general_fold) {
            return Iterator( _f, _t, _fold_dim, _order, _end_stripe_num, _stripe_num_inc, FoldPolicy<F,ValueType,T>::get());
        }

        Iterator end() const requires (!is_general_fold) {
            return Iterator( _f, _t, _fold_dim, _order, _end_stripe_num, _stripe_num_inc);
        }
    };

    // Stripe start/end/inc helper
    decltype(auto) get_stripe_info( std::size_t stripe_num, std::size_t dim, RCOrder order) const {
        std::size_t start_stripe_num, end_stripe_num, stripe_num_inc;
        std::size_t first_dim = ( order == RCOrder::col_major ? 0 : dims()-1);
        // Get increment first
        stripe_num_inc = 1;
        for(std::size_t ii=first_dim; ii != dim; ii += (order==RCOrder::col_major ? 1 : -1)){
            stripe_num_inc *= shape(ii);
        }
        // Get start stripe number
        if( dim==first_dim ){
            start_stripe_num = stripe_num*shape(first_dim);
        } else {
            start_stripe_num = (stripe_num % stripe_num_inc) + (stripe_num / stripe_num_inc)*stripe_num_inc*shape(dim);
        }
        // Combine the two to get end stripe number
        end_stripe_num = start_stripe_num + stripe_num_inc*shape(dim);
        return std::make_tuple(start_stripe_num,end_stripe_num,stripe_num_inc);
    }

    // Get stripe from _t
    decltype(auto) get_stripe( std::size_t stripe_num, std::size_t dim, RCOrder order) const requires (is_general_fold) {
        std::size_t start_stripe_num, end_stripe_num, stripe_num_inc;
        std::tie(start_stripe_num,end_stripe_num,stripe_num_inc) = get_stripe_info(stripe_num,dim,order);
        return Stripe(_f,_t,_fold_dim,order,FoldPolicy<F,ValueType,T>::get(),start_stripe_num,end_stripe_num,stripe_num_inc);
    }

    decltype(auto) get_stripe( std::size_t stripe_num, std::size_t dim, RCOrder order) const requires (!is_general_fold) {
        std::size_t start_stripe_num, end_stripe_num, stripe_num_inc;
        std::tie(start_stripe_num,end_stripe_num,stripe_num_inc) = get_stripe_info(stripe_num,dim,order);
        return Stripe(_f,_t,_fold_dim,order,start_stripe_num,end_stripe_num,stripe_num_inc);
    }
};

template<class F, class ValueType, class T> using FoldDenseExpression = FoldDenseExpressionImpl<F,ValueType,T,GeneralFoldPolicy>;
template<class F, class T> using AccumulateDenseExpression = FoldDenseExpressionImpl<F,typename std::remove_cvref_t<T>::value_type,T,AccumulatePolicy>;
template<class F, class T> using BooleanFoldDenseExpression = FoldDenseExpressionImpl<F,bool,T,BooleanFoldPolicy>;

// CumulativeDenseExpression
// Binary operation over a single arg. Returns something of the same shape.
// For multi-dimensional arrays, sums over the given dimension only. Defaults to zero.
// Unlike FoldDenseExpression, only an accumulating version exists.
// A more general implementation will only be included if a good use case can be demonstrated.

template<class F, class T>
class CumulativeDenseExpression : public DenseExpression<CumulativeDenseExpression<F,T>> {

public:

    using inner_value_type = typename std::remove_cvref_t<T>::value_type;
    using value_type = decltype(std::declval<F>()(std::declval<inner_value_type>(),std::declval<inner_value_type>()));
    static_assert( std::is_convertible<inner_value_type,value_type>::value );

private:

    using arg_t = std::conditional_t< std::is_lvalue_reference<T>::value, T, std::remove_cvref_t<T>>;
    using function_type = F;

    function_type _f;
    arg_t         _t;
    std::size_t   _dim;

public:

    CumulativeDenseExpression() = delete;
    CumulativeDenseExpression( const CumulativeDenseExpression& ) = delete;
    CumulativeDenseExpression( CumulativeDenseExpression&& ) = default;
    CumulativeDenseExpression& operator=( const CumulativeDenseExpression& ) = delete;
    CumulativeDenseExpression& operator=( CumulativeDenseExpression&& ) = default;

    CumulativeDenseExpression( const function_type& f, T&& t, std::size_t dim) : _f(f), _t(std::forward<T>(t)) , _dim(dim) {}

    decltype(auto) size() const { return _t.size(); }
    decltype(auto) dims() const { return _t.dims(); }
    decltype(auto) shape() const { return _t.shape(); }
    decltype(auto) shape(std::size_t ii) const { return _t.shape(ii); }
    decltype(auto) order() const noexcept { return _t.order(); }

    decltype(auto) num_stripes(std::size_t dim) const {
        if( dim != _dim ){
            std::string err = "Ultramat: CumulativeDenseExpressions may only be striped in the direction specified upon their creation.";
            err += " num_stripes was given dim of " + std::to_string(dim) + ", but expected " + std::to_string(_dim) + '.';
            throw std::runtime_error(err);
        }
        return _t.num_stripes(dim);
    }
    
    decltype(auto) required_stripe_dim() const { return _dim; }

    // CumulativeDenseExpressions cannot be performed in parallel and must make use of striped iteration, hence will
    // appear as non-contiguous and non-omp-parallel. Each stripe may still be determined in parallel however.
    constexpr bool is_contiguous() const noexcept { return false; }
    constexpr bool is_omp_parallelisable() const noexcept { return false; }

    // Define stripe class

    class Stripe {
        
        using Stripe_t = decltype(std::declval<std::add_const_t<std::remove_cvref_t<T>>>().get_stripe(0,0,RCOrder::col_major));

        function_type     _f;
        Stripe_t          _stripe;

        public:

        Stripe( const function_type& f, Stripe_t&& stripe) : _f(f), _stripe(std::move(stripe)) {}
        
        // Define iterator type

        class Iterator {
            
            using It_t = typename decltype(std::declval<std::add_const_t<std::remove_cvref_t<T>>>().get_stripe(0,0,RCOrder::col_major))::Iterator;
            
            function_type _f;
            It_t          _it;
            bool          _first;
            value_type    _val;
            
            public:
           
            Iterator( const function_type& f, It_t&& it) : _f(f), _it(std::move(it)), _first(true) {}
            Iterator& operator++() { ++_it; return *this; }
            decltype(auto) operator*() {
                if( _first ){
                    _first = false;
                    _val = *_it;
                } else {
                    _val = _f(*_it,_val); 
                }    
                return _val;
            }
        };
        
        Iterator begin() const { return Iterator(_f,_stripe.begin()); }
    };

    // Get stripe from _t
    decltype(auto) get_stripe( std::size_t stripe_num, std::size_t dim, RCOrder order) const {
        if( dim != _dim ){
            std::string err = "Ultramat: CumulativeDenseExpressions may only be striped in the direction specified upon their creation.";
            err += " get_stripe was given dim of " + std::to_string(dim) + ", but expected " + std::to_string(_dim) + '.';
            throw std::runtime_error(err);
        }
        return Stripe(_f,std::move(_t.get_stripe(stripe_num,dim,order)));
    }

    // Define const_iterator dummy class

    struct const_iterator {
        const_iterator() { throw std::runtime_error("Ultramat CumulativeDenseExpression: Must use striped iteration!");}
        decltype(auto) operator*() { return 0; }
        const_iterator& operator++() { return *this; }
    };
    const_iterator begin() const { return const_iterator(); }
};

// Define GeneratorExpression:

template<class F>
class GeneratorExpression : public DenseExpression<GeneratorExpression<F>> {
    
public:

    using function_type = std::remove_cvref_t<F>;
    using value_type = decltype(std::declval<function_type>()(static_cast<std::size_t>(0)));

private:

    function_type            _f;
    std::vector<std::size_t> _shape;
    std::size_t              _size;

public:

    GeneratorExpression() = delete;
    GeneratorExpression( const GeneratorExpression& ) = delete;
    GeneratorExpression( GeneratorExpression&& ) = default;
    GeneratorExpression& operator=( const GeneratorExpression& ) = delete;
    GeneratorExpression& operator=( GeneratorExpression&& ) = default;

    template<std::ranges::sized_range Shape>
    GeneratorExpression( F&& f, const Shape& shape) : _f(std::forward<F>(f)), _shape(shape.size()) {
        std::ranges::copy( shape, _shape.begin());
        _size = std::accumulate( _shape.begin(), _shape.end(), 1, std::multiplies<std::size_t>() );
    }

    decltype(auto) size() const { return _size; }
    decltype(auto) dims() const { return _shape.size(); }
    decltype(auto) shape() const { return _shape; }
    decltype(auto) shape(std::size_t ii) const { return _shape[ii]; }
    decltype(auto) order() const noexcept { return default_rc_order; }
    decltype(auto) num_stripes(std::size_t dim) const { return _size/_shape[dim]; }
    decltype(auto) required_stripe_dim() const { return dims(); }

    constexpr bool is_contiguous() const noexcept { return true; }
    constexpr bool is_omp_parallelisable() const noexcept { return true; }

    // Define iterator class

    class const_iterator {
        
        function_type _f;
        std::size_t   _count;

        public:
        
        const_iterator( const function_type& f) : _f(f), _count(0) {}
        decltype(auto) operator*() { return _f(_count); }
        const_iterator& operator++() { ++_count; return *this; }
    };

    const_iterator begin() const { return const_iterator(_f); }

    // Define stripe class
 
    class Stripe {

        function_type _f;

        public:

        Stripe( const function_type& f) : _f(f) {}

        using Iterator = const_iterator;
        const_iterator begin() const { return const_iterator(_f); }
    };

    // Get stripes from each Arg
    decltype(auto) get_stripe( std::size_t, std::size_t, RCOrder) const {
        return Stripe(_f);
    }
};

} // namespace
#endif
