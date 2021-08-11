#ifndef __ULTRA_DENSE_EXPRESSION_HPP
#define __ULTRA_DENSE_EXPRESSION_HPP

// DenseExpression.hpp
//
// See https://en.wikipedia.org/wiki/Expression_templates
// 
// Expressions are used to represent complex computations and to
// avoid unnecessary allocations and copies.

#include "DenseUtils.hpp"

namespace ultra {

// Base DenseExpression
// Requires that class T has a size, dims, shape, and stride.
// Additionally requires that T is iterable and 'stripeable'.

template<class T>
class DenseExpression {
    constexpr T& derived() noexcept { return static_cast<T&>(*this); }
    constexpr const T& derived() const noexcept { return static_cast<const T&>(*this); }
    
    public:

    constexpr decltype(auto) size() const { return derived().size(); }
    constexpr decltype(auto) dims() const { return derived().dims(); }
    constexpr decltype(auto) shape() const { return derived().shape(); }
    constexpr decltype(auto) shape(std::size_t ii) const { return derived().shape(ii); }
    static constexpr decltype(auto) order() { return T::order(); }

    constexpr bool is_contiguous() const noexcept { return derived().is_contiguous(); }
    constexpr bool is_broadcasting() const noexcept { return derived().is_broadcasting(); }
    constexpr bool is_omp_parallelisable() const noexcept { return derived().is_omp_parallelisable(); }

    constexpr decltype(auto) begin() { return derived().begin(); }
    constexpr decltype(auto) begin() const { return derived().begin(); }
    constexpr decltype(auto) end() { return derived().end(); }
    constexpr decltype(auto) end() const { return derived().end(); }

    decltype(auto) get_stripe( const DenseStriper& striper) { return derived().get_stripe(striper); }
    decltype(auto) get_stripe( const DenseStriper& striper) const { return derived().get_stripe(striper); }
    decltype(auto) required_stripe_dim() const { return derived().required_stripe_dim(); } // simply return dims() if no preference.
};

// Generic exception for when expressions go wrong.

class ExpressionException : public std::runtime_error {
    using std::runtime_error::runtime_error;
};

// eval
// Forces evaluation of an expession to a temporary, and returns it.

template<class T>
using eval_result = Dense<
    typename std::remove_cvref_t<T>::value_type,
    DenseType::nd,
    (std::remove_cvref_t<T>::order() == DenseOrder::mixed ? default_order : std::remove_cvref_t<T>::order())>;

template<class T>
decltype(auto) eval( const DenseExpression<T>& t){
    return eval_result<T>(static_cast<const T&>(t));
}

template<class T>
decltype(auto) eval( DenseExpression<T>&& t){
    return eval_result<T>(static_cast<T&&>(t));
}

// evaluating transforms
// Performs eval, then transforms. Creates a new Array. so not recommended in general.

template<class T, class... Slices> requires ( std::is_same<Slice,Slices>::value && ...)
eval_result<T> view( const DenseExpression<T>& t, const Slices&... slices){
    return eval(eval(static_cast<const T&>(t)).view(slices...));
}

template<class T, class... Slices> requires ( std::is_same<Slice,Slices>::value && ...)
eval_result<T> view( DenseExpression<T>&& t, const Slices&... slices){
    return eval(eval(static_cast<T&&>(t)).view(slices...));
}

template<class T, std::ranges::range Slices> requires ( std::is_same<Slice,typename Slices::value_type>::value )
eval_result<T> view( const DenseExpression<T>& t, const Slices& slices){
    return eval(eval(static_cast<const T&>(t)).view(slices));
}

template<class T, std::ranges::range Slices> requires ( std::is_same<Slice,typename Slices::value_type>::value )
eval_result<T> view( DenseExpression<T>&& t, const Slices& slices){
    return eval(eval(static_cast<T&&>(t)).view(slices));
}

template<class T, shapelike Shape>
eval_result<T> reshape( const DenseExpression<T>& t, const Shape& shape){
    return eval(eval(static_cast<const T&>(t)).reshape(shape));
}

template<class T, shapelike Shape>
eval_result<T> reshape( DenseExpression<T>&& t, const Shape& shape){
    return eval(eval(static_cast<T&&>(t)).reshape(shape));
}

template<class T, std::integral... Ints>
eval_result<T> reshape( const DenseExpression<T>& t, Ints... ints){
    return eval(eval(static_cast<const T&>(t)).reshape(ints...));
}

template<class T, std::integral... Ints>
eval_result<T> reshape( DenseExpression<T>&& t, Ints... ints){
    return eval(eval(static_cast<T&&>(t)).reshape(ints...));
}

template<class T, shapelike Shape>
eval_result<T> permute( const DenseExpression<T>& t, const Shape& shape){
    return eval(eval(static_cast<const T&>(t)).permute(shape));
}

template<class T, shapelike Shape>
eval_result<T> permute( DenseExpression<T>&& t, const Shape& shape){
    return eval(eval(static_cast<T&&>(t)).permute(shape));
}

template<class T, std::integral... Perm>
eval_result<T> permute( const DenseExpression<T>& t, Perm... permutations){
    return eval(eval(static_cast<const T&>(t)).permute(permutations...));
}

template<class T, std::integral... Perm>
eval_result<T> permute( DenseExpression<T>&& t, Perm... permutations){
    return eval(eval(static_cast<const T&&>(t)).permute(permutations...));
}

template<class T>
decltype(auto) transpose( const DenseExpression<T>& t) {
    return eval(eval(static_cast<const T&>(t)).transpose());
}

template<class T>
decltype(auto) transpose( DenseExpression<T>&& t) {
    return eval(eval(static_cast<T&&>(t)).transpose());
}

template<class T>
decltype(auto) hermitian( const DenseExpression<T>& t) {
    return conj(eval(eval(static_cast<const T&>(t)).transpose()));
}

template<class T>
decltype(auto) hermitian( DenseExpression<T>&& t) {
    return conj(eval(eval(static_cast<T&&>(t)).transpose()));
}

// ElementWiseDenseExpression
// Performs an operation taking an arbitray number of args. Operation acts independently over each arg.
// All inputs and outputs must be of the same shape.

template<class F, class... Args>
class ElementWiseDenseExpression : public DenseExpression<ElementWiseDenseExpression<F,Args...>> {

public:

    using value_type = decltype( std::apply( F{}, std::tuple<typename std::remove_cvref_t<Args>::value_type...>()));

private:
    
    std::vector<std::size_t> _shape;

    // tuple type: If receiving 'Args&' or 'const Args&' as lvalue references, use this. Otherwise, if receiving rvalue reference,
    //             store Args 'by-value'. This avoids dangling references when the user chooses to return an expression from a function.
    using tuple_t = std::tuple< std::conditional_t< std::is_lvalue_reference<Args>::value, Args, std::remove_cvref_t<Args>>...>;
    tuple_t _args;

public:

    ElementWiseDenseExpression() = delete;
    ElementWiseDenseExpression( const ElementWiseDenseExpression& ) = delete;
    ElementWiseDenseExpression( ElementWiseDenseExpression&& ) = default;
    ElementWiseDenseExpression& operator=( const ElementWiseDenseExpression& ) = delete;
    ElementWiseDenseExpression& operator=( ElementWiseDenseExpression&& ) = default;

    ElementWiseDenseExpression( Args&&... args) : 
        _shape( get_broadcast_shape<order()>(args.shape()...)),
        _args(std::forward<Args>(args)...)
    {}

    decltype(auto) size() const { return std::accumulate(_shape.begin(),_shape.end(),1,std::multiplies<std::size_t>{}); }
    decltype(auto) dims() const { return _shape.size(); }
    decltype(auto) shape() const { return _shape; }
    decltype(auto) shape(std::size_t ii) const { return _shape[ii]; }
    decltype(auto) required_stripe_dim() const { return dims(); }

    static constexpr DenseOrder order() { return get_common_order<Args...>::value; }

    constexpr bool is_contiguous() const noexcept { return all_of_tuple(apply_to_each(_IsContiguous{},_args)); }
    constexpr bool is_omp_parallelisable() const noexcept { return all_of_tuple(apply_to_each(_IsOmpParallelisable{},_args)); }

    bool is_broadcasting() const { 
        // Are args broadcasting?
        bool result = any_of_tuple(apply_to_each(_IsBroadcasting{},_args));
        if( result ) return result;
        // Do args have mismatching dims?
        result |= any_of_tuple(apply_to_each(_DimsNeq{dims()},_args));
        if( result ) return result;
        // Do args have mismatching shapes?
        result |= any_of_tuple(apply_to_each(_ShapeNeq{_shape},_args));
        return result;
    }

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
        
        const_iterator() = delete;
        const_iterator( const const_iterator& ) = default;
        const_iterator( const_iterator&& ) = default;
        const_iterator& operator=( const const_iterator& ) = default;
        const_iterator& operator=( const_iterator&& ) = default;

        const_iterator( ItTuple&& its) : f{}, _its(std::move(its)) {}

        decltype(auto) operator*() { return std::apply(f,apply_to_each(_Deref{},_its)); }
        const_iterator& operator++() { increment_tuple(_its); return *this; }
        const_iterator& operator--() { decrement_tuple(_its); return *this; }
        template<std::integral I> const_iterator& operator+=( const I& ii) { add_in_place_tuple(_its,ii); return *this; }
        template<std::integral I> const_iterator& operator-=( const I& ii) { sub_in_place_tuple(_its,ii); return *this; }
        template<std::integral I> const_iterator operator+( const I& ii) { auto result(*this); result+=ii; return result; }
        template<std::integral I> const_iterator operator-( const I& ii) { auto result(*this); result-=ii; return result; }
        bool operator==( const const_iterator& other) const { return std::get<0>(_its) == std::get<0>(other._its);}
        auto operator<=>( const const_iterator& other) const { return std::get<0>(_its) <=> std::get<0>(other._its);}
        std::ptrdiff_t operator-( const const_iterator& other) const { return std::get<0>(_its) - std::get<0>(other._its);}
    };

    const_iterator begin() const { return const_iterator(apply_to_each(_Begin{},_args)); }
    const_iterator end()   const { return const_iterator(apply_to_each(_End{},_args)); }

    // Define stripe class
    // As element-wise operations are strictly non-modifying, only read_only stripes are permitted.
 
    class Stripe {
        
        using StripeTuple = std::tuple< decltype(std::declval<std::add_const_t<std::remove_cvref_t<Args>>>().get_stripe(std::declval<DenseStriper>())) ...>;

        StripeTuple _stripes;

        public:

        Stripe( StripeTuple&& stripes) : _stripes(std::move(stripes)) {}
        
        // Define iterator type
        // Notes:
        // begin() should return a compound iterator over Args, containing _its.
        // Dereferencing this will return F(*_its[0],*_its[1],...).
        // Incrementing this will increment _its.

        class Iterator {
            
            using ItTuple = std::tuple<typename decltype(std::declval<std::add_const_t<std::remove_cvref_t<Args>>>().get_stripe(std::declval<DenseStriper>()))::Iterator ...>;
            
            F       _f;
            ItTuple _its;
            
            public:

            Iterator() = delete;
            Iterator( const Iterator& ) = default;
            Iterator( Iterator&& ) = default;
            Iterator& operator=( const Iterator& ) = default;
            Iterator& operator=( Iterator&& ) = default;
            
            Iterator( ItTuple&& its) : _f{}, _its(std::move(its)) {}

            decltype(auto) operator*() { return std::apply(_f,apply_to_each(_Deref{},_its)); }
            Iterator& operator++() { increment_tuple(_its); return *this; }
            Iterator& operator--() { decrement_tuple(_its); return *this; }
            template<std::integral I> Iterator& operator+=( const I& ii) { add_in_place_tuple(_its,ii); return *this; }
            template<std::integral I> Iterator& operator-=( const I& ii) { sub_in_place_tuple(_its,ii); return *this; }
            template<std::integral I> Iterator operator+( const I& ii) { auto result(*this); result+=ii; return result; }
            template<std::integral I> Iterator operator-( const I& ii) { auto result(*this); result-=ii; return result; }
            bool operator==( const Iterator& other) const { return std::get<0>(_its) == std::get<0>(other._its);}
            auto operator<=>( const Iterator& other) const { return std::get<0>(_its) <=> std::get<0>(other._its);}
            std::ptrdiff_t operator-( const Iterator& other) const { return std::get<0>(_its) - std::get<0>(other._its);}
        };
        
        Iterator begin() const { return Iterator(apply_to_each(_Begin{},_stripes)); }
        Iterator end()   const { return Iterator(apply_to_each(_End{},_stripes)); }
    };

    decltype(auto) get_stripe( const DenseStriper& striper) const {
        return Stripe(apply_to_each(_GetStripe{striper},_args));
    }
};

// FoldDenseExpression
// Binary operation over a single arg. Performs a 'fold' over a single dimension.
// Full reductions are performed by iteratively folding until only 1 dimension is left.

// Inherits from one of four 'policy' classes
// - GeneralFold: start_val is provided of type StartT, takes function of form `StartT f(T.begin(),T.end(),StartT) `
// - LinearFold: start_val is provided of type StartT, takes function of form `StartT f(StartT,T) `
// - Accumulate: start_val is not provided, takes function of form `T f(T,T)`. Includes min, max, sum, prod
// - BooleanFold: all_of, any_of, none_of, all may stop early under the right conditions

template<class F, class StartType, class T>
class GeneralFoldPolicy {
protected:
    using input_type = typename std::remove_cvref_t<T>;
    using input_value_type = typename input_type::value_type;
    using start_type = StartType;
    using value_type = decltype(std::declval<F>()(std::declval<input_type>().begin(),std::declval<input_type>().end(),std::declval<start_type>()));
    static_assert( std::is_convertible<start_type,std::remove_cvref_t<value_type>>::value );
    static constexpr bool is_general_fold = true;
    static constexpr bool is_linear_fold = false;
    static constexpr bool is_accumulate = false;
    static constexpr bool is_boolean_fold = false;
    start_type _start_val;
public:
    GeneralFoldPolicy( const start_type& start_val) : _start_val(start_val) {}
    start_type get() const { return _start_val; }
};

template<class F, class StartType, class T>
class LinearFoldPolicy {
protected:
    using input_value_type = typename std::remove_cvref_t<T>::value_type;
    using start_type = StartType;
    using value_type = decltype(std::declval<F>()(std::declval<start_type>(),std::declval<input_value_type>()));
    static_assert( std::is_convertible<start_type,std::remove_cvref_t<value_type>>::value );
    static constexpr bool is_general_fold = false;
    static constexpr bool is_linear_fold = true;
    static constexpr bool is_accumulate = false;
    static constexpr bool is_boolean_fold = false;
    start_type _start_val;
public:
    LinearFoldPolicy( const start_type& start_val) : _start_val(start_val) {}
    start_type get() const { return _start_val; }
};

template<class F, class ValueType, class T>
class AccumulatePolicy {
protected:
    using input_value_type = typename std::remove_cvref_t<T>::value_type;
    using value_type = decltype(std::declval<F>()(std::declval<input_value_type>(),std::declval<input_value_type>()));
    static_assert( std::is_convertible<ValueType,input_value_type>::value );
    static constexpr bool is_general_fold = false;
    static constexpr bool is_linear_fold = false;
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
    static constexpr bool is_linear_fold = false;
    static constexpr bool is_accumulate = false;
    static constexpr bool is_boolean_fold = true;
};

template<class F, class ValueType, class T, template<class,class,class> class FoldPolicy>
class FoldDenseExpressionImpl : public DenseExpression<FoldDenseExpressionImpl<F,ValueType,T,FoldPolicy>>, public FoldPolicy<F,ValueType,T> {
    
public:

    using input_value_type = FoldPolicy<F,ValueType,T>::input_value_type;
    using value_type = FoldPolicy<F,ValueType,T>::value_type;

    static constexpr bool is_general_fold = FoldPolicy<F,ValueType,T>::is_general_fold;
    static constexpr bool is_linear_fold = FoldPolicy<F,ValueType,T>::is_linear_fold;
    static constexpr bool is_accumulate = FoldPolicy<F,ValueType,T>::is_accumulate;
    static constexpr bool is_boolean_fold = FoldPolicy<F,ValueType,T>::is_boolean_fold;
    static constexpr bool requires_start_val = is_general_fold || is_linear_fold;

private:

    using arg_t = std::conditional_t< std::is_lvalue_reference<T>::value, T, std::remove_cvref_t<T>>;
    using ref_t = const std::remove_cvref_t<arg_t>&;
    using Stripe_t = decltype(std::declval<std::add_const_t<std::remove_cvref_t<T>>>().get_stripe(std::declval<DenseStriper>()));

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
        _fold_size(_t.shape(fold_dim))
    {
        if( _fold_dim >= _t.dims() ) throw ExpressionException("Ultramat: Fold dimension must be smaller than dims()");
    }

    FoldDenseExpressionImpl( const F& f, T&& t, std::size_t fold_dim ) : 
        _f(f),
        _t(std::forward<T>(t)),
        _fold_dim(fold_dim),
        _fold_size(_t.shape(fold_dim))
    {
        if( _fold_dim >= _t.dims() ) throw ExpressionException("Ultramat: Fold dimension must be smaller than dims()");
    }

    std::size_t dims() const { return std::max((std::size_t)1,_t.dims()-1); }
    decltype(auto) shape(std::size_t ii) const { return (_t.dims()==1 ? 1 : _t.shape(ii < _fold_dim ? ii : ii+1)); }
    constexpr bool is_contiguous() const noexcept { return _t.is_contiguous(); }
    constexpr bool is_broadcasting() const noexcept { return _t.is_broadcasting(); }
    constexpr bool is_omp_parallelisable() const noexcept { return _t.is_omp_parallelisable(); }
    decltype(auto) required_stripe_dim() const { return dims(); }

    static constexpr DenseOrder order() { return std::remove_cvref_t<T>::order(); }

    std::size_t size() const { 
        std::size_t result=1;
        for(std::size_t ii=0; ii<dims(); ++ii) result*=shape(ii);
        return result;
    }

    std::vector<std::size_t> shape() const {
        std::vector<std::size_t> result(dims());
        for( std::size_t ii=0; ii<dims(); ++ii) result[ii] = shape(ii);
        return result;
    }

    // Define const_iterator class
 
    // Notes:
    // begin() should return a const_iterator containing a stripe generator over _t, passing on the stripe dim
    // Dereferencing this will generate a stripe, perform the fold operation, and return the result.
    // Incrementing this will increment the stripe generator.

    class const_iterator : public FoldPolicy<F,ValueType,T> {

        F            _f;
        ref_t        _t;
        DenseStriper _striper;
        
        public:

        const_iterator() = delete;
        const_iterator( const const_iterator& ) = default;
        const_iterator( const_iterator&& ) = default;
        const_iterator& operator=( const const_iterator& ) = default;
        const_iterator& operator=( const_iterator&& ) = default;
        
        template<shapelike Shape> 
        const_iterator( const F& f, ref_t t, std::size_t fold_dim, DenseOrder order, const Shape& shape, bool end, value_type start_val) :
            FoldPolicy<F,ValueType,T>(start_val),
            _f(f),
            _t(t),
            _striper(fold_dim,order,shape,end)
        {}

        template<shapelike Shape> 
        const_iterator( const F& f, ref_t t, std::size_t fold_dim, DenseOrder order, const Shape& shape, bool end) :
            _f(f),
            _t(t),
            _striper(fold_dim,order,shape,end)
        {}

        template<shapelike Shape> 
        const_iterator( const F& f, ref_t t, std::size_t fold_dim, const Shape& shape, bool end, value_type start_val) :
            const_iterator(f,t,fold_dim,t.order(),shape,end,start_val)
        {}

        template<shapelike Shape> 
        const_iterator( const F& f, ref_t t, std::size_t fold_dim, const Shape& shape, bool end) :
            const_iterator(f,t,fold_dim,t.order(),shape,end)
        {}

#define ULTRA_FOLD_DEREF\
        decltype(auto) operator*() requires (is_general_fold) {\
            F f = _f;\
            value_type val = FoldPolicy<F,ValueType,T>::get();\
            auto stripe = _t.get_stripe(_striper);\
            val = f( stripe.begin(), stripe.end(), val);\
            return val;\
        }\
\
        decltype(auto) operator*() requires (is_linear_fold) {\
            F f = _f;\
            value_type val = FoldPolicy<F,ValueType,T>::get();\
            auto stripe = _t.get_stripe(_striper);\
            for( auto&& x : stripe ) val = f(val,x);\
            return val;\
        }\
\
        decltype(auto) operator*() requires (is_accumulate) {\
            F f = _f;\
            auto stripe = _t.get_stripe(_striper);\
            auto it = stripe.begin();\
            auto end = stripe.end();\
            std::remove_cvref_t<value_type> val = *it;\
            ++it;\
            for(; it != end; ++it ) val = f(val,*it);\
            return val;\
        }\
\
        decltype(auto) operator*() requires (is_boolean_fold) {\
            /* Rather than using F directly, we will instead make use of start_bool and early_exit_bool */ \
            bool result = F::start_bool;\
            auto stripe = _t.get_stripe(_striper);\
            for( auto&& x : stripe){\
                if( x == F::early_exit_bool ) return !result;\
            }\
            return result;\
        }
        ULTRA_FOLD_DEREF

        const_iterator& operator++() { ++_striper; return *this; }
        const_iterator& operator--() { --_striper; return *this; }
        template<std::integral I> const_iterator& operator+=( const I& ii) { _striper+=ii; return *this; }
        template<std::integral I> const_iterator& operator-=( const I& ii) { _striper-=ii; return *this; }
        template<std::integral I> const_iterator operator+( const I& ii) { auto result(*this); result+=ii; return result; }
        template<std::integral I> const_iterator operator-( const I& ii) { auto result(*this); result-=ii; return result; }
        bool operator==(const const_iterator& it) const { return _striper == it._striper; }
        auto operator<=>(const const_iterator& it) const { return _striper <=> it._striper; }
        std::ptrdiff_t operator-(const const_iterator& it) const { return _striper - it._striper; }
    };

    const_iterator begin() const requires (requires_start_val) {
        return const_iterator(_f,_t,_fold_dim,_t.shape(),0,FoldPolicy<F,ValueType,T>::get());
    }
    
    const_iterator end() const requires (requires_start_val) {
        return const_iterator(_f,_t,_fold_dim,_t.shape(),1,FoldPolicy<F,ValueType,T>::get());
    }

    const_iterator begin() const requires (!requires_start_val) {
        return const_iterator(_f,_t,_fold_dim,_t.shape(),0);
    }

    const_iterator end() const requires (!requires_start_val) {
        return const_iterator(_f,_t,_fold_dim,_t.shape(),1);
    }

    // Define stripe class

    class Stripe : public FoldPolicy<F,ValueType,T> {

        F            _f;
        ref_t        _t;
        std::size_t  _fold_dim;
        std::size_t  _fold_size;
        bool         _fold_1d;
        DenseStriper _striper;

        public:
        
        Stripe( const F& f, ref_t t, std::size_t fold_dim, std::size_t fold_size, const DenseStriper& striper, bool fold_1d, const value_type& val) :
            FoldPolicy<F,ValueType,T>(val),
            _f(f),
            _t(t),
            _fold_dim(fold_dim),
            _fold_size(fold_size),
            _fold_1d(fold_1d),
            _striper(striper)
        {}

        Stripe( const F& f, ref_t t, std::size_t fold_dim, std::size_t fold_size, const DenseStriper& striper, bool fold_1d) :
            _f(f),
            _t(t),
            _fold_dim(fold_dim),
            _fold_size(fold_size),
            _fold_1d(fold_1d),
            _striper(striper)
        {}

        // Define iterator type
        class Iterator : public FoldPolicy<F,ValueType,T> {

            F            _f;
            ref_t        _t;
            std::size_t  _stripe_dim;
            std::size_t  _fold_dim;
            DenseStriper _striper;

            public: 

            Iterator() = delete;
            Iterator( const Iterator& ) = default;
            Iterator( Iterator&& ) = default;
            Iterator& operator=( const Iterator& ) = default;
            Iterator& operator=( Iterator&& ) = default;
            
            Iterator( const F& f, ref_t t, std::size_t stripe_dim, std::size_t fold_dim, const DenseStriper& striper, value_type start_val) :
                FoldPolicy<F,ValueType,T>(start_val),
                _f(f),
                _t(t),
                _stripe_dim(stripe_dim + (striper.order()==DenseOrder::row_major) + (stripe_dim >= fold_dim)),
                _fold_dim(fold_dim + (striper.order()==DenseOrder::row_major)),
                _striper(striper)
            {}

            Iterator( const F& f, ref_t t, std::size_t stripe_dim, std::size_t fold_dim, const DenseStriper& striper) :
                _f(f),
                _t(t),
                _stripe_dim(stripe_dim + (striper.order()==DenseOrder::row_major) + (stripe_dim >= fold_dim)),
                _fold_dim(fold_dim + (striper.order()==DenseOrder::row_major)),
                _striper(striper)
            {}

            ULTRA_FOLD_DEREF
#undef ULTRA_FOLD_DEREF

            Iterator& operator++() { ++_striper.index(_stripe_dim); return *this; }
            Iterator& operator--() { --_striper.index(_stripe_dim); return *this; }
            template<std::integral I> Iterator& operator+=( const I& ii) { _striper.index(_stripe_dim) += ii; return *this; }
            template<std::integral I> Iterator& operator-=( const I& ii) { _striper.index(_stripe_dim) -= ii; return *this; }
            template<std::integral I> Iterator operator+( const I& ii) { auto result(*this); result+=ii; return result; }
            template<std::integral I> Iterator operator-( const I& ii) { auto result(*this); result-=ii; return result; }
            bool operator==( const Iterator& other) const { return _striper.index(_stripe_dim) == other._striper.index(_stripe_dim);}
            auto operator<=>( const Iterator& other) const { return _striper.index(_stripe_dim) <=> other._striper.index(_stripe_dim);}
            std::ptrdiff_t operator-( const Iterator& other) const { return _striper.index(_stripe_dim) - other._striper.index(_stripe_dim);}

        };

        DenseStriper get_inner_striper( bool end) const {
            std::vector<std::size_t> inner_shape(_striper.dims()+(!_fold_1d),0);
            if( _fold_1d){
                inner_shape[0] = _fold_size; 
            } else {
                for( std::size_t ii=0; ii<_striper.dims(); ++ii){
                    inner_shape[ii + (ii>=_fold_dim)] = _striper.shape(ii); 
                }
                inner_shape[_fold_dim] = _fold_size;
            }
            DenseStriper inner_striper( _fold_dim, _striper.order(), inner_shape, 0);
            if( _fold_1d){
                inner_striper.index(0) = _striper.index(0); 
            } else {
                for( std::size_t ii=0; ii<=_striper.dims(); ++ii){
                    inner_striper.index(ii + (ii>=(_fold_dim+(_striper.order()==DenseOrder::row_major)))) = _striper.index(ii); 
                }
            }
            if( end ){
                std::size_t end_dim = _striper.stripe_dim() + (_striper.stripe_dim() >= _fold_dim);
                if( _fold_1d) {
                    inner_striper.index(0) = inner_striper.shape(0);
                } else {
                    inner_striper.index(end_dim + (_striper.order() == DenseOrder::row_major)) = inner_striper.shape(end_dim);
                }
            }
            return inner_striper;
        }

        Iterator begin() const requires (requires_start_val) {
            auto inner_striper = get_inner_striper(0);
            return Iterator( _f, _t, _striper.stripe_dim(), _fold_dim, inner_striper, FoldPolicy<F,ValueType,T>::get());
        }

        Iterator begin() const requires (!requires_start_val) {
            auto inner_striper = get_inner_striper(0);
            return Iterator( _f, _t, _striper.stripe_dim() ,_fold_dim, inner_striper);
        }

        Iterator end() const requires (requires_start_val) {
            auto inner_striper = get_inner_striper(1);
            return Iterator( _f, _t, _striper.stripe_dim(), _fold_dim, inner_striper, FoldPolicy<F,ValueType,T>::get());
        }

        Iterator end() const requires (!requires_start_val) {
            auto inner_striper = get_inner_striper(1);
            return Iterator( _f, _t, _striper.stripe_dim(), _fold_dim, inner_striper);
        }
    };

    // Get stripe from _t
    decltype(auto) get_stripe( const DenseStriper& striper) const requires (requires_start_val) {
        return Stripe(_f,_t,_fold_dim,_fold_size,striper,_t.dims()==1,FoldPolicy<F,ValueType,T>::get());
    }

    decltype(auto) get_stripe( const DenseStriper& striper) const requires (!requires_start_val) {
        return Stripe(_f,_t,_fold_dim,_fold_size,striper,_t.dims()==1);
    }
};

template<class F, class ValueType, class T> using GeneralFoldDenseExpression = FoldDenseExpressionImpl<F,ValueType,T,GeneralFoldPolicy>;
template<class F, class ValueType, class T> using LinearFoldDenseExpression = FoldDenseExpressionImpl<F,ValueType,T,LinearFoldPolicy>;
template<class F, class T> using AccumulateDenseExpression = FoldDenseExpressionImpl<F,typename std::remove_cvref_t<T>::value_type,T,AccumulatePolicy>;
template<class F, class T> using BooleanFoldDenseExpression = FoldDenseExpressionImpl<F,bool,T,BooleanFoldPolicy>;

// CumulativeDenseExpression
// Binary operation over a single arg. Returns something of the same shape.
// For multi-dimensional arrays, sums over the given dimension only. Defaults to zero.
// Unlike FoldDenseExpression, only an accumulating version exists.
// A more general implementation will only be included if a good use case can be demonstrated.
// Only a foward iterator is implemented.

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
    static constexpr DenseOrder order() { return std::remove_cvref_t<T>::order(); }
    
    decltype(auto) required_stripe_dim() const { return _dim; }

    // CumulativeDenseExpressions cannot be performed in parallel and must make use of striped iteration, hence will
    // appear as non-contiguous and non-omp-parallel. Each stripe may still be determined in parallel however.
    constexpr bool is_contiguous() const noexcept { return false; }
    constexpr bool is_broadcasting() const noexcept { return _t.is_broadcasting(); }
    constexpr bool is_omp_parallelisable() const noexcept { return false; }

    // Define stripe class

    class Stripe {
        
        using Stripe_t = decltype(std::declval<std::add_const_t<std::remove_cvref_t<T>>>().get_stripe(std::declval<DenseStriper>()));

        function_type     _f;
        Stripe_t          _stripe;

        public:

        Stripe( const function_type& f, Stripe_t&& stripe) : _f(f), _stripe(std::move(stripe)) {}
        
        // Define iterator type

        class Iterator {
            
            using It_t = typename decltype(std::declval<std::add_const_t<std::remove_cvref_t<T>>>().get_stripe(std::declval<DenseStriper>()))::Iterator;
            
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
            bool operator==( const Iterator& other ) const { return _it == other._it; }
        };
        
        Iterator begin() const { return Iterator(_f,_stripe.begin()); }
        Iterator end()   const { return Iterator(_f,_stripe.end()); }
    };

    // Get stripe from _t
    decltype(auto) get_stripe( const DenseStriper& striper) const {
        if( striper.stripe_dim() != _dim ){
            std::string err = "Ultramat: CumulativeDenseExpressions may only be striped in the direction specified upon their creation.";
            err += " get_stripe was given dim of " + std::to_string(striper.stripe_dim()) + ", but expected " + std::to_string(_dim) + '.';
            throw std::runtime_error(err);
        }
        return Stripe(_f,std::move(_t.get_stripe(striper)));
    }

    // Define const_iterator dummy class

    struct const_iterator {
        const_iterator() { throw std::runtime_error("Ultramat CumulativeDenseExpression: Must use striped iteration!");}
        decltype(auto) operator*() { return 0; }
        const_iterator& operator++() { return *this; }
        const_iterator& operator--() { return *this; }
        template<std::integral I> const_iterator& operator+=( const I& ){ return *this;}
        template<std::integral I> const_iterator& operator-=( const I& ){ return *this;}
        template<std::integral I> const_iterator operator+( const I& ){ return *this;}
        template<std::integral I> const_iterator operator-( const I& ){ return *this;}
        bool operator==( const const_iterator& other) const { return true; }
        auto operator<=>( const const_iterator& other) const { return std::strong_ordering::equal;}
        std::ptrdiff_t operator-( const const_iterator& other) const { return 0;}
    };
    const_iterator begin() const { return const_iterator(); }
    const_iterator end()   const { return const_iterator(); }
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

    template<shapelike Shape>
    GeneratorExpression( F&& f, const Shape& shape) : 
        _f(std::forward<F>(f)), 
        _shape(shape.size()),
        _size(std::accumulate( shape.begin(), shape.end(), 1, std::multiplies<std::size_t>()))
    {
        std::ranges::copy( shape, _shape.begin());
    }

    decltype(auto) size() const { return _size; }
    decltype(auto) dims() const { return _shape.size(); }
    decltype(auto) shape() const { return _shape; }
    decltype(auto) shape(std::size_t ii) const { return _shape[ii]; }
    static constexpr DenseOrder order() { return default_order; }
    decltype(auto) required_stripe_dim() const { return dims(); }

    constexpr bool is_contiguous() const noexcept { return true; }
    constexpr bool is_broadcasting() const noexcept { return false; }
    constexpr bool is_omp_parallelisable() const noexcept { return true; }

    // Define iterator class

    class const_iterator {
        
        function_type _f;
        std::size_t   _count;

        public:

        const_iterator() = delete;
        const_iterator( const const_iterator& ) = default;
        const_iterator( const_iterator&& ) = default;
        const_iterator& operator=( const const_iterator& ) = default;
        const_iterator& operator=( const_iterator&& ) = default;
        
        const_iterator( const function_type& f, std::size_t count) : _f(f), _count(count) {}
        decltype(auto) operator*() { return _f(_count); }
        const_iterator& operator++() { ++_count; return *this; }
        const_iterator& operator--() { --_count; return *this; }
        template<std::integral I> const_iterator& operator+=( const I& ii) { _count+=ii; return *this; }
        template<std::integral I> const_iterator& operator-=( const I& ii) { _count-=ii; return *this; }
        template<std::integral I> const_iterator operator+( const I& ii) { auto result(*this); result+=ii; return result; }
        template<std::integral I> const_iterator operator-( const I& ii) { auto result(*this); result-=ii; return result; }
        bool operator==( const const_iterator& other) const { return _count == other._count; }
        auto operator<=>( const const_iterator& other) const { return _count <=> other._count; }
        std::ptrdiff_t operator-( const const_iterator& other) const { return (std::ptrdiff_t)_count - (std::ptrdiff_t)other._count; }
    };

    const_iterator begin() const { return const_iterator(_f,0); }
    const_iterator end()   const { return const_iterator(_f,_size); }

    // Define stripe class
 
    class Stripe {

        function_type _f;
        std::size_t _size;

        public:

        Stripe( const function_type& f, std::size_t size) : _f(f), _size(size) {}

        using Iterator = const_iterator;
        const_iterator begin() const { return const_iterator(_f,0); }
        const_iterator end()   const { return const_iterator(_f,_size); }
    };

    // Get stripes from each Arg
    decltype(auto) get_stripe( const DenseStriper& striper) const {
        return Stripe(_f,shape(striper.stripe_dim()));
    }
};

// WhereDenseExpression
// Takes three arguments: Cond, L, and R. Returns the result of L when Cond is true, or R when Cond is false.
// Could be more simply implemented using ElementWiseDenseExpression with the function (cond ? l : r).
// However, this would require both L and R to be evaluated, which is wasteful.

template<class Cond, class Left, class Right>
class WhereDenseExpression : public DenseExpression<WhereDenseExpression<Cond,Left,Right>> {

public:

    using value_type = decltype( std::declval<typename std::remove_cvref_t<Left>::value_type>() + std::declval<typename std::remove_cvref_t<Right>::value_type>());

private:

    std::vector<std::size_t> _shape;
    
    // arg_t: If receiving 'Args&' or 'const Args&' as lvalue references, use this. Otherwise, if receiving rvalue reference,
    //        store Args 'by-value'. This avoids dangling references when the user chooses to return an expression from a function.
    template<class T>
    using arg_t = std::conditional_t< std::is_lvalue_reference<T>::value, T, std::remove_cvref_t<T>>;

    arg_t<Cond> _condition;
    arg_t<Left> _left_expression;
    arg_t<Right> _right_expression;

public:

    WhereDenseExpression() = delete;
    WhereDenseExpression( const WhereDenseExpression& ) = delete;
    WhereDenseExpression( WhereDenseExpression&& ) = default;
    WhereDenseExpression& operator=( const WhereDenseExpression& ) = delete;
    WhereDenseExpression& operator=( WhereDenseExpression&& ) = default;

    // For reasons that escape me, this constructor must be templated, and explicit class template deduction guides are required.
    // This is not needed for ElementWiseDenseExpression, despite the fact that this class is essentially a copy.
    template<class C, class L, class R>
    WhereDenseExpression( C&& cond, L&& l, R&& r) :
        _shape(get_broadcast_shape<order()>(cond.shape(),l.shape(),r.shape())),
        _condition(std::forward<C>(cond)),
        _left_expression(std::forward<L>(l)),
        _right_expression(std::forward<R>(r))
    {}

    decltype(auto) size() const { return std::accumulate(_shape.begin(),_shape.end(),1,std::multiplies<std::size_t>{}); }
    decltype(auto) dims() const { return _shape.size(); }
    decltype(auto) shape() const { return _shape; }
    decltype(auto) shape(std::size_t ii) const { return _shape[ii]; }
    decltype(auto) required_stripe_dim() const { return dims(); }

    static constexpr DenseOrder order() {
        return get_common_order<Cond,Left,Right>::value; 
    }

    constexpr bool is_contiguous() const noexcept {
        return _condition.is_contiguous() && _left_expression.is_contiguous() && _right_expression.is_contiguous();
    }

    constexpr bool is_omp_parallelisable() const noexcept {
        return _condition.is_omp_parallelisable() && _left_expression.is_omp_parallelisable() && _right_expression.is_omp_parallelisable();
    }

    bool is_broadcasting() const { 
        // Are args broadcasting?
        bool result = _condition.is_broadcasting() || _left_expression.is_broadcasting() || _right_expression.is_broadcasting();
        if( result ) return result;
        // Do args have mismatching dims?
        result |= _condition.dims() != dims() || _left_expression.dims() != dims() || _right_expression.dims() != dims();
        if( result ) return result;
        // Do args have mismatching shapes?
        for( std::size_t ii=0; ii<dims(); ++ii){
            result |= _condition.shape(ii) != shape(ii) || _left_expression.shape(ii) != shape(ii) || _right_expression.shape(ii) != shape(ii);
        }
        return result;
    }

    // Define const_iterator class
 
    // Notes:
    // begin() should return a compound iterator over Cond, L and R
    // Dereferencing this will return ( *it_cond ? *it_l : *it_r).
    // Incrementing this will increment all iterators, regardless of whether they were dereferenced

    class const_iterator {
        template<class T>
        using it_t = typename std::remove_cvref_t<T>::const_iterator;
        
        it_t<Cond> _it_cond;
        it_t<Left> _it_l;
        it_t<Right> _it_r;
        
        public:

        const_iterator() = delete;
        const_iterator( const const_iterator& ) = default;
        const_iterator( const_iterator&& ) = default;
        const_iterator& operator=( const const_iterator& ) = default;
        const_iterator& operator=( const_iterator&& ) = default;
        
        const_iterator( it_t<Cond>&& it_cond, it_t<Left>&& it_l, it_t<Right>&& it_r) : _it_cond(std::move(it_cond)), _it_l(std::move(it_l)), _it_r(std::move(it_r)) {}
        decltype(auto) operator*() { return *_it_cond ? *_it_l : *_it_r; }
        const_iterator& operator++() { ++_it_cond; ++_it_l; ++_it_r; return *this; }
        const_iterator& operator--() { --_it_cond; --_it_l; --_it_r; return *this; }
        template<std::integral I> const_iterator& operator+=( const I& ii) { _it_cond+=ii; _it_l+=ii; _it_r+=ii; return *this; }
        template<std::integral I> const_iterator& operator-=( const I& ii) { _it_cond-=ii; _it_l-=ii; _it_r-=ii; return *this; }
        template<std::integral I> const_iterator operator+( const I& ii) { auto result(*this); result+=ii; return result; }
        template<std::integral I> const_iterator operator-( const I& ii) { auto result(*this); result-=ii; return result; }
        bool operator==( const const_iterator& other) const { return _it_cond == other._it_cond; }
        auto operator<=>( const const_iterator& other) const { return _it_cond <=> other._it_cond; }
        std::ptrdiff_t operator-( const const_iterator& other) const { return _it_cond - other._it_cond; }
    };

    const_iterator begin() const { return const_iterator(_condition.begin(),_left_expression.begin(),_right_expression.begin()); }
    const_iterator end()   const { return const_iterator(_condition.end(),_left_expression.end(),_right_expression.end()); }

    // Define stripe class
    // As element-wise operations are strictly non-modifying, only read_only stripes are permitted.
 
    class Stripe {
        
        template<class T>
        using stripe_t = decltype(std::declval<std::add_const_t<std::remove_cvref_t<T>>>().get_stripe(std::declval<DenseStriper>()));

        stripe_t<Cond> _stripe_cond;
        stripe_t<Left> _stripe_l;
        stripe_t<Right> _stripe_r;

        public:

        Stripe( stripe_t<Cond>&& stripe_cond, stripe_t<Left>&& stripe_l, stripe_t<Right>&& stripe_r) : 
            _stripe_cond(std::move(stripe_cond)),
            _stripe_l(std::move(stripe_l)),
            _stripe_r(std::move(stripe_r))
        {}
        
        // Define iterator type

        class Iterator {
            
            template<class T>
            using it_t = typename decltype(std::declval<std::add_const_t<std::remove_cvref_t<T>>>().get_stripe(std::declval<DenseStriper>()))::Iterator;
            
            it_t<Cond> _it_cond;
            it_t<Left> _it_l;
            it_t<Right> _it_r;
            
            public:

            Iterator() = delete;
            Iterator( const Iterator& ) = default;
            Iterator( Iterator&& ) = default;
            Iterator& operator=( const Iterator& ) = default;
            Iterator& operator=( Iterator&& ) = default;
            
            Iterator( it_t<Cond>&& it_cond, it_t<Left>&& it_l, it_t<Right>&& it_r) : _it_cond(std::move(it_cond)), _it_l(std::move(it_l)), _it_r(std::move(it_r)) {}
            decltype(auto) operator*() { return *_it_cond ? *_it_l : *_it_r; }
            Iterator& operator++() { ++_it_cond; ++_it_l; ++_it_r; return *this; }
            Iterator& operator--() { --_it_cond; --_it_l; --_it_r; return *this; }
            template<std::integral I> Iterator& operator+=( const I& ii) { _it_cond+=ii; _it_l+=ii; _it_r+=ii; return *this; }
            template<std::integral I> Iterator& operator-=( const I& ii) { _it_cond-=ii; _it_l-=ii; _it_r-=ii; return *this; }
            template<std::integral I> Iterator operator+( const I& ii) { auto result(*this); result+=ii; return result; }
            template<std::integral I> Iterator operator-( const I& ii) { auto result(*this); result-=ii; return result; }
            bool operator==( const Iterator* other) const { return _it_cond == other._it_cond; }
            auto operator<=>( const Iterator* other) const { return _it_cond <=> other._it_cond; }
            std::ptrdiff_t operator-( const Iterator& other) const { return _it_cond - other._it_cond; }
        };
        
        Iterator begin() const { return Iterator(_stripe_cond.begin(),_stripe_l.begin(),_stripe_r.begin()); }
        Iterator end()   const { return Iterator(_stripe_cond.end(),_stripe_l.end(),_stripe_r.end()); }
    };

    // Get stripes from each Arg
    decltype(auto) get_stripe( const DenseStriper& striper) const {
        return Stripe( _condition.get_stripe(striper), _left_expression.get_stripe(striper), _right_expression.get_stripe(striper));
    }
};

// explicit class template deduction guides

template< class C, class L, class R> WhereDenseExpression( const C& c, const L& l, const R& r) -> WhereDenseExpression<C,L,R>;
template< class C, class L, class R> WhereDenseExpression( C&& c, const L& l, const R& r) -> WhereDenseExpression<C,L,R>;
template< class C, class L, class R> WhereDenseExpression( const C& c, L&& l, const R& r) -> WhereDenseExpression<C,L,R>;
template< class C, class L, class R> WhereDenseExpression( const C& c, const L& l, R&& r) -> WhereDenseExpression<C,L,R>;
template< class C, class L, class R> WhereDenseExpression( C&& c, L&& l, const R& r) -> WhereDenseExpression<C,L,R>;
template< class C, class L, class R> WhereDenseExpression( C&& c, const L& l, R&& r) -> WhereDenseExpression<C,L,R>;
template< class C, class L, class R> WhereDenseExpression( const C& c, L&& l, R&& r) -> WhereDenseExpression<C,L,R>;
template< class C, class L, class R> WhereDenseExpression( C&& c, L&& l, R&& r) -> WhereDenseExpression<C,L,R>;

} // namespace
#endif
