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
// Additionally requires that T is 'stripeable'

template<class T>
struct DenseExpression {

    constexpr T& derived() noexcept { return static_cast<T&>(*this); }
    constexpr const T& derived() const noexcept { return static_cast<const T&>(*this); }

    decltype(auto) size() const { return derived().size(); }
    decltype(auto) dims() const { return derived().dims(); }
    decltype(auto) shape() const { return derived().shape(); }
    decltype(auto) shape(std::size_t ii) const { return derived().shape(ii); }
    decltype(auto) stride() const { return derived().stride(); }
    decltype(auto) stride(std::size_t ii) const { return derived().stride(ii); }

    decltype(auto) get_stripe(std::size_t stripe, std::size_t dim) { return derived().get_stripe(stripe,dim); }
    decltype(auto) get_stripe(std::size_t stripe, std::size_t dim) const { return derived().get_stripe(stripe,dim); }
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
decltype(auto) eval( DenseExpression<T>& t){
    return eval_result<T>(static_cast<T&>(t));
}

template<class T>
decltype(auto) eval( DenseExpression<T>&& t){
    return eval_result<T>(static_cast<T&&>(t));
}

// ElementWiseDenseExpression
// Performs an operation taking an arbitray number of args. Operation acts independently over each arg.
// All inputs and outputs must be of the same shape.

template<class F, class... Args>
class ElementWiseDenseExpression : public DenseExpression<ElementWiseDenseExpression<F,Args...>> {

public:

    using value_type = decltype( std::apply( F{}, std::tuple<typename std::remove_cvref_t<Args>::value_type...>()));

private:
    
    using tuple_t = decltype(std::forward_as_tuple(std::declval<Args>()...));
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
    decltype(auto) stride() const { return std::get<0>(_args).stride(); }
    decltype(auto) stride(std::size_t ii) const { return std::get<0>(_args).stride(ii); }

    // Define stripe class
    // As element-wise operations are strictly non-modifying, only read_only stripes are permitted.
 
    class Stripe {
        
        using StripeTuple = std::tuple< decltype(std::declval<std::add_const_t<std::remove_cvref_t<Args>>>().get_stripe(0,0)) ...>;

        StripeTuple _stripes;

        public:

        Stripe( StripeTuple&& stripes) : _stripes(std::move(stripes)) {}
        
        // Define iterator type
        // Notes:
        // begin() should return a compound iterator over Args, containing _its.
        // Dereferencing this will return F(*_its[0],*_its[1],...).
        // Incrementing this will increment _its.

        class Iterator {
            
            using ItTuple = std::tuple< typename decltype(std::declval<std::add_const_t<std::remove_cvref_t<Args>>>().get_stripe(0,0))::Iterator ...>;
            
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
    decltype(auto) get_stripe( std::size_t stripe_num, std::size_t dim) const {
        return Stripe(std::move(apply_to_each(GetStripe{stripe_num,dim},_args)));
    }

};

// CumulativeDenseDenseExpression
// Binary operation over a single arg. Returns something of the same shape.
// For multi-dimensional arrays, sums over the given dimension only. Defaults to zero.

template<class F, class T>
class CumulativeDenseExpression : public DenseExpression<CumulativeDenseExpression<F,T>> {
    
public:

    using inner_value_type = typename std::remove_cvref_t<T>::value_type;
    using value_type = decltype(F{}(inner_value_type(),inner_value_type()));

private:

    using ref_t = decltype(std::forward<T>(std::declval<T>()));

    ref_t _t;
    value_type _start_val;

public:

    CumulativeDenseExpression() = delete;
    CumulativeDenseExpression( const CumulativeDenseExpression& ) = delete;
    CumulativeDenseExpression( CumulativeDenseExpression&& ) = default;
    CumulativeDenseExpression& operator=( const CumulativeDenseExpression& ) = delete;
    CumulativeDenseExpression& operator=( CumulativeDenseExpression&& ) = default;

    CumulativeDenseExpression( T&& t, const value_type& start_val) : _t(std::forward<T>(t)) , _start_val(start_val) {}

    decltype(auto) size() const { return _t.size(); }
    decltype(auto) dims() const { return _t.dims(); }
    decltype(auto) shape() const { return _t.shape(); }
    decltype(auto) shape(std::size_t ii) const { return _t.shape(ii); }
    decltype(auto) stride() const { return _t.stride(); }
    decltype(auto) stride(std::size_t ii) const { return _t.stride(ii); }

    // Define stripe class

    class Stripe {
        
        using Stripe_t = decltype(std::declval<std::add_const_t<std::remove_cvref_t<T>>>().get_stripe(0,0));

        Stripe_t          _stripe;
        const value_type& _val;

        public:

        Stripe( Stripe_t&& stripe, const value_type& val) : _stripe(std::move(stripe)), _val(val) {}
        
        // Define iterator type

        class Iterator {
            
            using It_t = typename decltype(std::declval<std::add_const_t<std::remove_cvref_t<T>>>().get_stripe(0,0))::Iterator;
            
            F           _f;
            It_t        _it;
            value_type  _val;
            
            public:
           
            Iterator( It_t&& it, const value_type& val) : _f{}, _it(std::move(it)), _val(val)  {}
            decltype(auto) operator*() { _val = _f(*_it,_val); return _val; }
            Iterator& operator++() { ++_it; return *this; }
        };
        
        Iterator begin() const { return Iterator(_stripe.begin(),_val); }
    };

    // Get stripe from _t
    decltype(auto) get_stripe( std::size_t stripe_num, std::size_t dim) const {
        return Stripe(std::move(_t.get_stripe(stripe_num,dim)),_start_val);
    }
};

} // namespace
#endif
