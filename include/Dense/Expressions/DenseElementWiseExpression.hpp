#ifndef __ULTRA_DENSE_ELEMENT_WISE_EXPRESSION_HPP
#define __ULTRA_DENSE_ELEMENT_WISE_EXPRESSION_HPP

/*! \file DenseElementWiseExpression.hpp
 *  \brief Defines expression templates for simple operations that are applied independently to each component of a `Dense` object.
 *
 *  `DenseElementWiseExpression` is the most commonly used expression type, as it encapsulates all arithmetic/boolean operators,
 *  along with all N-dimensional versions of functions from the `cmath` library.
 */

#include "DenseExpression.hpp"

namespace ultra {

// ==============================================
// DenseElementWiseExpression

/*! \brief An expression used to represent simple operations that apply independently to each component of a `Dense` object.
 *  \tparam F A functor type defining the function applied over each \ref dense_object.
 *  \tparam Args The `Dense` objects over which `F` is applied.
 *
 * Applies a given function, defined by the functor class `F`, element-wise over an ordered collection of `Dense` objects.
 * The number of types represented by `Args...` must match the number of arguments taken by `F::operator()`. 
 * For example, if `F::operator()` defines simple addition between two arithmetic types, then `Args...` should consist of
 * two `Dense` objects containing arithmetic internal types.
 *
 * All inputs and outputs must be of the same shape, or broadcastable to some common shape. 
 */
template<class F, class... Args>
class DenseElementWiseExpression : public DenseExpression<DenseElementWiseExpression<F,Args...>> {

public:

    using value_type = decltype( std::apply( F{}, std::tuple<typename std::remove_cvref_t<Args>::value_type...>()));

private:
    
    std::vector<std::size_t> _shape;

    // tuple type: If receiving 'Args&' or 'const Args&' as lvalue references, use this. Otherwise, if receiving rvalue reference,
    //             store Args 'by-value'. This avoids dangling references when the user chooses to return an expression from a function.
    using tuple_t = std::tuple< std::conditional_t< std::is_lvalue_reference<Args>::value, Args, std::remove_cvref_t<Args>>...>;
    tuple_t _args;

    // Define dereferencing/function call utility function
    template<class Tuple, std::size_t... I>
    static decltype(auto) _deref_impl( const F& f, const Tuple& t, std::index_sequence<I...> ){
        return f( *std::get<I>(t) ...);
    }

    template<class Tuple>
    static decltype(auto) _deref( const F& f, const Tuple& t){
        return _deref_impl(f,t,std::make_index_sequence< std::tuple_size<Tuple>::value>{});
    }

    template<class Tuple, std::size_t... I>
    static decltype(auto) _deref_impl( const F& f, Tuple& t, std::index_sequence<I...> ){
        return f( *std::get<I>(t) ...);
    }

    template<class Tuple>
    static decltype(auto) _deref( const F& f, Tuple& t){
        return _deref_impl(f,t,std::make_index_sequence< std::tuple_size<Tuple>::value>{});
    }

    // define implementations for utility functions: is_contiguous, is_omp_parallelisable, is_broadcasting, get_stripe
    template<std::size_t... I>
    constexpr bool _is_contiguous_impl( std::index_sequence<I...> ) const noexcept { 
        return ( std::get<I>(_args).is_contiguous() && ... );
    }

    template<std::size_t... I>
    constexpr bool _is_omp_parallelisable_impl( std::index_sequence<I...> ) const noexcept { 
        return ( std::get<I>(_args).is_omp_parallelisable() && ... );
    }

    template<std::size_t... I>
    bool _is_broadcasting_impl( std::index_sequence<I...> ) const { 
        // Are args broadcasting?
        bool result = (std::get<I>(_args).is_broadcasting() || ...);
        if( result ) return result;
        // Do args have mismatching dims?
        std::size_t d = dims();
        result |= ((std::get<I>(_args).dims() != d) || ...);
        if( result ) return result;
        // Do args have mismatching shapes?
        for( std::size_t ii=0; ii<d; ++ii){
            std::size_t s = shape(ii);
            result |= ((std::get<I>(_args).shape(ii) != s) || ...);
            if( result) break;
        }
        return result;
    }

    template<std::size_t... I>
    decltype(auto) _get_stripe_impl( const DenseStripeIndex& striper, std::index_sequence<I...> ) const {
        return std::make_tuple(std::get<I>(_args).get_stripe(striper) ...);
    }


public:

    DenseElementWiseExpression() = delete;
    DenseElementWiseExpression( const DenseElementWiseExpression& ) = delete;
    DenseElementWiseExpression( DenseElementWiseExpression&& ) = default;
    DenseElementWiseExpression& operator=( const DenseElementWiseExpression& ) = delete;
    DenseElementWiseExpression& operator=( DenseElementWiseExpression&& ) = default;

    DenseElementWiseExpression( Args&&... args) : 
        _shape( get_broadcast_shape<order()>(args.shape()...)),
        _args(std::forward<Args>(args)...)
    {}

    decltype(auto) size() const { return std::accumulate(_shape.begin(),_shape.end(),1,std::multiplies<std::size_t>{}); }
    decltype(auto) dims() const { return _shape.size(); }
    decltype(auto) shape() const { return _shape; }
    decltype(auto) shape(std::size_t ii) const { return _shape[ii]; }
    decltype(auto) required_stripe_dim() const { return dims(); }

    static constexpr DenseOrder order() { return common_order<Args...>::value; }

    constexpr bool is_contiguous() const noexcept { return _is_contiguous_impl(std::make_index_sequence<std::tuple_size<tuple_t>::value>{});}
    constexpr bool is_omp_parallelisable() const noexcept { return _is_omp_parallelisable(std::make_index_sequence<std::tuple_size<tuple_t>::value>{});}
    bool is_broadcasting() const { return _is_broadcasting_impl(std::make_index_sequence<std::tuple_size<tuple_t>::value>{});}

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

        decltype(auto) operator*() { return _deref(f,_its); }
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

    const_iterator begin() const { return const_iterator(begin_tuple(_args)); }
    const_iterator end()   const { return const_iterator(end_tuple(_args)); }

    // Define stripe class
    // As element-wise operations are strictly non-modifying, only read_only stripes are permitted.
 
    class Stripe {
        
        using StripeTuple = std::tuple< decltype(std::declval<std::add_const_t<std::remove_cvref_t<Args>>>().get_stripe(std::declval<DenseStripeIndex>())) ...>;

        StripeTuple _stripes;

        public:

        Stripe( StripeTuple&& stripes) : _stripes(std::move(stripes)) {}
        
        // Define iterator type
        // Notes:
        // begin() should return a compound iterator over Args, containing _its.
        // Dereferencing this will return F(*_its[0],*_its[1],...).
        // Incrementing this will increment _its.

        class Iterator {
            
            using ItTuple = std::tuple<typename decltype(std::declval<std::add_const_t<std::remove_cvref_t<Args>>>().get_stripe(std::declval<DenseStripeIndex>()))::Iterator ...>;
            
            F       _f;
            ItTuple _its;
            
            public:

            Iterator() = delete;
            Iterator( const Iterator& ) = default;
            Iterator( Iterator&& ) = default;
            Iterator& operator=( const Iterator& ) = default;
            Iterator& operator=( Iterator&& ) = default;
            
            Iterator( ItTuple&& its) : _f{}, _its(std::move(its)) {}

            decltype(auto) operator*() { return _deref(_f,_its); }
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
        
        Iterator begin() const { return Iterator(begin_tuple(_stripes)); }
        Iterator end()   const { return Iterator(end_tuple(_stripes)); }
    };

    decltype(auto) get_stripe( const DenseStripeIndex& striper) const {
        return Stripe(_get_stripe_impl(striper,std::make_index_sequence<std::tuple_size<tuple_t>::value>{}));
    }
};

} // namespace ultra
#endif
