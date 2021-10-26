#ifndef __ULTRA_DENSE_CUMULATIVE_EXPRESSION_HPP
#define __ULTRA_DENSE_CUMULATIVE_EXPRESSION_HPP

#include "DenseExpression.hpp"

namespace ultra {

// DenseCumulativeExpression
// Binary operation over a single arg. Returns something of the same shape.
// For multi-dimensional arrays, sums over the given dimension only. Defaults to zero.
// Unlike FoldDenseExpression, only an accumulating version exists.
// A more general implementation will only be included if a good use case can be demonstrated.
// Only a foward iterator is implemented.

template<class F, class T>
class DenseCumulativeExpression : public DenseExpression<DenseCumulativeExpression<F,T>> {

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

    DenseCumulativeExpression() = delete;
    DenseCumulativeExpression( const DenseCumulativeExpression& ) = delete;
    DenseCumulativeExpression( DenseCumulativeExpression&& ) = default;
    DenseCumulativeExpression& operator=( const DenseCumulativeExpression& ) = delete;
    DenseCumulativeExpression& operator=( DenseCumulativeExpression&& ) = default;

    DenseCumulativeExpression( const function_type& f, T&& t, std::size_t dim) : _f(f), _t(std::forward<T>(t)) , _dim(dim) {}

    decltype(auto) size() const { return _t.size(); }
    decltype(auto) dims() const { return _t.dims(); }
    decltype(auto) shape() const { return _t.shape(); }
    decltype(auto) shape(std::size_t ii) const { return _t.shape(ii); }
    static constexpr DenseOrder order() { return std::remove_cvref_t<T>::order(); }
    
    decltype(auto) required_stripe_dim() const { return _dim; }

    // DenseCumulativeExpressions cannot be performed in parallel and must make use of striped iteration, hence will
    // appear as non-contiguous and non-omp-parallel. Each stripe may still be determined in parallel however.
    constexpr bool is_contiguous() const noexcept { return false; }
    constexpr bool is_broadcasting() const noexcept { return _t.is_broadcasting(); }
    constexpr bool is_omp_parallelisable() const noexcept { return false; }

    // Define stripe class

    class Stripe {
        
        using Stripe_t = decltype(std::declval<std::add_const_t<std::remove_cvref_t<T>>>().get_stripe(std::declval<DenseStripeIndex>()));

        function_type     _f;
        Stripe_t          _stripe;

        public:

        Stripe( const function_type& f, Stripe_t&& stripe) : _f(f), _stripe(std::move(stripe)) {}
        
        // Define iterator type

        class Iterator {
            
            using It_t = typename decltype(std::declval<std::add_const_t<std::remove_cvref_t<T>>>().get_stripe(std::declval<DenseStripeIndex>()))::Iterator;
            
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
    decltype(auto) get_stripe( const DenseStripeIndex& striper) const {
        if( striper.stripe_dim() != _dim ){
            std::string err = "Ultramat: DenseCumulativeExpressions may only be striped in the direction specified upon their creation.";
            err += " get_stripe was given dim of " + std::to_string(striper.stripe_dim()) + ", but expected " + std::to_string(_dim) + '.';
            throw std::runtime_error(err);
        }
        return Stripe(_f,std::move(_t.get_stripe(striper)));
    }

    // Define const_iterator dummy class

    struct const_iterator {
        const_iterator() { throw std::runtime_error("Ultramat DenseCumulativeExpression: Must use striped iteration!");}
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

} // namespace ultra
#endif
