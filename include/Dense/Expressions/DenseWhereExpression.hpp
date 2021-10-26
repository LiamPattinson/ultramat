#ifndef __ULTRA_DENSE_WHERE_EXPRESSION_HPP
#define __ULTRA_DENSE_WHERE_EXPRESSION_HPP

#include "DenseExpression.hpp"

namespace ultra {

// DenseWhereExpression
// Takes three arguments: Cond, L, and R. Returns the result of L when Cond is true, or R when Cond is false.
// Could be more simply implemented using ElementWiseDenseExpression with the function (cond ? l : r).
// However, this would require both L and R to be evaluated, which is wasteful.

template<class Cond, class Left, class Right>
class DenseWhereExpression : public DenseExpression<DenseWhereExpression<Cond,Left,Right>> {

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

    DenseWhereExpression() = delete;
    DenseWhereExpression( const DenseWhereExpression& ) = delete;
    DenseWhereExpression( DenseWhereExpression&& ) = default;
    DenseWhereExpression& operator=( const DenseWhereExpression& ) = delete;
    DenseWhereExpression& operator=( DenseWhereExpression&& ) = default;

    // For reasons that escape me, this constructor must be templated, and explicit class template deduction guides are required.
    // This is not needed for ElementWiseDenseExpression, despite the fact that this class is essentially a copy.
    template<class C, class L, class R>
    DenseWhereExpression( C&& cond, L&& l, R&& r) :
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
        return common_order<Cond,Left,Right>::value; 
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
        using stripe_t = decltype(std::declval<std::add_const_t<std::remove_cvref_t<T>>>().get_stripe(std::declval<DenseStripeIndex>()));

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
            using it_t = typename decltype(std::declval<std::add_const_t<std::remove_cvref_t<T>>>().get_stripe(std::declval<DenseStripeIndex>()))::Iterator;
            
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
    decltype(auto) get_stripe( const DenseStripeIndex& striper) const {
        return Stripe( _condition.get_stripe(striper), _left_expression.get_stripe(striper), _right_expression.get_stripe(striper));
    }
};

// explicit class template deduction guides

template< class C, class L, class R> DenseWhereExpression( const C& c, const L& l, const R& r) -> DenseWhereExpression<C,L,R>;
template< class C, class L, class R> DenseWhereExpression( C&& c, const L& l, const R& r) -> DenseWhereExpression<C,L,R>;
template< class C, class L, class R> DenseWhereExpression( const C& c, L&& l, const R& r) -> DenseWhereExpression<C,L,R>;
template< class C, class L, class R> DenseWhereExpression( const C& c, const L& l, R&& r) -> DenseWhereExpression<C,L,R>;
template< class C, class L, class R> DenseWhereExpression( C&& c, L&& l, const R& r) -> DenseWhereExpression<C,L,R>;
template< class C, class L, class R> DenseWhereExpression( C&& c, const L& l, R&& r) -> DenseWhereExpression<C,L,R>;
template< class C, class L, class R> DenseWhereExpression( const C& c, L&& l, R&& r) -> DenseWhereExpression<C,L,R>;
template< class C, class L, class R> DenseWhereExpression( C&& c, L&& l, R&& r) -> DenseWhereExpression<C,L,R>;

} // namespace ultra
#endif
