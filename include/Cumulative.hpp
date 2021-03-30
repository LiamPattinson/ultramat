#ifndef __ULTRA_CUMULATIVE_HPP
#define __ULTRA_CUMULATIVE_HPP

// Cumulative
//
// Defines expressions for cumulative functions.

#include "Expression.hpp"
#include "Arithmetic.hpp"

namespace ultra {

// CumulativeExpression
// Binary operation over a single arg. Returns something of the same shape.

template<class F, class T>
class CumulativeExpression : public Expression<CumulativeExpression<F,T>> {
    
public:

    using inner_contains = typename std::remove_cvref_t<T>::contains;
    using contains = decltype(F{}(inner_contains(),inner_contains()));

private:

    using ref_t = decltype(std::forward<T>(std::declval<T>()));

    ref_t _t;
    inner_contains _start_val;

public:

    CumulativeExpression( T&& t, const inner_contains& start_val) : _t(std::forward<T>(t)) , _start_val(start_val) {}

    std::size_t size() const { return _t.size(); }
    std::size_t shape(std::size_t ii) const { return _t.shape(ii); }
    std::size_t dims() const { return _t.dims(); }

    // Define iterator class
    // As element-wise operations are strictly non-modifying, only const_iterator will exist.

    class const_iterator {

        using it_t = typename std::remove_cvref_t<T>::const_iterator;
        F f;
        it_t _it;
        inner_contains _val;
        
        public:
        
        const_iterator( it_t&& it, const inner_contains& start_val) : f{}, _it(std::move(it)), _val(start_val) {}
        decltype(auto) operator*() { _val = f(*_it,_val); return _val; }
        const_iterator& operator++() { ++_it; return *this; }
    };

    const_iterator begin() const { return const_iterator(std::move(_t.begin()),_start_val); }
};

// We can reuse functors defined in Arithmetic

// Expressions

template<class T> using  CumSumExpression = CumulativeExpression<Plus,T>;
template<class T> using CumProdExpression = CumulativeExpression<Multiplies,T>;

// Functions

template<class T, class StartT=int>
decltype(auto) cumsum( const Expression<T>& t, const StartT& start = 0){
    return CumSumExpression(static_cast<const T&>(t),start);
}

template<class T, class StartT=int>
decltype(auto) cumsum( Expression<T>&& t, const StartT& start = 0){
    return CumSumExpression(static_cast<T&&>(t),start);
}

template<class T, class StartT=int>
decltype(auto) cumprod( const Expression<T>& t, const StartT& start = 1){
    return CumProdExpression(static_cast<const T&>(t),start);
}

template<class T, class StartT=int>
decltype(auto) cumprod( const Expression<T>&& t, const StartT& start = 1){
    return CumProdExpression(static_cast<T&&>(t),start);
}

} // namespace
#endif
