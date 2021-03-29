#ifndef __ULTRA_EXPRESSION_HPP
#define __ULTRA_EXPRESSION_HPP

// Expression.hpp
//
// See https://en.wikipedia.org/wiki/Expression_templates
// 
// Expressions are used to represent complex computations and to
// avoid unnecessary allocations and copies.

#include "Utils.hpp"

namespace ultra {

// Base Expression
// Satisfies requirements for all expression types. Namely, that they expose
// the functions size(), shape(std::size_t), dims(), and begin().
// Defers to whatever class it's templated over.

template<class T>
struct Expression {
    std::size_t size() const { return static_cast<const T&>(*this).size(); }
    std::size_t shape(std::size_t ii) const { return static_cast<const T&>(*this).shape(ii); }
    std::size_t dims() const { return static_cast<const T&>(*this).dims(); }
    decltype(auto) begin() { return static_cast<const T&>(*this).begin(); }
    decltype(auto) begin() const { return static_cast<const T&>(*this).begin(); }
};

// ElementWiseExpression
// Operates on arbitrary number of args, but all must be of the same shape.
// Returns something of the same shape.

template<class F, class... Args>
class ElementWiseExpression : public Expression<ElementWiseExpression<F,Args...>> {

public:

    using contains = decltype( std::apply( F{}, std::tuple<typename std::remove_cvref_t<Args>::contains...>()));

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

    ElementWiseExpression() = delete;
    ElementWiseExpression( const ElementWiseExpression<F,Args...>& ) = delete;
    ElementWiseExpression( ElementWiseExpression<F,Args...>&& ) = default;
    ElementWiseExpression<F,Args...>& operator=( const ElementWiseExpression<F,Args...>& ) = delete;
    ElementWiseExpression<F,Args...>& operator=( ElementWiseExpression<F,Args...>&& ) = default;

    ElementWiseExpression( Args&&... args) : _args(std::forward<Args>(args)...) {
        if( std::ranges::any_of(test_dims(),[](bool b){return b;}) ){
            throw std::runtime_error("ElementWiseExpression: args have incompatible dimensions.");
        }
        for( std::size_t ii=0, end=std::get<0>(_args).dims(); ii<end; ++ii){
            if( std::ranges::any_of(test_shape(ii),[](bool b){return b;}) ){
                throw std::runtime_error("ElementWiseExpression: args have incompatible shapes.");
            }
        }
        // No need to test size, this requirement is satisfied implicitly.
    }



    std::size_t size() const { return std::get<0>(_args).size(); }
    std::size_t shape(std::size_t ii) const { return std::get<0>(_args).shape(ii); }
    std::size_t dims() const { return std::get<0>(_args).dims(); }

    // Define iterator class
    // As element-wise operations are strictly non-modifying, only const_iterator will exist.
 
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
};

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

// GeneratorExpression
// Creates new elements on demand for a given shape.

template<class F>
class GeneratorExpression : public Expression<GeneratorExpression<F>> {
    
public:

    using contains = decltype(std::declval<F>()());

private:

    F _generator;
    std::vector<std::size_t> _shape;
    std::size_t _size;

public:

    template< std::ranges::sized_range Range>
    GeneratorExpression(const F& generator, const Range& range) : 
        _generator(generator), 
        _shape(std::ranges::size(range)),
        _size(std::accumulate(range.begin(),range.end(),1,std::multiplies<std::size_t>{})) 
    {
        std::copy(range.begin(),range.end(),_shape.begin());
    }


    std::size_t size() const { return _size; }
    std::size_t shape(std::size_t ii) const { return _shape[ii]; }
    std::size_t dims() const { return _shape.size(); }

    // Define iterator class

    class const_iterator {
        
        F _generator;
        
        public:
        
        const_iterator( const F& generator) : _generator(generator) {}
        decltype(auto) operator*() { return _generator(); }
        const_iterator& operator++() { /* do nothing! */ return *this; }
    };

    const_iterator begin() const { return const_iterator(_generator); }
};

} // namespace
#endif
