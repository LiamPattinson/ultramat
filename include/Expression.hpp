#ifndef __ULTRA_EXPRESSION_HPP
#define __ULTRA_EXPRESSION_HPP

// Expression.hpp
//
// See https://en.wikipedia.org/wiki/Expression_templates
// 
// Expressions are used to represent complex computations and to
// avoid unnecessary allocations and copies.

#include <type_traits>
#include <functional>

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

// Binary Expression

template<class F, class L, class R>
class BinaryExpression : public Expression<BinaryExpression<F,L,R>> {
    
    const L& _l;
    const R& _r;

    public:

    BinaryExpression( const L& l, const R& r) : _l(l), _r(r) {
        if( _l.dims() != _r.dims() ){
            throw std::runtime_error("BinaryExpression: lhs and rhs have incompatible dimensions.");
        }
        for( std::size_t ii=0; ii<_l.dims(); ++ii){
            if( _l.shape(ii) != _r.shape(ii) ){
                throw std::runtime_error("BinaryExpression: lhs and rhs have incompatible shapes.");
            }
        }
        // No need to test size, this requirement is satisfied implicitly.
    }

    std::size_t size() const { return _l.size(); }
    std::size_t shape(std::size_t ii) const { return _l.shape(ii); }
    std::size_t dims() const { return _l.dims(); }

    // Define iterator class
    // As this is for non-modifying operations, only const_iterator will exist.
 
    // Notes:
    // begin() should return a compound iterator over L and R, containing _l_it and _r_it.
    // Dereferencing this will return F(*_l_it,*_r_it).
    // Incrementing this will increment _l_it and _r_it.

    class const_iterator {
        L::const_iterator _l_it;
        R::const_iterator _r_it;
        F f;
        public:
        const_iterator( L::const_iterator l_it, R::const_iterator r_it) : _l_it(l_it), _r_it(r_it), f{} {}
        decltype(auto) operator*() { return f(*_l_it,*_r_it); }
        const_iterator& operator++() { ++_l_it; ++_r_it; return *this; }
    };

    const_iterator begin() const { return const_iterator(_l.begin(),_r.begin()); }
};

} // namespace
#endif
