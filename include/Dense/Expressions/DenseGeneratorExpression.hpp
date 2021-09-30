#ifndef __ULTRA_DENSE_GENERATOR_EXPRESSION_HPP
#define __ULTRA_DENSE_GENERATOR_EXPRESSION_HPP

#include "DenseExpression.hpp"

namespace ultra {

template<class F>
class DenseGeneratorExpression : public DenseExpression<DenseGeneratorExpression<F>> {
    
public:

    using function_type = std::remove_cvref_t<F>;
    using value_type = decltype(std::declval<function_type>()(static_cast<std::size_t>(0)));

private:

    function_type            _f;
    std::vector<std::size_t> _shape;
    std::size_t              _size;

public:

    DenseGeneratorExpression() = delete;
    DenseGeneratorExpression( const DenseGeneratorExpression& ) = delete;
    DenseGeneratorExpression( DenseGeneratorExpression&& ) = default;
    DenseGeneratorExpression& operator=( const DenseGeneratorExpression& ) = delete;
    DenseGeneratorExpression& operator=( DenseGeneratorExpression&& ) = default;

    template<shapelike Shape>
    DenseGeneratorExpression( F&& f, const Shape& shape) : 
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

} // namespace ultra
#endif
