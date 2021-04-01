#ifndef __ULTRA_FIXED_ARRAY_HPP
#define __ULTRA_FIXED_ARRAY_HPP

// FixedArray.hpp
//
// A generic multidimensional array with fixed dimensions.

#include "FixedArrayUtils.hpp"
#include "Expression.hpp"

#include <stdexcept>
#include <string>
#include <algorithm>
#include <numeric>
#include <concepts>
#include <ranges>

namespace ultra {

// Declare Array and preferred interface

template<class T, RCOrder Order, std::size_t... Dims>
class FixedArrayImpl;

template<class T,std::size_t... Dims>
using FixedArray = FixedArrayImpl<T,default_rc_order,Dims...>;

// Definitions

template<class T, RCOrder Order, std::size_t... Dims>
class FixedArrayImpl : public Expression<FixedArrayImpl<T,Order,Dims...>> {

    static_assert(sizeof...(Dims) >= 1, "FixedArray must have at least one dimension.");

public:

    using value_type = T;
    using size_type = std::size_t;
    using difference_type = std::ptrdiff_t;
    using shape_type = std::array<size_type,sizeof...(Dims)>;
    using stride_type = std::array<size_type,sizeof...(Dims)+1>;

    // For convenience, specify row/col major via FixedArray<T,Dims...>::row/col_major
    using row_major = FixedArrayImpl<T,RCOrder::row_major,Dims...>;
    using col_major = FixedArrayImpl<T,RCOrder::col_major,Dims...>;
  
protected:  
    
    static constexpr std::size_t _size = variadic_product<Dims...>::value;
    static constexpr shape_type  _shape = {{Dims...}};
    static constexpr stride_type _stride = ( Order == RCOrder::row_major ? variadic_stride<Dims...>::row_major : variadic_stride<Dims...>::col_major );

    using data_type = std::array<T,_size>;
    data_type _data;

    // Helper functions
    
    // variadic_memjump
    // used with round-bracket indexing.

    // Base case
    template<std::size_t N, std::integral Int>
    static constexpr std::size_t variadic_memjump_impl( Int coord) noexcept {
        return _stride[N] * coord; 
    }

    // Recursive step
    template<std::size_t N, std::integral Int, std::integral... Ints>
    static constexpr std::size_t variadic_memjump_impl( Int coord, Ints... coords) noexcept {
        return (_stride[N] * coord) + variadic_memjump_impl<N+1,Ints...>(coords...);
    }

    template<std::integral... Ints>
    static constexpr std::size_t variadic_memjump( Ints... coords) noexcept {
        static_assert(sizeof...(Ints)==sizeof...(Dims),"FixedArray: Must index with correct number of indices.");
        // if row major, must skip first element of stride
        return variadic_memjump_impl<(Order==RCOrder::row_major?1:0),Ints...>(coords...);
    }

    // check_expression
    // checks if expression matches dims and shape, throws an exception if it doesn't.

    template<class U>
    void check_expression( const Expression<U>& expression) const {
        if( expression.dims() != dims()){
            throw std::runtime_error("FixedArray: Tried to construct/assign array of dims " + std::to_string(dims()) 
                    + " with expression of dims " + std::to_string(expression.dims()));
        }
        for( std::size_t ii=0; ii<dims(); ++ii){
            if( expression.shape(ii) != shape(ii) ){
                std::string expression_shape("( ");
                std::string array_shape("( ");
                for( std::size_t ii=0; ii<dims(); ++ii){
                    expression_shape += std::to_string(expression.shape(ii)) + ' ';
                    array_shape += std::to_string(shape(ii)) + ' ';
                }
                expression_shape += ')';
                array_shape += ')';
                throw std::runtime_error("FixedArray: Tried to construct/assign array of shape " + array_shape 
                        + "with expression of shape " + expression_shape);
            }
        }
    }

public:

    // ===============================================
    // Constructors

    FixedArrayImpl() = default;
    FixedArrayImpl( const FixedArrayImpl& other) = default;
    FixedArrayImpl( FixedArrayImpl&& other) = default;
    FixedArrayImpl& operator=( const FixedArrayImpl& other) = default;
    FixedArrayImpl& operator=( FixedArrayImpl&& other) = default;
    
    // With fill
    FixedArrayImpl( const T& fill) { _data.fill(fill); }

    // Swap
    constexpr void swap( FixedArrayImpl& other) noexcept { _data.swap(other._data); }

    constexpr friend void swap( FixedArrayImpl& a,FixedArrayImpl& b) noexcept { a.swap(b); }

    // Construct from an expression
    template<class U>
    FixedArrayImpl( const Expression<U>& expression) {
        check_expression(expression);
        auto expr=expression.begin();
        for(auto it=begin(), it_end=end(); it != it_end; ++it, ++expr) *it = *expr;
    }

    template<class U>
    FixedArrayImpl& operator=( const Expression<U>& expression) {
        check_expression(expression);
        auto expr=expression.begin();
        for(auto it=begin(), it_end=end(); it != it_end; ++it, ++expr) *it = *expr;
        return *this;
    }

    // ===============================================
    // Attributes

    constexpr std::size_t dims() const noexcept { return _shape.size(); }
    constexpr std::size_t size() const noexcept { return _size; }
    constexpr std::size_t shape( std::size_t dim) const noexcept { return _shape[dim]; }
    constexpr std::size_t stride( std::size_t dim) const noexcept { return _stride[dim]; }

    // ===============================================
    // Data access

    // Pointer to first element

    constexpr T* data() noexcept { return _data.data(); }
    constexpr const T* data() const noexcept { return _data.data(); }

    // Access via unsigned ints.
    
    template<std::integral... Ints> 
    T operator()( Ints... coords ) const noexcept {
        return _data[variadic_memjump(coords...)];
    }
    
    template<std::integral... Ints> 
    T& operator()( Ints... coords ) noexcept {
        return _data[variadic_memjump(coords...)];
    }

    // Access via range
    
    template<std::ranges::range Coords> requires std::integral<typename Coords::value_type>
    T operator()( const Coords& coords) const {
        return _data[std::inner_product(coords.begin(),coords.end(),_stride.begin()+(Order==RCOrder::row_major),0)];
    }

    template<std::ranges::range Coords> requires std::integral<typename Coords::value_type>
    T& operator()( const Coords& coords) {
        return _data[std::inner_product(coords.begin(),coords.end(),_stride.begin()+(Order==RCOrder::row_major),0)];
    }

    // Access via square brackets
    // Treats all arrays as 1D containers.
    
    constexpr T operator[](std::size_t ii) const noexcept { return _data[ii]; }
    constexpr T& operator[](std::size_t ii) noexcept { return _data[ii]; }

    // ===============================================
    // Iteration

    using iterator = data_type::iterator;
    using const_iterator = data_type::const_iterator;

    constexpr iterator begin() noexcept { return _data.begin(); }
    constexpr const_iterator begin() const noexcept { return _data.begin(); }
    constexpr iterator end() noexcept { return _data.end(); }
    constexpr const_iterator end() const noexcept { return _data.end(); }

    // TODO striped iteration
};

} // namespace
#endif
