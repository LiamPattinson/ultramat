#ifndef __ULTRA_FIXED_ARRAY_HPP
#define __ULTRA_FIXED_ARRAY_HPP

// FixedArray.hpp
//
// A generic multidimensional array with fixed dimensions.

#include "FixedArrayUtils.hpp"
#include "Expression.hpp"
#include "View.hpp"

namespace ultra {

// Declare Array and preferred interface

template<class T, RCOrder Order, std::size_t... Dims>
class FixedArrayImpl;

template<class T,std::size_t... Dims>
using FixedArray = FixedArrayImpl<T,default_rc_order,Dims...>;

// Definitions

template<class T, RCOrder Order, std::size_t... Dims>
class FixedArrayImpl : public Expression<FixedArrayImpl<T,Order,Dims...>> {

    static constexpr std::size_t _dims = sizeof...(Dims);
    static constexpr std::size_t _size = variadic_product<Dims...>::value;

    static_assert(_dims >= 1, "FixedArray must have at least one dimension.");

public:

    using value_type = T;
    static constexpr RCOrder rc_order = Order;
    using shape_type = std::array<std::size_t,_dims>;
    using stride_type = std::array<std::size_t,_dims+1>;
    using data_type = std::array<T,_size>;

    // For convenience, specify row/col major via FixedArray<T,Dims...>::row/col_major
    using row_major = FixedArrayImpl<T,RCOrder::row_major,Dims...>;
    using col_major = FixedArrayImpl<T,RCOrder::col_major,Dims...>;

    // View of self
    using View = ultra::View<FixedArrayImpl<T,Order,Dims...>>;

protected:  

    static constexpr shape_type  _shape = {{Dims...}};
    static constexpr stride_type _stride = ( Order == RCOrder::row_major ? variadic_stride<Dims...>::row_major : variadic_stride<Dims...>::col_major );
    data_type _data;

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
        check_expression(*this,expression);
        auto expr=expression.begin();
        for(auto it=begin(), it_end=end(); it != it_end; ++it, ++expr) *it = *expr;
    }

    template<class U>
    FixedArrayImpl& operator=( const Expression<U>& expression) {
        check_expression(*this,expression);
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
    constexpr const shape_type& shape() const noexcept { return _shape; }
    constexpr const stride_type& stride() const noexcept { return _stride; }

    constexpr bool is_contiguous() const { return true;}
    constexpr bool is_read_only() const  { return false;}

    // ===============================================
    // Data access

    // Pointer to first element

    constexpr T* data() noexcept { return _data.data(); }
    constexpr const T* data() const noexcept { return _data.data(); }

    // Access via unsigned ints.
    
    template<std::integral... Ints> 
    T operator()( Ints... coords ) const noexcept {
        static_assert(sizeof...(Ints)==_dims,"FixedArray: Must index with correct number of indices.");
        return _data[variadic_memjump<Order>(_stride,coords...)];
    }
    
    template<std::integral... Ints> 
    T& operator()( Ints... coords ) noexcept {
        static_assert(sizeof...(Ints)==_dims,"FixedArray: Must index with correct number of indices.");
        return _data[variadic_memjump<Order>(_stride,coords...)];
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
    // TODO have this take a slice and return a view over the array
    
    constexpr T operator[](std::size_t ii) const noexcept { return _data[ii]; }
    constexpr T& operator[](std::size_t ii) noexcept { return _data[ii]; }

    // ===============================================
    // View creation

    View view() { return View(*this);}

    template<class... Slices> requires ( std::is_same<Slice,Slices>::value && ... )
    View view(const Slices&... slices) {
        return View(*this).slice(slices...);
    }

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
