#ifndef __ULTRA_DENSE_HPP
#define __ULTRA_DENSE_HPP

// Dense.hpp
//
// Defines generic dense array-like containers, including Array, Matrix, Vector, and their fixed-size counterparts.

#include "DenseBase.hpp"
#include "DenseView.hpp"

namespace ultra {

// Definitions

template<class T, RCOrder Order>
class ArrayImpl : public Expression<ArrayImpl<T,Order>>, public DenseBase<ArrayImpl<T,Order>,Order> {

    friend DenseBase<ArrayImpl<T,Order>,Order>;

public:

    using value_type = T;
    static constexpr RCOrder rc_order = Order;
    using shape_type = std::vector<std::size_t>;
    using stride_type = std::vector<std::size_t>;
    using data_type = std::vector<T>;
    using iterator = data_type::iterator;
    using const_iterator = data_type::const_iterator;


    // For convenience, specify row/col major via Array<T>::row/col_major
    using row_major = ArrayImpl<T,RCOrder::row_major>;
    using col_major = ArrayImpl<T,RCOrder::col_major>;

    // View of self
    using View = DenseView<ArrayImpl<T,Order>>;

protected:  

    shape_type  _shape;
    stride_type _stride;
    data_type   _data;

public:

    // ===============================================
    // Constructors

    ArrayImpl() = default;
    ArrayImpl( const ArrayImpl& other) = default;
    ArrayImpl( ArrayImpl&& other) = default;
    ArrayImpl& operator=( const ArrayImpl& other) = default;
    ArrayImpl& operator=( ArrayImpl&& other) = default;
    
    // Swap
    void swap( ArrayImpl& other) noexcept { 
        _shape.swap(other._shape);
        _stride.swap(other._stride);
        _data.swap(other._data);
    }

    friend void swap( ArrayImpl& a, ArrayImpl& b) noexcept { a.swap(b); }

    // Construct from shape
    template<std::ranges::range Shape>
    requires std::integral<typename Shape::value_type>
    ArrayImpl( const Shape& shape ) :
        _shape(shape.size()),
        _stride(shape.size()+1),
        _data(std::accumulate(shape.begin(),shape.end(),1,std::multiplies<typename Shape::value_type>{}))
    {
        std::ranges::copy( shape, _shape.begin());
        set_stride();
    }

    // Construct from shape with fill
    template<std::ranges::range Shape>
    requires std::integral<typename Shape::value_type>
    ArrayImpl( const Shape& shape, const value_type& val) :
        _shape(shape.size()),
        _stride(shape.size()+1),
        _data(std::accumulate(shape.begin(),shape.end(),1,std::multiplies<typename Shape::value_type>{}),val)
    {
        std::ranges::copy( shape, _shape.begin());
        set_stride();
    }


    // Construct from an expression
    template<class U>
    ArrayImpl( const Expression<U>& expression) :
        _shape(expression.dims()),
        _stride(expression.dims()+1),
        _data(expression.size())
    {
        for( std::size_t ii = 0; ii < dims(); ++ii) _shape[ii] = expression.shape(ii);
        set_stride();
        auto expr=expression.begin();
        for(auto it=begin(), it_end=end(); it != it_end; ++it, ++expr) *it = *expr;
    }

    template<class U>
    ArrayImpl& operator=( const Expression<U>& expression) {
        // check expression shape matches self. If not, resize and reshape in place
        try {
            check_expression(expression);
        } catch(const ExpressionException&) {
            _shape.resize(expression.dims());
            _stride.resize(expression.dims()+1);
            _data.resize(expression.size());
            for( std::size_t ii = 0; ii < dims(); ++ii) _shape[ii] = expression.shape(ii);
            set_stride();
        }
        auto expr=expression.begin();
        for(auto it=begin(), it_end=end(); it != it_end; ++it, ++expr) *it = *expr;
        return *this;
    }

    // ===============================================
    // Pull in methods from base class

    using DenseBase<ArrayImpl<T,Order>,Order>::dims;
    using DenseBase<ArrayImpl<T,Order>,Order>::size;
    using DenseBase<ArrayImpl<T,Order>,Order>::shape;
    using DenseBase<ArrayImpl<T,Order>,Order>::stride;
    using DenseBase<ArrayImpl<T,Order>,Order>::data;
    using DenseBase<ArrayImpl<T,Order>,Order>::view;
    using DenseBase<ArrayImpl<T,Order>,Order>::begin;
    using DenseBase<ArrayImpl<T,Order>,Order>::end;
    using DenseBase<ArrayImpl<T,Order>,Order>::operator();
    using DenseBase<ArrayImpl<T,Order>,Order>::operator[];
    using DenseBase<ArrayImpl<T,Order>,Order>::check_expression;
    using DenseBase<ArrayImpl<T,Order>,Order>::set_stride;
};

template<class T, RCOrder Order, std::size_t... Dims>
class FixedArrayImpl : public Expression<FixedArrayImpl<T,Order,Dims...>> , public DenseBase<FixedArrayImpl<T,Order,Dims...>,Order> {

    friend DenseBase<FixedArrayImpl<T,Order,Dims...>,Order>;

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
    using View = DenseView<FixedArrayImpl<T,Order,Dims...>>;

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

    // Pull in methods from CRTP base

    using DenseBase<FixedArrayImpl<T,Order,Dims...>,Order>::dims;
    using DenseBase<FixedArrayImpl<T,Order,Dims...>,Order>::size;
    using DenseBase<FixedArrayImpl<T,Order,Dims...>,Order>::shape;
    using DenseBase<FixedArrayImpl<T,Order,Dims...>,Order>::stride;
    using DenseBase<FixedArrayImpl<T,Order,Dims...>,Order>::data;
    using DenseBase<FixedArrayImpl<T,Order,Dims...>,Order>::view;
    using DenseBase<FixedArrayImpl<T,Order,Dims...>,Order>::begin;
    using DenseBase<FixedArrayImpl<T,Order,Dims...>,Order>::end;
    using DenseBase<FixedArrayImpl<T,Order,Dims...>,Order>::operator();
    using DenseBase<FixedArrayImpl<T,Order,Dims...>,Order>::operator[];
    using DenseBase<FixedArrayImpl<T,Order,Dims...>,Order>::check_expression;
};

} // namespace
#endif