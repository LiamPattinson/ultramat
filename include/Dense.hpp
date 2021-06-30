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
class ArrayImpl : public DenseExpression<ArrayImpl<T,Order>>, public DenseBase<ArrayImpl<T,Order>,Order> {

    friend DenseBase<ArrayImpl<T,Order>,Order>;

public:

    using value_type = std::conditional_t<std::is_same<T,bool>::value,Bool,T>;
    static constexpr RCOrder rc_order = Order;
    using shape_type = std::vector<std::size_t>;
    using stride_type = std::vector<std::size_t>;
    using data_type = std::vector<value_type>;
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
    ArrayImpl( const DenseExpression<U>& expression) :
        _shape(expression.dims()),
        _stride(expression.dims()+1),
        _data(expression.size())
    {
        for( std::size_t ii = 0; ii < dims(); ++ii) _shape[ii] = expression.shape(ii);
        set_stride();
        equal_expression(expression);
    }

    template<class U>
    ArrayImpl( DenseExpression<U>&& expression) :
        _shape(expression.dims()),
        _stride(expression.dims()+1),
        _data(expression.size())
    {
        for( std::size_t ii = 0; ii < dims(); ++ii) _shape[ii] = expression.shape(ii);
        set_stride();
        equal_expression(std::move(expression));
    }

    template<class U>
    ArrayImpl& operator=( const DenseExpression<U>& expression) {
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
        return equal_expression(expression);
    }

    template<class U>
    ArrayImpl& operator=( DenseExpression<U>&& expression) {
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
        return equal_expression(std::move(expression));
    }

    // ===============================================
    // Pull in methods from base class

    using DenseBase<ArrayImpl<T,Order>,Order>::dims;
    using DenseBase<ArrayImpl<T,Order>,Order>::size;
    using DenseBase<ArrayImpl<T,Order>,Order>::shape;
    using DenseBase<ArrayImpl<T,Order>,Order>::stride;
    using DenseBase<ArrayImpl<T,Order>,Order>::order;
    using DenseBase<ArrayImpl<T,Order>,Order>::data;
    using DenseBase<ArrayImpl<T,Order>,Order>::fill;
    using DenseBase<ArrayImpl<T,Order>,Order>::view;
    using DenseBase<ArrayImpl<T,Order>,Order>::reshape;
    using DenseBase<ArrayImpl<T,Order>,Order>::broadcast;
    using DenseBase<ArrayImpl<T,Order>,Order>::permute;
    using DenseBase<ArrayImpl<T,Order>,Order>::transpose;
    using DenseBase<ArrayImpl<T,Order>,Order>::t;
    using DenseBase<ArrayImpl<T,Order>,Order>::begin;
    using DenseBase<ArrayImpl<T,Order>,Order>::end;
    using DenseBase<ArrayImpl<T,Order>,Order>::num_stripes;
    using DenseBase<ArrayImpl<T,Order>,Order>::get_stripe;
    using DenseBase<ArrayImpl<T,Order>,Order>::required_stripe_dim;
    using DenseBase<ArrayImpl<T,Order>,Order>::operator();
    using DenseBase<ArrayImpl<T,Order>,Order>::operator[];
    using DenseBase<ArrayImpl<T,Order>,Order>::operator=;
    using DenseBase<ArrayImpl<T,Order>,Order>::operator+=;
    using DenseBase<ArrayImpl<T,Order>,Order>::operator-=;
    using DenseBase<ArrayImpl<T,Order>,Order>::operator*=;
    using DenseBase<ArrayImpl<T,Order>,Order>::operator/=;
    using DenseBase<ArrayImpl<T,Order>,Order>::equal_expression;
    using DenseBase<ArrayImpl<T,Order>,Order>::check_expression;
    using DenseBase<ArrayImpl<T,Order>,Order>::set_stride;
    using DenseBase<ArrayImpl<T,Order>,Order>::is_contiguous;
    using DenseBase<ArrayImpl<T,Order>,Order>::is_omp_parallelisable;
};

template<class T, RCOrder Order, std::size_t... Dims>
class FixedArrayImpl : public DenseExpression<FixedArrayImpl<T,Order,Dims...>>, public DenseBase<FixedArrayImpl<T,Order,Dims...>,Order> {

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
    using iterator = data_type::iterator;
    using const_iterator = data_type::const_iterator;

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

    template<class U>
    FixedArrayImpl( const DenseExpression<U>& expression) { operator=(expression);}

    template<class U>
    FixedArrayImpl( DenseExpression<U>&& expression) { operator=(std::move(expression));}

    // Swap
    constexpr void swap( FixedArrayImpl& other) noexcept { _data.swap(other._data); }

    constexpr friend void swap( FixedArrayImpl& a,FixedArrayImpl& b) noexcept { a.swap(b); }

    // Pull in methods from CRTP base

    using DenseBase<FixedArrayImpl<T,Order,Dims...>,Order>::dims;
    using DenseBase<FixedArrayImpl<T,Order,Dims...>,Order>::size;
    using DenseBase<FixedArrayImpl<T,Order,Dims...>,Order>::shape;
    using DenseBase<FixedArrayImpl<T,Order,Dims...>,Order>::stride;
    using DenseBase<FixedArrayImpl<T,Order,Dims...>,Order>::order;
    using DenseBase<FixedArrayImpl<T,Order,Dims...>,Order>::data;
    using DenseBase<FixedArrayImpl<T,Order,Dims...>,Order>::fill;
    using DenseBase<FixedArrayImpl<T,Order,Dims...>,Order>::view;
    using DenseBase<FixedArrayImpl<T,Order,Dims...>,Order>::broadcast;
    using DenseBase<FixedArrayImpl<T,Order,Dims...>,Order>::permute;
    using DenseBase<FixedArrayImpl<T,Order,Dims...>,Order>::transpose;
    using DenseBase<FixedArrayImpl<T,Order,Dims...>,Order>::t;
    using DenseBase<FixedArrayImpl<T,Order,Dims...>,Order>::begin;
    using DenseBase<FixedArrayImpl<T,Order,Dims...>,Order>::end;
    using DenseBase<FixedArrayImpl<T,Order,Dims...>,Order>::num_stripes;
    using DenseBase<FixedArrayImpl<T,Order,Dims...>,Order>::get_stripe;
    using DenseBase<FixedArrayImpl<T,Order,Dims...>,Order>::required_stripe_dim;
    using DenseBase<FixedArrayImpl<T,Order,Dims...>,Order>::operator();
    using DenseBase<FixedArrayImpl<T,Order,Dims...>,Order>::operator[];
    using DenseBase<FixedArrayImpl<T,Order,Dims...>,Order>::operator=;
    using DenseBase<FixedArrayImpl<T,Order,Dims...>,Order>::operator+=;
    using DenseBase<FixedArrayImpl<T,Order,Dims...>,Order>::operator-=;
    using DenseBase<FixedArrayImpl<T,Order,Dims...>,Order>::operator*=;
    using DenseBase<FixedArrayImpl<T,Order,Dims...>,Order>::operator/=;
    using DenseBase<FixedArrayImpl<T,Order,Dims...>,Order>::check_expression;
    using DenseBase<FixedArrayImpl<T,Order,Dims...>,Order>::is_contiguous;
    using DenseBase<FixedArrayImpl<T,Order,Dims...>,Order>::is_omp_parallelisable;
};

} // namespace
#endif
