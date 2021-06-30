#ifndef __ULTRA_DENSE_HPP
#define __ULTRA_DENSE_HPP

// Dense.hpp
//
// Defines generic dense array-like containers, including Array, Matrix, Vector, and their fixed-size counterparts.

#include "DenseImpl.hpp"
#include "DenseView.hpp"

namespace ultra {

// Definitions

template<class T, DenseOrder Order>
class Dense : public DenseExpression<Dense<T,Order>>, public DenseImpl<Dense<T,Order>,Order> {

    friend DenseImpl<Dense<T,Order>,Order>;

public:

    using value_type = std::conditional_t<std::is_same<T,bool>::value,Bool,T>;
    using shape_type = std::vector<std::size_t>;
    using stride_type = std::vector<std::size_t>;
    using data_type = std::vector<value_type>;
    using iterator = data_type::iterator;
    using const_iterator = data_type::const_iterator;
    static constexpr DenseOrder order() { return Order; }

    // For convenience, specify row/col major via Array<T>::row/col_major
    using row_major = Dense<T,DenseOrder::row_major>;
    using col_major = Dense<T,DenseOrder::col_major>;

    // View of self
    using View = DenseView<Dense<T,Order>>;

protected:  

    shape_type  _shape;
    stride_type _stride;
    data_type   _data;

public:

    // ===============================================
    // Constructors

    Dense() = default;
    Dense( const Dense& other) = default;
    Dense( Dense&& other) = default;
    Dense& operator=( const Dense& other) = default;
    Dense& operator=( Dense&& other) = default;
    
    // Swap
    void swap( Dense& other) noexcept { 
        _shape.swap(other._shape);
        _stride.swap(other._stride);
        _data.swap(other._data);
    }

    friend void swap( Dense& a, Dense& b) noexcept { a.swap(b); }

    // Construct from shape
    template<std::ranges::range Shape>
    requires std::integral<typename Shape::value_type>
    Dense( const Shape& shape ) :
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
    Dense( const Shape& shape, const value_type& val) :
        _shape(shape.size()),
        _stride(shape.size()+1),
        _data(std::accumulate(shape.begin(),shape.end(),1,std::multiplies<typename Shape::value_type>{}),val)
    {
        std::ranges::copy( shape, _shape.begin());
        set_stride();
    }

    // Construct from an expression

    template<class U>
    Dense( const DenseExpression<U>& expression) :
        _shape(expression.dims()),
        _stride(expression.dims()+1),
        _data(expression.size())
    {
        for( std::size_t ii = 0; ii < dims(); ++ii) _shape[ii] = expression.shape(ii);
        set_stride();
        equal_expression(expression);
    }

    template<class U>
    Dense( DenseExpression<U>&& expression) :
        _shape(expression.dims()),
        _stride(expression.dims()+1),
        _data(expression.size())
    {
        for( std::size_t ii = 0; ii < dims(); ++ii) _shape[ii] = expression.shape(ii);
        set_stride();
        equal_expression(std::move(expression));
    }

    template<class U>
    Dense& operator=( const DenseExpression<U>& expression) {
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
    Dense& operator=( DenseExpression<U>&& expression) {
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

    using DenseImpl<Dense<T,Order>,Order>::dims;
    using DenseImpl<Dense<T,Order>,Order>::size;
    using DenseImpl<Dense<T,Order>,Order>::shape;
    using DenseImpl<Dense<T,Order>,Order>::stride;
    using DenseImpl<Dense<T,Order>,Order>::data;
    using DenseImpl<Dense<T,Order>,Order>::fill;
    using DenseImpl<Dense<T,Order>,Order>::view;
    using DenseImpl<Dense<T,Order>,Order>::reshape;
    using DenseImpl<Dense<T,Order>,Order>::broadcast;
    using DenseImpl<Dense<T,Order>,Order>::permute;
    using DenseImpl<Dense<T,Order>,Order>::transpose;
    using DenseImpl<Dense<T,Order>,Order>::t;
    using DenseImpl<Dense<T,Order>,Order>::begin;
    using DenseImpl<Dense<T,Order>,Order>::end;
    using DenseImpl<Dense<T,Order>,Order>::num_stripes;
    using DenseImpl<Dense<T,Order>,Order>::get_stripe;
    using DenseImpl<Dense<T,Order>,Order>::required_stripe_dim;
    using DenseImpl<Dense<T,Order>,Order>::operator();
    using DenseImpl<Dense<T,Order>,Order>::operator[];
    using DenseImpl<Dense<T,Order>,Order>::operator=;
    using DenseImpl<Dense<T,Order>,Order>::operator+=;
    using DenseImpl<Dense<T,Order>,Order>::operator-=;
    using DenseImpl<Dense<T,Order>,Order>::operator*=;
    using DenseImpl<Dense<T,Order>,Order>::operator/=;
    using DenseImpl<Dense<T,Order>,Order>::equal_expression;
    using DenseImpl<Dense<T,Order>,Order>::check_expression;
    using DenseImpl<Dense<T,Order>,Order>::set_stride;
    using DenseImpl<Dense<T,Order>,Order>::is_contiguous;
    using DenseImpl<Dense<T,Order>,Order>::is_omp_parallelisable;
};

template<class T, DenseOrder Order, std::size_t... Dims>
class DenseFixed : public DenseExpression<DenseFixed<T,Order,Dims...>>, public DenseImpl<DenseFixed<T,Order,Dims...>,Order> {

    friend DenseImpl<DenseFixed<T,Order,Dims...>,Order>;

    static constexpr std::size_t _dims = sizeof...(Dims);
    static constexpr std::size_t _size = variadic_product<Dims...>::value;
    static_assert(_dims >= 1, "FixedArray must have at least one dimension.");

public:

    using value_type = T;
    using shape_type = std::array<std::size_t,_dims>;
    using stride_type = std::array<std::size_t,_dims+1>;
    using data_type = std::array<T,_size>;
    using iterator = data_type::iterator;
    using const_iterator = data_type::const_iterator;
    static constexpr DenseOrder order() { return Order; }

    // For convenience, specify row/col major via FixedArray<T,Dims...>::row/col_major
    using row_major = DenseFixed<T,DenseOrder::row_major,Dims...>;
    using col_major = DenseFixed<T,DenseOrder::col_major,Dims...>;

    // View of self
    using View = DenseView<DenseFixed<T,Order,Dims...>>;

protected:  

    static constexpr shape_type  _shape = {{Dims...}};
    static constexpr stride_type _stride = ( Order == DenseOrder::row_major ? variadic_stride<Dims...>::row_major : variadic_stride<Dims...>::col_major );
    data_type _data;

public:

    // ===============================================
    // Constructors

    DenseFixed() = default;
    DenseFixed( const DenseFixed& other) = default;
    DenseFixed( DenseFixed&& other) = default;
    DenseFixed& operator=( const DenseFixed& other) = default;
    DenseFixed& operator=( DenseFixed&& other) = default;
    
    // With fill
    DenseFixed( const T& fill) { _data.fill(fill); }

    template<class U>
    DenseFixed( const DenseExpression<U>& expression) { operator=(expression);}

    template<class U>
    DenseFixed( DenseExpression<U>&& expression) { operator=(std::move(expression));}

    // Swap
    constexpr void swap( DenseFixed& other) noexcept { _data.swap(other._data); }

    constexpr friend void swap( DenseFixed& a,DenseFixed& b) noexcept { a.swap(b); }

    // Pull in methods from CRTP base

    using DenseImpl<DenseFixed<T,Order,Dims...>,Order>::dims;
    using DenseImpl<DenseFixed<T,Order,Dims...>,Order>::size;
    using DenseImpl<DenseFixed<T,Order,Dims...>,Order>::shape;
    using DenseImpl<DenseFixed<T,Order,Dims...>,Order>::stride;
    using DenseImpl<DenseFixed<T,Order,Dims...>,Order>::data;
    using DenseImpl<DenseFixed<T,Order,Dims...>,Order>::fill;
    using DenseImpl<DenseFixed<T,Order,Dims...>,Order>::view;
    using DenseImpl<DenseFixed<T,Order,Dims...>,Order>::broadcast;
    using DenseImpl<DenseFixed<T,Order,Dims...>,Order>::permute;
    using DenseImpl<DenseFixed<T,Order,Dims...>,Order>::transpose;
    using DenseImpl<DenseFixed<T,Order,Dims...>,Order>::t;
    using DenseImpl<DenseFixed<T,Order,Dims...>,Order>::begin;
    using DenseImpl<DenseFixed<T,Order,Dims...>,Order>::end;
    using DenseImpl<DenseFixed<T,Order,Dims...>,Order>::num_stripes;
    using DenseImpl<DenseFixed<T,Order,Dims...>,Order>::get_stripe;
    using DenseImpl<DenseFixed<T,Order,Dims...>,Order>::required_stripe_dim;
    using DenseImpl<DenseFixed<T,Order,Dims...>,Order>::operator();
    using DenseImpl<DenseFixed<T,Order,Dims...>,Order>::operator[];
    using DenseImpl<DenseFixed<T,Order,Dims...>,Order>::operator=;
    using DenseImpl<DenseFixed<T,Order,Dims...>,Order>::operator+=;
    using DenseImpl<DenseFixed<T,Order,Dims...>,Order>::operator-=;
    using DenseImpl<DenseFixed<T,Order,Dims...>,Order>::operator*=;
    using DenseImpl<DenseFixed<T,Order,Dims...>,Order>::operator/=;
    using DenseImpl<DenseFixed<T,Order,Dims...>,Order>::check_expression;
    using DenseImpl<DenseFixed<T,Order,Dims...>,Order>::is_contiguous;
    using DenseImpl<DenseFixed<T,Order,Dims...>,Order>::is_omp_parallelisable;
};

} // namespace
#endif
