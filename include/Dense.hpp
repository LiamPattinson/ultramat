#ifndef __ULTRA_DENSE_HPP
#define __ULTRA_DENSE_HPP

// Dense.hpp
//
// Defines generic dense array-like containers, including Array, Matrix, Vector, and their fixed-size counterparts.

#include "DenseImpl.hpp"
#include "DenseView.hpp"

namespace ultra {

template<class T, DenseType Type, DenseOrder Order>
class Dense : public DenseExpression<Dense<T,Type,Order>>, public DenseImpl<Dense<T,Type,Order>> {

    friend DenseImpl<Dense<T,Type,Order>>;

    static constexpr bool is_nd = ( Type == DenseType::nd );
    static constexpr std::size_t fixed_dims = static_cast<std::size_t>(Type);

public:

    using value_type = std::conditional_t<std::is_same<T,bool>::value,Bool,T>;
    using shape_type = std::conditional_t<is_nd,std::vector<std::size_t>,std::array<std::size_t,fixed_dims>>;
    using stride_type = std::conditional_t<is_nd,std::vector<std::size_t>,std::array<std::size_t,fixed_dims+1>>;
    using data_type = std::vector<value_type>;
    using iterator = data_type::iterator;
    using const_iterator = data_type::const_iterator;
    static constexpr DenseOrder order() { return Order; }

    // For convenience, specify row/col major via Array<T>::row/col_major
    using row_major = Dense<T,Type,DenseOrder::row_major>;
    using col_major = Dense<T,Type,DenseOrder::col_major>;

    // View of self
    using View = DenseView<Dense<T,Type,Order>>;

private: 

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
    template<shapelike Shape> requires ( is_nd )
    Dense( const Shape& shape ) :
        _shape(shape.size()),
        _stride(shape.size()+1),
        _data(std::accumulate(shape.begin(),shape.end(),1,std::multiplies<typename Shape::value_type>{}))
    {
        std::ranges::copy( shape, _shape.begin());
        set_stride();
    }

    template<shapelike Shape> requires ( is_nd )
    Dense( const Shape& shape, const value_type& fill) :
        _shape(shape.size()),
        _stride(shape.size()+1),
        _data(std::accumulate(shape.begin(),shape.end(),1,std::multiplies<typename Shape::value_type>{}),fill)
    {
        std::ranges::copy( shape, _shape.begin());
        set_stride();
    }

    template<shapelike Shape> requires ( !is_nd )
    Dense( const Shape& shape ) :
        _data(std::accumulate(shape.begin(),shape.end(),1,std::multiplies<typename Shape::value_type>{}))
    {
        test_fixed_dims(shape.size());
        std::ranges::copy( shape, _shape.begin());
        set_stride();
    }

    template<shapelike Shape> requires ( !is_nd )
    Dense( const Shape& shape, const value_type& fill) :
        _data(std::accumulate(shape.begin(),shape.end(),1,std::multiplies<typename Shape::value_type>{}),fill)
    {
        test_fixed_dims(shape.size());
        std::ranges::copy( shape, _shape.begin());
        set_stride();
    }

    Dense( std::size_t size ) requires ( !is_nd && fixed_dims == 1 ) : _shape{size}, _data(size) { set_stride(); }
    
    Dense( std::size_t size, const value_type& fill ) requires ( !is_nd && fixed_dims == 1 ) : _shape{size}, _data(size,fill) { set_stride(); }

    Dense( std::size_t rows, std::size_t cols ) requires ( !is_nd && fixed_dims == 2 ) : _shape{rows,cols}, _data(rows*cols) { set_stride(); }
    
    Dense( std::size_t rows, std::size_t cols, const value_type& fill ) requires ( !is_nd && fixed_dims == 2 ) :
        _shape{rows,cols},
        _data(rows*cols,fill) 
    {
        set_stride();
    }

    // Construct from an expression

    template<class U> requires ( is_nd )
    Dense( const DenseExpression<U>& expression) :
        _shape(expression.dims()),
        _stride(expression.dims()+1),
        _data(expression.size())
    {
        for( std::size_t ii = 0; ii < dims(); ++ii) _shape[ii] = expression.shape(ii);
        set_stride();
        equal_expression(expression);
    }

    template<class U> requires ( is_nd )
    Dense( DenseExpression<U>&& expression) :
        _shape(expression.dims()),
        _stride(expression.dims()+1),
        _data(expression.size())
    {
        for( std::size_t ii = 0; ii < dims(); ++ii) _shape[ii] = expression.shape(ii);
        set_stride();
        equal_expression(std::move(expression));
    }

    template<class U> requires ( !is_nd )
    Dense( const DenseExpression<U>& expression) :
        _data(expression.size())
    {
        test_fixed_dims(shape.size());
        for( std::size_t ii = 0; ii < dims(); ++ii) _shape[ii] = expression.shape(ii);
        set_stride();
        equal_expression(expression);
    }

    template<class U> requires ( !is_nd )
    Dense( DenseExpression<U>&& expression) :
        _data(expression.size())
    {
        test_fixed_dims(shape.size());
        for( std::size_t ii = 0; ii < dims(); ++ii) _shape[ii] = expression.shape(ii);
        set_stride();
        equal_expression(std::move(expression));
    }

    template<class U> requires ( is_nd )
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

    template<class U> requires ( is_nd )
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

    template<class U> requires ( !is_nd )
    Dense& operator=( const DenseExpression<U>& expression) {
        // check expression shape matches self. If not, resize and reshape in place
        try {
            check_expression(expression);
        } catch(const ExpressionException&) {
            test_fixed_dims(expression.dims());
            _data.resize(expression.size());
            for( std::size_t ii = 0; ii < dims(); ++ii) _shape[ii] = expression.shape(ii);
            set_stride();
        }
        return equal_expression(expression);
    }

    template<class U> requires ( !is_nd )
    Dense& operator=( DenseExpression<U>&& expression) {
        // check expression shape matches self. If not, resize and reshape in place
        try {
            check_expression(expression);
        } catch(const ExpressionException&) {
            test_fixed_dims(expression.dims());
            _data.resize(expression.size());
            for( std::size_t ii = 0; ii < dims(); ++ii) _shape[ii] = expression.shape(ii);
            set_stride();
        }
        return equal_expression(std::move(expression));
    }

    // ===============================================
    // Pull in methods from base class

    using DenseImpl<Dense<T,Type,Order>>::dims;
    using DenseImpl<Dense<T,Type,Order>>::size;
    using DenseImpl<Dense<T,Type,Order>>::shape;
    using DenseImpl<Dense<T,Type,Order>>::stride;
    using DenseImpl<Dense<T,Type,Order>>::data;
    using DenseImpl<Dense<T,Type,Order>>::fill;
    using DenseImpl<Dense<T,Type,Order>>::view;
    using DenseImpl<Dense<T,Type,Order>>::reshape;
    using DenseImpl<Dense<T,Type,Order>>::broadcast;
    using DenseImpl<Dense<T,Type,Order>>::permute;
    using DenseImpl<Dense<T,Type,Order>>::transpose;
    using DenseImpl<Dense<T,Type,Order>>::t;
    using DenseImpl<Dense<T,Type,Order>>::begin;
    using DenseImpl<Dense<T,Type,Order>>::end;
    using DenseImpl<Dense<T,Type,Order>>::num_stripes;
    using DenseImpl<Dense<T,Type,Order>>::get_stripe;
    using DenseImpl<Dense<T,Type,Order>>::required_stripe_dim;
    using DenseImpl<Dense<T,Type,Order>>::operator();
    using DenseImpl<Dense<T,Type,Order>>::operator[];
    using DenseImpl<Dense<T,Type,Order>>::operator=;
    using DenseImpl<Dense<T,Type,Order>>::operator+=;
    using DenseImpl<Dense<T,Type,Order>>::operator-=;
    using DenseImpl<Dense<T,Type,Order>>::operator*=;
    using DenseImpl<Dense<T,Type,Order>>::operator/=;
    using DenseImpl<Dense<T,Type,Order>>::equal_expression;
    using DenseImpl<Dense<T,Type,Order>>::check_expression;
    using DenseImpl<Dense<T,Type,Order>>::set_stride;
    using DenseImpl<Dense<T,Type,Order>>::is_contiguous;
    using DenseImpl<Dense<T,Type,Order>>::is_omp_parallelisable;

private:

    void test_fixed_dims( std::size_t d ) const {
        if( d != dims() ){
            if( fixed_dims==1 ) throw std::runtime_error("Ultra: Tried to construct Vector with shape of size " + std::to_string(d) + '.');
            if( fixed_dims==2 ) throw std::runtime_error("Ultra: Tried to construct Matrix with shape of size " + std::to_string(d) + '.');
        }
    }

    void resize_shape_and_stride( std::size_t size) requires ( is_nd ) {
        _shape.resize(size);
        _stride.resize(size+1);
    }

    void resize_shape_and_stride( std::size_t size) requires ( !is_nd ) {
        test_fixed_dims(size);
    }

};

template<class T, DenseOrder Order, std::size_t... Dims>
class DenseFixed : public DenseExpression<DenseFixed<T,Order,Dims...>>, public DenseImpl<DenseFixed<T,Order,Dims...>> {

    friend DenseImpl<DenseFixed<T,Order,Dims...>>;

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

private:

    static constexpr shape_type  _shape = {{Dims...}};
    static constexpr stride_type _stride = (Order == DenseOrder::row_major ? variadic_stride<Dims...>::row_major : variadic_stride<Dims...>::col_major);
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

    using DenseImpl<DenseFixed<T,Order,Dims...>>::dims;
    using DenseImpl<DenseFixed<T,Order,Dims...>>::size;
    using DenseImpl<DenseFixed<T,Order,Dims...>>::shape;
    using DenseImpl<DenseFixed<T,Order,Dims...>>::stride;
    using DenseImpl<DenseFixed<T,Order,Dims...>>::data;
    using DenseImpl<DenseFixed<T,Order,Dims...>>::fill;
    using DenseImpl<DenseFixed<T,Order,Dims...>>::view;
    using DenseImpl<DenseFixed<T,Order,Dims...>>::broadcast;
    using DenseImpl<DenseFixed<T,Order,Dims...>>::permute;
    using DenseImpl<DenseFixed<T,Order,Dims...>>::transpose;
    using DenseImpl<DenseFixed<T,Order,Dims...>>::t;
    using DenseImpl<DenseFixed<T,Order,Dims...>>::begin;
    using DenseImpl<DenseFixed<T,Order,Dims...>>::end;
    using DenseImpl<DenseFixed<T,Order,Dims...>>::num_stripes;
    using DenseImpl<DenseFixed<T,Order,Dims...>>::get_stripe;
    using DenseImpl<DenseFixed<T,Order,Dims...>>::required_stripe_dim;
    using DenseImpl<DenseFixed<T,Order,Dims...>>::operator();
    using DenseImpl<DenseFixed<T,Order,Dims...>>::operator[];
    using DenseImpl<DenseFixed<T,Order,Dims...>>::operator=;
    using DenseImpl<DenseFixed<T,Order,Dims...>>::operator+=;
    using DenseImpl<DenseFixed<T,Order,Dims...>>::operator-=;
    using DenseImpl<DenseFixed<T,Order,Dims...>>::operator*=;
    using DenseImpl<DenseFixed<T,Order,Dims...>>::operator/=;
    using DenseImpl<DenseFixed<T,Order,Dims...>>::check_expression;
    using DenseImpl<DenseFixed<T,Order,Dims...>>::is_contiguous;
    using DenseImpl<DenseFixed<T,Order,Dims...>>::is_omp_parallelisable;
};

} // namespace
#endif
