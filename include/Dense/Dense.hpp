#ifndef __ULTRA_DENSE_HPP
#define __ULTRA_DENSE_HPP

#include "DenseImpl.hpp"
#include "DenseView.hpp"

namespace ultra {

// ===============================================
// Dense
//
// Defines generic N-d dense containers.
// Preferred interface is via the Array alias.

template<class T, DenseOrder Order>
class Dense : public DenseExpression<Dense<T,Order>>, public DenseImpl<Dense<T,Order>> {

    friend DenseImpl<Dense<T,Order>>;

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
    template<shapelike Shape>
    Dense( const Shape& shape ) :
        _shape(shape.size()),
        _stride(shape.size()+1),
        _data(std::accumulate(shape.begin(),shape.end(),1,std::multiplies<typename Shape::value_type>{}))
    {
        std::ranges::copy( shape, _shape.begin());
        set_stride();
    }

    template<shapelike Shape>
    Dense( const Shape& shape, const value_type& fill) :
        _shape(shape.size()),
        _stride(shape.size()+1),
        _data(std::accumulate(shape.begin(),shape.end(),1,std::multiplies<typename Shape::value_type>{}),fill)
    {
        std::ranges::copy( shape, _shape.begin());
        set_stride();
    }

    Dense( std::size_t size ) : _shape{size}, _data(size) { set_stride(); }
    
    Dense( std::size_t size, const value_type& fill ) : _shape{size}, _data(size,fill) { set_stride(); }

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

    using DenseImpl<Dense<T,Order>>::dims;
    using DenseImpl<Dense<T,Order>>::size;
    using DenseImpl<Dense<T,Order>>::shape;
    using DenseImpl<Dense<T,Order>>::stride;
    using DenseImpl<Dense<T,Order>>::data;
    using DenseImpl<Dense<T,Order>>::fill;
    using DenseImpl<Dense<T,Order>>::view;
    using DenseImpl<Dense<T,Order>>::reshape;
    using DenseImpl<Dense<T,Order>>::permute;
    using DenseImpl<Dense<T,Order>>::transpose;
    using DenseImpl<Dense<T,Order>>::t;
    using DenseImpl<Dense<T,Order>>::begin;
    using DenseImpl<Dense<T,Order>>::end;
    using DenseImpl<Dense<T,Order>>::get_stripe;
    using DenseImpl<Dense<T,Order>>::required_stripe_dim;
    using DenseImpl<Dense<T,Order>>::operator();
    using DenseImpl<Dense<T,Order>>::operator[];
    using DenseImpl<Dense<T,Order>>::operator=;
    using DenseImpl<Dense<T,Order>>::operator+=;
    using DenseImpl<Dense<T,Order>>::operator-=;
    using DenseImpl<Dense<T,Order>>::operator*=;
    using DenseImpl<Dense<T,Order>>::operator/=;
    using DenseImpl<Dense<T,Order>>::equal_expression;
    using DenseImpl<Dense<T,Order>>::check_expression;
    using DenseImpl<Dense<T,Order>>::set_stride;
    using DenseImpl<Dense<T,Order>>::is_contiguous;
    using DenseImpl<Dense<T,Order>>::is_omp_parallelisable;
    using DenseImpl<Dense<T,Order>>::is_broadcasting;

};

} // namespace
#endif
