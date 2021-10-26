#ifndef __ULTRA_DENSE_HPP
#define __ULTRA_DENSE_HPP

/*! \file Dense.hpp
 *  \brief Defines class for dynamically-sized arrays.
 */

#include "DenseImpl.hpp"
#include "DenseView.hpp"

namespace ultra {

// ===============================================
// Dense


/*! \brief Dynamically-sized N-dimensional arrays.
 *
 *  A dynamically-sized, heap-allocated, N-dimensional array.
 *  Unlike #ultra::DenseFixed, \ref dense_shape%s and \ref dense_stride%s are determined at runtime and must
 *  be locally stored. Generally speaking, #ultra::Dense offers slightly lower performance than #ultra::DenseFixed, but much greater flexibility, similar
 *  to the advantages of `std::vector` versus `std::array`.
 *
 *  Most functionality is implemented via #ultra::DenseImpl.
 *
 *  The preferred interface to this class is the `ultra::Array` alias.
 */
template<class T, DenseOrder Order = default_order>
class Dense : public DenseExpression<Dense<T,Order>>, public DenseImpl<Dense<T,Order>> {

    friend DenseImpl<Dense<T,Order>>;

public:

    /*! \brief The type of each element of the array, usually arithmetic or complex types.
     *
     *  As data is stored internally using `std::vector`, `bool` is replaced with the #ultra::Bool class to avoids the weirdness of `std::vector<bool>`
     */
    using value_type = std::conditional_t<std::is_same<T,bool>::value,Bool,T>;

    using shape_type = std::vector<std::size_t>;      //!< The internal type of the array's \ref dense_shape
    using stride_type = std::vector<std::size_t>;     //!< The internal type of the array's \ref dense_stride
    using data_type = std::vector<value_type>;        //!< The internal 1D array type used to store the contents of `Dense`.
    using iterator = data_type::iterator;             //!< Non-const (modifying) iterator type
    using const_iterator = data_type::const_iterator; //!< Const (read-only) iterator type

    //! Returns the \link dense_order row/column-major ordering \endlink.
    static constexpr DenseOrder order() { return Order; }

    using row_major = Dense<T,DenseOrder::row_major>; //!< Alias for a \link dense_order row-major ordered \endlink `Dense` with the same `T`.
    using col_major = Dense<T,DenseOrder::col_major>; //!< Alias for a \link dense_order column-major ordered \endlink `Dense` with the same `T`.

    using View = DenseView<Dense<T,Order>>; //!< Alias for a #ultra::DenseView over this class.

private: 

    shape_type  _shape;  //!< The \ref dense_shape of the array.
    stride_type _stride; //!< The \ref dense_stride of the array.
    data_type   _data;   //!< The internal 1D array which stores the contents of `Dense`.

public:

    // ===============================================
    // Constructors

    Dense() = default;                                //!< Default constructor
    Dense( const Dense& other) = default;             //!< Copy constructor
    Dense( Dense&& other) = default;                  //!< Move constructor
    Dense& operator=( const Dense& other) = default;  //!< Copy assignment
    Dense& operator=( Dense&& other) = default;       //!< Move assignment
    
    //! Swap internal data
    void swap( Dense& other) noexcept { 
        _shape.swap(other._shape);
        _stride.swap(other._stride);
        _data.swap(other._data);
    }

    //! Friend swap function
    friend void swap( Dense& a, Dense& b) noexcept { a.swap(b); }

    //! Construct from \ref dense_shape
    template<shapelike Shape>
    Dense( const Shape& shape ) :
        _shape(shape.size()),
        _stride(shape.size()+1),
        _data(std::accumulate(shape.begin(),shape.end(),1,std::multiplies<typename Shape::value_type>{}))
    {
        std::ranges::copy( shape, _shape.begin());
        set_stride();
    }

    //! Construct from \ref dense_shape and fill.
    template<shapelike Shape>
    Dense( const Shape& shape, const value_type& fill) :
        _shape(shape.size()),
        _stride(shape.size()+1),
        _data(std::accumulate(shape.begin(),shape.end(),1,std::multiplies<typename Shape::value_type>{}),fill)
    {
        std::ranges::copy( shape, _shape.begin());
        set_stride();
    }

    //! Construct 1D array of with `size` elements
    Dense( std::size_t size ) : _shape{size}, _data(size) { set_stride(); }
    
    //! Construct 1D array of with `size` elements, and fill
    Dense( std::size_t size, const value_type& fill ) : _shape{size}, _data(size,fill) { set_stride(); }

    /*! \brief Construct from a #ultra::DenseExpression
     *
     *  Rather than using the implementation from #ultra::DenseImpl directly, `Dense` permits
     *  reshaping before assigning from the expression.
     */
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

    /*! \brief Construct from an rvalue #ultra::DenseExpression
     *
     *  Rather than using the implementation from #ultra::DenseImpl directly, `Dense` permits
     *  reshaping before assigning from the expression.
     */
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

    /*! \brief Assignment from a #ultra::DenseExpression
     *
     *  Rather than using the implementation from #ultra::DenseImpl directly, `Dense` permits
     *  reshaping before assigning from the expression.
     */
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

    /*! \brief Assignment from an rvalue #ultra::DenseExpression
     *
     *  Rather than using the implementation from #ultra::DenseImpl directly, `Dense` permits
     *  reshaping before assigning from the expression.
     */
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
