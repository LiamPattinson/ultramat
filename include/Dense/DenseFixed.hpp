#ifndef __ULTRA_DENSE_FIXED_HPP
#define __ULTRA_DENSE_FIXED_HPP

/*! \file DenseFixed.hpp
 *  \brief Defines class for fixed-size N-dimensional arrays, and associated compile-time utilities.
 */

#include "DenseImpl.hpp"
#include "DenseView.hpp"

namespace ultra {

// ==========================
// Fixed-Size Dense Utils
// Lots of compile-time template nonsense ahead.

// ==========================
// variadic_product

template<std::size_t... Ints> struct variadic_product;

template<> struct variadic_product<> { static constexpr std::size_t value = 1; };

//! Given of variadic template list of`std::size_t`, computes their product at compile time.
template<std::size_t Int,std::size_t... Ints>
struct variadic_product<Int,Ints...> {
    //!  Contains product of provided `std::size_t`
    static constexpr std::size_t value = Int*variadic_product<Ints...>::value;
};

// ==========================
// index_sequence_to_array

//! Creates `std::array<std::size_t,N>` from a variadic template list of an arbitrary number of `std::size_t`.
template<std::size_t... Ints>
constexpr auto index_sequence_to_array( std::index_sequence<Ints...> ) noexcept {
    return std::array<std::size_t,sizeof...(Ints)>{{Ints...}};
}

// ==========================
// reverse_index_sequence

template<class T1,class T2> struct reverse_index_sequence_impl;

template<std::size_t Int1, std::size_t... Ints1, std::size_t Int2, std::size_t... Ints2>
struct reverse_index_sequence_impl< std::index_sequence<Int1,Ints1...>, std::index_sequence<Int2,Ints2...>> {
    using type = 
        reverse_index_sequence_impl<std::index_sequence<Ints1...>,std::index_sequence<Int1,Int2,Ints2...>>::type;
};

template<std::size_t... Ints2>
struct reverse_index_sequence_impl< std::index_sequence<>, std::index_sequence<Ints2...>> {
    using type = std::index_sequence<Ints2...>;
};

template<std::size_t Int1, std::size_t... Ints1>
struct reverse_index_sequence_impl< std::index_sequence<Int1,Ints1...>, std::index_sequence<>> {
    using type = reverse_index_sequence_impl<std::index_sequence<Ints1...>,std::index_sequence<Int1>>::type;
};

template<class T> struct reverse_index_sequence;

//! Given a `std::index_sequence`, gives a `std::index_sequence` with all contained `std::size_t` in reverse order.
template<std::size_t... Ints>
struct reverse_index_sequence<std::index_sequence<Ints...>> {
    //! Defines the reversed `index_sequence`
    using type = reverse_index_sequence_impl<std::index_sequence<Ints...>,std::index_sequence<>>::type;
};

//! Typedef shortcut for `reverse_index_sequence<T>::type`
template<class T>
using reverse_index_sequence_t = reverse_index_sequence<T>::type;

// ==========================
/*! \struct variadic_stride
 *  \brief Given a \ref dense_shape in the form of a variadic template list of `std::size_t`, 
 *  gives a `std::array<std::size_t,N>` of the corresponding \ref dense_stride for each dimension.
 *
 * The member variable `row_major` provides the stride array for row-major ordered arrays, while the `col_major` member variable
 * gives the column-major ordered variant.
 */

template<class T1,class T2> struct variadic_stride_impl;

template<std::size_t ShapeInt, std::size_t... ShapeInts, std::size_t StrideInt, std::size_t... StrideInts>
struct variadic_stride_impl< std::index_sequence<ShapeInt,ShapeInts...>, std::index_sequence<StrideInt,StrideInts...>> {
    using type = 
        variadic_stride_impl<std::index_sequence<ShapeInts...>,std::index_sequence<ShapeInt*StrideInt,StrideInt,StrideInts...>>::type;
};

template<std::size_t... StrideInts>
struct variadic_stride_impl< std::index_sequence<>, std::index_sequence<StrideInts...>> {
    using type = std::index_sequence<StrideInts...>;
};

template<std::size_t ShapeInt, std::size_t... ShapeInts>
struct variadic_stride_impl< std::index_sequence<ShapeInt,ShapeInts...>, std::index_sequence<>> {
    using type = variadic_stride_impl<std::index_sequence<ShapeInts...>,std::index_sequence<ShapeInt>>::type;
};

template<std::size_t... ShapeInts>
struct variadic_stride {
    static constexpr auto col_major = index_sequence_to_array(
        typename reverse_index_sequence<typename variadic_stride_impl<std::index_sequence<1,ShapeInts...>,std::index_sequence<>>::type>::type()
    );
    static constexpr auto row_major = index_sequence_to_array(
        typename variadic_stride_impl<typename reverse_index_sequence<std::index_sequence<ShapeInts...,1>>::type,std::index_sequence<>>::type()
    );
};

// ===============================================
// DenseFixed

/*! \brief Fixed-size N-dimensional arrays.
 *
 *  A fixed-size, stack-allocated, N-dimensional array. Shapes and strides are determined at compile time, and do not take up memory in
 *  instantiations of the class.
 *
 *  Most functionality is implemented via #ultra::DenseImpl.
 *
 *  The preferred interface to this class is the ultra::Array alias.
 */
template<class T, DenseOrder Order, std::size_t... Dims>
class DenseFixed : public DenseExpression<DenseFixed<T,Order,Dims...>>, public DenseImpl<DenseFixed<T,Order,Dims...>> {

    friend DenseImpl<DenseFixed<T,Order,Dims...>>;

    static constexpr std::size_t _dims = sizeof...(Dims);                   //!< The number of dimensions of the `DenseFixed`. 
    static constexpr std::size_t _size = variadic_product<Dims...>::value;  //!< The total number of elements in the `DenseFixed`.
    static_assert(_dims >= 1, "FixedArray must have at least one dimension.");

public:

    using value_type = T;                                  //!< The type of each element of the array, usually arithmetic or complex types.
    using shape_type = std::array<std::size_t,_dims>;      //!< The internal type of the array's \ref dense_shape
    using stride_type = std::array<std::size_t,_dims+1>;   //!< The internal type of the array's \ref dense_stride
    using data_type = std::array<T,_size>;                 //!< The internal 1D array type used to store the contents of `DenseFixed`.
    using iterator = data_type::iterator;                  //!< Non-const (modifying) iterator type
    using const_iterator = data_type::const_iterator;      //!< Const (read-only) iterator type

    //! Returns the \link dense_order row/column-major ordering \endlink.
    static constexpr DenseOrder order() { 
        return Order;
    }
    //! Alias for a \link dense_order row-major ordered \endlink `DenseFixed` with the same `T` and dimensions.
    using row_major = DenseFixed<T,DenseOrder::row_major,Dims...>; 
    //! Alias for a column-major ordered `DenseFixed` with the same `T` and dimensions.
    using col_major = DenseFixed<T,DenseOrder::col_major,Dims...>;
    //! Alias for a #ultra::DenseView over this class.
    using View = DenseView<DenseFixed<T,Order,Dims...>>;

private:

    /*! \brief The internal \ref dense_shape.
     *
     * Static and determined at compile time, so does not add to the memory footprint.
     */
    static constexpr shape_type  _shape = {{Dims...}};

    /*! \brief The internal \ref dense_stride. 
     *
     * Static and determined at compile time, so does not add to the memory footprint.
     */
    static constexpr stride_type _stride = (Order == DenseOrder::row_major ? variadic_stride<Dims...>::row_major : variadic_stride<Dims...>::col_major);
    
    //! The internal 1D array which stores the contents of `DenseFixed`.
    data_type _data;

public:

    // ===============================================
    // Constructors

    DenseFixed() = default;                                     //!< Default constructor
    DenseFixed( const DenseFixed& other) = default;             //!< Copy constructor
    DenseFixed( DenseFixed&& other) = default;                  //!< Move constructor
    DenseFixed& operator=( const DenseFixed& other) = default;  //!< Copy assignment
    DenseFixed& operator=( DenseFixed&& other) = default;       //!< Move assignment
    
    //! Construct and fill. Copies `fill` to all elements.
    DenseFixed( const T& fill) {
        _data.fill(fill);
    }

    /*! \brief Construct from expression
     *
     *  Relies on #ultra::DenseImpl::operator=().
     */
    template<class U>
    DenseFixed( const DenseExpression<U>& expression) {
        operator=(expression);
    }

    /*! \brief Construct from rvalue expression
     *
     *  Relies on #ultra::DenseImpl::operator=().
     */
    template<class U>
    DenseFixed( DenseExpression<U>&& expression) {
        operator=(std::move(expression));
    }

    //! Swap internal data
    constexpr void swap( DenseFixed& other) noexcept {
        _data.swap(other._data);
    }

    //! Friend swap function
    constexpr friend void swap( DenseFixed& a,DenseFixed& b) noexcept {
        a.swap(b);
    }

    // Pull in methods from CRTP base

    using DenseImpl<DenseFixed<T,Order,Dims...>>::dims;
    using DenseImpl<DenseFixed<T,Order,Dims...>>::size;
    using DenseImpl<DenseFixed<T,Order,Dims...>>::shape;
    using DenseImpl<DenseFixed<T,Order,Dims...>>::stride;
    using DenseImpl<DenseFixed<T,Order,Dims...>>::data;
    using DenseImpl<DenseFixed<T,Order,Dims...>>::fill;
    using DenseImpl<DenseFixed<T,Order,Dims...>>::view;
    using DenseImpl<DenseFixed<T,Order,Dims...>>::permute;
    using DenseImpl<DenseFixed<T,Order,Dims...>>::transpose;
    using DenseImpl<DenseFixed<T,Order,Dims...>>::t;
    using DenseImpl<DenseFixed<T,Order,Dims...>>::begin;
    using DenseImpl<DenseFixed<T,Order,Dims...>>::end;
    using DenseImpl<DenseFixed<T,Order,Dims...>>::get_stripe;
    using DenseImpl<DenseFixed<T,Order,Dims...>>::required_stripe_dim;
    using DenseImpl<DenseFixed<T,Order,Dims...>>::operator();
    using DenseImpl<DenseFixed<T,Order,Dims...>>::operator[];
    using DenseImpl<DenseFixed<T,Order,Dims...>>::operator=;
    using DenseImpl<DenseFixed<T,Order,Dims...>>::operator+=;
    using DenseImpl<DenseFixed<T,Order,Dims...>>::operator-=;
    using DenseImpl<DenseFixed<T,Order,Dims...>>::operator*=;
    using DenseImpl<DenseFixed<T,Order,Dims...>>::operator/=;
    using DenseImpl<DenseFixed<T,Order,Dims...>>::is_contiguous;
    using DenseImpl<DenseFixed<T,Order,Dims...>>::is_omp_parallelisable;
    using DenseImpl<DenseFixed<T,Order,Dims...>>::is_broadcasting;
};

} // namespace ultra
#endif
