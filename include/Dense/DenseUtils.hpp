#ifndef __ULTRA_DENSE_UTILS_HPP
#define __ULTRA_DENSE_UTILS_HPP

/*! \file DenseUtils.hpp
 *  \brief Defines any utility functions/structs deemed useful for Dense objects.
 *
 *  This file contains utilities widely used in the definitions and internal workings
 *  of Dense objects. Most features are not expected to be used by the end-user, but
 *  one important feature is the definition of Array -- the recommended alias for
 *  Dense objects.
 */

#include "ultramat/include/Utils/Utils.hpp"

namespace ultra {

// ==============================================
// DenseOrder

//! An Enum Class used to indicate whether a Dense object is row-major or column-major ordered.
/*! `DenseOrder` is commonly used as a template argument to `Dense` objects, and is used to indicate
 *  whether a given `Dense` object is row-major ordered (C-style) or column-major ordered
 *  (Fortran-style). If an object is row-major ordered, then the last dimension is the fastest
 *  incrementing, while if an object is column-major ordered, the first dimension is the fastest
 *  incrementing. The option `DenseOrder::mixed` exists to indicate that a given object is neither
 *  row-major nor column-major. Examples include an expression in which a row-major `Array` is
 *  added to a column-major `Array`.
 */
enum class DenseOrder { 
    //! Row-major ordering, last dimension increments fastest
    row_major, 
    //! Column-major ordering, first dimension increments fastest
    col_major, 
    //! Object is neither row-major nor column-major
    mixed
};

//! The default row/column-major ordering used throughout ultramat. Normally set to row-major.
const DenseOrder default_order = DenseOrder::row_major;


// ==============================================
// DenseType

//! An Enum Class used to indicate whether a `Dense` object is a vector, matrix, or is N-dimensional.
/*! Used internally by `Dense` to determine whether to store shape/stride as `std::vector` or `std::array`, and whether
 *  to restrict reshaping. This is currently in the firing line for deprecation.
 */
enum class DenseType : std::size_t { 
    //! `Dense` object is 1D
    vec=1, 
    //! `Dense` object is 2D
    mat=2, 
    //! `Dense` object is N-Dimensional
    nd=0 
};

// ==============================================
// DenseType

//! An Enum Class used to indicate whether an object is read-only or if it may be written to.
/*! Used internally by objects such as `DenseView` and `DenseStripe` to indicate whether they should reference
 *  their data using a regular pointer (writeable) or a const pointer (read_only).
 */
enum class ReadWrite {
    //! Object may be freely read from or written to
    writeable,
    //! Object may only be read from
    read_only
};

// ==============================================
// Array

template<class, DenseType, DenseOrder> class Dense;
template<class, DenseOrder, std::size_t...> class DenseFixed;
template<class, ReadWrite = ReadWrite::writeable> class DenseView;
template<class, ReadWrite = ReadWrite::writeable> class DenseStripe;

/*! \brief The alias used to access either dynamically-sized Dense objects or fixed-size `Dense` objects.
 *  \tparam T the value_type contained by the `Array`
 *  \tparam Dims (Optional) A list of unsigned integers giving the dimensions of the Array. Omitting this results in a dynamically sized array.
 *
 *  `Array`'s should be considered the primary objects in the ultramat library, but in actuality `Array`
 *  is actually an alias. If `Array` is supplied with only a single template argument, such as `Array<int>`
 *  or `Array<double>`, it is an alias for an N-dimensional dynamically-sized `Dense` object. If it is provided
 *  with a list of dimension sizes after the value type, such as `Array<float,3,3>`, it instead aliases a
 *  fixed-size \f$3\times3\f$ `DenseFixed` object. Both objects may be used interchangeably in ultramat expressions,
 *  though the latter may not be resized in any way.
 */
template<class T,std::size_t... Dims>
using Array = std::conditional_t<sizeof...(Dims),DenseFixed<T,default_order,Dims...>,Dense<T,DenseType::nd,default_order>>;

// define intermediate impl aliases as doxygen doesn't seem to like 'requires'
template<class T,std::size_t... Dims> requires ( sizeof...(Dims) == 0 || sizeof...(Dims) == 1)
using VectorImpl = std::conditional_t<sizeof...(Dims),DenseFixed<T,default_order,Dims...>,Dense<T,DenseType::vec,default_order>>;

template<class T,std::size_t... Dims> requires ( sizeof...(Dims) == 0 || sizeof...(Dims) == 2)
using MatrixImpl = std::conditional_t<sizeof...(Dims),DenseFixed<T,default_order,Dims...>,Dense<T,DenseType::mat,default_order>>;

/*! \brief The alias used to access either dynamically-sized 1D `Dense` objects or fixed-size 1D `Dense` objects.
 *  \tparam T The value_type contained by the `Vector`
 *  \tparam Size (Optional) Unsigned int representing fixed size. Omitting this results in a dynamically sized `Vector` 
 *
 *  Similar to the `Array` alias, though restricted to 1D arrays. Due for deprecation.
 */
template<class T,std::size_t... Dims>
using Vector = VectorImpl<T,Dims...>;

/*!  \brief The alias used to access either dynamically-sized 2D `Dense` objects or fixed-size 2D `Dense` objects.
 *  \tparam T the value_type contained by the `Matrix`
 *  \tparam Rows (Optional) Unsigned integer giving the number of rows. Omitting this results in a dynamically sized array. Must also provide `Cols`.
 *  \tparam Cols (Optional) Unsigned integer giving the number of columns. Omitting this results in a dynamically sized array.
 * 
 *  Similar to the Array alias, though restricted to 2D arrays. Due for deprecation.
 */
template<class T,std::size_t... Dims>
using Matrix = MatrixImpl<T,Dims...>;

// ==============================================
// is_dense

//! Type-trait which determines whether a given type is an ultramat `Dense` object.
template<class T>
struct is_dense {
    //! Evaluates to `true` if the template argument `T` is a `Dense` Object. Evaluates to `false` otherwise.
    static constexpr bool value = false; 
};

template<class T, DenseType Type, DenseOrder Order> struct is_dense<Dense<T,Type,Order>> { static constexpr bool value = true; };
template<class T, DenseOrder Order, std::size_t... dims> struct is_dense<DenseFixed<T,Order,dims...>> { static constexpr bool value = true; };
template<class T, ReadWrite RW> struct is_dense<DenseView<T,RW>> { static constexpr bool value = true; };

// ==============================================
// shapelike

//! Defines a sized range of integers, and excludes ultramat `Dense` objects, e.g. `std::vector<std::size_t>`
/*! For the sake of compatibility, the `shapelike` concept applies to signed integer ranges as well as unsigned ranges. */
template<class T> concept shapelike = std::ranges::sized_range<T> && std::integral<typename T::value_type> && !is_dense<T>::value;

// ==============================================
// common_order

template<class... Ts> struct CommonOrderImpl;

template<class T1, class T2, class... Ts>
struct CommonOrderImpl<T1,T2,Ts...> {
    static constexpr DenseOrder Order = std::remove_cvref_t<T1>::order() == CommonOrderImpl<T2,Ts...>::order() ? std::remove_cvref_t<T1>::order() : DenseOrder::mixed;
    static constexpr DenseOrder order() { return Order; }
};

template<class T1>
struct CommonOrderImpl<T1> {
    static constexpr DenseOrder Order = std::remove_cvref_t<T1>::order();
    static constexpr DenseOrder order() { return Order; }
};

/*! \brief A type-trait-like struct that returns the common order of a collection of `Dense` objects.
 * 
 *  Given a collection of `Dense` objects, this struct provides a way to compute their 'common order' 
 *  at compile time, via `common_order<Dense1,Dense2,...>::value`. If all objects are
 *  row-major, then this returns `DenseOrder::row_major`. Similarly, if all objects are column-major,
 *  then this returns `DenseOrder::col_major`. If there are a mix of orderings in the template argument
 *  list, it returns `DenseOrder::mixed`.
 */
template<class... Ts>
struct common_order {
    //! Gives the common order of the template arguments given.
    static constexpr DenseOrder Order = CommonOrderImpl<Ts...>::order(); 
    
    //! constexpr function used to access `Order`
    static constexpr DenseOrder order(){ 
        return Order;
    }
    
    //! Alias for `Order`
    static constexpr DenseOrder value = Order;
};

// ==============================================
// Slice

//! A tool for generating views of `Dense` objects.
/*! `Slice` is an 'aggregate'/'pod' type, so it has a relatively intuitive interface by default.
 *  For example, a `Slice` that takes all but the last element may be created via `Slice{0,-1}`.
 *  The same in reverse is written `Slice{0,-1,-1}`. In Python, it is possible to signify
 *  that a slice should give all elements with the syntax `x[:]`. This is achieved with `Slice`
 *  using the `Slice::all` static member: `Slice{Slice::all,Slice::all}`. Note that both the
 *  beginning and the end must be specified. The step size defaults to 1.
 */
struct Slice { 
    //! Beginning of the slice, inclusive. `Slice::all` has the same effect as 0. Negative numbers count backwards from the end.
    std::ptrdiff_t start;
    /*! \brief End of the slice, exclusive. `Slice::all` indicates that all elements from start onwards are included. 
     * Negative numbers count backwards from the end.
     */
    std::ptrdiff_t end;
    //! Step size of the slice. Negative steps mean the slice goes from `end-1` to `start`.
    std::ptrdiff_t step=1;
    /*! \brief Static member. When used for `start`, all elements up to `end` are included. 
     *  When used for end, all elements from `start` onwards are used. When used for both `start` and `end`, all elements are included.
     */
    static constexpr std::ptrdiff_t all = std::numeric_limits<std::ptrdiff_t>::max();
};

// ==============================================
// DenseStriper

/*! \brief A utility class used by `Dense` objects to generate 1D 'stripes' for iteration purposes.
 *
 *  Striped iteration is a core concept in the ultramat library. The C++ standard library is
 *  heavily dependent on iterators, but as iteration is inherently a 1D operation, it does not
 *  always apply well to N-dimensional objects. An exception is when operations occur between
 *  `Dense` objects of the same shapes and row/column-major ordering, provided the operation is 'simple',
 *  such as element-wise arithmetic.
 *
 *  `DenseStriper` keeps track of the shape of a `Dense` object or expression, a coordinate, and a striping
 *  dimension. If row-major ordered, incrementing a `DenseStriper` will update the coordinate in 
 *  the last dimension until it equals the shape in the last dimension. It then increments the coordinate in the
 *  second-to-last dimension, resets the last dimension to zero, and so on. Similar behaviour can be expected if
 *  a `DenseStriper` is column-major ordered, although in this case the first dimension increments first, then
 *  the second, etc. Note that the coordinate in the striping dimension is always zero, and is skipped over when
 *  finding the next dimension to increment.
 *
 *  A `DenseStriper` behaves like a random access iterator, so also provides decrement, in-place addition and
 *  subtraction, and distance calculations.
 *
 *  `DenseStriper` is a core component in the following features:
 *  - Automatic broadcasting: If a `Dense` object is asked to provide a stripe, but is given a `DenseStriper`
 *    with a broadcasted shape, it may return a stripe with zero stride.
 *  - Non-contiguous iteration: Iterating directly over a non-contiguous array is a costly operation (see
 *    `DenseViewIterator` for proof). Striped iteration reduces the amount of checks that must be performed,
 *    as a non-contiguous object may instead be viewed as a series of contiguous (or strided) 1D chunks.
 *  - Mixed row/column-major ordered operations: When providing a coordinate from which to generate a stripe,
 *    it doesn't matter if the target is row-major or column-major ordered (although striping over a row-major
 *    object in a column-major manner, or vice versa, will likely be less efficient).
 *  - OpenMP parallelisation: Striped iteration was designed with OpenMP-style parallelism in mind. In this model,
 *    each thread generates a single stripe at a time, and iterates over it.
 */
class DenseStriper {

    //! The striping dimension. This index in this dimension is always zero.
    std::size_t _dim;

    //! Determines whether the index is incremented starting from the last dimension (row-major) or first dimension (col-major)
    DenseOrder _order;

    //! The current index, or coordinate. The integer in each dimension must be less than the shape.
    std::vector<std::ptrdiff_t> _idx;

    //! The scalar index tracks the distance from the beginning, and is incremented by 1 each time the `DenseStriper` is incremented.
    std::size_t _scalar_idx;

    //! The maximum coordinate in each dimension. This should be the shape of the target object, which may be broadcasted.
    std::vector<std::size_t> _shape;

    //! Sets the index from the scalar index. Used to handle random access jumps.
    void _set_from_scalar_index() {
        std::size_t scalar_index = _scalar_idx;
        if( _order == DenseOrder::col_major) {
            for( std::size_t ii=0; ii != dims(); ++ii) {
                if( ii == stripe_dim() ) continue;
                index(ii) = scalar_index % shape(ii);
                scalar_index /= shape(ii);
            }
            index(dims()) = (scalar_index > 0);
        } else {
            for( std::size_t ii=dims(); ii != 0; --ii) {
                if( ii == stripe_dim() +1 ) continue;
                index(ii) = scalar_index % shape(ii-1);
                scalar_index /= shape(ii-1);
            }
            index(0) = (scalar_index > 0);
        }
    }

    public:

    //! Default constructor disabled.
    DenseStriper() = delete;

    //! Copy constructor set to default.
    DenseStriper( const DenseStriper& ) = default;

    //! Move constructor set to default.
    DenseStriper( DenseStriper&& ) = default;

    //! Copy assignment set to default.
    DenseStriper& operator=( const DenseStriper& ) = default;

    //! Move assignment set to default.
    DenseStriper& operator=( DenseStriper&& ) = default;

    /*! /brief Construct a new `DenseStriper`.
     *  \var dim The striping dimension
     *  \var order Sets the `DenseStriper` to row-major or column-major mode
     *  \var shape The shape of the target object (may be broadcasted)
     *  \var end When `false`, initialise the index to all zeros. When `true`, set to 1 past the last valid coordinate.
     */
    template<shapelike Shape>
    DenseStriper( std::size_t dim, DenseOrder order, const Shape& shape, bool end=false) :
        _dim(dim),
        _order(order),
        _idx(shape.size()+1,0),
        _scalar_idx(0),
        _shape(shape.size())
    {
        std::ranges::copy( shape, _shape.begin());
        if( end ){
            _idx[ _order==DenseOrder::col_major ? dims() : 0 ] = 1;
            _scalar_idx = num_stripes();
        }
    }

    //! Returns the number of stripes that can be generated
    std::size_t num_stripes() const {
        return std::accumulate( _shape.begin(), _shape.end(), 1, std::multiplies<std::size_t>{})/_shape[_dim];
    }

    //! Returns the striping dimension
    std::size_t stripe_dim() const {
        return _dim;
    }

    //! Returns the length of generated stripes
    std::size_t stripe_size() const {
        return _shape[_dim];
    }

    //! Returns the number of dimensions
    std::size_t dims() const {
        return _shape.size();
    }

    //! Returns the row/column-major ordering of the `DenseStriper`
    DenseOrder order() const {
        return _order;
    }

    //! Returns a copy of _shape
    std::vector<std::size_t> shape() const {
        return _shape;
    }

    //! Returns the shape in the given dimension, by value
    std::size_t shape( std::size_t ii) const {
        return _shape[ii];
    }

    //! Returns the shape in the given dimension, by reference
    std::size_t& shape( std::size_t ii) {
        return _shape[ii];
    }

    //! Returns the current index/coordinate, by const reference
    const std::vector<std::ptrdiff_t>& index() const {
        return _idx;
    }

    //! Returns the current index/coordinate in the given dimension, by value
    std::ptrdiff_t index( std::size_t ii) const {
        return _idx[ii];
    }

    //! Returns the current index/coordinate in the given dimension, by reference
    std::ptrdiff_t& index( std::size_t ii) {
        return _idx[ii];
    }

    //! Increment the `DenseStriper` by 1
    /*! If row major, increment in the last dimension first. Once reaching the end of this dimension, set to zero and increment the second-last 
     *  dimension. If column major, instead start with the first dimension and carry on to the second. Skip the striping dimension in either case.
     */
    DenseStriper& operator++() {
        if( _order == DenseOrder::col_major ){
            for( std::size_t ii=0; ii <= dims(); ++ii) {
                if( ii == stripe_dim()  ) continue;
                ++index(ii);
                if( ii < dims() && index(ii) == shape(ii) ) {
                    index(ii) = 0;
                } else {
                    break;
                }
            }
        } else {
            for( std::ptrdiff_t ii=dims(); ii >= 0; --ii) {
                if( ii == stripe_dim()+1  ) continue;
                ++index(ii);
                if( ii > 0 && index(ii) == shape(ii-1) ) {
                    index(ii) = 0;
                } else {
                    break;
                }
            }
        }
        ++_scalar_idx;
        return *this;
    }

    //! Increment the `DenseStriper` by 1
    /*! If row major, decrement in the last dimension first. Once reaching -1 in this dimension, set to the max value in this dimension 
     *  and decrement the second-last dimension. If column major, instead start with the first dimension and carry on to the second. Skip the striping
     *  dimension in either case.
     */
    DenseStriper& operator--() {
        if( _order == DenseOrder::col_major ){
            for( std::size_t ii=0; ii <= dims(); ++ii) {
                if( ii == stripe_dim()  ) continue;
                --index(ii);
                if( ii < dims() && index(ii) == -1 ) {
                    index(ii) = shape(ii)-1;
                } else {
                    break;
                }
            }
        } else {
            for( std::ptrdiff_t ii=dims(); ii >= 0; --ii) {
                if( ii == stripe_dim()+1  ) continue;
                --index(ii);
                if( ii>0 && index(ii) == -1 ) {
                    index(ii) = shape(ii-1)-1;
                } else {
                    break;
                }
            }
        }
        --_scalar_idx;
        return *this;
    }


    //! Increment the `DenseStriper` by a given amount
    DenseStriper& operator+=( std::ptrdiff_t diff) {
        // Determine current scalar index, add diff, set index accordingly
        _scalar_idx += diff;
        _set_from_scalar_index();
        return *this;
    }

    //! Decrement the `DenseStriper` by a given amount
    DenseStriper& operator-=( std::ptrdiff_t diff) {
        // Determine current scalar index, subtract diff, set index accordingly
        _scalar_idx -= diff;
        _set_from_scalar_index();
        return *this;
    }

    //! Create a new `DenseStriper`, incremented by a given amount
    DenseStriper operator+( std::ptrdiff_t diff) const {
        DenseStriper copy(*this);
        copy += diff;
        return copy;
    }

    //! Create a new `DenseStriper`, decremented by a given amount
    DenseStriper operator-( std::ptrdiff_t diff) const {
        DenseStriper copy(*this);
        copy -= diff;
        return copy;
    }

    //! Get the 'distance' between two `DenseStriper`'s.
    /*! Makes use of scalar index for speed.
     *  Both must have the same shape, striping dimension, and order. Otherwise, results are undefined.
     */
    std::ptrdiff_t operator-( const DenseStriper& other) const {
        return static_cast<std::ptrdiff_t>(_scalar_idx) - static_cast<std::ptrdiff_t>(other._scalar_idx);
    }

    //! Test whether two `DenseStriper`'s have the same coordinate.
    /*! Makes use of scalar index for speed. 
     *  Both must have the same shape, striping dimension, and order. Otherwise, results are undefined.
     */
    bool operator==( const DenseStriper& other) const {
        for( std::size_t ii=0; ii <= dims(); ++ii) {
            if( index(ii) != other.index(ii) ) return false;
        }
        return true;
    }

    //! Gives ordering of two `DenseStriper`'s.
    /*! Makes use of scalar index for speed. 
     *  Both must have the same shape, striping dimension, and order. Otherwise, results are undefined.
     */
    auto operator<=>( const DenseStriper& other) const {
        return _scalar_idx <=> other._scalar_idx;
    }
};

// ==============================================
// get_broadcast_shape

/*! \brief Given a selection of shapes, returns the result of broadcasting all of them together.
 *  \tparam order (Required) Determines whether new broadcasting dimensions should be appended (col-major) or prepended (row-major)
 *  \tparam Shapes (Automatically deduced) Types of the provided shapes
 *  \var shapes A list of shapelike objects
 */
template<DenseOrder order, shapelike... Shapes> 
std::vector<std::size_t> get_broadcast_shape( const Shapes&... shapes) {
    // If row_major, prepend dims
    // If col_major, apend dims
    std::size_t max_dims = std::max({shapes.size()...});
    std::vector<std::size_t> bcast_shape(max_dims,1);
    if( order == DenseOrder::col_major ){
        for( std::size_t ii=0; ii<max_dims; ++ii){
            bcast_shape[ii] = std::max({ (ii < shapes.size() ? shapes[ii] : 0) ...});
            // throw exception if any of the shapes included have a dimension which is neither bcast_shape[ii] nor 1.
            auto errors = std::array<bool,sizeof...(Shapes)>{
                ( ii < shapes.size() ? ( shapes[ii] == 1 || shapes[ii] == bcast_shape[ii] ? false : true) : false)...
            };
            if( std::ranges::any_of(errors,[](bool b){return b;}) ) throw std::runtime_error("Ultramat: Cannot broadcast shapes");   
        }
    } else {
        for( std::size_t ii=0; ii<max_dims; ++ii){
            bcast_shape[max_dims-ii-1] = std::max({ (ii < shapes.size() ? shapes[shapes.size()-ii-1] : 0) ...});
            // throw exception if any of the shapes included have a dimension which is neither bcast_shape[ii] nor 1.
            auto errors = std::array<bool,sizeof...(Shapes)>{
                ( ii < shapes.size() ? ( shapes[shapes.size()-ii-1] == 1 || shapes[shapes.size()-ii-1] == bcast_shape[max_dims-ii-1] ? false : true) : false)...
            };
            if( std::ranges::any_of(errors,[](bool b){return b;}) ) throw std::runtime_error("Ultramat: Cannot broadcast shapes");   
        }
    }
    return bcast_shape;
}

// ==========================
// Fixed-Size Dense Utils
// Lots of compile-time template nonsense ahead.

// variadic_product

template<std::size_t... Ints> struct variadic_product;

template<> struct variadic_product<> { static constexpr std::size_t value = 1; };

template<std::size_t Int,std::size_t... Ints>
struct variadic_product<Int,Ints...> {
    static constexpr std::size_t value = Int*variadic_product<Ints...>::value;
};

// index_sequence_to_array

template<std::size_t... Ints>
constexpr auto index_sequence_to_array( std::index_sequence<Ints...> ) noexcept {
    return std::array<std::size_t,sizeof...(Ints)>{{Ints...}};
}

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

template<std::size_t... Ints>
struct reverse_index_sequence<std::index_sequence<Ints...>> {
    using type = reverse_index_sequence_impl<std::index_sequence<Ints...>,std::index_sequence<>>::type;
};

// variadic stride

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

} // namespace
#endif
