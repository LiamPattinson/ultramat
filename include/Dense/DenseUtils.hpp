#ifndef __ULTRA_DENSE_UTILS_HPP
#define __ULTRA_DENSE_UTILS_HPP

/*! \file DenseUtils.hpp
 *  \brief Defines any utility functions/structs deemed useful for Dense objects.
 *
 *  This file contains utilities widely used in the definitions and internal workings
 *  of Dense objects. Most features are not expected to be used by the end-user.
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
 *  row-major nor column-major. For example, an expression in which a row-major `Dense` is
 *  added to a column-major `Dense` is regarded as having mixed ordering.
 */
enum class DenseOrder { 
    row_major, //!< Row-major ordering, last dimension increments fastest
    col_major, //!< Column-major ordering, first dimension increments fastest
    mixed      //!< Object is neither row-major nor column-major
};

//! The default row/column-major ordering used throughout ultramat. Normally set to row-major.
const DenseOrder default_order = DenseOrder::row_major;

// ==============================================
// ReadWrite

//! An Enum Class used to indicate whether an object is read-only or if it may be written to.
/*! Used internally by objects such as `DenseView` and `DenseStripe` to indicate whether they should reference
 *  their data using a regular pointer (writeable) or a const pointer (read_only).
 */
enum class ReadWrite {
    writeable, //!< Object may be freely read from or written to
    read_only  //!< Object may only be read from
};

// ==============================================
// Pre-declare Dense things
// Needed to define things like is_dense

/*! @name Pre-declarations
 * Defines the signature of `Dense` types.
 */
///@{

template<class, DenseOrder> class Dense;                             //!< Dynamically-sized N-d arrays.
template<class, DenseOrder, std::size_t...> class DenseFixed;        //!< Fixed-size N-d arrays.
template<class, ReadWrite = ReadWrite::writeable> class DenseView;   //!< Non-owning and generally non-contiguous reference to dense data.
template<class, ReadWrite = ReadWrite::writeable> class DenseStripe; //!< A 1D strided view over one dimension of dense data, used for iteration.

///@}

// ==============================================
// is_dense

//! Type-trait which determines whether a given type is an ultramat `Dense` object.
template<class T>
struct is_dense {
    static constexpr bool value = false; //!< Evaluates to `true` if the template argument `T` is a `Dense` Object. Evaluates to `false` otherwise.
};

// Specialisation for Dense
template<class T, DenseOrder Order> struct is_dense<Dense<T,Order>> { static constexpr bool value = true; };
// Specialisation for DenseFixed
template<class T, DenseOrder Order, std::size_t... dims> struct is_dense<DenseFixed<T,Order,dims...>> { static constexpr bool value = true; };
// Specialisation for DenseView
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

} // namespace
#endif
