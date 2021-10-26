#ifndef __ULTRA_DENSE_UTILS_HPP
#define __ULTRA_DENSE_UTILS_HPP

/*! \file DenseUtils.hpp
 *  \brief Defines any utility functions/structs deemed useful for \ref DenseObject%s.
 *
 *  This file contains utilities widely used in the definitions and internal workings
 *  of \ref DenseObject%s. Most features are not expected to be used by the end-user.
 */

#include "ultramat/include/Utils/Utils.hpp"

namespace ultra {

// ==============================================
// DenseOrder

/*!  \brief An Enum Class used to indicate whether a Dense object is row-major or column-major ordered.
 * 
 *  `DenseOrder` is commonly used as a template argument to \ref DenseObject%s, and is used to indicate
 *  if the object is row-major ordered (C-style) or column-major ordered (Fortran-style).
 *  If an object is row-major ordered, then the last dimension is the fastest
 *  incrementing, while if an object is column-major ordered, the first dimension is the fastest
 *  incrementing. The option `DenseOrder::mixed` exists to indicate that a given object is neither
 *  row-major nor column-major. For example, an expression in which a row-major object is
 *  added to a column-major object is regarded as having mixed ordering.
 *
 *  For further information, see a more detailed explanation of \link dense_order row- and column-major ordering \endlink.
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

/*! \brief An Enum Class used to indicate whether an object is read-only or if it may be written to.
 * 
 * Used internally by objects such as #ultra::DenseView and #ultra::DenseStripe to indicate whether they should reference
 * their data using a regular pointer (writeable) or a const pointer (read_only).
 */
enum class ReadWrite {
    writeable, //!< Object may be freely read from or written to
    read_only  //!< Object may only be read from
};

// ==============================================
// Pre-declare Dense things
// Needed to define things like is_dense


template<class, DenseOrder> class Dense;
template<class, DenseOrder, std::size_t...> class DenseFixed;
template<class, ReadWrite = ReadWrite::writeable> class DenseView;
template<class, ReadWrite = ReadWrite::writeable> class DenseStripe;

// ==============================================
// is_dense

//! Type-trait which determines whether a given type is a \ref DenseObject.
template<class T>
struct is_dense {
    static constexpr bool value = false; //!< Evaluates to `true` if the template argument `T` is a \ref DenseObject. Evaluates to `false` otherwise.
};

// Specialisation for Dense
template<class T, DenseOrder Order> struct is_dense<Dense<T,Order>> { static constexpr bool value = true; };
// Specialisation for DenseFixed
template<class T, DenseOrder Order, std::size_t... dims> struct is_dense<DenseFixed<T,Order,dims...>> { static constexpr bool value = true; };
// Specialisation for DenseView
template<class T, ReadWrite RW> struct is_dense<DenseView<T,RW>> { static constexpr bool value = true; };

// ==============================================
// Slice

/*! \brief  A tool for generating #ultra::DenseView%s from \ref DenseObject%s.
 *
 * `Slice` is an 'aggregate'/'pod' type, so it has a relatively intuitive interface by default.
 *  For example, a `Slice` that takes all but the last element may be created via `Slice{0,-1}`.
 *  The same in reverse is written `Slice{0,-1,-1}`. In Python, it is possible to signify
 *  that a slice should give all elements with the syntax `x[:]`. This is achieved with `Slice`
 *  using the `Slice::all` static member: `Slice{Slice::all,Slice::all}`. Note that both the
 *  beginning and the end must be specified. The step size defaults to 1.
 */
struct Slice { 
    /*! \brief Beginning of the slice, inclusive.
     *  A value of 0 indicates that all elements up to `end` are included.
     *  Negative numbers count backwards from the end.
     */
    std::ptrdiff_t start=0;
    /*! \brief End of the slice, exclusive.
     *  A value of 0 indicates that all elements from `start` onwards are included. 
     *  Negative numbers count backwards from the end.
     */
    std::ptrdiff_t end=0;
    /*! \brief Step size. 
     *  Negative steps mean the slice goes from `end-1` to `start` inclusive.
     */
    std::ptrdiff_t step=1;
};

//! Generates a slice from a single integer
template<std::integral I>
constexpr Slice to_slice( I idx) {
    return Slice{idx,idx+1};
}

template<class T> requires std::is_same<T,Slice>::value
constexpr Slice to_slice( T t ) {
    return t;
}

// ==============================================
// shapelike

/*! \brief Defines a sized range of integers, and excludes Ultramat \ref DenseObject%s, e.g. `std::vector<std::size_t>`
 * 
 *  For the sake of compatibility, the `shapelike` concept applies to signed integer ranges as well as unsigned ranges.
 *  Note that supplying a shape with negative integers is unlikely to be caught, but is highly likely to break everything.
 */
template<class T> concept shapelike = std::ranges::sized_range<T> && 
                                      std::integral<typename T::value_type> && 
                                      !is_dense<T>::value &&
                                      !std::is_same<T,Slice>::value;

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

/*! \brief A type-trait-like struct that returns the common order of a collection of \ref DenseObject%s.
 * 
 *  Given a collection of \ref DenseObject%s, this struct provides a way to compute their 'common order' 
 *  at compile time, via `common_order<Dense1,Dense2,...>::value`. If all objects are
 *  row-major, then this returns `DenseOrder::row_major`. Similarly, if all objects are column-major,
 *  then this returns `DenseOrder::col_major`. If there are a mix of orderings in the template argument
 *  list, it returns `DenseOrder::mixed`.
 */
template<class... Ts>
struct common_order {
    //! Gives the common order of the template arguments given.
    static constexpr DenseOrder Order = CommonOrderImpl<Ts...>::order(); 
    
    //! constexpr function used to access #Order
    static constexpr DenseOrder order(){ 
        return Order;
    }
    
    //! Alias for #Order
    static constexpr DenseOrder value = Order;
};

// ==============================================
// get_broadcast_shape

/*! \brief Given a selection of \link dense_shape shapes \endlink, returns the result of broadcasting all of them together.
 *  \tparam order (Required) Determines whether new broadcasting dimensions should be appended (col-major) or prepended (row-major)
 *  \tparam Shapes (Automatically deduced) Types of the provided \link dense_shape shapes \endlink
 *  \var shapes A list of #ultra::shapelike objects
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
