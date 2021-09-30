#ifndef __ULTRA_ARRAY_HPP
#define __ULTRA_ARRAY_HPP

#include "Dense/Dense.hpp"
#include "Dense/DenseFixed.hpp"

namespace ultra {

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


} // namespace ultra
#endif
