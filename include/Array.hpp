#ifndef __ULTRA_ARRAY_HPP
#define __ULTRA_ARRAY_HPP

/*! \file Array.hpp
 *  \brief Defines the preferred alias for `Dense` objects.
 */

#include "Dense/Dense.hpp"
#include "Dense/DenseFixed.hpp"

namespace ultra {

/*! \brief The alias used to access either dynamically-sized Dense objects or fixed-size `Dense` objects.
 *  \tparam T the value_type contained by the `Array`
 *  \tparam Dims (Optional) A list of unsigned integers giving the dimensions of the Array. Omitting this results in a dynamically sized array.
 *
 *  `Array`'s are the primary objects in the ultramat library, but in actuality `Array` is an alias.
 *  If `Array` is supplied with only a single template argument, such as `Array<int>`
 *  or `Array<double>`, it is an alias for an N-dimensional dynamically-sized `Dense` object. If it is provided
 *  with a list of dimension sizes after the value type, such as `Array<float,3,3>`, it instead aliases a
 *  fixed-size \f$3\times3\f$ `DenseFixed` object. Both objects may be used interchangeably in ultramat expressions,
 *  though the latter may not be resized in any way.
 */
template<class T,std::size_t... Dims>
using Array = std::conditional_t<
    sizeof...(Dims),
    DenseFixed<T,default_order,Dims...>,
    Dense<T,default_order>
>;

} // namespace ultra
#endif
