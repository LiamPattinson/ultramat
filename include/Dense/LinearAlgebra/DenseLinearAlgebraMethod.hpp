#ifndef __ULTRA_DENSE_LINEAR_ALGEBRA_METHOD_HPP
#define __ULTRA_DENSE_LINEAR_ALGEBRA_METHOD_HPP

/*! \file DenseLinearAlgebraMethod.hpp
 *  \brief Defines generic linear algebra method that applies a routine over a stack of matrices/vectors.
 */

#include "ultramat/include/Dense/DenseView.hpp"

namespace ultra {

// ==============================================
// DenseLinearAlgebraMethods

// TODO doc page on linear algebra, include:
// - stacks
// - broadcasting rules

/*! \brief An expression used to represent generic linear algebra operations.
 *  \tparam F A functor type defining the function applied over each \ref dense_object.
 *  \tparam Args The \ref dense_object%s operated over.
 *
 * Applies a given function, defined by the functor class `F`, element-wise over a stack of vectors/matrices.
 * The number of types represented by `Args...` must match the number of arguments taken by `F::operator()`. 
 *
 * The required \ref dense_shape of each input is determined by the choice of `F`, following specific broadcasting rules.
 * The output shape is also determined by `F`.
 */

} // namespace ultra
#endif
