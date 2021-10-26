#ifndef __ULTRA_DENSE_LINEAR_ALGEBRA_GENERATORS_HPP
#define __ULTRA_DENSE_LINEAR_ALGEBRA_GENERATORS_HPP

/*! \file DenseLinearAlgebraGenerators.hpp
 *  \brief Defines functions for generating matrices
 */

#include "ultramat/include/Array.hpp"
#include "ultramat/include/Dense/Math/DenseArithmetic.hpp"
#include "ultramat/include/Dense/Math/DenseGenerators.hpp"

namespace ultra {

//! Produces NxM array with ones down the kth diagonal
decltype(auto) eye( std::size_t rows, std::size_t cols, std::ptrdiff_t k=0){
    return (reshape(arange<std::ptrdiff_t>(0,rows,1),rows,1) == arange<std::ptrdiff_t>(-k,static_cast<std::ptrdiff_t>(cols)-k,1));
}

//! Produces NxN array with ones down the main diagonal
decltype(auto) identity( std::size_t N){
    return eye(N,N,0);
}

} // namespace
#endif
