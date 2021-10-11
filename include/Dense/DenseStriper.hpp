#ifndef __ULTRA_DENSE_STRIPER_HPP
#define __ULTRA_DENSE_STRIPER_HPP

/*! \file DenseStriper.hpp
 *  \brief Defines the #ultra::DenseStriper class, widely used for generic iteration over \ref DenseObject%s.
 */

#include "DenseUtils.hpp"

namespace ultra {

// ==============================================
// DenseStriper

/*! \brief A utility class used by \ref DenseObject%s to generate 1D 'stripes' for iteration purposes.
 *
 *  Striped iteration is a core concept in the Ultramat library. The C++ standard library is
 *  heavily dependent on iterators, but as iteration is inherently a 1D operation, it does not
 *  always apply well to N-dimensional objects. An exception is when operations occur between
 *  \ref DenseObject%s of the same \ref dense_shape and \link dense_order row/column-major ordering \endlink, 
 *  provided the operation is 'simple', such as element-wise arithmetic.
 *
 *  #ultra::DenseStriper keeps track of the \ref dense_shape of a \ref DenseObject or expression, a coordinate, and a striping
 *  dimension. If row-major ordered, incrementing a `DenseStriper` will update the coordinate in 
 *  the last dimension until it equals the shape in the last dimension. It then increments the coordinate in the
 *  second-to-last dimension, resets the last dimension to zero, and so on. Similar behaviour can be expected if
 *  a `DenseStriper` is column-major ordered, although in this case the first dimension increments first, then
 *  the second, etc. Note that the coordinate in the striping dimension is always zero, and is skipped over when
 *  finding the next dimension to increment.
 *
 *  #ultra::DenseStriper behaves like a random access iterator, so also provides decrement, in-place addition and
 *  subtraction, and distance calculations.
 *
 *  #ultra::DenseStriper is a core component in the following features:
 *  - Automatic broadcasting: If a \ref DenseObject is asked to provide a stripe, but is given a #ultra::DenseStriper
 *    with a broadcasted shape, it may return a stripe with zero stride.
 *  - \link dense_semicontiguous Semi-contiguous iteration \endlink: Iterating directly over a 
 *    \link dense_semicontiguous semi-contiguous \endlink array is a costly operation (see
 *    #ultra::DenseViewIterator for proof). Striped iteration reduces the amount of checks that must be performed during
 *    iteration, as N-dimensional arrays are instead decomposed into a series of 1D strided arrays.
 *  - Mixed \link dense_order row/column-major ordered \endlink operations: When providing a coordinate from which to generate a stripe,
 *    it doesn't matter if the target is row-major or column-major ordered (although striping over a row-major
 *    object in a column-major manner, or vice versa, will likely be less efficient).
 *  - OpenMP parallelisation: Striped iteration was designed with OpenMP-style parallelism in mind. In this model,
 *    each thread generates a single stripe at a time, and iterates over it.
 */
class DenseStriper {

    //! The striping dimension. The index in this dimension is always zero.
    std::size_t _dim;

    /*! \brief Determines whether the index is incremented starting from the last dimension (row-major) or first dimension (col-major)
     *  
     *  Note that column-major ordered arrays may be striped in a row-major manner, and vice versa -- it's just slower that way.
     */
    DenseOrder _order;

    //! The current index, or coordinate. `_idx[i] <= _shape[i]`.
    std::vector<std::ptrdiff_t> _idx;

    //! The scalar index tracks the distance from the beginning, and is incremented by 1 each time the `DenseStriper` is incremented.
    std::size_t _scalar_idx;

    //! The \ref dense_shape of the target object, which may be broadcasted.
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

    DenseStriper() = delete;                                  //! Default constructor disabled.
    DenseStriper( const DenseStriper& ) = default;            //! Copy constructor set to default.
    DenseStriper( DenseStriper&& ) = default;                 //! Move constructor set to default.
    DenseStriper& operator=( const DenseStriper& ) = default; //! Copy assignment set to default.
    DenseStriper& operator=( DenseStriper&& ) = default;      //! Move assignment set to default.

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

    //! Returns the \link dense_order row/column-major ordering \endlink of the `DenseStriper`
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

    //! Decrement the `DenseStriper` by 1
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

} // namespace ultra
#endif
