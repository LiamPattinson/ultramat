#ifndef __ULTRA_DENSE_STRIPE_HPP
#define __ULTRA_DENSE_STRIPE_HPP

#include "DenseUtils.hpp"

namespace ultra {

// ===============================================
// DenseStripe
//
// A 1D view, used extensively within the library to iterate over higher-dimensional arrays.

template<class T, ReadWrite rw>
class DenseStripe {

    static constexpr ReadWrite other_rw = ( rw == ReadWrite::writeable ? ReadWrite::read_only : ReadWrite::writeable);
    friend DenseStripe<T,other_rw>;

public:

    using value_type = typename T::value_type;
    using difference_type = std::ptrdiff_t;
    using pointer = std::conditional_t<rw==ReadWrite::writeable,value_type*, const value_type*>;
    using reference = std::conditional_t<rw==ReadWrite::writeable,value_type&,const value_type&>;

protected:

    pointer        _ptr;
    std::ptrdiff_t _stride;
    std::size_t    _size;

public:

    // ===============================================
    // Constructors

    DenseStripe() = delete;
    DenseStripe( const DenseStripe& ) = default;
    DenseStripe( DenseStripe&& ) = default;
    DenseStripe& operator=( const DenseStripe& ) = default;
    DenseStripe& operator=( DenseStripe&& ) = default;

    DenseStripe( pointer ptr, std::ptrdiff_t stride, std::size_t size) :
        _ptr(ptr),
        _stride(stride),
        _size(size)
    {}

    // Copy/assign from other read/write type
    DenseStripe( const DenseStripe<T,other_rw>& other) requires ( rw == ReadWrite::read_only ) :
        _ptr(other._ptr),
        _stride(other._stride),
        _size(other._size)
    {}

    DenseStripe& operator=( const DenseStripe<T,other_rw>& other) requires ( rw == ReadWrite::read_only ) {
        _ptr = other._ptr;
        _stride = other._stride;
        _size = other._size;
    }

    // ===============================================
    // Iteration

    class Iterator {

        pointer        _ptr;
        std::ptrdiff_t _stride;

        public:

        Iterator() = delete;
        Iterator( const Iterator& ) = default;
        Iterator( Iterator&& ) = default;
        Iterator& operator=( const Iterator& ) = default;
        Iterator& operator=( Iterator&& ) = default;

        Iterator( pointer ptr, std::ptrdiff_t stride) : _ptr(ptr), _stride(stride) {}

        // ===============================================
        // Standard iterator interface

        // Dereference
        reference operator*() const { return *_ptr; }
        
        // Increment/decrement
        Iterator& operator++() {
            _ptr += _stride;
            return *this;
        }

        Iterator& operator--(){
            _ptr -= _stride;
            return *this;
        }

        Iterator operator++(int) const {
            return ++Iterator(*this);
        }

        Iterator operator--(int) const {
            return --Iterator(*this);
        }

        // Random-access
        Iterator& operator+=( difference_type diff){
            _ptr += diff*_stride;
            return *this;
        }

        Iterator& operator-=( difference_type diff){
            _ptr -= diff*_stride;
            return *this;
        }
        
        Iterator operator+( difference_type diff) const {
           Iterator result(*this);
           result += diff;
           return result;
        }

        Iterator operator-( difference_type diff) const {
           Iterator result(*this);
           result -= diff;
           return result;
        }

        // Distance
        std::ptrdiff_t operator-( const Iterator& it_r) const {
            // Assumes both pointers are looking at the same thing. If not, the results are undefined.
            return (_ptr - it_r._ptr)/_stride;
        }

        // Boolean comparisons
        bool operator==( const Iterator& it_r) const {
            return _ptr == it_r._ptr;
        }

        auto operator<=>( const Iterator& it_r) const {
            return (*this - it_r) <=> 0;
        }
    };

    Iterator begin() { return Iterator(_ptr,_stride); }
    Iterator begin() const { return Iterator(_ptr,_stride); }
    Iterator end() { return Iterator(_ptr+_size*_stride,_stride); }
    Iterator end() const { return Iterator(_ptr+_size*_stride,_stride); }
};

// ==============================================
// DenseStripeIndex

/*! \brief A utility class used by \ref DenseObject%s to generate 1D \link striped_iteration stripes \endlink for iteration purposes.
 *
 *  #ultra::DenseStripeIndex keeps track of the \ref dense_shape of a \ref DenseObject or expression, a coordinate, and a striping
 *  dimension. If row-major ordered, incrementing a will update the coordinate in the last dimension until it equals the shape in the 
 *  last dimension. It then increments the coordinate in the second-to-last dimension, resets the last dimension to zero, and so on. 
 *  Similar behaviour can be expected for column-major ordered objects, although in this case the first dimension increments first, then
 *  the second, etc. Note that the coordinate in the striping dimension is always zero, and is skipped over when
 *  finding the next dimension to increment.
 *
 *  #ultra::DenseStripeIndex behaves like a random access iterator, so also provides decrement, in-place addition and
 *  subtraction, and distance calculations.
 */
class DenseStripeIndex {

    //! The striping dimension. The index in this dimension is always zero.
    std::size_t _dim;

    /*! \brief Determines whether the index is incremented starting from the last dimension (row-major) or first dimension (col-major)
     *  
     *  Note that column-major ordered arrays may be striped in a row-major manner, and vice versa -- it's just slower that way.
     */
    DenseOrder _order;

    //! The current index, or coordinate. `_idx[i] <= _shape[i]`.
    std::vector<std::ptrdiff_t> _idx;

    //! The scalar index tracks the distance from the beginning, and is incremented by 1 each time the `DenseStripeIndex` is incremented.
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

    DenseStripeIndex() = delete;                                  //! Default constructor disabled.
    DenseStripeIndex( const DenseStripeIndex& ) = default;            //! Copy constructor set to default.
    DenseStripeIndex( DenseStripeIndex&& ) = default;                 //! Move constructor set to default.
    DenseStripeIndex& operator=( const DenseStripeIndex& ) = default; //! Copy assignment set to default.
    DenseStripeIndex& operator=( DenseStripeIndex&& ) = default;      //! Move assignment set to default.

    /*! /brief Construct a new `DenseStripeIndex`.
     *  \var dim The striping dimension
     *  \var order Sets the `DenseStripeIndex` to row-major or column-major mode
     *  \var shape The shape of the target object (may be broadcasted)
     *  \var end When `false`, initialise the index to all zeros. When `true`, set to 1 past the last valid coordinate.
     */
    template<shapelike Shape>
    DenseStripeIndex( std::size_t dim, DenseOrder order, const Shape& shape, bool end=false) :
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

    //! Returns the \link dense_order row/column-major ordering \endlink of the `DenseStripeIndex`
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

    //! Increment the `DenseStripeIndex` by 1
    /*! If row major, increment in the last dimension first. Once reaching the end of this dimension, set to zero and increment the second-last 
     *  dimension. If column major, instead start with the first dimension and carry on to the second. Skip the striping dimension in either case.
     */
    DenseStripeIndex& operator++() {
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

    //! Decrement the `DenseStripeIndex` by 1
    /*! If row major, decrement in the last dimension first. Once reaching -1 in this dimension, set to the max value in this dimension 
     *  and decrement the second-last dimension. If column major, instead start with the first dimension and carry on to the second. Skip the striping
     *  dimension in either case.
     */
    DenseStripeIndex& operator--() {
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


    //! Increment the `DenseStripeIndex` by a given amount
    DenseStripeIndex& operator+=( std::ptrdiff_t diff) {
        // Determine current scalar index, add diff, set index accordingly
        _scalar_idx += diff;
        _set_from_scalar_index();
        return *this;
    }

    //! Decrement the `DenseStripeIndex` by a given amount
    DenseStripeIndex& operator-=( std::ptrdiff_t diff) {
        // Determine current scalar index, subtract diff, set index accordingly
        _scalar_idx -= diff;
        _set_from_scalar_index();
        return *this;
    }

    //! Create a new `DenseStripeIndex`, incremented by a given amount
    DenseStripeIndex operator+( std::ptrdiff_t diff) const {
        DenseStripeIndex copy(*this);
        copy += diff;
        return copy;
    }

    //! Create a new `DenseStripeIndex`, decremented by a given amount
    DenseStripeIndex operator-( std::ptrdiff_t diff) const {
        DenseStripeIndex copy(*this);
        copy -= diff;
        return copy;
    }

    //! Get the 'distance' between two `DenseStripeIndex`'s.
    /*! Makes use of scalar index for speed.
     *  Both must have the same shape, striping dimension, and order. Otherwise, results are undefined.
     */
    std::ptrdiff_t operator-( const DenseStripeIndex& other) const {
        return static_cast<std::ptrdiff_t>(_scalar_idx) - static_cast<std::ptrdiff_t>(other._scalar_idx);
    }

    //! Test whether two `DenseStripeIndex`'s have the same coordinate.
    /*! Makes use of scalar index for speed. 
     *  Both must have the same shape, striping dimension, and order. Otherwise, results are undefined.
     */
    bool operator==( const DenseStripeIndex& other) const {
        for( std::size_t ii=0; ii <= dims(); ++ii) {
            if( index(ii) != other.index(ii) ) return false;
        }
        return true;
    }

    //! Gives ordering of two `DenseStripeIndex`'s.
    /*! Makes use of scalar index for speed. 
     *  Both must have the same shape, striping dimension, and order. Otherwise, results are undefined.
     */
    auto operator<=>( const DenseStripeIndex& other) const {
        return _scalar_idx <=> other._scalar_idx;
    }
};

} // namespace ultra
#endif
