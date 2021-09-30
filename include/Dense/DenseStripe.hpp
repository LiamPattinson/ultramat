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

} // namespace ultra
#endif
