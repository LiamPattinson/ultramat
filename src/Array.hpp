#ifndef __ARRAY_HPP
#define __ARRAY_HPP

#include <string>
#include <algorithm>
#include <stdexcept>

// Array.hpp
// A generic multidimensional array, templated over the type it contains.
// Can be contiguous or non-contiguous, row-major or column-major.
// Dynamically allocated.

namespace ultra {

template<class T>
class Array {

    char         _status;
    std::size_t  _dims;
    std::size_t* _shape;
    std::size_t* _stride;
    T*           _data;
    
public:


    // ===============================================
    // Constructors

    // Default constructor and destructor
    Array();
    ~Array();

    // Copy and move
    Array( const Array& other);
    Array( Array&& other);

    // Build with given size
    // Templated over std::vector-like structures
    template<class V>
    Array( const V& shape, char rc_order=row_major);

    // Assignment and move assignment
    Array& operator=( const Array& other);
    // TODO Array& operator=( Array&& other);

    // ===============================================
    // Attributes

    std::size_t size() const;
    std::size_t shape( std::size_t dim) const;

    // ===============================================
    // Data access

    // Access via anything that looks like a std::vector
    template<class Coords>
    T operator()( const Coords& coords) const; 
    template<class Coords>
    T& operator()( const Coords& coords); 

    // Access via many ints.
    // No checks are performed to ensure the correct version has been called.
    // Works up to _dims==10. Anything above that should probably be considered a warcrime anyway.
    // (There's probably a neat way to do this with variadic templates, but I can't figure it out right now)
    T operator()( std::size_t i0) const;
    T operator()( std::size_t i0, std::size_t i1) const;
    T operator()( std::size_t i0, std::size_t i1, std::size_t i2) const;
    T operator()( std::size_t i0, std::size_t i1, std::size_t i2, std::size_t i3) const;
    T operator()( std::size_t i0, std::size_t i1, std::size_t i2, std::size_t i3, std::size_t i4) const;
    T operator()( std::size_t i0, std::size_t i1, std::size_t i2, std::size_t i3, std::size_t i4, std::size_t i5) const;
    T operator()( std::size_t i0, std::size_t i1, std::size_t i2, std::size_t i3, std::size_t i4, std::size_t i5
        std::size_t i6) const;
    T operator()( std::size_t i0, std::size_t i1, std::size_t i2, std::size_t i3, std::size_t i4, std::size_t i5
        std::size_t i6, std::size_t i7) const;
    T operator()( std::size_t i0, std::size_t i1, std::size_t i2, std::size_t i3, std::size_t i4, std::size_t i5
        std::size_t i6, std::size_t i7, std::size_t i8) const;
    T operator()( std::size_t i0, std::size_t i1, std::size_t i2, std::size_t i3, std::size_t i4, std::size_t i5
        std::size_t i6, std::size_t i7, std::size_t i8, std::size_t i9) const;
    T& operator()( std::size_t i0);
    T& operator()( std::size_t i0, std::size_t i1);
    T& operator()( std::size_t i0, std::size_t i1, std::size_t i2);
    T& operator()( std::size_t i0, std::size_t i1, std::size_t i2, std::size_t i3);
    T& operator()( std::size_t i0, std::size_t i1, std::size_t i2, std::size_t i3, std::size_t i4);
    T& operator()( std::size_t i0, std::size_t i1, std::size_t i2, std::size_t i3, std::size_t i4, std::size_t i5);
    T& operator()( std::size_t i0, std::size_t i1, std::size_t i2, std::size_t i3, std::size_t i4, std::size_t i5
        std::size_t i6);
    T& operator()( std::size_t i0, std::size_t i1, std::size_t i2, std::size_t i3, std::size_t i4, std::size_t i5
        std::size_t i6, std::size_t i7);
    T& operator()( std::size_t i0, std::size_t i1, std::size_t i2, std::size_t i3, std::size_t i4, std::size_t i5
        std::size_t i6, std::size_t i7, std::size_t i8);
    T& operator()( std::size_t i0, std::size_t i1, std::size_t i2, std::size_t i3, std::size_t i4, std::size_t i5
        std::size_t i6, std::size_t i7, std::size_t i8, std::size_t i9);

    // ===============================================
    // Iteration

    // Several types of iterator are available.
    // 
    // Contiguous:      fast_iterator
    // Semi-contiguous: stripe_iterator
    // Non-contiguous:  iterator
    // 
    // The non-contiguous 'iterator' will work in all cases, and will be reasonably fast,
    // though it is unlikely to make use of any vectorisation.
    // 
    // The semi-contiguous 'stripe_iterator' may be used with arrays that are contiguous in 
    // the last dimension (if row_major) or first dimension (if column_major). This
    // frequently occurs when taking views of an Array. Each stripe_iterator operates over
    // a limited strip of contiguous memory. A nested for-loop is required: one over the
    // total number of stripes, and one over contiguous memory within each stripe.
    //
    // The contiguous 'fast_iterator' should only be used with Arrays that are fully
    // contiguous i.e. not a view of an existing Array, and not sliced.
    //
    // Note that iterators will behave differently for row-major and column-major Arrays, 
    // and this may lead to confusing behaviour.

    // ===============================================
    // Status handling.

    static constexpr char uninitialised   = 0x00;
    static constexpr char initialised     = 0x01;
    static constexpr char own_data        = 0x02;
    static constexpr char contiguous      = 0x04;
    static constexpr char semi_contiguous = 0x04;
    static constexpr char row_major       = 0x10;
    static constexpr char col_major       = 0x20;
    static constexpr char c_contiguous    = contiguous | row_major;
    static constexpr char f_contiguous    = contiguous | col_major;

    inline char is_initialised(){ return _status & initialised;}
    inline char is_initialized(){ return _status & initialised;}
    inline char owns_data(){ return _status & own_data;}
    inline char is_contiguous(){ return _status & contiguous;}
    inline char is_semi_contiguous(){ return _status & semi_contiguous;}
    inline char is_c_contiguous(){ return _status & c_contiguous;}
    inline char is_f_contiguous(){ return _status & f_contiguous;}
    inline char is_row_major(){ return _status & row_major;}
    inline char is_col_major(){ return _status & col_major;}
    inline void set_status( char status){ _status = status;}
    inline void update_status( char status){ _status |= status;}
};

// Function definitions follow

template<class T>
Array<T>::Array() :
    _status(uninitialised),
    _dims(0),
    _shape(nullptr),
    _stride(nullptr),
    _data(nullptr)
{}

template<class T>
Array<T>::~Array() {
    if( is_initialised() ){
        delete[] _shape;
        delete[] _stride;
    }
    if( owns_data() ){
        delete[] _data;
    }
}

template<class T>
Array<T>::Array( const Array<T>& other) :
    _status(other._status),
    _dims(other._status)
{
    if( !other.is_initialised() ){
        _shape  = new std::size_t[_dims];
        _stride = new std::size_t[_dims];
        std::copy( other._shape, other._shape+_dims, _shape);
        std::copy( other._stride, other._stride+_dims, _stride);
        // If 'other' owns its own data, perform a full copy.
        // An array that owns its own data will also be contiguous, so can used std::copy
        // Otherwise, simply copy over the data pointer.
        if( other.owns_data() ){
            std::size_t n_elements = size();
            _data = new T[n_elements];
            std::copy( other._data, other._data+n_elements, _data);
        } else {
            _data = other._data;
        }
    } else {
        _shape = nullptr;
        _stride = nullptr;
        _data = nullptr;
    }
}

template<class T>
Array<T>::Array( Array<T>&& other ):
    _status(other._status),
    _dims(other._dims),
    _shape(other._shape),
    _stride(other._stride),
    _data(other._data)
{
    // Set status of other to uninitialised.
    // The new object now handles the lifetimes of _data, _shape, and _stride.
    other.set_status(uninitialised);
}

template<class T>
template<class V>
Array<T>::Array( const V& shape, char rc_order) :
    _status( initialised | own_data | contiguous | rc_order ),
    _dims(shape.size())
{
    // Ensure 'order' is either row_major, col_major, c_contiguous, or f_contiguous
    if( order!=row_major && order!=col_major && order!=c_contiguous && order!=f_contiguous ){
        std::string err = "UltraArray: Construction with explicit row/col major ordering.";
        err += " Argument 'rc_order' must be one of Array<T>::row_major, Array<T>::col_major, Array<T>::c_contiguous, or Array<T>::f_contiguous.";
        throw std::invalid_argument(err.c_str());
    }
    // if _dims==0, reset status to uninitialised
    if( _dims==0 ){
        _shape=nullptr;
        _stride=nullptr;
        _data=nullptr;
        set_status(uninitialised);
    } else {
        _shape = new std::size_t[_dims];
        _stride = new std::size_t[_dims];
        std::copy( shape.begin(), shape.end(), _shape);
        // Determine strides
        _stride[0] = 1;
        if( is_row_major() ){
            for( std::size_t ii=1; ii<_dims; ++ii){
                _stride[ii] = _shape[dims-ii] * _stride[ii-1];
            }
        } else { // assume col_major
            for( std::size_t ii=1; ii<_dims; ++ii){
                _stride[ii] = _shape[ii-1] * _stride[ii-1];
            }
        }
    }
}

template<class T>
Array<T>& Array<T>::operator=( const Array<T>& other){
    if( other.is_initialised() ){
        // If 'other' owns its own data, perform a full copy.
        // Otherwise, copy only _data pointer and other details.
        if( other.owns_data() ){
            // check if this and other have the same shape. If so, copy elements.
            // Otherwise, rebuild everything
            // TODO
        } else {
            // TODO
        }
    } else {
        // delete everything, return to uninitialised state
        // TODO
    }
}

template<class T>
std::size_t Array<T>::size() const {
    std::size_t result=1;
    for( unsigned ii=0; ii<_dims; ++ii) result *= shape[ii];
    return result;
}

template<class T>
std::size_t Array<T>::shape( std::size_t dim) const {
    return _shape[dim];
}

template<class T>
template<class Coords>
T Array<T>::operator()( const Coords& coords) const {
    std::size_t memjump=0;
    for( unsigned ii=0; ii<_dims; ++ii) memjump+= coords[ii]*_stride[ii];
    return *(_data+memjump);
}

template<class T>
template<class Coords>
T& Array<T>::operator()( const Coords& coords) {
    std::size_t memjump=0;
    for( unsigned ii=0; ii<_dims; ++ii) memjump+= coords[ii]*_stride[ii];
    return *(_data+memjump);
}

template<class T>
T Array<T>::operator()( std::size_t i0) const {
    return *(_data + i0*_stride[0]);
}

template<class T>
T Array<T>::operator()( std::size_t i0, std::size_t i1) const {
    return *(_data + i0*_stride[0] + i1*_stride[1]);
}

template<class T>
T Array<T>::operator()( std::size_t i0, std::size_t i1, std::size_t i2) const {
    return *(_data + i0*_stride[0] + i1*_stride[1] + i2*_stride[2]);
}

template<class T>
T Array<T>::operator()( std::size_t i0, std::size_t i1, std::size_t i2, std::size_t i3) const {
    return *(_data + i0*_stride[0] + i1*_stride[1] + i2*_stride[2] + i3*_stride[3]);
}

template<class T>
T Array<T>::operator()( std::size_t i0, std::size_t i1, std::size_t i2, std::size_t i3, std::size_t i4) const {
    return *(_data + i0*_stride[0] + i1*_stride[1] + i2*_stride[2] + i3*_stride[3] + i4*_stride[4]);
}

template<class T>
T Array<T>::operator()( std::size_t i0, std::size_t i1, std::size_t i2, std::size_t i3, std::size_t i4, std::size_t i5) const {
    return *(_data + i0*_stride[0] + i1*_stride[1] + i2*_stride[2] + i3*_stride[3] + i4*_stride[4] + i5*_stride[5]);
}

template<class T>
T Array<T>::operator()( std::size_t i0, std::size_t i1, std::size_t i2, std::size_t i3, std::size_t i4, std::size_t i5
    std::size_t i6) const {
    return *(_data + i0*_stride[0] + i1*_stride[1] + i2*_stride[2] + i3*_stride[3] + i4*_stride[4] + i5*_stride[5] +
        i6*_stride[6]);
}

template<class T>
T Array<T>::operator()( std::size_t i0, std::size_t i1, std::size_t i2, std::size_t i3, std::size_t i4, std::size_t i5
    std::size_t i6, std::size_t i7) const {
    return *(_data + i0*_stride[0] + i1*_stride[1] + i2*_stride[2] + i3*_stride[3] + i4*_stride[4] + i5*_stride[5] +
        i6*_stride[6] + i7*_stride[7]);
}

template<class T>
T Array<T>::operator()( std::size_t i0, std::size_t i1, std::size_t i2, std::size_t i3, std::size_t i4, std::size_t i5
    std::size_t i6, std::size_t i7, std::size_t i8) const {
    return *(_data + i0*_stride[0] + i1*_stride[1] + i2*_stride[2] + i3*_stride[3] + i4*_stride[4] + i5*_stride[5] +
        i6*_stride[6] + i7*_stride[7] + i8*_stride[8]);
}

template<class T>
T Array<T>::operator()( std::size_t i0, std::size_t i1, std::size_t i2, std::size_t i3, std::size_t i4, std::size_t i5
    std::size_t i6, std::size_t i7, std::size_t i8, std::size_t i9) const {
    return *(_data + i0*_stride[0] + i1*_stride[1] + i2*_stride[2] + i3*_stride[3] + i4*_stride[4] + i5*_stride[5] +
        i6*_stride[6] + i7*_stride[7] + i8*_stride[8] + i9*_stride[9]);
}

template<class T>
T& Array<T>::operator()( std::size_t i0){
    return *(_data + i0*_stride[0]);
}

template<class T>
T& Array<T>::operator()( std::size_t i0, std::size_t i1){
    return *(_data + i0*_stride[0] + i1*_stride[1]);
}

template<class T>
T& Array<T>::operator()( std::size_t i0, std::size_t i1, std::size_t i2){
    return *(_data + i0*_stride[0] + i1*_stride[1] + i2*_stride[2]);
}

template<class T>
T& Array<T>::operator()( std::size_t i0, std::size_t i1, std::size_t i2, std::size_t i3){
    return *(_data + i0*_stride[0] + i1*_stride[1] + i2*_stride[2] + i3*_stride[3]);
}

template<class T>
T& Array<T>::operator()( std::size_t i0, std::size_t i1, std::size_t i2, std::size_t i3, std::size_t i4){
    return *(_data + i0*_stride[0] + i1*_stride[1] + i2*_stride[2] + i3*_stride[3] + i4*_stride[4]);
}

template<class T>
T& Array<T>::operator()( std::size_t i0, std::size_t i1, std::size_t i2, std::size_t i3, std::size_t i4, std::size_t i5){
    return *(_data + i0*_stride[0] + i1*_stride[1] + i2*_stride[2] + i3*_stride[3] + i4*_stride[4] + i5*_stride[5]);
}

template<class T>
T& Array<T>::operator()( std::size_t i0, std::size_t i1, std::size_t i2, std::size_t i3, std::size_t i4, std::size_t i5
    std::size_t i6){
    return *(_data + i0*_stride[0] + i1*_stride[1] + i2*_stride[2] + i3*_stride[3] + i4*_stride[4] + i5*_stride[5] +
        i6*_stride[6]);
}

template<class T>
T& Array<T>::operator()( std::size_t i0, std::size_t i1, std::size_t i2, std::size_t i3, std::size_t i4, std::size_t i5
    std::size_t i6, std::size_t i7){
    return *(_data + i0*_stride[0] + i1*_stride[1] + i2*_stride[2] + i3*_stride[3] + i4*_stride[4] + i5*_stride[5] +
        i6*_stride[6] + i7*_stride[7]);
}

template<class T>
T& Array<T>::operator()( std::size_t i0, std::size_t i1, std::size_t i2, std::size_t i3, std::size_t i4, std::size_t i5
    std::size_t i6, std::size_t i7, std::size_t i8){
    return *(_data + i0*_stride[0] + i1*_stride[1] + i2*_stride[2] + i3*_stride[3] + i4*_stride[4] + i5*_stride[5] +
        i6*_stride[6] + i7*_stride[7] + i8*_stride[8]);
}

template<class T>
T& Array<T>::operator()( std::size_t i0, std::size_t i1, std::size_t i2, std::size_t i3, std::size_t i4, std::size_t i5
    std::size_t i6, std::size_t i7, std::size_t i8, std::size_t i9){
    return *(_data + i0*_stride[0] + i1*_stride[1] + i2*_stride[2] + i3*_stride[3] + i4*_stride[4] + i5*_stride[5] +
        i6*_stride[6] + i7*_stride[7] + i8*_stride[8] + i9*_stride[9]);
}

} // namespace
#endif
