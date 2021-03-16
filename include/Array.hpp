#ifndef __ULTRA_ARRAY_HPP
#define __ULTRA_ARRAY_HPP

#include <cmath>
#include <stdexcept>
#include <string>
#include <array>
#include <vector>
#include <algorithm>
#include <numeric>
#include <limits>
#include <functional>
#include <type_traits>

// Array.hpp
//
// A generic multidimensional array, templated over the type it contains.
// Can be contiguous or non-contiguous, row-major or column-major.
// 
// Dynamically allocated, though unlike std::vector, it cannot be resized
// after creation.

namespace ultra {

// Define slice : a tool for generating views of Arrays and related objects.
// Is an 'aggregate'/'pod' type.

struct Slice { 
    static constexpr std::ptrdiff_t all = std::numeric_limits<std::ptrdiff_t>::max();
    std::ptrdiff_t start;
    std::ptrdiff_t end;
    std::ptrdiff_t step=1;
};

// Define Array

template<class T>
class Array {

protected:

    char            _status;
    std::size_t     _dims;
    std::size_t     _size;
    std::size_t*    _shape;
    std::ptrdiff_t* _stride;
    T*              _data;

    // ===============================================
    // Memory helpers

    template<std::size_t N, class Int, class... Ints>
    std::ptrdiff_t variadic_memjump( Int coord, Ints... coords) const;

    template<std::size_t N, class Int>
    std::ptrdiff_t variadic_memjump( Int coord) const;

    std::ptrdiff_t idx_to_memjump(std::size_t idx) const;
    
    std::ptrdiff_t jump_to_stripe( std::size_t stripe, bool jump_to_end) const;

public:

    // ===============================================
    // Constructors

    // Default constructor and destructor
    Array();
    ~Array();

    // Copy and move
    
    Array( const Array<T>& other);

    template<class U>
    Array( const Array<U>& other);

    Array( Array&& other);

    // Build with given size
    
    // Templated over std::vector-like structures
    template<class V, std::enable_if_t<std::is_class<V>::value,bool> = true>
    Array( const V& shape, char rc_order=row_major);

    // Build 1D array with single integer arg
    template<class Int, std::enable_if_t<std::is_integral<Int>::value,bool> = true>
    Array( Int shape, char rc_order=row_major);

    // Build with c-arrays
    template<class CArray, std::enable_if_t<std::is_array<CArray>::value && std::rank<CArray>::value==1,bool> = true>
    Array( const CArray& shape, char rc_order=row_major);

    // Build with dynamic c-arrays
    // (does not take ownership of int_ptr)
    template<class Int>
    Array( Int* int_ptr, std::size_t N, char rc_order=row_major);

    // Assignment and move assignment
    
    Array& operator=( const Array<T>& other);
    
    template<class U>
    Array& operator=( const Array<U>& other);

    Array& operator=( Array&& other);

    // Additional utils
    void reset();

    // Specialised grid building methods
    
    Array<T> view() const;

    template<class... Slices>
    Array<T> view( const Slices&... ) const;

    // ===============================================
    // Attributes

    std::size_t dims() const;
    std::size_t size() const;
    std::size_t size( std::size_t dim) const;
    std::size_t shape( std::size_t dim) const; // alias for size(std::size_t)

    // return raw pointer, useful for interfacing with C libraries.
    // Will lead to strange behaviour if called by a non-contiguous Array.
    T* data() const;

    // ===============================================
    // Status handling.

    static constexpr char uninitialised   = 0x00;
    static constexpr char uninitialized   = 0x00;
    static constexpr char initialised     = 0x01;
    static constexpr char initialized     = 0x01;
    static constexpr char own_data        = 0x02;
    static constexpr char contiguous      = 0x04;
    static constexpr char semicontiguous  = 0x08;
    static constexpr char row_major       = 0x10;
    static constexpr char col_major       = 0x20;
    static constexpr char bcast           = 0x40;

    inline char get_status() const { return _status;}
    inline void set_status( char status){ _status = status;}

    inline bool is_initialised() const { return _status & initialised;}
    inline bool is_initialized() const { return _status & initialised;}
    inline bool owns_data() const { return _status & own_data;}
    inline bool is_contiguous() const { return _status & contiguous;}
    inline bool is_semicontiguous() const { return _status & semicontiguous;}
    inline bool is_row_major() const { return _status & row_major;}
    inline bool is_col_major() const { return _status & col_major;}
    inline bool is_broadcast() const { return _status & bcast;}

    // ===============================================
    // Data access

    // Access via many ints.
    // Warning: No checks are performed to ensure the correct version has been called.

    template<class... Ints> 
    T operator()( Ints... coords ) const;
    
    template<class... Ints> 
    T& operator()( Ints... coords );

    // Access via anything that looks like a std::vector

    template<class Coords, std::enable_if_t<std::is_class<Coords>::value,bool> = true>
    T operator()( const Coords& coords) const; 

    template<class Coords, std::enable_if_t<std::is_class<Coords>::value,bool> = true>
    T& operator()( const Coords& coords); 

    // Access via C array

    template<class Coords, std::enable_if_t<std::is_array<Coords>::value && std::rank<Coords>::value==1,bool> = true>
    T operator()( const Coords& coords) const; 

    template<class Coords, std::enable_if_t<std::is_array<Coords>::value && std::rank<Coords>::value==1,bool> = true>
    T& operator()( const Coords& coords); 

    // Access via dynamic C array (for the truly perverted)

    template<class Int>
    T operator()( Int* coords, std::size_t N) const; 

    template<class Int>
    T& operator()( Int* coords, std::size_t N); 

    // Access via square brackets
    // Treats all arrays as 1D containers.

    T operator[](std::size_t ii) const;
    T& operator[](std::size_t ii);

    // ===============================================
    // Iteration

    // Two types of iterator are available, covering 3 distinct iteration strategies.
    // 
    // (Semi-)Contiguous: fast_iterator
    // Non-contiguous:    iterator
    // 
    // The non-contiguous 'iterator' will work in all cases, and will be reasonably fast,
    // though it is unlikely to make use of any vectorisation. It may also be used for
    // 'broadcasting'. For example, this may be treating a 1xn array as an mxn array.
    // 
    // 'fast_iterator' should only be used with Arrays that are (semi-)contiguous. If
    // fully contiguous (i.e. not a view of an existing Array and not sliced), one should
    // call 'begin_fast()' and 'end_fast()'. The 'fast_iterator' is actually little more
    // than a pointer to T, similar to std::vector<T>::iterator.
    //
    // If semi-contiguous, the user should instead make use of 'begin_stripe(size_t stripe)'
    // and 'end_stripe(size_t stripe)'. This returns the same type of iterator, but the positions
    // given by begin and end will be the start and finish of one contiguous 'stripe' of the
    // fastest incrementing dimension (the last when row-major or first when column-major).
    //
    // Note that all iteration strategies will behave differently for row-major and column-major
    // Arrays, and this may lead to confusing behaviour. Avoid mixing the two wherever possible.
    
    // (Semi-)Contiguous access
    template<bool constness>
    class base_fast_iterator;

    using fast_iterator = base_fast_iterator<false>;
    using const_fast_iterator = base_fast_iterator<true>;

    fast_iterator begin_fast(); 
    fast_iterator end_fast();
    const_fast_iterator begin_fast() const;
    const_fast_iterator end_fast() const;

    std::size_t num_stripes() const;

    fast_iterator begin_stripe( std::size_t stripe);
    fast_iterator end_stripe( std::size_t stripe);
    const_fast_iterator begin_stripe( std::size_t stripe) const;
    const_fast_iterator end_stripe( std::size_t stripe) const;

    // Generic access
    template<bool constness>
    class base_iterator;
    
    using iterator = base_iterator<false>;
    using const_iterator = base_iterator<true>;
    
    iterator begin();
    iterator end();
    const_iterator begin() const;
    const_iterator end() const;

    // TODO
    //const_iterator begin_broadcast( ... something ...) const;
    //const_iterator end_broadcast( ... something ...) const;
};

// ===============================================
// Define custom iterators

template<class T>
template<bool constness>
class Array<T>::base_fast_iterator {

    friend typename Array<T>::base_fast_iterator<!constness>;

public:

    // TODO figure out C++20 iterator concepts
    using iterator_category = std::contiguous_iterator_tag;
    using difference_type   = std::ptrdiff_t;
    using value_type        = T;
    using pointer           = std::conditional_t<constness,const T*, T*>;
    using reference         = std::conditional_t<constness,const T&, T&>;

private:

    pointer _ptr;

public:

    // ===============================================
    // Constructors
 
    base_fast_iterator( pointer ptr);
    base_fast_iterator( const base_fast_iterator& other);
    base_fast_iterator( base_fast_iterator&& other);
    base_fast_iterator& operator=( const base_fast_iterator& other);
    base_fast_iterator& operator=( base_fast_iterator&& other);

    // ===============================================
    // Conversion from non-const to const

    template<bool C=!constness, std::enable_if_t<C,bool> = true>
    operator base_fast_iterator<C>() const;

    // ===============================================
    // Standard iterator interface

    // Dereference
    reference operator*();
    
    // Increment/decrement
    base_fast_iterator<constness>& operator++();
    base_fast_iterator<constness> operator++(int) const;

    base_fast_iterator<constness>& operator--();
    base_fast_iterator<constness> operator--(int) const;

    // Random-access
    base_fast_iterator<constness>& operator+=( difference_type diff);
    base_fast_iterator<constness>& operator-=( difference_type diff);
    base_fast_iterator<constness> operator+( difference_type diff) const;
    base_fast_iterator<constness> operator-( difference_type diff) const;

    // Distance
    template<bool constness_r> difference_type operator-( const base_fast_iterator<constness_r>& it_r) const;

    // Boolean comparisons
    template<bool constness_r> bool operator==( const base_fast_iterator<constness_r>& it_r) const;
    template<bool constness_r> bool operator!=( const base_fast_iterator<constness_r>& it_r) const;
    template<bool constness_r> bool operator>=( const base_fast_iterator<constness_r>& it_r) const;
    template<bool constness_r> bool operator<=( const base_fast_iterator<constness_r>& it_r) const;
    template<bool constness_r> bool operator<( const base_fast_iterator<constness_r>& it_r) const;
    template<bool constness_r> bool operator>( const base_fast_iterator<constness_r>& it_r) const;
};

template<class T>
template<bool constness>
class Array<T>::base_iterator {
    
    friend typename Array<T>::base_iterator<!constness>;

public:

    // TODO figure out C++20 iterator concepts
    using iterator_category = std::random_access_iterator_tag;
    using difference_type   = std::ptrdiff_t;
    using value_type        = T;
    using pointer           = std::conditional_t<constness,const T*, T*>;
    using reference         = std::conditional_t<constness,const T&, T&>;

private:

    pointer         _ptr;
    std::size_t     _dims;
    std::size_t*    _shape;
    std::ptrdiff_t* _stride;
    std::ptrdiff_t* _pos;
    bool            _col_major;

public:

    // ===============================================
    // Constructors
 
    base_iterator( pointer ptr, std::size_t dims, std::size_t* shape, std::ptrdiff_t* stride, bool end, bool col_major);
    base_iterator( pointer ptr, std::size_t dims, std::size_t* shape, std::ptrdiff_t* stride, std::ptrdiff_t* pos, bool col_major);
    ~base_iterator();
    base_iterator( const base_iterator<constness>& other);
    base_iterator( base_iterator<constness>&& other);
    base_iterator& operator=( const base_iterator<constness>& other);
    base_iterator& operator=( base_iterator<constness>&& other);

    // ===============================================
    // Conversion from non-const to const

    template<bool C=!constness, std::enable_if_t<C,bool> = true>
    operator base_iterator<C>() const;

    // ===============================================
    // Standard iterator interface

    // Dereference
    reference operator*();
    
    // Increment/decrement
    base_iterator<constness>& operator++();
    base_iterator<constness> operator++(int) const;

    base_iterator<constness>& operator--();
    base_iterator<constness> operator--(int) const;

    // Random-access
    base_iterator<constness>& operator+=( difference_type diff);
    base_iterator<constness>& operator-=( difference_type diff);
    base_iterator<constness> operator+( difference_type diff) const;
    base_iterator<constness> operator-( difference_type diff) const;

    // Distance
    template<bool constness_r> difference_type operator-( const base_iterator<constness_r>& it_r) const;

    // Boolean comparisons
    template<bool constness_r> bool operator==( const base_iterator<constness_r>& it_r) const;
    template<bool constness_r> bool operator!=( const base_iterator<constness_r>& it_r) const;
    template<bool constness_r> bool operator>=( const base_iterator<constness_r>& it_r) const;
    template<bool constness_r> bool operator<=( const base_iterator<constness_r>& it_r) const;
    template<bool constness_r> bool operator<( const base_iterator<constness_r>& it_r) const;
    template<bool constness_r> bool operator>( const base_iterator<constness_r>& it_r) const;
};

// ===============================================
// Function definitions follow

// ===============================================
// Array

template<class T>
Array<T>::Array() :
    _status(uninitialised),
    _dims(0),
    _size(0),
    _shape(nullptr),
    _stride(nullptr),
    _data(nullptr)
{}

template<class T>
Array<T>::~Array() {
    reset();
}

template<class T>
Array<T>::Array( const Array<T>& other) :
    _status(other._status),
    _dims(other._dims),
    _size(other._size)
{
    if( other.is_initialised() ){
        _shape  = new std::size_t[_dims];
        _stride = new std::ptrdiff_t[_dims];
        std::copy( other._shape, other._shape+_dims, _shape);
        std::copy( other._stride, other._stride+_dims, _stride);
        // If 'other' owns its own data, perform a full copy.
        // An array that owns its own data will also be contiguous, so can used std::copy
        // Otherwise, simply copy over the data pointer: a copy of a view is also a view.
        // Use the explicit 'copy' function to get a new contiguous array from a view.
        if( other.owns_data() ){
            _data = new T[_size];
            std::copy( other._data, other._data+_size, _data);
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
template<class U>
Array<T>::Array( const Array<U>& other) :
    _status(other._status),
    _dims(other._dims),
    _size(other._size)
{
    if( other.is_initialised() ){
        if( other.owns_data() ){
            _data = new T[_size];
            std::copy( other._data, other._data+_size, _data);
        } else {
            throw std::invalid_argument("UltraArray: Cannot copy Array<T1> to Array<T2> if Array<T1> does not own its own data.");
        }
        _shape  = new std::size_t[_dims];
        _stride = new std::ptrdiff_t[_dims];
        std::copy( other._shape, other._shape+_dims, _shape);
        std::copy( other._stride, other._stride+_dims, _stride);
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
    _size(other._size),
    _shape(other._shape),
    _stride(other._stride),
    _data(other._data)
{
    // Set status of other to uninitialised.
    // The new object now handles the lifetimes of _data, _shape, and _stride.
    other.set_status(uninitialised);
}

template<class T>
template<class Int, std::enable_if_t<std::is_integral<Int>::value,bool>>
Array<T>::Array( Int shape, char rc_order) : 
    Array( std::array<Int,1>{shape}, rc_order )
{}

template<class T>
template<class CArray, std::enable_if_t<std::is_array<CArray>::value && std::rank<CArray>::value==1,bool>>
Array<T>::Array( const CArray& shape, char rc_order) :
    Array( std::to_array(shape), rc_order)
{
    static_assert(std::is_integral<std::remove_extent_t<CArray>>::value,"UltraArray: C array constructor must have integral type.");
}

template<class T>
template<class Int>
Array<T>::Array( Int* int_ptr, std::size_t N, char rc_order) :
    Array( std::vector<Int>(int_ptr,int_ptr+N), rc_order)
{
    static_assert(std::is_integral<Int>::value,"UltraArray: Dynamic array constructor must take pointer to integral type.");
}

template<class T>
template<class V, std::enable_if_t<std::is_class<V>::value,bool>>
Array<T>::Array( const V& shape, char rc_order) :
    _status( initialised | own_data | contiguous | semicontiguous | rc_order ),
    _dims(shape.size())
{
    // Ensure 'rc_order' is either row_major or col_major
    if( rc_order!=row_major && rc_order!=col_major ){
        std::string err = "UltraArray: Construction with explicit row/col major ordering.";
        err += " Argument 'rc_order' must be either Array<T>::row_major or Array<T>::col_major.";
        throw std::invalid_argument(err.c_str());
    }
    // if _dims==0, reset status to uninitialised
    // if _dims==1, set both c- and f-contiguous
    switch(_dims){
        case 0:
            _size=0;
            _shape=nullptr;
            _stride=nullptr;
            _data=nullptr;
            set_status(uninitialised);
            break;
        case 1:
            set_status(_status | row_major | col_major);
            [[fallthrough]];
        default:
            _shape = new std::size_t[_dims];
            _stride = new std::ptrdiff_t[_dims];
            std::copy( shape.begin(), shape.end(), _shape);
            _size = std::accumulate( _shape, _shape+_dims, 1, std::multiplies<std::size_t>() );
            _data = new T[_size];
            // Determine strides
            if( is_row_major() ){
                _stride[_dims-1] = 1;
                for( std::size_t ii=1; ii<_dims; ++ii){
                    _stride[_dims-1-ii] = _shape[_dims-ii] * _stride[_dims-ii];
                }
            } else {
                _stride[0] = 1;
                for( std::size_t ii=1; ii<_dims; ++ii){
                    _stride[ii] = _shape[ii-1] * _stride[ii-1];
                }
            }
    }
}

template<class T>
Array<T>& Array<T>::operator=( const Array<T>& other){
    if( other.is_initialised() ){
        if( is_initialised() ){
            // Determine if we need to destroy and/or (re)build _data
            if( owns_data() ){
                if( other.owns_data()){
                    if( _size != other._size){
                        delete[] _data;
                        _data = new T[other._size];
                    } else {
                        // Do nothing! Can simply copy data across
                    }
                } else {
                    delete[] _data;
                }
            } else {
                if( other.owns_data()){
                    _data = new T[other._size];
                } else {
                    // Do nothing! Can copy data pointer across
                }
            }
            // If this and other have different dims, rebuild shape and stride.
            if( _dims != other._dims){
                delete[] _shape;
                delete[] _stride;
                _shape = new std::size_t[other._dims];
                _stride = new std::ptrdiff_t[other._dims];
            }
        } else {
            _shape = new std::size_t[other._dims];
            _stride = new std::ptrdiff_t[other._dims];
            if( other.owns_data() ) _data = new T[other._size];
        }
        _dims = other._dims;
        _size = other._size;
        // Copy over shape and stride
        std::copy( other._shape, other._shape+_dims, _shape);
        std::copy( other._stride, other._stride+_dims, _stride);
        // Copy data
        if( other.owns_data()){
            std::copy( other._data, other._data+size(), _data);
        } else {
            _data = other._data;
        }
        // Copy status
        set_status( other._status );
    } else {
        // delete everything, return to uninitialised state
        reset();
    }
    return *this;
}

template<class T>
template<class U>
Array<T>& Array<T>::operator=( const Array<U>& other){
    // Very similar to the standard copy assignment, but disallows copy of an array which does not own its own data
    if( other.is_initialised() ){
        if( !other.owns_data() ){
            throw std::invalid_argument("UltraArray: Cannot copy Array<T1> to Array<T2> if Array<T1> does not own its own data.");
        }
        if( is_initialised() ){
            if( owns_data() ){
                if( _size != other._size){
                    delete[] _data;
                    _data = new T[other._size];
                } else {
                    // Do nothing! Can simply copy data across
                }
            } else {
                _data = new T[other._size];
            }
        } else {
            _shape = new std::size_t[other._dims];
            _stride = new std::ptrdiff_t[other._dims];
            if( other.owns_data() ) _data = new T[other._size];
        }
        _dims = other._dims;
        _size = other._size;
        // Copy over shape, stride, and data
        std::copy( other._shape, other._shape+_dims, _shape);
        std::copy( other._stride, other._stride+_dims, _stride);
        std::copy( other._data, other._data+size(), _data);
        // Copy status
        set_status( other._status );
    } else {
        // delete everything, return to uninitialised state
        reset();
    }
    return *this;
}

template<class T>
Array<T>& Array<T>::operator=( Array<T>&& other){
    // Delete anything currently held, take control of other.
    reset();
    _status = other._status;
    _dims = other._dims;
    _size = other._size;
    _shape = other._shape;
    _stride = other._stride;
    _data = other._data;
    other.set_status(uninitialised);
    return *this;
}


template<class T>
void Array<T>::reset(){
    if( is_initialised() ){
        delete[] _shape;
        delete[] _stride;
        if( owns_data() ){
            delete[] _data;
        }
    }
    _status = uninitialised;
    _dims=0;
    _shape=nullptr;
    _stride=nullptr;
    _data=nullptr;
}

template<class T>
Array<T> Array<T>::view() const {
    // Create an uninitialised array, return this if 'viewing' something uninitialised.
    Array<T> result;
    if( !is_initialised() ) return result;
    // Copy over relevant info
    result._status = _status;
    if( owns_data() ) result._status -= own_data;
    result._dims = _dims;
    result._size = _size;
    result._shape = new std::size_t[_dims];
    std::copy( _shape, _shape+_dims, result._shape);
    result._stride = new ptrdiff_t[_dims];
    std::copy( _stride, _stride+_dims, result._stride);
    result._data = _data;
    return result;
}

template<class T>
template<class... Slices>
Array<T> Array<T>::view( const Slices&... var_slices) const {
    std::array<Slice,sizeof...(Slices)> slices = {{ var_slices... }};
    // Create an uninitialised array, return this if 'viewing' something uninitialised.
    Array<T> result;
    if( !is_initialised() ) return result;
    // Copy over relevant info
    result._dims = _dims;
    result._shape = new std::size_t[_dims];
    result._stride = new ptrdiff_t[_dims];
    result._status = _status;
    result._data = _data;
    if( owns_data() ) result._status -= own_data;
    // Modify shape and stride
    // Create copy of slice which we can edit
    Slice slice;
    for( std::size_t ii=0; ii<_dims; ++ii){
        if( ii < slices.size() ){
            slice = slices[ii];
        } else { // if not enough slices provided, assume start=all, end=all, step=1
            slice = { Slice::all, Slice::all, 1 };
        }
        // Account for negative start/end
        if( slice.start < 0 ) slice.start = _shape[ii] + slice.start;
        if( slice.end < 0 ) slice.end = _shape[ii] + slice.end;
        // Account for 'all' specifiers
        if( slice.start == Slice::all ) slice.start = 0;
        if( slice.end == Slice::all ) slice.end = _shape[ii];
        // Throw exceptions if slice is impossible
        if( slice.start < 0 || slice.end > static_cast<std::ptrdiff_t>(_shape[ii]) ) throw std::invalid_argument("UltraArray: Slice out of bounds.");
        if( slice.end <= slice.start ) throw std::invalid_argument("UltraArray: Slice end is less than or equal to start.");
        if( slice.step == 0 ) throw std::invalid_argument("UltraArray: Slice has zero step.");
        // Account for the case of step size larger than shape
        if( slice.end - slice.start < abs(slice.step) ) slice.step = (slice.end - slice.start) * (slice.step < 0 ? -1 : 1);
        // Set shape and stride of result. Shape is (slice.end-slice.start)/abs(slice.step), but rounding up rather than down.
        result._shape[ii] = (slice.end - slice.start + ((slice.end-slice.start)%abs(slice.step)))/abs(slice.step);
        result._stride[ii] = _stride[ii]*slice.step;
        // Move data to start of slice (be sure to use this stride rather than result stride)
        if( slice.step > 0 ){
            result._data += slice.start * _stride[ii];
        } else {
            result._data += (slice.end-1) * _stride[ii];
        }
        // Set contiguity status depending on how shape/stride have been changed
        if( result.is_contiguous() && ( slice.start != 0 || slice.end != static_cast<std::ptrdiff_t>(_shape[ii]) || slice.step != 1)){
            result._status -= contiguous;
        }
        if( result.is_semicontiguous() && slice.step != 1 && ((is_row_major() && ii==_dims-1) || (is_col_major() && ii==0) )){
            result._status -= semicontiguous;
        }
    }
    // Set remaining info and return
    result._size = std::accumulate( result._shape, result._shape+_dims, 1, std::multiplies<std::size_t>() );
    return result;
}

template<class T>
std::size_t Array<T>::dims() const {
    return _dims;
}

template<class T>
std::size_t Array<T>::size() const {
    return _size;
}

template<class T>
std::size_t Array<T>::size( std::size_t dim) const {
    return _shape[dim];
}

template<class T>
std::size_t Array<T>::shape( std::size_t dim) const {
    return _shape[dim];
}

template<class T>
T* Array<T>::data() const {
    return _data;
}
    
// Base case(s)
template<class T>
template<std::size_t N, class Int>
std::ptrdiff_t Array<T>::variadic_memjump( Int coord) const {
    static_assert(std::is_integral<std::remove_reference_t<Int>>::value); 
    return _stride[N] * coord; 
}

// Recursive step
template<class T>
template<std::size_t N, class Int, class... Ints>
std::ptrdiff_t Array<T>::variadic_memjump( Int coord, Ints... coords) const {
    static_assert(std::is_integral<std::remove_reference_t<Int>>::value); 
    return (_stride[N] * coord) + variadic_memjump<N+1,Ints...>(coords...);
}

template<class T>
template<class... Ints> 
T Array<T>::operator()( Ints... coords ) const {
    return *(_data + variadic_memjump<0,Ints...>(coords...));
}
    
template<class T>
template<class... Ints> 
T& Array<T>::operator()( Ints... coords ) {
    return *(_data + variadic_memjump<0,Ints...>(coords...));
}

template<class T>
template<class Coords, std::enable_if_t<std::is_class<Coords>::value,bool>>
T Array<T>::operator()( const Coords& coords) const {
    return *(_data+std::inner_product(coords.begin(),coords.end(),_stride,0));
}

template<class T>
template<class Coords, std::enable_if_t<std::is_class<Coords>::value,bool>>
T& Array<T>::operator()( const Coords& coords) {
    return *(_data+std::inner_product(coords.begin(),coords.end(),_stride,0));
}

template<class T>
template<class Coords, std::enable_if_t<std::is_array<Coords>::value && std::rank<Coords>::value==1,bool>>
T Array<T>::operator()( const Coords& coords) const {
    return *(_data+std::inner_product(coords,coords+std::extent<Coords>::value,_stride,0));
}

template<class T>
template<class Coords, std::enable_if_t<std::is_array<Coords>::value && std::rank<Coords>::value==1,bool>>
T& Array<T>::operator()( const Coords& coords) {
    return *(_data+std::inner_product(coords,coords+std::extent<Coords>::value,_stride,0));
}

template<class T>
template<class Int>
T Array<T>::operator()( Int* coords, std::size_t N) const {
    return *(_data+std::inner_product(coords,coords+N,_stride,0));
}

template<class T>
template<class Int>
T& Array<T>::operator()( Int* coords, std::size_t N) {
    return *(_data+std::inner_product(coords,coords+N,_stride,0));
}

template<class T>
std::ptrdiff_t Array<T>::idx_to_memjump(std::size_t idx) const {
    std::ptrdiff_t jump = 0;
    if( is_col_major() ){
        for(std::size_t ii=0; ii<_dims; ++ii){
            jump += _stride[ii]*(idx%_shape[ii]);
            idx /= _shape[ii];
        }
    } else {
        for(std::size_t ii=_dims; ii>0; --ii){
            jump += _stride[ii-1]*(idx%_shape[ii-1]);
            idx /= _shape[ii-1];
        }
    }
    return jump;
}

template<class T>
T Array<T>::operator[](std::size_t idx) const {
    return *(_data+idx_to_memjump(idx));
}

template<class T>
T& Array<T>::operator[](std::size_t idx) {
    return *(_data+idx_to_memjump(idx));
}

template<class T>
typename Array<T>::fast_iterator Array<T>::begin_fast() {
    return fast_iterator(_data);
}

template<class T>
typename Array<T>::fast_iterator Array<T>::end_fast() {
    return fast_iterator(_data + size());
}

template<class T>
typename Array<T>::const_fast_iterator Array<T>::begin_fast() const {
    return const_fast_iterator(_data);
}

template<class T>
typename Array<T>::const_fast_iterator Array<T>::end_fast() const {
    return const_fast_iterator(_data + size());
}

template<class T>
std::size_t Array<T>::num_stripes() const {
    if( _dims > 1 ){
        return std::accumulate( _shape+is_col_major(), _shape+_dims-is_row_major(), 1, std::multiplies<std::size_t>() );
    } else {
        return 1;
    }
}

template<class T>
std::ptrdiff_t Array<T>::jump_to_stripe( std::size_t stripe, bool jump_to_end) const {
    std::ptrdiff_t jump = 0;
    if( is_row_major() ){
        for( std::size_t ii=_dims-1; ii>0; --ii ){
            jump += _stride[ii-1]*(stripe % _shape[ii-1]);
            stripe /= _shape[ii-1];
        }
       if( jump_to_end ) jump += _shape[_dims-1] * _stride[_dims-1]; 
    } else {
        for( std::size_t ii=1; ii<_dims; ++ii ){
            jump += _stride[ii]*(stripe % _shape[ii]);
            stripe /= _shape[ii];
        }
       if( jump_to_end ) jump += _shape[0] * _stride[0]; 
    }
    return jump;
}

template<class T>
typename Array<T>::fast_iterator Array<T>::begin_stripe( std::size_t stripe) {
    return fast_iterator(_data + jump_to_stripe(stripe,false));
}

template<class T>
typename Array<T>::fast_iterator Array<T>::end_stripe( std::size_t stripe) {
    return fast_iterator(_data + jump_to_stripe(stripe,true));
}

template<class T>
typename Array<T>::const_fast_iterator Array<T>::begin_stripe( std::size_t stripe) const {
    return const_fast_iterator(_data + jump_to_stripe(stripe,false));
}

template<class T>
typename Array<T>::const_fast_iterator Array<T>::end_stripe( std::size_t stripe) const {
    return const_fast_iterator(_data + jump_to_stripe(stripe,true));
}

template<class T>
Array<T>::iterator Array<T>::begin() {
    return iterator( _data, _dims, _shape, _stride, false, is_col_major());
}

template<class T>
Array<T>::iterator Array<T>::end() {
    return iterator( _data, _dims, _shape, _stride, true, is_col_major());
}

template<class T>
Array<T>::const_iterator Array<T>::begin() const {
    return const_iterator( _data, _dims, _shape, _stride, false, is_col_major());
}

template<class T>
Array<T>::const_iterator Array<T>::end() const {
    return const_iterator( _data, _dims, _shape, _stride, true, is_col_major());
}

// ===============================================
// fast_iterator

template<class T> template<bool constness>
Array<T>::base_fast_iterator<constness>::base_fast_iterator( typename Array<T>::base_fast_iterator<constness>::pointer ptr ) :
    _ptr(ptr)
{}

template<class T> template<bool constness>
Array<T>::base_fast_iterator<constness>::base_fast_iterator( const typename Array<T>::base_fast_iterator<constness>& other ) :
    _ptr(other._ptr)
{}

template<class T> template<bool constness>
Array<T>::base_fast_iterator<constness>::base_fast_iterator( typename Array<T>::base_fast_iterator<constness>&& other) :
    _ptr(other._ptr)
{}

template<class T> template<bool constness>
typename Array<T>::base_fast_iterator<constness>& Array<T>::base_fast_iterator<constness>::operator=( const typename Array<T>::base_fast_iterator<constness>& other) {
    _ptr = other._ptr;
    return *this;
}

template<class T> template<bool constness>
typename Array<T>::base_fast_iterator<constness>& Array<T>::base_fast_iterator<constness>::operator=( typename Array<T>::base_fast_iterator<constness>&& other) {
    _ptr = other._ptr;
    return *this;
}

template<class T> template<bool constness> template<bool C, std::enable_if_t<C,bool>>
Array<T>::base_fast_iterator<constness>::operator base_fast_iterator<C>() const {
    return base_fast_iterator<C>(_ptr);
}

template<class T> template<bool constness>
typename Array<T>::base_fast_iterator<constness>::reference Array<T>::base_fast_iterator<constness>::operator*() {
    return *_ptr;
}
    
template<class T> template<bool constness>
typename Array<T>::base_fast_iterator<constness>& Array<T>::base_fast_iterator<constness>::operator++(){
    ++_ptr;
    return *this;
}

template<class T> template<bool constness>
typename Array<T>::base_fast_iterator<constness> Array<T>::base_fast_iterator<constness>::operator++(int) const {
    return Array<T>::base_fast_iterator<constness>(_ptr+1);
}

template<class T> template<bool constness>
typename Array<T>::base_fast_iterator<constness>& Array<T>::base_fast_iterator<constness>::operator--(){
    --_ptr;
    return *this;
}

template<class T> template<bool constness>
typename Array<T>::base_fast_iterator<constness> Array<T>::base_fast_iterator<constness>::operator--(int) const {
    return Array<T>::base_fast_iterator<constness>(_ptr-1);
}

template<class T> template<bool constness>
typename Array<T>::base_fast_iterator<constness>& Array<T>::base_fast_iterator<constness>::operator+=( typename Array<T>::base_fast_iterator<constness>::difference_type diff) {
    _ptr += diff;
    return *this;
}

template<class T> template<bool constness>
typename Array<T>::base_fast_iterator<constness>& Array<T>::base_fast_iterator<constness>::operator-=( typename Array<T>::base_fast_iterator<constness>::difference_type diff) {
    _ptr -= diff;
    return *this;
}

template<class T> template<bool constness>
typename Array<T>::base_fast_iterator<constness> Array<T>::base_fast_iterator<constness>::operator+( typename Array<T>::base_fast_iterator<constness>::difference_type diff) const {
    return Array<T>::base_fast_iterator<constness>( _ptr + diff);
}

template<class T> template<bool constness>
typename Array<T>::base_fast_iterator<constness> Array<T>::base_fast_iterator<constness>::operator-( typename Array<T>::base_fast_iterator<constness>::difference_type diff) const {
    return Array<T>::base_fast_iterator<constness>( _ptr - diff);
}

template<class T> template<bool constness> template<bool constness_r>
typename Array<T>::base_fast_iterator<constness>::difference_type Array<T>::base_fast_iterator<constness>::operator-( const typename Array<T>::base_fast_iterator<constness_r>& it_r) const {
    return _ptr - it_r._ptr;
}

template<class T> template<bool constness> template<bool constness_r>
bool Array<T>::base_fast_iterator<constness>::operator==( const typename Array<T>::base_fast_iterator<constness_r>& it_r) const {
    return _ptr == it_r._ptr;
}

template<class T> template<bool constness> template<bool constness_r>
bool Array<T>::base_fast_iterator<constness>::operator!=( const typename Array<T>::base_fast_iterator<constness_r>& it_r) const {
    return _ptr != it_r._ptr;
}

template<class T> template<bool constness> template<bool constness_r>
bool Array<T>::base_fast_iterator<constness>::operator>=( const typename Array<T>::base_fast_iterator<constness_r>& it_r) const {
    return _ptr >= it_r._ptr;
}

template<class T> template<bool constness> template<bool constness_r>
bool Array<T>::base_fast_iterator<constness>::operator<=( const typename Array<T>::base_fast_iterator<constness_r>& it_r) const {
    return _ptr <= it_r._ptr;
}

template<class T> template<bool constness> template<bool constness_r>
bool Array<T>::base_fast_iterator<constness>::operator>( const typename Array<T>::base_fast_iterator<constness_r>& it_r) const {
    return _ptr > it_r._ptr;
}

template<class T> template<bool constness> template<bool constness_r>
bool Array<T>::base_fast_iterator<constness>::operator<( const typename Array<T>::base_fast_iterator<constness_r>& it_r) const {
    return _ptr < it_r._ptr;
}

// ===============================================
// iterator

template<class T> template<bool constness>
Array<T>::base_iterator<constness>::base_iterator( typename Array<T>::base_iterator<constness>::pointer ptr, std::size_t dims, std::size_t* shape, std::ptrdiff_t* stride, bool end, bool col_major) :
    _ptr(ptr),
    _dims(dims),
    _shape(shape),
    _stride(stride),
    _pos(new std::ptrdiff_t[dims]),
    _col_major(col_major)
{
    for( std::size_t ii=0; ii<_dims; ++ii) _pos[ii] = 0;

    // If this is an 'end' iterator, pos should be zero in all dimensions except the slowest incrementing.
    if(end){
        if(_col_major){
            _pos[_dims-1] = _shape[_dims-1];
            _ptr += _stride[_dims-1]*_shape[_dims-1];
        } else {
            _pos[0] = _shape[0];
            _ptr += _stride[0]*_shape[0];
        }
    }
}

template<class T> template<bool constness>
Array<T>::base_iterator<constness>::base_iterator( typename Array<T>::base_iterator<constness>::pointer ptr, std::size_t dims, std::size_t* shape, std::ptrdiff_t* stride, std::ptrdiff_t* pos, bool col_major) :
    _ptr(ptr),
    _dims(dims),
    _shape(shape),
    _stride(stride),
    _pos(new std::ptrdiff_t[dims]),
    _col_major(col_major)
{
    std::copy( pos, pos+_dims, _pos);
}


template<class T> template<bool constness>
Array<T>::base_iterator<constness>::~base_iterator() {
    if( _ptr != nullptr ){
        delete[] _pos;
    }
}

template<class T> template<bool constness>
Array<T>::base_iterator<constness>::base_iterator( const typename Array<T>::base_iterator<constness>& other ) :
    _ptr(other._ptr),
    _dims(other._dims),
    _shape(other._shape),
    _stride(other._stride),
    _pos(new std::ptrdiff_t[other._dims]),
    _col_major(other._col_major)
{
    std::copy( other._pos, other._pos+_dims, _pos);
}

template<class T> template<bool constness>
Array<T>::base_iterator<constness>::base_iterator( typename Array<T>::base_iterator<constness>&& other) :
    _ptr(other._ptr),
    _dims(other._dims),
    _shape(other._shape),
    _stride(other._stride),
    _pos(other._pos),
    _col_major(other._col_major)
{
    // invalidate other so it won't delete[] shape/stride/pos.
    other._ptr = nullptr;
}

template<class T> template<bool constness>
typename Array<T>::base_iterator<constness>& Array<T>::base_iterator<constness>::operator=( const typename Array<T>::base_iterator<constness>& other) {
    // if other has different dims, need to reallocate pos
    if( _dims != other._dims ){
        _dims = other._dims;
        if( _ptr != nullptr ) delete[] _pos;
        _pos = new std::ptrdiff_t[_dims];
    }
    _ptr = other._ptr;
    _shape = other._shape;
    _stride = other._stride;
    std::copy( other._pos, other._pos+_dims, _pos);
    _col_major = other._col_major;
    return *this;
}

template<class T> template<bool constness>
typename Array<T>::base_iterator<constness>& Array<T>::base_iterator<constness>::operator=( typename Array<T>::base_iterator<constness>&& other) {
    _ptr = other._ptr;
    _dims = other._dims;
    _shape = other._shape;
    _stride = other._stride;
    _pos = other._pos;
    _col_major = other._col_major;
    // invalidate other so it won't delete[] shape/stride/pos.
    other._ptr = nullptr;
    return *this;
}

template<class T> template<bool constness> template<bool C, std::enable_if_t<C,bool>>
Array<T>::base_iterator<constness>::operator base_iterator<C>() const {
    return base_fast_iterator<C>(_ptr,_dims,_shape,_stride,_pos,_col_major);
}

template<class T> template<bool constness>
typename Array<T>::base_iterator<constness>::reference Array<T>::base_iterator<constness>::operator*() {
    return *_ptr;
}
    
template<class T> template<bool constness>
typename Array<T>::base_iterator<constness>& Array<T>::base_iterator<constness>::operator++(){
    if( _col_major ){
        std::size_t idx = 0;
        _ptr += _stride[idx];
        ++_pos[idx];
        while( _pos[idx] == static_cast<std::ptrdiff_t>(_shape[idx]) && idx != _dims-1 ){
            // Go back to start of current dimension
            _ptr -= _stride[idx] * _shape[idx];
            _pos[idx]=0;
            // Increment one in next dimension
            _ptr += _stride[idx+1];
            ++_pos[idx+1];
            // Repeat for remaining dimensions
            ++idx;
        }
    } else {
        std::size_t idx = _dims-1;
        _ptr += _stride[idx];
        ++_pos[idx];
        while( _pos[idx] == static_cast<std::ptrdiff_t>(_shape[idx]) && idx != 0 ){
            // Go back to start of current dimension
            _ptr -= _stride[idx] * _shape[idx];
            _pos[idx]=0;
            // Increment one in next dimension
            _ptr += _stride[idx-1];
            ++_pos[idx-1];
            // Repeat for remaining dimensions
            --idx;
        }
    }
    return *this;
}

template<class T> template<bool constness>
typename Array<T>::base_iterator<constness> Array<T>::base_iterator<constness>::operator++(int) const {
    return ++Array<T>::base_iterator<constness>(*this);
}

template<class T> template<bool constness>
typename Array<T>::base_iterator<constness>& Array<T>::base_iterator<constness>::operator--(){
    if( _col_major ){
        std::size_t idx = 0;
        _ptr -= _stride[idx];
        --_pos[idx];
        while( _pos[idx] == -1 && idx != _dims-1 ){
            // Go to end of current dimension
            _ptr += _stride[idx] * _shape[idx];
            _pos[idx]=_shape[idx]-1;
            // Decrement one in next dimension
            _ptr -= _stride[idx+1];
            --_pos[idx+1];
            // Repeat for remaining dimensions
            ++idx;
        }
    } else {
        std::size_t idx = _dims-1;
        _ptr -= _stride[idx];
        --_pos[idx];
        while( _pos[idx] == -1 && idx != 0 ){
            // Go to end of current dimension
            _ptr += _stride[idx] * _shape[idx];
            _pos[idx]=_shape[idx]-1;
            // Decrement one in next dimension
            _ptr -= _stride[idx-1];
            --_pos[idx-1];
            // Repeat for remaining dimensions
            --idx;
        }
    }
    return *this;
}

template<class T> template<bool constness>
typename Array<T>::base_iterator<constness> Array<T>::base_iterator<constness>::operator--(int) const {
    return --Array<T>::base_iterator<constness>(*this);
}

template<class T> template<bool constness>
typename Array<T>::base_iterator<constness>& Array<T>::base_iterator<constness>::operator+=( std::ptrdiff_t diff) {
    // If diff is less than 0, call the in-place subtract method instead
    if( diff < 0){
        return (*this -= (-diff));
    } else {
        if( _col_major) {
            std::size_t idx = 0;
            while( diff != 0 && idx != _dims ) {
                // Go back to start of current dimension, add the difference onto diff
                _ptr -= _pos[idx] * _stride[idx];
                diff += _pos[idx];
                _pos[idx] = 0;
                // Go forward diff % shape, then divide diff by shape
                // If at the last dimension, don't wrap around
                if( idx < _dims-1 ){
                    _ptr += (diff % _shape[idx]) * _stride[idx];
                    _pos[idx] += (diff % _shape[idx]);
                } else {
                    _ptr += diff * _stride[idx];
                    _pos[idx] += diff;
                }
                diff /= _shape[idx];
                // Repeat for remaining dimensions or until diff == 0
                ++idx;
            }
        } else {
            std::size_t idx = _dims;
            while( diff != 0 && idx != 0 ) {
                // Go back to start of current dimension, add the difference onto diff
                _ptr -= _pos[idx-1] * _stride[idx-1];
                diff += _pos[idx-1];
                _pos[idx-1] = 0;
                // Go forward diff % shape, then divide diff by shape
                // If at the last dimension, don't wrap around
                if( idx > 1 ) {
                    _ptr += (diff % _shape[idx-1]) * _stride[idx-1];
                    _pos[idx-1] += (diff % _shape[idx-1]);
                } else {
                    _ptr += diff * _stride[idx-1];
                    _pos[idx-1] += diff;
                }
                diff /= _shape[idx-1];
                // Repeat for remaining dimensions or until diff == 0
                --idx;
            }
        }
        return *this;
    }
}

template<class T> template<bool constness>
typename Array<T>::base_iterator<constness>& Array<T>::base_iterator<constness>::operator-=( std::ptrdiff_t diff) {
    // If diff is less than 0, call the in-place add method instead
    if( diff < 0){
        return (*this += (-diff));
    } else {
        if( _col_major ){
            std::size_t idx = 0;
            while( diff != 0 && idx != _dims ) {
                // Go to end of current dimension, add the difference onto diff
                _ptr += (_shape[idx]-_pos[idx]) * _stride[idx];
                diff += (_shape[idx]-_pos[idx]);
                _pos[idx] = _shape[idx];
                // Go back diff % shape, then divide diff by shape
                // If at the last dimension, don't wrap around
                if( idx < _dims-1 ){
                    _ptr -= (diff % _shape[idx]) * _stride[idx];
                    _pos[idx] -= (diff % _shape[idx]);
                } else {
                    _ptr -= diff * _stride[idx];
                    _pos[idx] -= diff;
                }
                diff /= _shape[idx];
                // Repeat for remaining dimensions or until diff == 0
                ++idx;
            }
        } else {
            std::size_t idx = _dims;
            while( diff != 0 && idx != 0 ) {
                // Go to end of current dimension, add the difference onto diff
                _ptr += (_shape[idx-1]-_pos[idx-1]) * _stride[idx-1];
                diff += (_shape[idx-1]-_pos[idx-1]);
                _pos[idx-1] = _shape[idx-1];
                // Go back diff % shape, then divide diff by shape
                // If at the last dimension, don't wrap around
                if( idx > 1 ){
                    _ptr -= (diff % _shape[idx-1]) * _stride[idx-1];
                    _pos[idx-1] -= (diff % _shape[idx-1]);
                } else {
                    _ptr -= diff * _stride[idx-1];
                    _pos[idx-1] -= diff;
                }
                diff /= _shape[idx-1];
                // Repeat for remaining dimensions or until diff == 0
                --idx;
            }
        }
        return *this;
    }
}

template<class T> template<bool constness>
typename Array<T>::base_iterator<constness> Array<T>::base_iterator<constness>::operator+( typename Array<T>::base_iterator<constness>::difference_type diff) const {
    Array<T>::base_iterator<constness> it(*this);
    it += diff;
    return it;
}

template<class T> template<bool constness>
typename Array<T>::base_iterator<constness> Array<T>::base_iterator<constness>::operator-( typename Array<T>::base_iterator<constness>::difference_type diff) const {
    Array<T>::base_iterator<constness> it(*this);
    it -= diff;
    return it;
}

template<class T> template<bool constness> template<bool constness_r>
typename Array<T>::base_iterator<constness>::difference_type Array<T>::base_iterator<constness>::operator-( const typename Array<T>::base_iterator<constness_r>& it_r) const {
    // Assumes both pointers are looking at the same thing. If not, the results are undefined.
    std::ptrdiff_t distance = 0;
    std::size_t shape_cum_prod = 1;
    if( _col_major ){
        for( std::size_t ii = 0; ii != _dims; ++ii){
            distance += shape_cum_prod*(_pos[ii] - it_r._pos[ii]);
            shape_cum_prod *= _shape[ii];
        }
    } else {
        for( std::size_t ii = _dims; ii != 0; --ii){
            distance += shape_cum_prod*(_pos[ii-1] - it_r._pos[ii-1]);
            shape_cum_prod *= _shape[ii-1];
        }
    }
    return distance;
}

template<class T> template<bool constness> template<bool constness_r>
bool Array<T>::base_iterator<constness>::operator==( const typename Array<T>::base_iterator<constness_r>& it_r) const {
    return _ptr == it_r._ptr;
}

template<class T> template<bool constness> template<bool constness_r>
bool Array<T>::base_iterator<constness>::operator!=( const typename Array<T>::base_iterator<constness_r>& it_r) const {
    return _ptr != it_r._ptr;
}

template<class T> template<bool constness> template<bool constness_r>
bool Array<T>::base_iterator<constness>::operator>=( const typename Array<T>::base_iterator<constness_r>& it_r) const {
    return !(*this < it_r);
}

template<class T> template<bool constness> template<bool constness_r>
bool Array<T>::base_iterator<constness>::operator<=( const typename Array<T>::base_iterator<constness_r>& it_r) const {
    return (*this == it_r) || (*this < it_r);
}

template<class T> template<bool constness> template<bool constness_r>
bool Array<T>::base_iterator<constness>::operator>( const typename Array<T>::base_iterator<constness_r>& it_r) const {
    return !(*this <= it_r);
}

template<class T> template<bool constness> template<bool constness_r>
bool Array<T>::base_iterator<constness>::operator<( const typename Array<T>::base_iterator<constness_r>& it_r) const {
    bool result = false;
    if( _col_major ){
        for( std::size_t ii=_dims; ii>0; --ii ){
            if( _pos[ii-1] == it_r._pos[ii-1] ) continue;
            if( _pos[ii-1] < it_r._pos[ii-1] ) result = true;
            break; // leaves result=false if _pos[ii-1] > it_r._pos[ii-1]
        }
    } else {
        for( std::size_t ii=0; ii<_dims; ++ii ){
            if( _pos[ii] == it_r._pos[ii] ) continue;
            if( _pos[ii] < it_r._pos[ii] ) result = true;
            break; // leaves result=false if _pos[ii] > it_r._pos[ii]
        }
    }
    return result;
}

} // namespace
#endif
