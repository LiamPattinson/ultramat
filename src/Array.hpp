#ifndef __ARRAY_HPP
#define __ARRAY_HPP

#include <cmath>
#include <stdexcept>
#include <string>
#include <array>
#include <vector>
#include <algorithm>
#include <numeric>
#include <functional>
#include <type_traits>

// Array.hpp
// A generic multidimensional array, templated over the type it contains.
// Can be contiguous or non-contiguous, row-major or column-major.
// 
// Dynamically allocated, though unlike std::vector, it cannot be resized
// after creation.

namespace ultra {

template<class T>
class Array {

    char            _status;
    std::size_t     _dims;
    std::size_t     _size;
    std::size_t*    _shape;
    std::ptrdiff_t* _stride;
    T*              _data;
    
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
    Array& operator=( const Array& other);
    Array& operator=( Array&& other);
    
    // Additional utils
    void reset();

    // Specialised grid building methods

    //TODO
    //Array<T> view( const slice& );
    //Array<T> operator()( const slice& );
    //Array<T> broadcast()( ...something... );

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

    inline bool is_initialised() const { return _status & initialised;}
    inline bool is_initialized() const { return _status & initialised;}
    inline bool owns_data() const { return _status & own_data;}
    inline bool is_contiguous() const { return _status & contiguous;}
    inline bool is_semicontiguous() const { return _status & semicontiguous;}
    inline bool is_row_major() const { return _status & row_major;}
    inline bool is_col_major() const { return _status & col_major;}

    inline char get_status() const { return _status;}
    inline void set_status( char status){ _status = status;}

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
    // Data access

    // Access via many ints.
    // Warning: No checks are performed to ensure the correct version has been called.

private:
    template<std::size_t N, class Int, class... Ints>
    std::ptrdiff_t variadic_memjump( Int coord, Ints... coords) const;

    template<std::size_t N, class Int>
    std::ptrdiff_t variadic_memjump( Int coord) const;
public:

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

    // ===============================================
    // Iteration

    // Two types of iterator are available, covering 3 distinct iteration strategies.
    // 
    // (Semi-)Contiguous: fast_iterator
    // Non-contiguous:    iterator
    // 
    // The non-contiguous 'iterator' will work in all cases, and will be reasonably fast,
    // though it is unlikely to make use of any vectorisation.
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

};

// ===============================================
// Define custom iterators

}// temporarily close namespace to pre-declare operator overloads

template<class U, bool C>
typename ultra::Array<U>::base_fast_iterator<C> operator+( const typename ultra::Array<U>::base_fast_iterator<C>& it_l, std::ptrdiff_t diff);

template<class U, bool C>
typename ultra::Array<U>::base_fast_iterator<C> operator-( const typename ultra::Array<U>::base_fast_iterator<C>& it_l, std::ptrdiff_t diff);

template<class U, bool constness_l, bool constness_r>
std::ptrdiff_t operator-( const typename ultra::Array<U>::base_fast_iterator<constness_l>& it_l, const typename ultra::Array<U>::base_fast_iterator<constness_r>& it_r);

template<class T, bool constness_l, bool constness_r>
bool operator==( const typename ultra::Array<T>::base_fast_iterator<constness_l>& it_l, const typename ultra::Array<T>::base_fast_iterator<constness_r>& it_r);

template<class T, bool constness_l, bool constness_r>
bool operator!=( const typename ultra::Array<T>::base_fast_iterator<constness_l>& it_l, const typename ultra::Array<T>::base_fast_iterator<constness_r>& it_r);

template<class T, bool constness_l, bool constness_r>
bool operator<=( const typename ultra::Array<T>::base_fast_iterator<constness_l>& it_l, const typename ultra::Array<T>::base_fast_iterator<constness_r>& it_r);

template<class T, bool constness_l, bool constness_r>
bool operator>=( const typename ultra::Array<T>::base_fast_iterator<constness_l>& it_l, const typename ultra::Array<T>::base_fast_iterator<constness_r>& it_r);

template<class T, bool constness_l, bool constness_r>
bool operator<( const typename ultra::Array<T>::base_fast_iterator<constness_l>& it_l, const typename ultra::Array<T>::base_fast_iterator<constness_r>& it_r);

template<class T, bool constness_l, bool constness_r>
bool operator>( const typename ultra::Array<T>::base_fast_iterator<constness_l>& it_l, const typename ultra::Array<T>::base_fast_iterator<constness_r>& it_r);

template<class U, bool C>
typename ultra::Array<U>::base_iterator<C> operator+( const typename ultra::Array<U>::base_iterator<C>& it_l, std::ptrdiff_t diff);

template<class U, bool C>
typename ultra::Array<U>::base_iterator<C> operator-( const typename ultra::Array<U>::base_iterator<C>& it_l, std::ptrdiff_t diff);

template<class U, bool constness_l, bool constness_r>
std::ptrdiff_t operator-( const typename ultra::Array<U>::base_iterator<constness_l>& it_l, const typename ultra::Array<U>::base_iterator<constness_r>& it_r);

template<class T, bool constness_l, bool constness_r>
bool operator==( const typename ultra::Array<T>::base_iterator<constness_l>& it_l, const typename ultra::Array<T>::base_iterator<constness_r>& it_r);

template<class T, bool constness_l, bool constness_r>
bool operator!=( const typename ultra::Array<T>::base_iterator<constness_l>& it_l, const typename ultra::Array<T>::base_iterator<constness_r>& it_r);

template<class T, bool constness_l, bool constness_r>
bool operator<=( const typename ultra::Array<T>::base_iterator<constness_l>& it_l, const typename ultra::Array<T>::base_iterator<constness_r>& it_r);

template<class T, bool constness_l, bool constness_r>
bool operator>=( const typename ultra::Array<T>::base_iterator<constness_l>& it_l, const typename ultra::Array<T>::base_iterator<constness_r>& it_r);

template<class T, bool constness_l, bool constness_r>
bool operator<( const typename ultra::Array<T>::base_iterator<constness_l>& it_l, const typename ultra::Array<T>::base_iterator<constness_r>& it_r);

template<class T, bool constness_l, bool constness_r>
bool operator>( const typename ultra::Array<T>::base_iterator<constness_l>& it_l, const typename ultra::Array<T>::base_iterator<constness_r>& it_r);

// reopen namespace
namespace ultra {

template<class T>
template<bool constness>
class Array<T>::base_fast_iterator {

public:

    using ptr_t = std::conditional_t<constness,const T*, T*>;

private:

    ptr_t _ptr;

public:

    // ===============================================
    // Constructors
 
    base_fast_iterator( ptr_t ptr);

    template<bool other_constness>
    base_fast_iterator( const base_fast_iterator<other_constness>& other);

    template<bool other_constness>
    base_fast_iterator( base_fast_iterator<other_constness>&& other);

    template<bool other_constness>
    base_fast_iterator& operator=( const base_fast_iterator<other_constness>& other);

    template<bool other_constness>
    base_fast_iterator& operator=( base_fast_iterator<other_constness>&& other);

    // ===============================================
    // Standard iterator interface

    T operator*() const;
    
    template<bool not_const=!constness, std::enable_if_t<not_const,bool> = true>
    T& operator*();
    
    base_fast_iterator<constness>& operator++();
    base_fast_iterator<constness> operator++(int) const;

    base_fast_iterator<constness>& operator--();
    base_fast_iterator<constness> operator--(int) const;

    // ===============================================
    // Arithmetic

    base_fast_iterator<constness>& operator+=( std::ptrdiff_t diff);
    base_fast_iterator<constness>& operator-=( std::ptrdiff_t diff);

    template<class U, bool C>
    friend typename Array<U>::base_fast_iterator<C> (::operator+)( const typename Array<U>::base_fast_iterator<C>& it, std::ptrdiff_t diff);

    template<class U, bool C>
    friend typename Array<U>::base_fast_iterator<C> (::operator-)( const typename Array<U>::base_fast_iterator<C>& it, std::ptrdiff_t diff);

    template<class U, bool constness_l, bool constness_r>
    friend std::ptrdiff_t ::operator-( const typename Array<U>::base_fast_iterator<constness_l>& it_l, const typename Array<U>::base_fast_iterator<constness_r>& it_r);

    // ===============================================
    // Comparisons

    template<class U, bool constness_l, bool constness_r>
    friend bool ::operator==( const typename Array<U>::base_fast_iterator<constness_l>& it_l, const typename Array<U>::base_fast_iterator<constness_r>& it_r);

    template<class U, bool constness_l, bool constness_r>
    friend bool ::operator!=( const typename Array<U>::base_fast_iterator<constness_l>& it_l, const typename Array<U>::base_fast_iterator<constness_r>& it_r);

    template<class U, bool constness_l, bool constness_r>
    friend bool ::operator<=( const typename Array<U>::base_fast_iterator<constness_l>& it_l, const typename Array<U>::base_fast_iterator<constness_r>& it_r);

    template<class U, bool constness_l, bool constness_r>
    friend bool ::operator>=( const typename Array<U>::base_fast_iterator<constness_l>& it_l, const typename Array<U>::base_fast_iterator<constness_r>& it_r);

    template<class U, bool constness_l, bool constness_r>
    friend bool ::operator<( const typename Array<U>::base_fast_iterator<constness_l>& it_l, const typename Array<U>::base_fast_iterator<constness_r>& it_r);

    template< class U, bool constness_l, bool constness_r>
    friend bool ::operator>( const typename Array<U>::base_fast_iterator<constness_l>& it_l, const typename Array<U>::base_fast_iterator<constness_r>& it_r);
};

template<class T>
template<bool constness>
class Array<T>::base_iterator {

public:

    using ptr_t = std::conditional_t<constness, const T*, T*>;

private:

    ptr_t        _ptr;
    std::size_t  _dims;
    std::size_t* _shape;
    std::ptrdiff_t* _stride;
    std::ptrdiff_t* _pos;

public:

    // ===============================================
    // Constructors
 
    base_iterator( ptr_t ptr, std::size_t dim, std::size_t* shape, std::ptrdiff_t* stride, bool end, bool col_major);

    ~base_iterator();

    template<bool other_constness>
    base_iterator( const base_iterator<other_constness>& other);

    template<bool other_constness>
    base_iterator( base_iterator<other_constness>&& other);

    template<bool other_constness>
    base_iterator& operator=( const base_iterator<other_constness>& other);

    template<bool other_constness>
    base_iterator& operator=( base_iterator<other_constness>&& other);

    // ===============================================
    // Standard iterator interface

    T operator*() const;
    
    template<bool not_const=!constness, std::enable_if_t<not_const,bool> = true>
    T& operator*();
    
    base_iterator<constness>& operator++();
    base_iterator<constness> operator++(int) const;

    base_iterator<constness>& operator--();
    base_iterator<constness> operator--(int) const;

    // ===============================================
    // Arithmetic

    base_iterator<constness>& operator+=( std::ptrdiff_t diff);
    base_iterator<constness>& operator-=( std::ptrdiff_t diff);

    template<class U, bool C>
    friend typename Array<U>::base_iterator<C> (::operator+)( const typename Array<U>::base_iterator<C>& it, std::ptrdiff_t diff);

    template<class U, bool C>
    friend typename Array<U>::base_iterator<C> (::operator-)( const typename Array<U>::base_iterator<C>& it, std::ptrdiff_t diff);

    template<class U, bool constness_l, bool constness_r>
    friend std::ptrdiff_t ::operator-( const typename Array<U>::base_iterator<constness_l>& it_l, const typename Array<U>::base_iterator<constness_r>& it_r);

    // ===============================================
    // Comparisons

    template<class U, bool constness_l, bool constness_r>
    friend bool ::operator==( const typename Array<U>::base_iterator<constness_l>& it_l, const typename Array<U>::base_iterator<constness_r>& it_r);

    template<class U, bool constness_l, bool constness_r>
    friend bool ::operator!=( const typename Array<U>::base_iterator<constness_l>& it_l, const typename Array<U>::base_iterator<constness_r>& it_r);

    template<class U, bool constness_l, bool constness_r>
    friend bool ::operator<=( const typename Array<U>::base_iterator<constness_l>& it_l, const typename Array<U>::base_iterator<constness_r>& it_r);

    template<class U, bool constness_l, bool constness_r>
    friend bool ::operator>=( const typename Array<U>::base_iterator<constness_l>& it_l, const typename Array<U>::base_iterator<constness_r>& it_r);

    template<class U, bool constness_l, bool constness_r>
    friend bool ::operator<( const typename Array<U>::base_iterator<constness_l>& it_l, const typename Array<U>::base_iterator<constness_r>& it_r);

    template< class U, bool constness_l, bool constness_r>
    friend bool ::operator>( const typename Array<U>::base_iterator<constness_l>& it_l, const typename Array<U>::base_iterator<constness_r>& it_r);
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
            // If we own data and either: A) sizes do not match, or B) other does not own data, delete data.
            if( owns_data() && (_size != other._size || !other.owns_data()) ){
                delete[] _data;
                if( other.owns_data() ) _data = new T[other._size];
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
}

template<class T>
Array<T>& Array<T>::operator=( Array<T>&& other){
    // Delete anything currently held, take control of other.
    reset();
    _status = other._status;
    _dims = other._dims;
    _shape = other._shape;
    _stride = other._stride;
    _data = other._data;
    other.set_status(uninitialised);
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

// ===============================================
// fast_iterator

template<class T>
template<bool constness>
Array<T>::base_fast_iterator<constness>::base_fast_iterator( typename Array<T>::base_fast_iterator<constness>::ptr_t ptr ) :
    _ptr(ptr)
{}

template<class T>
template<bool constness>
template<bool other_constness>
Array<T>::base_fast_iterator<constness>::base_fast_iterator( const typename Array<T>::base_fast_iterator<other_constness>& other ) :
    _ptr(other._ptr)
{}
template<class T>
template<bool constness>
template<bool other_constness>
Array<T>::base_fast_iterator<constness>::base_fast_iterator( typename Array<T>::base_fast_iterator<other_constness>&& other) :
    _ptr(other._ptr)
{}

template<class T>
template<bool constness>
template<bool other_constness>
typename Array<T>::base_fast_iterator<constness>&
Array<T>::base_fast_iterator<constness>::operator=( const typename Array<T>::base_fast_iterator<other_constness>& other) {
    _ptr = other._ptr;
    return *this;
}

template<class T>
template<bool constness>
template<bool other_constness>
typename Array<T>::base_fast_iterator<constness>&
Array<T>::base_fast_iterator<constness>::operator=( typename Array<T>::base_fast_iterator<other_constness>&& other) {
    _ptr = other._ptr;
    return *this;
}

template<class T>
template<bool constness>
T Array<T>::base_fast_iterator<constness>::operator*() const {
    return *_ptr;
}
    
template<class T>
template<bool constness>
template<bool not_const, std::enable_if_t<not_const,bool>>
T& Array<T>::base_fast_iterator<constness>::operator*() {
    return *_ptr;
}
    
template<class T>
template<bool constness>
typename Array<T>::base_fast_iterator<constness>& Array<T>::base_fast_iterator<constness>::operator++(){
    ++_ptr;
    return *this;
}

template<class T>
template<bool constness>
typename Array<T>::base_fast_iterator<constness> Array<T>::base_fast_iterator<constness>::operator++(int) const {
    return Array<T>::base_fast_iterator<constness>(_ptr+1);
}

template<class T>
template<bool constness>
typename Array<T>::base_fast_iterator<constness>& Array<T>::base_fast_iterator<constness>::operator--(){
    --_ptr;
    return *this;
}

template<class T>
template<bool constness>
typename Array<T>::base_fast_iterator<constness> Array<T>::base_fast_iterator<constness>::operator--(int) const {
    return Array<T>::base_fast_iterator<constness>(_ptr-1);
}

template<class T>
template<bool constness>
typename Array<T>::base_fast_iterator<constness>& Array<T>::base_fast_iterator<constness>::operator+=( std::ptrdiff_t diff) {
    _ptr += diff;
    return *this;
}

template<class T>
template<bool constness>
typename Array<T>::base_fast_iterator<constness>& Array<T>::base_fast_iterator<constness>::operator-=( std::ptrdiff_t diff) {
    _ptr -= diff;
    return *this;
}

} // namespace

template<class U, bool C>
typename ultra::Array<U>::base_fast_iterator<C> operator+( const typename ultra::Array<U>::base_fast_iterator<C>& it, std::ptrdiff_t diff) {
    return ultra::Array<U>::base_fast_iterator<C>( it._ptr + diff);
}

template<class U, bool C>
typename ultra::Array<U>::base_fast_iterator<C> operator-( const typename ultra::Array<U>::base_fast_iterator<C>& it, std::ptrdiff_t diff) {
    return ultra::Array<U>::base_fast_iterator<C>( it._ptr - diff);
}

template<class U, bool constness_l, bool constness_r>
std::ptrdiff_t operator-( const typename ultra::Array<U>::base_fast_iterator<constness_l>& it_l, const typename ultra::Array<U>::base_fast_iterator<constness_r>& it_r) {
    return it_l._ptr - it_r._ptr;
}

template<class T, bool constness_l, bool constness_r>
bool operator==( const typename ultra::Array<T>::base_fast_iterator<constness_l>& it_l, const typename ultra::Array<T>::base_fast_iterator<constness_r>& it_r) {
    return it_l._ptr == it_r._ptr;
}

template<class T, bool constness_l, bool constness_r>
bool operator!=( const typename ultra::Array<T>::base_fast_iterator<constness_l>& it_l, const typename ultra::Array<T>::base_fast_iterator<constness_r>& it_r) {
    return it_l._ptr != it_r._ptr;
}

template<class T, bool constness_l, bool constness_r>
bool operator<=( const typename ultra::Array<T>::base_fast_iterator<constness_l>& it_l, const typename ultra::Array<T>::base_fast_iterator<constness_r>& it_r) {
    return it_l._ptr <= it_r._ptr;
}

template<class T, bool constness_l, bool constness_r>
bool operator>=( const typename ultra::Array<T>::base_fast_iterator<constness_l>& it_l, const typename ultra::Array<T>::base_fast_iterator<constness_r>& it_r) {
    return it_l._ptr >= it_r._ptr;
}

template<class T, bool constness_l, bool constness_r>
bool operator<( const typename ultra::Array<T>::base_fast_iterator<constness_l>& it_l, const typename ultra::Array<T>::base_fast_iterator<constness_r>& it_r) {
    return it_l._ptr < it_r._ptr;
}

template<class T, bool constness_l, bool constness_r>
bool operator>( const typename ultra::Array<T>::base_fast_iterator<constness_l>& it_l, const typename ultra::Array<T>::base_fast_iterator<constness_r>& it_r) {
    return it_l._ptr > it_r._ptr;
}

// ===============================================
// iterator

namespace ultra {

template<class T>
template<bool constness>
Array<T>::base_iterator<constness>::base_iterator( typename Array<T>::base_iterator<constness>::ptr_t ptr, std::size_t dims, std::size_t* shape, std::ptrdiff_t* stride, bool end, bool col_major) :
    _ptr(ptr),
    _dims(dims),
    _shape(new std::size_t[dims]),
    _stride(new std::ptrdiff_t[dims]),
    _pos(new std::ptrdiff_t[dims])
{
    for( std::size_t ii=0; ii<_dims; ++ii) _pos[ii] = 0;

    // if col major, we need to reverse shape and stride so that it looks row major.
    if( col_major ){
        std::reverse_copy(shape,shape+dims,_shape);
        std::reverse_copy(stride,stride+dims,_stride);
    } else {
        std::copy(shape,shape+dims,_shape);
        std::copy(stride,stride+dims,_stride);
    }

    // If this is an 'end' iterator, pos should be zero in all dimensions except the slowest incrementing.
    // Since this will look like row major (first dimension is slowest), this means pos[0];
    if(end) _pos[0] = _shape[0]+1;
}

template<class T>
template<bool constness>
Array<T>::base_iterator<constness>::~base_iterator() {
    if( _ptr != nullptr ){
        delete[] _shape;
        delete[] _stride;
        delete[] _pos;
    }
}

template<class T>
template<bool constness>
template<bool other_constness>
Array<T>::base_iterator<constness>::base_iterator( const typename Array<T>::base_iterator<other_constness>& other ) :
    _ptr(other._ptr),
    _dims(other._dims),
    _shape(new std::size_t[other._dims]),
    _stride(new std::ptrdiff_t[other._dims]),
    _pos(new std::ptrdiff_t[other._dims])
{
    std::copy( other._shape, other._shape+_dims, _shape);
    std::copy( other._stride, other._stride+_dims, _stride);
    std::copy( other._pos, other._pos+_dims, _pos);
}
template<class T>
template<bool constness>
template<bool other_constness>
Array<T>::base_iterator<constness>::base_iterator( typename Array<T>::base_iterator<other_constness>&& other) :
    _ptr(other._ptr),
    _dims(other._dims),
    _shape(other._shape),
    _stride(other._stride),
    _pos(other._pos)
{
    // invalidate other so it won't delete[] shape/stride/pos.
    other._ptr = nullptr;
}

template<class T>
template<bool constness>
template<bool other_constness>
typename Array<T>::base_iterator<constness>&
Array<T>::base_iterator<constness>::operator=( const typename Array<T>::base_iterator<other_constness>& other) {
    // if other has different dims, need to reallocate internals
    if( _dims != other._dims ){
        _dims = other._dims;
        if( _ptr != nullptr ){
            delete[] _shape;
            delete[] _stride;
            delete[] _pos;
        }
        _shape = new std::size_t[_dims];
        _stride = new std::ptrdiff_t[_dims];
        _pos = new std::ptrdiff_t[_dims];
    }
    _ptr = other._ptr;
    std::copy( other._shape, other._shape+_dims, _shape);
    std::copy( other._stride, other._stride+_dims, _stride);
    std::copy( other._pos, other._pos+_dims, _pos);
    return *this;
}

template<class T>
template<bool constness>
template<bool other_constness>
typename Array<T>::base_iterator<constness>&
Array<T>::base_iterator<constness>::operator=( typename Array<T>::base_iterator<other_constness>&& other) {
    _ptr = other._ptr;
    _dims = other._dims;
    _shape = other._shape;
    _stride = other._stride;
    _pos = other._pos;
    // invalidate other so it won't delete[] shape/stride/pos.
    other._ptr = nullptr;
    return *this;
}

template<class T>
template<bool constness>
T Array<T>::base_iterator<constness>::operator*() const {
    return *_ptr;
}
    
template<class T>
template<bool constness>
template<bool not_const, std::enable_if_t<not_const,bool>>
T& Array<T>::base_iterator<constness>::operator*() {
    return *_ptr;
}
    
template<class T>
template<bool constness>
typename Array<T>::base_iterator<constness>& Array<T>::base_iterator<constness>::operator++(){
    std::size_t idx = _dims-1;
    _ptr += _stride[idx];
    ++_pos[idx];
    while( _pos[idx] == _shape[idx] && idx != 0 ){
        // Go back to start of current dimension
        _ptr -= _stride[idx] * _shape[idx];
        _pos[idx]=0;
        // Increment one in next dimension
        _ptr += _stride[idx-1];
        ++_pos[idx-1];
        // Repeat for remaining dimensions
        --idx;
    }
    return *this;
}

template<class T>
template<bool constness>
typename Array<T>::base_iterator<constness> Array<T>::base_iterator<constness>::operator++(int) const {
    return ++Array<T>::base_iterator<constness>(*this);
}

template<class T>
template<bool constness>
typename Array<T>::base_iterator<constness>& Array<T>::base_iterator<constness>::operator--(){
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
    return *this;
}

template<class T>
template<bool constness>
typename Array<T>::base_iterator<constness> Array<T>::base_iterator<constness>::operator--(int) const {
    return --Array<T>::base_iterator<constness>(*this);
}

template<class T>
template<bool constness>
typename Array<T>::base_iterator<constness>& Array<T>::base_iterator<constness>::operator+=( std::ptrdiff_t diff) {
    // If diff is less than 0, call the in-place subtract method instead
    if( diff < 0){
        return (*this -= (-diff));
    } else {
        std::size_t idx = _dims-1;
        do{
            // Go back to start of current dimension, add the difference onto diff
            _ptr -= _pos[idx] * _stride[idx];
            diff += _pos[idx];
            // Go forward diff % shape, then divide diff by shape
            _ptr += (diff % _shape[idx]) * _stride[idx] ;
            _pos[idx] += (diff % _shape[idx]);
            diff /= _shape[idx];
            // Repeat for remaining dimensions until diff == 0 or idx=0
            --idx;
        } while( diff != 0 && idx != 0 );
        return *this;
    }
}

template<class T>
template<bool constness>
typename Array<T>::base_iterator<constness>& Array<T>::base_iterator<constness>::operator-=( std::ptrdiff_t diff) {
    // If diff is less than 0, call the in-place add method instead
    if( diff < 0){
        return (*this += (-diff));
    } else {
        std::size_t idx = _dims-1;
        do {
            // Go to end of current dimension, add the difference onto diff
            _ptr -= (_shape[idx]-_pos[idx]) * _stride[idx];
            diff += (_shape[idx]-_pos[idx]);
            // Go back diff % shape, then divide diff by shape
            _ptr -= (diff % _shape[idx]) * _stride[idx] ;
            _pos[idx] -= (diff % _shape[idx]);
            diff /= _shape[idx];
            // Repeat for remaining dimensions until diff == 0 or idx ==0
            --idx;
        } while( diff != 0 && idx != 0 );
        return *this;
    }
}

} // namespace

template<class U, bool C>
typename ultra::Array<U>::base_iterator<C> operator+( const typename ultra::Array<U>::base_iterator<C>& it, std::ptrdiff_t diff) {
    typename ultra::Array<U>::base_iterator<C> result(it);
    result += diff;
    return result;
}

template<class U, bool C>
typename ultra::Array<U>::base_iterator<C> operator-( const typename ultra::Array<U>::base_iterator<C>& it, std::ptrdiff_t diff) {
    typename ultra::Array<U>::base_iterator<C> result(it);
    result -= diff;
    return result;
}

template<class U, bool constness_l, bool constness_r>
std::ptrdiff_t operator-( const typename ultra::Array<U>::base_iterator<constness_l>& it_l, const typename ultra::Array<U>::base_iterator<constness_r>& it_r) {
    // Assumes both pointers are looking at the same thing. If not, the results are undefined.
    std::ptrdiff_t distance = 0;
    std::size_t shape_cum_prod = 1;
    for( std::size_t ii = it_l._dims; ii != 0; ++ii){
        distance += shape_cum_prod*(it_l._pos[ii-1] - it_r._pos[ii-1]);
        shape_cum_prod *= it_l._shape[ii-1];
    }
    return distance;
}

template<class T, bool constness_l, bool constness_r>
bool operator==( const typename ultra::Array<T>::base_iterator<constness_l>& it_l, const typename ultra::Array<T>::base_iterator<constness_r>& it_r) {
    return it_l._ptr == it_r._ptr;
}

template<class T, bool constness_l, bool constness_r>
bool operator!=( const typename ultra::Array<T>::base_iterator<constness_l>& it_l, const typename ultra::Array<T>::base_iterator<constness_r>& it_r) {
    return it_l._ptr != it_r._ptr;
}

template<class T, bool constness_l, bool constness_r>
bool operator<=( const typename ultra::Array<T>::base_iterator<constness_l>& it_l, const typename ultra::Array<T>::base_iterator<constness_r>& it_r) {
    return (it_l - it_r) <= 0;
}

template<class T, bool constness_l, bool constness_r>
bool operator>=( const typename ultra::Array<T>::base_iterator<constness_l>& it_l, const typename ultra::Array<T>::base_iterator<constness_r>& it_r) {
    return (it_l - it_r) >= 0;
}

template<class T, bool constness_l, bool constness_r>
bool operator<( const typename ultra::Array<T>::base_iterator<constness_l>& it_l, const typename ultra::Array<T>::base_iterator<constness_r>& it_r) {
    return (it_l - it_r) < 0;
}

template<class T, bool constness_l, bool constness_r>
bool operator>( const typename ultra::Array<T>::base_iterator<constness_l>& it_l, const typename ultra::Array<T>::base_iterator<constness_r>& it_r) {
    return (it_l - it_r._ptr) > 0;
}

#endif
