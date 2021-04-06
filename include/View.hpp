#ifndef __ULTRA_VIEW_HPP
#define __ULTRA_VIEW_HPP

// View.hpp
//
// A container that refers to data belonging to some other container.
// A view does not control the lifetime of its contents. It may be
// writeable or read-only, contiguous or non-contiguous.
// Copies/moves in constant time.

#include "Expression.hpp"

namespace ultra {

template<class T, ReadWriteStatus ReadWrite = ReadWriteStatus::writeable>
class View : public Expression<View<T,ReadWrite>> {

public:

    using value_type = typename T::value_type;
    using difference_type = std::ptrdiff_t;
    using pointer = std::conditional_t<ReadWrite==ReadWriteStatus::writeable,value_type*, const value_type*>;
    using reference = std::conditional_t<ReadWrite==ReadWriteStatus::writeable,value_type&,const value_type&>;
    using shape_type = std::vector<std::size_t>;
    using stride_type = std::vector<std::ptrdiff_t>;
    static constexpr RCOrder rc_order = T::rc_order;

protected:

    std::size_t  _size;
    shape_type   _shape;
    stride_type  _stride;
    pointer      _data;

public:

    // ===============================================
    // Constructors

    View() = delete;
    View( const View& ) = default;
    View( View&& ) = default;
    View& operator=( const View& ) = default;
    View& operator=( View&& ) = default;

    // Full view of a container
    View( T& container) :
        _size( container.size() ),
        _shape( container.dims() ),
        _stride( container.dims()+1 ),
        _data( container.data() )
    {
        std::ranges::copy(container.shape(),_shape.begin());
        std::ranges::copy(container.stride(),_stride.begin());
    }

    // Manual view creation
    // Recommended interface for reshape/transpose/slices
    template<std::ranges::range Shape, std::ranges::range Stride>
    requires std::unsigned_integral<typename Shape::value_type> && std::integral<typename Stride::value_type>
    View( value_type* data, const Shape& shape, const Stride& stride, char status) :
        _size(std::accumulate(shape.begin(),shape.end(),1,std::multiplies<std::size_t>{})),
        _shape(shape.size()),
        _stride(stride.size()),
        _data(data)
    {
        std::ranges::copy(shape,_shape.begin());
        std::ranges::copy(stride,_stride.begin());
    }

    // Assign from Expression
    // (cannot in general construct from an expression, as no data pointer is available)
    
    template<class U>
    View& operator=( const Expression<U>& expression) {
        check_expression(*this,expression);
        auto expr=expression.begin();
        for(auto it=begin(), it_end=end(); it != it_end; ++it, ++expr) *it = *expr;
        return *this;
    }

    // ===============================================
    // Attributes

    std::size_t dims() const { return _shape.size();}
    std::size_t size() const { return _size;}
    std::size_t shape( std::size_t dim) const { return _shape[dim];}
    std::ptrdiff_t stride( std::size_t dim) const { return _stride[dim];}
    const shape_type&  shape() const { return _shape;}
    const stride_type& stride() const { return _stride;}

    // ===============================================
    // Data access

    // return raw pointer -- use with care!
    pointer data() const { return _data; }

    // Access via ints.
    // Warning: No compile-time checks are performed to ensure the correct version has been called.

    template<std::integral... Ints> 
    value_type operator()( Ints... coords ) const {
        return _data[variadic_memjump<rc_order>(_stride,coords...)];
    }
    
    template<std::integral... Ints>
    requires (ReadWrite == ReadWriteStatus::writeable)
    reference operator()( Ints... coords ) {
        return _data[variadic_memjump<rc_order>(_stride,coords...)];
    }

    // Access via anything that looks like a std::vector

    template<std::ranges::range Coords> requires std::integral<typename Coords::value_type>
    value_type operator()( const Coords& coords) const {
        return _data[std::inner_product(coords.begin(),coords.end(),_stride.begin()+(rc_order==RCOrder::row_major),0)];
    }

    template<std::ranges::range Coords>
    requires std::integral<typename Coords::value_type> && (ReadWrite == ReadWriteStatus::writeable)
    reference operator()( const Coords& coords) {
        return _data[std::inner_product(coords.begin(),coords.end(),_stride.begin()+(rc_order==RCOrder::row_major),0)];
    }

    // Access via square brackets
    // Treats all arrays as 1D containers.
    // TODO have this take a slice and return a new view

    T operator[](std::size_t ii) const;
    T& operator[](std::size_t ii);

    // ===============================================
    // Iteration

    template<bool constness>
    class iterator_impl;
    
    using iterator = iterator_impl<false>;
    using const_iterator = iterator_impl<true>;
    
    iterator begin();
    iterator end();
    const_iterator begin() const;
    const_iterator end() const;
    
    // ===============================================
    // Special view methods

    template<class... Slices> requires ( std::is_same<Slices,Slice>::value && ... )
    View slice( const Slices&... var_slices) {
        std::array<Slice,sizeof...(Slices)> slices = {{ var_slices... }};
        // Create copy to work with
        View result(*this);
        std::size_t stride_offset = ( rc_order == RCOrder::row_major );
        for( std::size_t ii=0; ii<dims(); ++ii){
            // if not enough slices provided, assume start=all, end=all, step=1
            Slice slice = ( ii < slices.size() ? slices[ii] : Slice{Slice::all,Slice::all,1});
            // Account for negative start/end
            if( slice.start < 0 ) slice.start = _shape[ii] + slice.start;
            if( slice.end < 0 ) slice.end = _shape[ii] + slice.end;
            // Account for 'all' specifiers
            if( slice.start == Slice::all ) slice.start = 0;
            if( slice.end == Slice::all ) slice.end = _shape[ii];
            // Throw exceptions if slice is impossible
            if( slice.start < 0 || slice.end > static_cast<std::ptrdiff_t>(_shape[ii])) throw std::runtime_error("Ultramat: Slice out of bounds.");
            if( slice.end <= slice.start ) throw std::runtime_error("Ultramat: Slice end is less than or equal to start.");
            if( slice.step == 0 ) throw std::runtime_error("UltraArray: Slice has zero step.");
            // Account for the case of step size larger than shape
            if( slice.end - slice.start < std::abs(slice.step) ) slice.step = (slice.end - slice.start) * (slice.step < 0 ? -1 : 1);
            // Set shape and stride of result. Shape is (slice.end-slice.start)/std::abs(slice.step), but rounding up rather than down.
            result._shape[ii] = (slice.end - slice.start + ((slice.end-slice.start)%std::abs(slice.step)))/std::abs(slice.step);
            result._stride[ii+stride_offset] = _stride[ii+stride_offset]*slice.step;
            // Move data to start of slice (be sure to use this stride rather than result stride)
            if( slice.step > 0 ){
                result._data += slice.start * _stride[ii+stride_offset];
            } else {
                result._data += (slice.end-1) * _stride[ii+stride_offset];
            }
        }
        // Set remaining info and return
        result._size = std::accumulate( result._shape.begin(), result._shape.end(), 1, std::multiplies<std::size_t>());
        if( rc_order == RCOrder::row_major ){
            result._stride[0] = result._stride[1] * result._shape[0];
        } else {
            result._stride[dims()] = result._stride[dims()-1] * result._shape[dims()-1];
        }
        return result;
    }
};

// Define view iterator

/* TODO. So far copied from Array with minor edits.
template<class T>
template<bool constness>
class View<T>::iterator_impl {
    
    friend typename View<T>::iterator_impl<!constness>;

public:

    using iterator_category = std::random_access_iterator_tag;
    using difference_type   = std::ptrdiff_t;
    using value_type        = View<T>::value_type;
    using pointer           = View<T>::pointer;
    using reference         = View<T>::reference;
    static constexpr RCOrder rc_order = View<T>::rc_order;

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
 
    iterator_impl( pointer ptr, , bool end);
    iterator_impl( const iterator_impl<constness>& other);
    iterator_impl( iterator_impl<constness>&& other);
    iterator_impl& operator=( const iterator_impl<constness>& other);
    iterator_impl& operator=( iterator_impl<constness>&& other);

    // ===============================================
    // Conversion from non-const to const

    template<bool C=!constness, std::enable_if_t<C,bool> = true>
    operator iterator_impl<C>() const;

    // ===============================================
    // Standard iterator interface

    // Dereference
    reference operator*();
    
    // Increment/decrement
    iterator_impl<constness>& operator++();
    iterator_impl<constness> operator++(int) const;

    iterator_impl<constness>& operator--();
    iterator_impl<constness> operator--(int) const;

    // Random-access
    iterator_impl<constness>& operator+=( difference_type diff);
    iterator_impl<constness>& operator-=( difference_type diff);
    iterator_impl<constness> operator+( difference_type diff) const;
    iterator_impl<constness> operator-( difference_type diff) const;

    // Distance
    template<bool constness_r> difference_type operator-( const iterator_impl<constness_r>& it_r) const;

    // Boolean comparisons
    template<bool constness_r> bool operator==( const iterator_impl<constness_r>& it_r) const;
    template<bool constness_r> bool operator!=( const iterator_impl<constness_r>& it_r) const;
    template<bool constness_r> bool operator>=( const iterator_impl<constness_r>& it_r) const;
    template<bool constness_r> bool operator<=( const iterator_impl<constness_r>& it_r) const;
    template<bool constness_r> bool operator<( const iterator_impl<constness_r>& it_r) const;
    template<bool constness_r> bool operator>( const iterator_impl<constness_r>& it_r) const;
};
*/
} // namespace
#endif
