#ifndef __ULTRA_DENSE_VIEW_HPP
#define __ULTRA_DENSE_VIEW_HPP

// DenseView.hpp
//
// A container that refers to data belonging to some other dense container.
// A view does not control the lifetime of its contents. It may be
// writeable or read-only, contiguous or non-contiguous.
// Copies/moves in constant time.

#include "DenseBase.hpp"

namespace ultra {

template<class T, ReadWriteStatus ReadWrite>
class DenseView : public Expression<DenseView<T,ReadWrite>>, public DenseBase<DenseView<T,ReadWrite>,T::rc_order> {

    friend DenseBase<DenseView<T,ReadWrite>,T::rc_order>;

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

    DenseView() = delete;
    DenseView( const DenseView& ) = default;
    DenseView( DenseView&& ) = default;
    DenseView& operator=( const DenseView& ) = default;
    DenseView& operator=( DenseView&& ) = default;

    // Full view of a container
    DenseView( T& container) :
        _size( container.size() ),
        _shape( container.dims() ),
        _stride( container.dims()+1 ),
        _data( container.data() )
    {
        std::ranges::copy(container.shape(),_shape.begin());
        std::ranges::copy(container.stride(),_stride.begin());
    }

    // Assign from Expression
    // (cannot in general construct from an expression, as no data pointer is available)
    
    template<class U>
    DenseView& operator=( const Expression<U>& expression) {
        check_expression(*this,expression);
        auto expr=expression.begin();
        for(auto it=begin(), it_end=end(); it != it_end; ++it, ++expr) *it = *expr;
        return *this;
    }

    // ===============================================
    // Pull methods from base
    // Some methods are shadowed, as the default behaviour is not appropriate

    using DenseBase<DenseView<T,ReadWrite>,T::rc_order>::dims;
    using DenseBase<DenseView<T,ReadWrite>,T::rc_order>::shape;
    using DenseBase<DenseView<T,ReadWrite>,T::rc_order>::stride;
    using DenseBase<DenseView<T,ReadWrite>,T::rc_order>::fill;
    using DenseBase<DenseView<T,ReadWrite>,T::rc_order>::operator();
    using DenseBase<DenseView<T,ReadWrite>,T::rc_order>::reshape;
    using DenseBase<DenseView<T,ReadWrite>,T::rc_order>::check_expression;

    std::size_t size() const noexcept { return _size;}
    pointer data() const noexcept { return _data; }

    bool is_contiguous() const noexcept requires (rc_order==RCOrder::row_major) {
        ptrdiff_t stride = 1;
        if( stride != _stride[dims()]) return false;
        for( std::size_t ii=dims(); ii != 0; --ii){
            stride *= _shape[ii-1];
            if( stride != _stride[ii-1]) return false;
        }
        return true;
    }

    bool is_contiguous() const noexcept requires (rc_order==RCOrder::col_major) {
        ptrdiff_t stride = 1;
        if( stride != _stride[0]) return false;
        for( std::size_t ii=0; ii != dims(); ++ii){
            stride *= _shape[ii];
            if( stride != _stride[ii+1]) return false;
        }
        return true;
    }

    // Access via square brackets
    // Treats all arrays as 1D containers.
    // TODO have this take a slice and return a new view
    //T operator[](std::size_t ii) const;
    //T& operator[](std::size_t ii);

    // ===============================================
    // View within a View
    // (inception noises)
    
    DenseView view() const noexcept {
        return *this;
    }

    template<class... Slices> requires ( std::is_same<Slices,Slice>::value && ... )
    DenseView view( const Slices&... slices) const {
        return slice(slices...);
    }

    // ===============================================
    // Iteration

    template<bool constness> class iterator_impl;
    
    using iterator = iterator_impl<false>;
    using const_iterator = iterator_impl<true>;
    
    iterator begin() {
        return iterator(data(),_shape,_stride);
    }
    
    const_iterator begin() const {
        return const_iterator(data(),_shape,_stride);
    }
    
    iterator end() {
        return iterator(data() + _stride[rc_order==RCOrder::col_major? dims() : 0],_shape,_stride,true);
    }
    
    const_iterator end() const {
        return const_iterator(data() + _stride[rc_order==RCOrder::col_major? dims() : 0],_shape,_stride,true);
    }
    
    // ===============================================
    // Special view methods

    template<class... Slices> requires ( std::is_same<Slices,Slice>::value && ... )
    DenseView slice( const Slices&... var_slices) const {
        std::array<Slice,sizeof...(Slices)> slices = {{ var_slices... }};
        // Create copy to work with
        DenseView result(*this);
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

template<class T,ReadWriteStatus ReadWrite>
template<bool constness>
class DenseView<T,ReadWrite>::iterator_impl {
    
    friend typename DenseView<T>::iterator_impl<!constness>;

public:

    using iterator_category = std::random_access_iterator_tag;
    using difference_type   = std::ptrdiff_t;
    using value_type        = DenseView<T>::value_type;
    using shape_type        = DenseView<T>::shape_type;
    using stride_type       = DenseView<T>::stride_type;
    using pointer           = DenseView<T>::pointer;
    using reference         = DenseView<T>::reference;
    static constexpr RCOrder rc_order = DenseView<T>::rc_order;

private:

    pointer            _ptr;
    const shape_type&  _shape;
    const stride_type& _stride;
    stride_type        _pos;

public:

    // ===============================================
    // Constructors
 
    iterator_impl() = delete;
    iterator_impl( const iterator_impl<constness>& other) = default;
    iterator_impl( iterator_impl<constness>&& other) = default;
    iterator_impl& operator=( const iterator_impl<constness>& other) = default;
    iterator_impl& operator=( iterator_impl<constness>&& other) = default;

    iterator_impl( pointer ptr, const shape_type& shape, const stride_type& stride, bool end = false) :
        _ptr(ptr),
        _shape(shape),
        _stride(stride),
        _pos(stride.size(),0)
    {
        _pos[rc_order==RCOrder::row_major ? stride.size()-1 : 0] = end;
    }

    // Construct with explicit pos. Used to convert between iterator types, not recommended otherwise
    iterator_impl( pointer ptr, const shape_type& shape, const stride_type& stride, const stride_type& pos) :
        _ptr(ptr),
        _shape(shape),
        _stride(stride),
        _pos(pos)
    {}

    // ===============================================
    // Conversion from non-const to const

    operator iterator_impl<!constness>() const requires (!constness) {
        return iterator_impl<!constness>(_ptr,_shape,_stride,_pos);
    }

    // ===============================================
    // Standard iterator interface

    // Dereference
    reference operator*() const { return *_ptr; }
    
    // Increment/decrement
    iterator_impl<constness>& operator++() requires ( rc_order == RCOrder::col_major ) {
        std::size_t idx = 0;
        _ptr += _stride[idx];
        ++_pos[idx];
        while( idx != _shape.size() && _pos[idx] == static_cast<std::ptrdiff_t>(_shape[idx])){
            // Go back to start of current dimension
            _ptr -= _stride[idx] * _shape[idx];
            _pos[idx]=0;
            // Increment one in next dimension
            _ptr += _stride[idx+1];
            ++_pos[idx+1];
            // Repeat for remaining dimensions
            ++idx;
        }
        return *this;
    }

    iterator_impl<constness>& operator++() requires ( rc_order == RCOrder::row_major ) {
        std::size_t idx = _shape.size();
        _ptr += _stride[idx];
        ++_pos[idx];
        while( idx != 0 && _pos[idx] == static_cast<std::ptrdiff_t>(_shape[idx-1])){
            // Go back to start of current dimension
            _ptr -= _stride[idx] * _shape[idx-1];
            _pos[idx]=0;
            // Increment one in next dimension
            _ptr += _stride[idx-1];
            ++_pos[idx-1];
            // Repeat for remaining dimensions
            --idx;
        }
        return *this;
    }
    
    iterator_impl<constness>& operator--() requires ( rc_order == RCOrder::col_major ) {
        std::size_t idx = 0;
        _ptr -= _stride[idx];
        --_pos[idx];
        while( idx != _shape.size() && _pos[idx] == -1){
            // Go to end of current dimension
            _ptr += _stride[idx] * _shape[idx];
            _pos[idx] = _shape[idx]-1;
            // Decrement one in next dimension
            _ptr -= _stride[idx+1];
            --_pos[idx+1];
            // Repeat for remaining dimensions
            ++idx;
        }
        return *this; 
    }

    iterator_impl<constness>& operator--() requires ( rc_order == RCOrder::row_major ) {
        std::size_t idx = _shape.size();
        _ptr -= _stride[idx];
        --_pos[idx];
        while( idx != 0 && _pos[idx] == -1 ){
            // Go to end of current dimension
            _ptr += _stride[idx] * _shape[idx-1];
            _pos[idx] = _shape[idx-1]-1;
            // Decrement one in next dimension
            _ptr -= _stride[idx-1];
            --_pos[idx-1];
            // Repeat for remaining dimensions
            --idx;
        }
        return *this;
    }

    iterator_impl<constness> operator++(int) const {
        return ++iterator_impl<constness>(*this);
    }
    
    iterator_impl<constness> operator--(int) const {
        return --iterator_impl<constness>(*this);
    }

    // Random-access
    iterator_impl<constness>& operator+=( difference_type diff) requires ( rc_order == RCOrder::col_major ) {
        // If diff is less than 0, call the in-place subtract method instead
        if( diff < 0){
            return (*this -= (-diff));
        } else {
            std::size_t idx = 0;
            while( diff != 0 && idx != _shape.size() ) {
                // Go back to start of current dimension, add the difference onto diff
                _ptr -= _pos[idx] * _stride[idx];
                diff += _pos[idx];
                _pos[idx] = 0;
                _ptr += (diff % _shape[idx]) * _stride[idx];
                _pos[idx] += (diff % _shape[idx]);
                diff /= _shape[idx];
                // Repeat for remaining dimensions or until diff == 0
                ++idx;
            }
            return *this;
        }
    }

    iterator_impl<constness>& operator+=( difference_type diff) requires ( rc_order == RCOrder::row_major ) {
        // If diff is less than 0, call the in-place subtract method instead
        if( diff < 0){
            return (*this -= (-diff));
        } else {
            std::size_t idx = _shape.size();
            while( diff != 0 && idx != 0 ) {
                // Go back to start of current dimension, add the difference onto diff
                _ptr -= _pos[idx] * _stride[idx];
                diff += _pos[idx];
                _pos[idx] = 0;
                // Go forward diff % shape, then divide diff by shape
                _ptr += (diff % _shape[idx-1]) * _stride[idx];
                _pos[idx] += (diff % _shape[idx-1]);
                diff /= _shape[idx-1];
                // Repeat for remaining dimensions or until diff == 0
                --idx;
            }
            return *this;
        }
    }

    iterator_impl<constness>& operator-=( difference_type diff) requires (rc_order == RCOrder::col_major ) {
        // If diff is less than 0, call the in-place add method instead
        if( diff < 0){
            return (*this += (-diff));
        } else {
            std::size_t idx = 0;
            while( diff != 0 && idx != _shape.size() ) {
                // Go to end of current dimension, add the difference onto diff
                _ptr += (_shape[idx]-_pos[idx]) * _stride[idx];
                diff += (_shape[idx]-_pos[idx]);
                _pos[idx] = _shape[idx];
                // Go back diff % shape, then divide diff by shape
                _ptr -= (diff % _shape[idx]) * _stride[idx];
                _pos[idx] -= (diff % _shape[idx]);
                diff /= _shape[idx];
                // Repeat for remaining dimensions or until diff == 0
                ++idx;
            }
            return *this;
        }
    }

    iterator_impl<constness>& operator-=( difference_type diff) requires (rc_order == RCOrder::row_major ) {
        // If diff is less than 0, call the in-place add method instead
        if( diff < 0){
            return (*this += (-diff));
        } else {
            std::size_t idx = _shape.size();
            while( diff != 0 && idx != 0 ) {
                // Go to end of current dimension, add the difference onto diff
                _ptr += (_shape[idx-1]-_pos[idx]) * _stride[idx];
                diff += (_shape[idx-1]-_pos[idx]);
                _pos[idx] = _shape[idx-1];
                // Go back diff % shape, then divide diff by shape
                _ptr -= (diff % _shape[idx-1]) * _stride[idx];
                _pos[idx] -= (diff % _shape[idx-1]);
                diff /= _shape[idx-1];
                // Repeat for remaining dimensions or until diff == 0
                --idx;
            }
            return *this;
        }
    }

    
    iterator_impl<constness> operator+( difference_type diff) const {
       iterator_impl<constness> result(*this);
       result += diff;
       return result;
    }

    iterator_impl<constness> operator-( difference_type diff) const {
       iterator_impl<constness> result(*this);
       result -= diff;
       return result;
    }

    // Distance
    template<bool constness_r>
    std::ptrdiff_t operator-( const iterator_impl<constness_r>& it_r) const {
        // Assumes both pointers are looking at the same thing. If not, the results are undefined.
        std::ptrdiff_t distance = 0;
        std::size_t shape_cum_prod = 1;
        if( rc_order == RCOrder::col_major ){
            for( std::size_t ii = 0; ii != _shape.size(); ++ii){
                distance += shape_cum_prod*(_pos[ii] - it_r._pos[ii]);
                shape_cum_prod *= _shape[ii];
            }
            distance += shape_cum_prod*(_pos[_shape.size()] - it_r._pos[_shape.size()]);
        } else {
            for( std::size_t ii = _shape.size(); ii != 0; --ii){
                distance += shape_cum_prod*(_pos[ii] - it_r._pos[ii]);
                shape_cum_prod *= _shape[ii-1];
            }
            distance += shape_cum_prod*(_pos[0] - it_r._pos[0]);
        }
        return distance;
    }

    // Boolean comparisons
    template<bool constness_r>
    bool operator==( const iterator_impl<constness_r>& it_r) const {
        return _ptr == it_r._ptr;
    }

    template<bool constness_r>
    auto operator<=>( const iterator_impl<constness_r>& it_r) const requires ( rc_order == RCOrder::row_major ) {
        for( std::size_t ii=0; ii<_pos.size(); ++ii ){
            if( _pos[ii] == it_r._pos[ii] ) continue;
            return ( _pos[ii] < it_r._pos[ii] ? std::strong_ordering::less : std::strong_ordering::greater );
        }
        return std::strong_ordering::equal;
    }

    template<bool constness_r>
    auto operator<=>( const iterator_impl<constness_r>& it_r) const requires ( rc_order == RCOrder::col_major ) {
        for( int ii=_pos.size()-1; ii>=0; --ii ){
            if( _pos[ii] == it_r._pos[ii] ) continue;
            return ( _pos[ii] < it_r._pos[ii] ? std::strong_ordering::less : std::strong_ordering::greater );
        }
        return std::strong_ordering::equal;
    }
};


} // namespace
#endif
