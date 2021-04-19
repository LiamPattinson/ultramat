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

template<class T, ReadWrite rw>
class DenseView : public DenseExpression<DenseView<T,rw>>, public DenseBase<DenseView<T,rw>,T::rc_order> {

    friend DenseBase<DenseView<T,rw>,T::rc_order>;

    static constexpr ReadWrite other_rw = (rw == ReadWrite::read_only ? ReadWrite::writeable : ReadWrite::read_only);
    friend DenseView<T,other_rw>;

public:

    using value_type = typename T::value_type;
    using difference_type = std::ptrdiff_t;
    using pointer = std::conditional_t<rw==ReadWrite::writeable,value_type*, const value_type*>;
    using reference = std::conditional_t<rw==ReadWrite::writeable,value_type&,const value_type&>;
    using shape_type = std::vector<std::size_t>;
    using stride_type = std::vector<std::ptrdiff_t>;
    static constexpr RCOrder rc_order = T::rc_order;

protected:

    std::size_t  _size;
    shape_type   _shape;
    stride_type  _stride;
    pointer      _data;
    bool         _is_contiguous;

public:

    // ===============================================
    // Constructors

    DenseView() = delete;
    DenseView( const DenseView& ) = default;
    DenseView( DenseView&& ) = default;
    DenseView& operator=( const DenseView& ) = default;
    DenseView& operator=( DenseView&& ) = default;

    // Copy from other read/write type
    DenseView( const DenseView<T,other_rw>& other ) :
        _size(other._size),
        _shape(other._shape),
        _stride(other._stride),
        _data(other._data),
        _is_contiguous(other._is_contiguous)
    {}

    // Full view of a container
    DenseView( T& container) :
        _size( container.size() ),
        _shape( container.dims() ),
        _stride( container.dims()+1 ),
        _data( container.data() ),
        _is_contiguous( container.is_contiguous())
    {
        std::ranges::copy(container.shape(),_shape.begin());
        std::ranges::copy(container.stride(),_stride.begin());
    }

    DenseView( const T& container) :
        _size( container.size() ),
        _shape( container.dims() ),
        _stride( container.dims()+1 ),
        _data( container.data() ),
        _is_contiguous( container.is_contiguous())
    {
        std::ranges::copy(container.shape(),_shape.begin());
        std::ranges::copy(container.stride(),_stride.begin());
    }

    // ===============================================
    // Pull methods from base
    // Some methods are shadowed, as the default behaviour is not appropriate

    using DenseBase<DenseView<T,rw>,T::rc_order>::dims;
    using DenseBase<DenseView<T,rw>,T::rc_order>::shape;
    using DenseBase<DenseView<T,rw>,T::rc_order>::stride;
    using DenseBase<DenseView<T,rw>,T::rc_order>::order;
    using DenseBase<DenseView<T,rw>,T::rc_order>::fill;
    using DenseBase<DenseView<T,rw>,T::rc_order>::stripes;
    using DenseBase<DenseView<T,rw>,T::rc_order>::num_stripes;
    using DenseBase<DenseView<T,rw>,T::rc_order>::get_stripe;
    using DenseBase<DenseView<T,rw>,T::rc_order>::reshape;
    using DenseBase<DenseView<T,rw>,T::rc_order>::operator();
    using DenseBase<DenseView<T,rw>,T::rc_order>::operator[];
    using DenseBase<DenseView<T,rw>,T::rc_order>::operator=;
    using DenseBase<DenseView<T,rw>,T::rc_order>::operator+=;
    using DenseBase<DenseView<T,rw>,T::rc_order>::operator-=;
    using DenseBase<DenseView<T,rw>,T::rc_order>::operator*=;
    using DenseBase<DenseView<T,rw>,T::rc_order>::operator/=;
    using DenseBase<DenseView<T,rw>,T::rc_order>::check_expression;
    using DenseBase<DenseView<T,rw>,T::rc_order>::is_omp_parallelisable;

    std::size_t size() const noexcept { return _size;}
    pointer data() const noexcept { return _data; }
    pointer data() noexcept requires ( rw == ReadWrite::writeable ){ return _data; }

    bool is_contiguous() const noexcept { return _is_contiguous; }

    bool test_contiguous() const noexcept requires (rc_order==RCOrder::row_major) {
        ptrdiff_t stride = 1;
        if( stride != _stride[dims()]) return false;
        for( std::size_t ii=dims(); ii != 0; --ii){
            stride *= _shape[ii-1];
            if( stride != _stride[ii-1]) return false;
        }
        return true;
    }

    bool test_contiguous() const noexcept requires (rc_order==RCOrder::col_major) {
        ptrdiff_t stride = 1;
        if( stride != _stride[0]) return false;
        for( std::size_t ii=0; ii != dims(); ++ii){
            stride *= _shape[ii];
            if( stride != _stride[ii+1]) return false;
        }
        return true;
    }

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
    // Striped Iteration

    auto begin_stripe( std::size_t stripe, std::size_t dim) {
         return _data + jump_to_stripe(stride,dim);   
    }

    auto begin_stripe( std::size_t stripe, std::size_t dim) const {
         return _data + jump_to_stripe(stride,dim);   
    }

    auto end_stripe( std::size_t stripe, std::size_t dim) {
         return _data + jump_to_stripe(stride,dim) + _shape[dim] * _stride[dim+(rc_order==RCOrder::row_major)];
    }

    auto end_stripe( std::size_t stripe, std::size_t dim) const {
         return _data + jump_to_stripe(stride,dim) + _shape[dim] * _stride[dim+(rc_order==RCOrder::row_major)];
    }

    auto begin_stripe( std::size_t stripe) { return begin_stripe(stripe,(dims()-1)*(rc_order == RCOrder::row_major)); }
    auto begin_stripe( std::size_t stripe) const { return begin_stripe(stripe,(dims()-1)*(rc_order == RCOrder::row_major)); }
    auto end_stripe( std::size_t stripe) { return end_stripe(stripe,(dims()-1)*(rc_order == RCOrder::row_major)); }
    auto end_stripe( std::size_t stripe) const { return end_stripe(stripe,(dims()-1)*(rc_order == RCOrder::row_major)); }
    
    // ===============================================
    // Special view methods:
    // - Slicing
    // - Broadcasting
    // - Permuting/Transposing

    template<std::ranges::range Slices> requires ( std::is_same<typename Slices::value_type,Slice>::value )
    DenseView slice( const Slices& slices ) const {
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
        result._is_contiguous = result.test_contiguous();
        return result;
    }

    template<class... Slices> requires ( std::is_same<Slices,Slice>::value && ... )
    DenseView slice( const Slices&... var_slices) const {
        return slice(std::array<Slice,sizeof...(Slices)>{ var_slices... });
    }


    template<std::ranges::range... Shapes> requires ( std::integral<typename Shapes::value_type> && ... )
    static std::vector<std::size_t> get_broadcast_shape( const Shapes&... shapes) {
        std::size_t max_dims = std::max({shapes.size()...});
        std::vector<std::size_t> bcast_shape(max_dims,1);
        for( std::size_t ii=0; ii<max_dims; ++ii){
            bcast_shape[ii] = std::max({ (ii < shapes.size() ? shapes[ii] : 0) ...});
            // throw exception if any of the shapes included have a dimension which is neither bcast_shape[ii] nor 1.
            auto errors = std::array<bool,sizeof...(Shapes)>{
                ( ii < shapes.size() ? ( shapes[ii] == 1 || shapes[ii] == bcast_shape[ii] ? false : true) : false)...
            };
            if( std::ranges::any_of(errors,[](bool b){return b;}) ) throw std::runtime_error("Ultramat: Cannot broadcast shapes");   
        }
        return bcast_shape;
    }

    template<std::ranges::range... Shapes>
    requires (( !std::is_base_of<DenseTag,Shapes>::value && std::integral<typename Shapes::value_type>) && ... )
    DenseView<T,ReadWrite::read_only> broadcast( const Shapes&... shapes) const {
        static const std::string err = "Ultramat: Cannot broadcast to given shape";
        auto bcast_shape = get_broadcast_shape(_shape,shapes...);
        // Check bcast_shape is valid
        for(std::size_t ii=0; ii<dims(); ++ii){
            // Account for broadcasting down
            if( ii > bcast_shape.size() ){
                if( _shape[ii] > 1 ){
                    throw std::runtime_error(err);
                } else {
                    continue;
                }
            }
            // Check that shapes agree, or that this view has shape 1
            if( _shape[ii] != bcast_shape[ii] && _shape[ii] != 1 ) throw std::runtime_error(err);
        }
        // Create copy to work with
        DenseView<T,ReadWrite::read_only> bcast_view(*this);
        bcast_view._shape = bcast_shape;
        bcast_view._size = std::accumulate( bcast_shape.begin(), bcast_shape.end(), 1, std::multiplies<std::size_t>{});
        bcast_view._stride.resize(bcast_shape.size()+1);
        // Broadcasting stride rules:
        // - If _shape[ii] == 1 and bcast_shape[ii] > 1, stride=0
        // - If ii > dims(), stride=0
        // - If _shape[ii] == bcast_shape[ii], stride[ii] = _stride[ii]
        if( rc_order == RCOrder::col_major ){
            for( std::size_t ii=0; ii<bcast_shape.size(); ++ii){
                bcast_view._stride[ii] = ( (_shape[ii]==1 && bcast_shape[ii]>1) || ii>dims() ? 0 : _stride[ii]); 
            }
            bcast_view._stride[ bcast_shape.size() ] = bcast_view._size;
        } else {
            for( std::size_t ii=bcast_shape.size(); ii!=0; --ii){
                bcast_view._stride[ii] = ( (_shape[ii-1]==1 && bcast_shape[ii-1]>1) || ii>dims() ? 0 : _stride[ii]); 
            }
            bcast_view._stride[0] = bcast_view._size;
        }
        // Set contiguous
        bcast_view._is_contiguous = bcast_view.test_contiguous();
        return bcast_view;
    }

    template<class... Denses> requires ( std::is_base_of<DenseTag,Denses>::value && ... )
    DenseView<T,ReadWrite::read_only> broadcast( const Denses&... denses) const {
        return broadcast(denses.shape()...);
    }

    template<std::ranges::range Perm> requires std::integral<typename Perm::value_type>
    DenseView permute( const Perm& permutations) const {
        static const std::string permute_err = "Ultramat: Permute should be given ints in range [0,dims()) without repeats";
        // Require length of pemutations to be same as dims(), and should contain all of the ints in the range [0,dims()) without repeats.
        if( permutations.size() != dims() ) throw std::runtime_error("Ultramat: Permute given wrong number of dimensions");
        std::vector<bool> dims_included(dims(),false);
        for( auto&& x : permutations ){
            if( x < 0 || x >dims() ) throw std::runtime_error(permute_err);
            dims_included[x] = true;
        }
        if(!std::ranges::all_of(dims_included,[](bool b){return b;})){
            throw std::runtime_error(permute_err);
        }
        // Create copy and apply permutations accordingly. Greatest stride should not be affected.
        auto copy(*this);
        for( std::size_t ii=0; ii<dims(); ++ii){
            copy._shape[ii] = _shape[permutations[ii]];
            copy._stride[ii + (rc_order==RCOrder::row_major)] = _stride[permutations[ii] + (rc_order==RCOrder::row_major)];
        }
        return copy;
    }

    template<std::integral... Perm>
    DenseView permute( Perm... permutations) const {
        return permute(std::array<std::size_t,sizeof...(Perm)>{permutations...});
    }

    DenseView transpose() const {
        if( dims() != 2 ) throw std::runtime_error("Ultramat: transpose() requires dims() == 2. Perhaps you wanted permute()?");
        return permute(1,0);
    }

    DenseView t() const { return transpose(); }
};

// Define view iterator

template<class T,ReadWrite rw>
template<bool constness>
class DenseView<T,rw>::iterator_impl {
    
    friend typename DenseView<T,rw>::iterator_impl<!constness>;

public:

    using iterator_category = std::random_access_iterator_tag;
    using difference_type   = std::ptrdiff_t;
    using value_type        = DenseView<T,rw>::value_type;
    using shape_type        = DenseView<T,rw>::shape_type;
    using stride_type       = DenseView<T,rw>::stride_type;
    using pointer           = DenseView<T,rw>::pointer;
    using reference         = DenseView<T,rw>::reference;
    static constexpr RCOrder rc_order = DenseView<T,rw>::rc_order;

private:

    pointer     _ptr;
    shape_type  _shape;
    stride_type _stride;
    stride_type _pos;

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

// DenseStripe
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

// Define StripeGenerator

template<class T, ReadWrite rw>
class StripeGenerator {

    static constexpr ReadWrite other_rw = (rw == ReadWrite::read_only ? ReadWrite::writeable : ReadWrite::read_only);
    friend StripeGenerator<T,other_rw>;

public:

    using value_type = typename T::value_type;
    using difference_type = std::ptrdiff_t;
    using pointer = std::conditional_t<rw==ReadWrite::writeable,value_type*, const value_type*>;
    using reference = std::conditional_t<rw==ReadWrite::writeable,value_type&,const value_type&>;
    using shape_type = typename T::shape_type;
    using stride_type = typename T::stride_type;
    static constexpr RCOrder rc_order = T::rc_order;

protected:

    shape_type   _shape;
    stride_type  _stride;
    pointer      _data;
    std::size_t  _dim;
    std::size_t  _stride_dim;
    std::size_t  _stripe;
    std::size_t  _num_stripes;

public:

    // ===============================================
    // Constructors

    StripeGenerator() = delete;
    StripeGenerator( const StripeGenerator& ) = default;
    StripeGenerator( StripeGenerator&& ) = default;
    StripeGenerator& operator=( const StripeGenerator& ) = default;
    StripeGenerator& operator=( StripeGenerator&& ) = default;

    StripeGenerator( T& container, std::size_t dim) :
        _shape( container.shape() ),
        _stride( container.stride() ),
        _data( container.data() ),
        _dim(dim),
        _stride_dim(dim+(rc_order==RCOrder::row_major)),
        _stripe(0),
        _num_stripes(std::accumulate(_shape.begin(),_shape.end(),1,std::multiplies<std::size_t>{}) / _shape[_dim])
    {}

    StripeGenerator( const T& container, std::size_t dim) :
        _shape( container.shape() ),
        _stride( container.stride() ),
        _data( container.data() ),
        _dim(dim),
        _stride_dim(dim+(rc_order==RCOrder::row_major)),
        _stripe(0),
        _num_stripes(std::accumulate(_shape.begin(),_shape.end(),1,std::multiplies<std::size_t>{}) / _shape[_dim])
    {}

    // ===============================================
    // Iteration
    
    std::size_t stripe() const noexcept { return _stripe; }
    std::size_t num_stripes() const noexcept { return _num_stripes; }

    class StripeIterator;

    std::ptrdiff_t jump_to_stripe() const requires (rc_order == RCOrder::row_major) {
        std::ptrdiff_t result=0;
        std::size_t stripe = _stripe;
        for(std::size_t ii=_shape.size(); ii!=0; --ii){
            if( ii-1 == _dim ) continue;
            if( !stripe ) break;
            result += ( stripe % _shape[ii-1]) * _stride[ii];
            stripe /= _shape[ii-1];
        }
        return result;
    }

    std::ptrdiff_t jump_to_stripe() const requires (rc_order == RCOrder::col_major) {
        std::ptrdiff_t result=0;
        std::size_t stripe = _stripe;
        for(std::size_t ii=0; ii!=_shape.size(); ++ii){
            if( ii == _dim ) continue;
            if( !stripe ) break;
            result += ( stripe % _shape[ii]) * _stride[ii];
            stripe /= _shape[ii];
        }
        return result;
    }

    auto begin(){ return StripeIterator( _data + jump_to_stripe(), _stride[_stride_dim]); }
    auto begin() const { return StripeIterator( _data + jump_to_stripe(), _stride[_stride_dim]); }
    auto end(){ return StripeIterator( _data + jump_to_stripe() + _shape[_dim] * _stride[_stride_dim], _stride[_stride_dim]); }
    auto end() const { return StripeIterator( _data + jump_to_stripe() + _shape[_dim] * _stride[_stride_dim], _stride[_stride_dim]); }

    StripeGenerator& operator*() {
        return *this;
    }

    StripeGenerator& operator++() {
        ++_stripe;
        return *this;
    }   

    bool operator==( std::size_t stripe) const {
        return _stripe == stripe;
    }
};

template<class T, ReadWrite rw>
class StripeGeneratorImpl {

    // simply forwards details to StripeGenerator
    // begin returns StripeGenerator
    // end returns num_stripes

    using reference = std::conditional_t<rw==ReadWrite::writeable,T&,const T&>;
    static constexpr RCOrder rc_order = T::rc_order;

    reference   _t;
    std::size_t _dim;
    
    public:

    StripeGeneratorImpl( T& container, std::size_t dim) : _t(container), _dim(dim) {}
    StripeGeneratorImpl( const T& container, std::size_t dim) : _t(container), _dim(dim) {}
    StripeGeneratorImpl( T& container) : StripeGeneratorImpl(container,(rc_order==RCOrder::row_major ? container.dims()-1 : 0)) {}
    StripeGeneratorImpl( const T& container) : StripeGeneratorImpl(container,(rc_order==RCOrder::row_major ? container.dims()-1 : 0)) {}

    auto begin() { return StripeGenerator<T,rw>(_t,_dim);}
    auto begin() const { return StripeGenerator<T,rw>(_t,_dim);}
    auto end() { return StripeGenerator<T,rw>(_t,_dim).num_stripes();}
    auto end() const { return StripeGenerator<T,rw>(_t,_dim).num_stripes();}
};

template<class T, ReadWrite rw>
class StripeGenerator<T,rw>::StripeIterator {

    pointer        _ptr;
    std::ptrdiff_t _stride;

    public:

    StripeIterator() = delete;
    StripeIterator( const StripeIterator& ) = default;
    StripeIterator( StripeIterator&& ) = default;
    StripeIterator& operator=( const StripeIterator& ) = default;
    StripeIterator& operator=( StripeIterator&& ) = default;

    StripeIterator( pointer ptr, std::ptrdiff_t stride) : _ptr(ptr), _stride(stride) {}

    // ===============================================
    // Standard iterator interface

    // Dereference
    reference operator*() const { return *_ptr; }
    
    // Increment/decrement
    StripeIterator& operator++() {
        _ptr += _stride;
        return *this;
    }

    StripeIterator& operator--(){
        _ptr -= _stride;
        return *this;
    }

    StripeIterator operator++(int) const {
        return ++StripeIterator(*this);
    }

    StripeIterator operator--(int) const {
        return --StripeIterator(*this);
    }

    // Random-access
    StripeIterator& operator+=( difference_type diff){
        _ptr += diff*_stride;
        return *this;
    }

    StripeIterator& operator-=( difference_type diff){
        _ptr -= diff*_stride;
        return *this;
    }
    
    StripeIterator operator+( difference_type diff) const {
       StripeIterator result(*this);
       result += diff;
       return result;
    }

    StripeIterator operator-( difference_type diff) const {
       StripeIterator result(*this);
       result -= diff;
       return result;
    }

    // Distance
    std::ptrdiff_t operator-( const StripeIterator& stripe_r) const {
        // Assumes both pointers are looking at the same thing. If not, the results are undefined.
        return (_ptr - stripe_r._ptr)/_stride;
    }

    // Boolean comparisons
    bool operator==( const StripeIterator& stripe_r) const {
        return _ptr == stripe_r._ptr;
    }

    auto operator<=>( const StripeIterator& stripe_r) const {
        return (*this - stripe_r) <=> 0;
    }

};


} // namespace
#endif
