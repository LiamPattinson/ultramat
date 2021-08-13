#ifndef __ULTRA_DENSE_UTILS_HPP
#define __ULTRA_DENSE_UTILS_HPP

// DenseUtils.hpp
//
// Defines any utility functions/structs deemed useful for Dense objects.

#include "ultramat/include/Utils/Utils.hpp"

namespace ultra {

// Define row/col order enum class

enum class DenseOrder { row_major, col_major, mixed };
const DenseOrder default_order = DenseOrder::row_major;

// get_common_order
// return row_major if all row_major, col_major if all col_major, and default otherwise

template<class... Ts> struct GetCommonOrderImpl;

template<class T1, class T2, class... Ts>
struct GetCommonOrderImpl<T1,T2,Ts...> {
    static constexpr DenseOrder Order = std::remove_cvref_t<T1>::order() == GetCommonOrderImpl<T2,Ts...>::order() ? std::remove_cvref_t<T1>::order() : DenseOrder::mixed;
    static constexpr DenseOrder order() { return Order; }
};

template<class T1>
struct GetCommonOrderImpl<T1> {
    static constexpr DenseOrder Order = std::remove_cvref_t<T1>::order();
    static constexpr DenseOrder order() { return Order; }
};

template<class... Ts>
struct get_common_order {
    static constexpr DenseOrder Order = GetCommonOrderImpl<Ts...>::order();
    static constexpr DenseOrder order(){ return Order; }
    static constexpr DenseOrder value = Order;
};

// Define 'DenseType' enum class.
// Used by Dense to determine whether to store shape/stride as std::vector or std::array, and whether to restrict reshaping

enum class DenseType : std::size_t { vec=1, mat=2, nd=0 };

// Define read-only enum class

enum class ReadWrite { writeable, read_only };

// Declare Array types and preferred interface

template<class, DenseType, DenseOrder> class Dense;
template<class, DenseOrder, std::size_t...> class DenseFixed;
template<class, ReadWrite = ReadWrite::writeable> class DenseView;
template<class, ReadWrite = ReadWrite::writeable> class DenseStripe;

template<class T,std::size_t... Dims>
using Array = std::conditional_t<sizeof...(Dims),DenseFixed<T,default_order,Dims...>,Dense<T,DenseType::nd,default_order>>;

template<class T,std::size_t... Dims> requires ( sizeof...(Dims) == 0 || sizeof...(Dims) == 1)
using Vector = std::conditional_t<sizeof...(Dims),DenseFixed<T,default_order,Dims...>,Dense<T,DenseType::vec,default_order>>;

template<class T,std::size_t... Dims> requires ( sizeof...(Dims) == 0 || sizeof...(Dims) == 2)
using Matrix = std::conditional_t<sizeof...(Dims),DenseFixed<T,default_order,Dims...>,Dense<T,DenseType::mat,default_order>>;

// is_dense type trait
template<class T> struct is_dense { static constexpr bool value = false; };
template<class T, DenseType Type, DenseOrder Order> struct is_dense<Dense<T,Type,Order>> { static constexpr bool value = true; };
template<class T, DenseOrder Order, std::size_t... dims> struct is_dense<DenseFixed<T,Order,dims...>> { static constexpr bool value = true; };
template<class T, ReadWrite RW> struct is_dense<DenseView<T,RW>> { static constexpr bool value = true; };

// shapelike concept: a range of integers, but not a dense object, e.g. std::vector<std::size_t>
template<class T> concept shapelike = std::ranges::sized_range<T> && std::integral<typename T::value_type> && !is_dense<T>::value;

// Define slice : a tool for generating views of Arrays and related objects.
// Is an 'aggregate'/'pod' type, so should have a relatively intuitive interface by default.

struct Slice { 
    static constexpr std::ptrdiff_t all = std::numeric_limits<std::ptrdiff_t>::max();
    std::ptrdiff_t start;
    std::ptrdiff_t end;
    std::ptrdiff_t step=1;
};

// Bool class
// Looks like a bool, acts like a bool. Used in place of regular bool in dynamic dense objects to avoid the horrors of std::vector<bool>

class Bool {
    bool _x;
    public:
    Bool() = default;
    Bool( const Bool& ) = default;
    Bool( Bool&& ) = default;
    Bool& operator=( const Bool& ) = default;
    Bool& operator=( Bool&& ) = default;
    inline Bool( bool x) : _x(x) {}
    inline operator bool() const { return _x; }
    inline operator bool&() { return _x; }
};

// ==========================
// Striped iteration utils

class DenseStriper {

    std::size_t                 _dim;
    DenseOrder                  _order;
    std::vector<std::ptrdiff_t> _idx;
    std::vector<std::size_t>    _shape;

    public:

    DenseStriper() = delete;
    DenseStriper( const DenseStriper& ) = default;
    DenseStriper( DenseStriper&& ) = default;
    DenseStriper& operator=( const DenseStriper& ) = default;
    DenseStriper& operator=( DenseStriper&& ) = default;

    template<shapelike Shape>
    DenseStriper( std::size_t dim, DenseOrder order, const Shape& shape, bool end=false) :
        _dim(dim),
        _order(order),
        _idx(shape.size()+1,0),
        _shape(shape.size())
    {
        std::ranges::copy( shape, _shape.begin());
        if( end ){
            _idx[ _order==DenseOrder::col_major ? dims() : 0 ] = 1;
        }
    }

    std::size_t num_stripes() const {
        return std::accumulate( _shape.begin(), _shape.end(), 1, std::multiplies<std::size_t>{})/_shape[_dim];
    }

    std::size_t stripe_dim() const {
        return _dim;
    }

    std::size_t stripe_size() const {
        return _shape[_dim];
    }

    std::size_t dims() const {
        return _shape.size();
    }

    DenseOrder order() const {
        return _order;
    }

    std::vector<std::size_t> shape() const {
        return _shape;
    }

    std::size_t shape( std::size_t ii) const {
        return _shape[ii];
    }

    std::size_t& shape( std::size_t ii) {
        return _shape[ii];
    }

    const std::vector<std::ptrdiff_t>& index() const {
        return _idx;
    }

    std::ptrdiff_t index( std::size_t ii) const {
        return _idx[ii];
    }

    std::ptrdiff_t& index( std::size_t ii) {
        return _idx[ii];
    }

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
        return *this;
    }

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
        return *this;
    }

    std::size_t get_scalar_index() const {
        std::size_t scalar_index = 0;
        std::size_t scalar_index_factor = 1;
        if( _order == DenseOrder::col_major) {
            for( std::size_t ii=0; ii != dims(); ++ii) {
                if( ii == stripe_dim() ) continue;
                scalar_index += scalar_index_factor * index(ii);
                scalar_index_factor *= shape(ii);
            }
            scalar_index += scalar_index_factor * index(dims());
        } else {
            for( std::size_t ii=dims(); ii != 0; --ii) {
                if( ii == stripe_dim()+1 ) continue;
                scalar_index += scalar_index_factor * index(ii);
                scalar_index_factor *= shape(ii-1);
            }
            scalar_index += scalar_index_factor * index(0);
        }
        return scalar_index;
    }

    void set_from_scalar_index( std::size_t scalar_index) {
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

    void set_from_index( const std::vector<std::size_t>& idx) {
        if( idx.size() != _idx.size() ) throw std::runtime_error("Ultramat DenseStrider: Tried to set with index of incorrect size");
        std::ranges::copy( idx, _idx.begin());
    }

    DenseStriper& operator+=( std::ptrdiff_t diff) {
        // Determine current scalar index, add diff, set index accordingly
        std::size_t scalar_index = get_scalar_index();
        scalar_index += diff;
        set_from_scalar_index(scalar_index);
        return *this;
    }

    DenseStriper& operator-=( std::ptrdiff_t diff) {
        // Determine current scalar index, subtract diff, set index accordingly
        std::size_t scalar_index = get_scalar_index();
        scalar_index -= diff;
        set_from_scalar_index(scalar_index);
        return *this;
    }

    DenseStriper operator+( std::ptrdiff_t diff) const {
        DenseStriper copy(*this);
        copy += diff;
        return copy;
    }

    DenseStriper operator-( std::ptrdiff_t diff) const {
        DenseStriper copy(*this);
        copy -= diff;
        return copy;
    }

    std::ptrdiff_t operator-( const DenseStriper& other) const {
        return static_cast<std::ptrdiff_t>(get_scalar_index()) - static_cast<std::ptrdiff_t>(other.get_scalar_index());
    }

    bool operator==( const DenseStriper& other) const {
        for( std::size_t ii=0; ii <= dims(); ++ii) {
            if( index(ii) != other.index(ii) ) return false;
        }
        return true;
    }

    auto operator<=>( const DenseStriper& other) const {
        if( _order == DenseOrder::col_major ) {
            for( std::ptrdiff_t ii=dims(); ii >= 0; --ii) {
                if( index(ii) == other.index(ii) ) {
                    continue;
                } else {
                    return index(ii) <=> other.index(ii);
                }
            }
        } else {
            for( std::size_t ii=0; ii <= dims(); ++ii) {
                if( index(ii) == other.index(ii) ) {
                    continue;
                } else {
                    return index(ii) <=> other.index(ii);
                }
            }

        }
        return std::strong_ordering::equal;
    }
};

template<DenseOrder order, shapelike... Shapes> 
std::vector<std::size_t> get_broadcast_shape( const Shapes&... shapes) {
    // If row_major, prepend dims
    // If col_major, apend dims
    std::size_t max_dims = std::max({shapes.size()...});
    std::vector<std::size_t> bcast_shape(max_dims,1);
    if( order == DenseOrder::col_major ){
        for( std::size_t ii=0; ii<max_dims; ++ii){
            bcast_shape[ii] = std::max({ (ii < shapes.size() ? shapes[ii] : 0) ...});
            // throw exception if any of the shapes included have a dimension which is neither bcast_shape[ii] nor 1.
            auto errors = std::array<bool,sizeof...(Shapes)>{
                ( ii < shapes.size() ? ( shapes[ii] == 1 || shapes[ii] == bcast_shape[ii] ? false : true) : false)...
            };
            if( std::ranges::any_of(errors,[](bool b){return b;}) ) throw std::runtime_error("Ultramat: Cannot broadcast shapes");   
        }
    } else {
        for( std::size_t ii=0; ii<max_dims; ++ii){
            bcast_shape[max_dims-ii-1] = std::max({ (ii < shapes.size() ? shapes[shapes.size()-ii-1] : 0) ...});
            // throw exception if any of the shapes included have a dimension which is neither bcast_shape[ii] nor 1.
            auto errors = std::array<bool,sizeof...(Shapes)>{
                ( ii < shapes.size() ? ( shapes[shapes.size()-ii-1] == 1 || shapes[shapes.size()-ii-1] == bcast_shape[max_dims-ii-1] ? false : true) : false)...
            };
            if( std::ranges::any_of(errors,[](bool b){return b;}) ) throw std::runtime_error("Ultramat: Cannot broadcast shapes");   
        }
    }
    return bcast_shape;
}

// ==========================
// Fixed-Size Dense Utils
// Lots of compile-time template nonsense ahead.

// variadic_product

template<std::size_t... Ints> struct variadic_product;

template<> struct variadic_product<> { static constexpr std::size_t value = 1; };

template<std::size_t Int,std::size_t... Ints>
struct variadic_product<Int,Ints...> {
    static constexpr std::size_t value = Int*variadic_product<Ints...>::value;
};

// index_sequence_to_array

template<std::size_t... Ints>
constexpr auto index_sequence_to_array( std::index_sequence<Ints...> ) noexcept {
    return std::array<std::size_t,sizeof...(Ints)>{{Ints...}};
}

// reverse_index_sequence

template<class T1,class T2> struct reverse_index_sequence_impl;

template<std::size_t Int1, std::size_t... Ints1, std::size_t Int2, std::size_t... Ints2>
struct reverse_index_sequence_impl< std::index_sequence<Int1,Ints1...>, std::index_sequence<Int2,Ints2...>> {
    using type = 
        reverse_index_sequence_impl<std::index_sequence<Ints1...>,std::index_sequence<Int1,Int2,Ints2...>>::type;
};

template<std::size_t... Ints2>
struct reverse_index_sequence_impl< std::index_sequence<>, std::index_sequence<Ints2...>> {
    using type = std::index_sequence<Ints2...>;
};

template<std::size_t Int1, std::size_t... Ints1>
struct reverse_index_sequence_impl< std::index_sequence<Int1,Ints1...>, std::index_sequence<>> {
    using type = reverse_index_sequence_impl<std::index_sequence<Ints1...>,std::index_sequence<Int1>>::type;
};

template<class T> struct reverse_index_sequence;

template<std::size_t... Ints>
struct reverse_index_sequence<std::index_sequence<Ints...>> {
    using type = reverse_index_sequence_impl<std::index_sequence<Ints...>,std::index_sequence<>>::type;
};

// variadic stride

template<class T1,class T2> struct variadic_stride_impl;

template<std::size_t ShapeInt, std::size_t... ShapeInts, std::size_t StrideInt, std::size_t... StrideInts>
struct variadic_stride_impl< std::index_sequence<ShapeInt,ShapeInts...>, std::index_sequence<StrideInt,StrideInts...>> {
    using type = 
        variadic_stride_impl<std::index_sequence<ShapeInts...>,std::index_sequence<ShapeInt*StrideInt,StrideInt,StrideInts...>>::type;
};

template<std::size_t... StrideInts>
struct variadic_stride_impl< std::index_sequence<>, std::index_sequence<StrideInts...>> {
    using type = std::index_sequence<StrideInts...>;
};

template<std::size_t ShapeInt, std::size_t... ShapeInts>
struct variadic_stride_impl< std::index_sequence<ShapeInt,ShapeInts...>, std::index_sequence<>> {
    using type = variadic_stride_impl<std::index_sequence<ShapeInts...>,std::index_sequence<ShapeInt>>::type;
};

template<std::size_t... ShapeInts>
struct variadic_stride {
    static constexpr auto col_major = index_sequence_to_array(
        typename reverse_index_sequence<typename variadic_stride_impl<std::index_sequence<1,ShapeInts...>,std::index_sequence<>>::type>::type()
    );
    static constexpr auto row_major = index_sequence_to_array(
        typename variadic_stride_impl<typename reverse_index_sequence<std::index_sequence<ShapeInts...,1>>::type,std::index_sequence<>>::type()
    );
};

} // namespace
#endif