#ifndef __ULTRA_DENSE_FIXED_HPP
#define __ULTRA_DENSE_FIXED_HPP

#include "DenseImpl.hpp"
#include "DenseView.hpp"

namespace ultra {

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

// ===============================================
// DenseFixed
//
// Dense object with size fixed at compile time.
// Preferred interface is the Array alias.


template<class T, DenseOrder Order, std::size_t... Dims>
class DenseFixed : public DenseExpression<DenseFixed<T,Order,Dims...>>, public DenseImpl<DenseFixed<T,Order,Dims...>> {

    friend DenseImpl<DenseFixed<T,Order,Dims...>>;

    static constexpr std::size_t _dims = sizeof...(Dims);
    static constexpr std::size_t _size = variadic_product<Dims...>::value;
    static_assert(_dims >= 1, "FixedArray must have at least one dimension.");

public:

    using value_type = T;
    using shape_type = std::array<std::size_t,_dims>;
    using stride_type = std::array<std::size_t,_dims+1>;
    using data_type = std::array<T,_size>;
    using iterator = data_type::iterator;
    using const_iterator = data_type::const_iterator;
    static constexpr DenseOrder order() { return Order; }

    // For convenience, specify row/col major via FixedArray<T,Dims...>::row/col_major
    using row_major = DenseFixed<T,DenseOrder::row_major,Dims...>;
    using col_major = DenseFixed<T,DenseOrder::col_major,Dims...>;

    // View of self
    using View = DenseView<DenseFixed<T,Order,Dims...>>;

private:

    static constexpr shape_type  _shape = {{Dims...}};
    static constexpr stride_type _stride = (Order == DenseOrder::row_major ? variadic_stride<Dims...>::row_major : variadic_stride<Dims...>::col_major);
    data_type _data;

public:

    // ===============================================
    // Constructors

    DenseFixed() = default;
    DenseFixed( const DenseFixed& other) = default;
    DenseFixed( DenseFixed&& other) = default;
    DenseFixed& operator=( const DenseFixed& other) = default;
    DenseFixed& operator=( DenseFixed&& other) = default;
    
    // With fill
    DenseFixed( const T& fill) { _data.fill(fill); }

    template<class U>
    DenseFixed( const DenseExpression<U>& expression) { operator=(expression);}

    template<class U>
    DenseFixed( DenseExpression<U>&& expression) { operator=(std::move(expression));}

    // Swap
    constexpr void swap( DenseFixed& other) noexcept { _data.swap(other._data); }

    constexpr friend void swap( DenseFixed& a,DenseFixed& b) noexcept { a.swap(b); }

    // Pull in methods from CRTP base

    using DenseImpl<DenseFixed<T,Order,Dims...>>::dims;
    using DenseImpl<DenseFixed<T,Order,Dims...>>::size;
    using DenseImpl<DenseFixed<T,Order,Dims...>>::shape;
    using DenseImpl<DenseFixed<T,Order,Dims...>>::stride;
    using DenseImpl<DenseFixed<T,Order,Dims...>>::data;
    using DenseImpl<DenseFixed<T,Order,Dims...>>::fill;
    using DenseImpl<DenseFixed<T,Order,Dims...>>::view;
    using DenseImpl<DenseFixed<T,Order,Dims...>>::permute;
    using DenseImpl<DenseFixed<T,Order,Dims...>>::transpose;
    using DenseImpl<DenseFixed<T,Order,Dims...>>::t;
    using DenseImpl<DenseFixed<T,Order,Dims...>>::begin;
    using DenseImpl<DenseFixed<T,Order,Dims...>>::end;
    using DenseImpl<DenseFixed<T,Order,Dims...>>::get_stripe;
    using DenseImpl<DenseFixed<T,Order,Dims...>>::required_stripe_dim;
    using DenseImpl<DenseFixed<T,Order,Dims...>>::operator();
    using DenseImpl<DenseFixed<T,Order,Dims...>>::operator[];
    using DenseImpl<DenseFixed<T,Order,Dims...>>::operator=;
    using DenseImpl<DenseFixed<T,Order,Dims...>>::operator+=;
    using DenseImpl<DenseFixed<T,Order,Dims...>>::operator-=;
    using DenseImpl<DenseFixed<T,Order,Dims...>>::operator*=;
    using DenseImpl<DenseFixed<T,Order,Dims...>>::operator/=;
    using DenseImpl<DenseFixed<T,Order,Dims...>>::is_contiguous;
    using DenseImpl<DenseFixed<T,Order,Dims...>>::is_omp_parallelisable;
    using DenseImpl<DenseFixed<T,Order,Dims...>>::is_broadcasting;
};

} // namespace ultra
#endif
