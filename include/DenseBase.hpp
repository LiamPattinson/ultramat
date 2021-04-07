#ifndef __ULTRA_DENSE_BASE_HPP
#define __ULTRA_DENSE_BASE_HPP

// DenseBase.hpp
//
// Defines a base class for all dense array-like objects, including Arrays, Vectors, Matrices, and their fixed-size counterparts.

#include "Expression.hpp"

namespace ultra {

// Declare DenseView class

template<class T, ReadWriteStatus ReadWrite=ReadWriteStatus::writeable> class DenseView;

// CRTP Base Class
template<class T,RCOrder Order>
class DenseBase : public DenseTag {

    protected:

    // Attributes

    constexpr auto dims() const noexcept { return static_cast<const T&>(*this)._shape.size(); }
    constexpr auto size() const noexcept { return static_cast<const T&>(*this)._data.size(); }
    constexpr auto shape( std::size_t dim) const noexcept { return static_cast<const T&>(*this)._shape[dim]; }
    constexpr auto stride( std::size_t dim) const noexcept { return static_cast<const T&>(*this)._stride[dim]; }
    constexpr const auto& shape() const noexcept { return static_cast<const T&>(*this)._shape; }
    constexpr const auto& stride() const noexcept { return static_cast<const T&>(*this)._stride; }

    // Data access

    constexpr auto* data() noexcept { return static_cast<T&>(*this)._data.data(); }
    constexpr const auto* data() const noexcept { return static_cast<const T&>(*this)._data.data(); }

    // Access via unsigned ints.
    // Warning: no check that the correct number of ints has been provided.
    
    template<std::integral... Ints> 
    auto operator()( Ints... coords ) const noexcept {
        return static_cast<const T&>(*this)._data[variadic_memjump(coords...)];
    }
    
    template<std::integral... Ints> 
    auto& operator()( Ints... coords ) noexcept {
        return static_cast<T&>(*this)._data[variadic_memjump(coords...)];
    }

    // Access via range
    
    template<std::ranges::range Coords> requires std::integral<typename Coords::value_type>
    auto operator()( const Coords& coords) const {
        return static_cast<const T&>(*this)._data[
            std::inner_product(coords.begin(),coords.end(),static_cast<const T&>(*this)._stride.begin()+(Order==RCOrder::row_major),0)
        ];
    }

    template<std::ranges::range Coords> requires std::integral<typename Coords::value_type>
    auto& operator()( const Coords& coords) {
        return static_cast<T&>(*this)._data[
            std::inner_product(coords.begin(),coords.end(),static_cast<const T&>(*this)._stride.begin()+(Order==RCOrder::row_major),0)
        ];
    }

    // Access via square brackets
    // Treats all arrays as 1D containers.
    // TODO have this take a slice and return a view over the array
    
    constexpr auto operator[](std::size_t ii) const noexcept { return static_cast<const T&>(*this)._data[ii]; }
    constexpr auto& operator[](std::size_t ii) noexcept { return static_cast<T&>(*this)._data[ii]; }

    // ===============================================
    // View creation

    DenseView<T> view() { return DenseView<T>(static_cast<T&>(*this));}

    template<class... Slices> requires ( std::is_same<Slice,Slices>::value && ... )
    DenseView<T> view(const Slices&... slices) {
        return DenseView<T>(static_cast<T&>(*this)).slice(slices...);
    }

    // ===============================================
    // Iteration

    constexpr auto begin() noexcept { return static_cast<T&>(*this)._data.begin(); }
    constexpr auto begin() const noexcept { return static_cast<const T&>(*this)._data.cbegin(); }
    constexpr auto cbegin() const noexcept { return static_cast<const T&>(*this)._data.cbegin(); }
    constexpr auto end() noexcept { return static_cast<T&>(*this)._data.end(); }
    constexpr auto end() const noexcept { return static_cast<const T&>(*this)._data.cend(); }
    constexpr auto cend() const noexcept { return static_cast<const T&>(*this)._data.cend(); }

    // ===============================================
    // Utils
    
    // check_expression
    // Tests whether an expression is compatible.

    template<class U>
    void check_expression( const Expression<U>& expression){
        if( dims() != expression.dims()){
            throw ExpressionException("Ultramat: Tried to construct/assign array-like object of dims " + std::to_string(dims()) 
                    + " with expression of dims " + std::to_string(expression.dims()));
        }
        for( std::size_t ii=0; ii<dims(); ++ii){
            if( shape(ii) != expression.shape(ii) ){
                std::string expression_shape("( ");
                std::string array_shape("( ");
                for( std::size_t ii=0; ii<dims(); ++ii){
                    array_shape += std::to_string(shape(ii)) + ' ';
                    expression_shape += std::to_string(expression.shape(ii)) + ' ';
                }
                array_shape += ')';
                expression_shape += ')';
                throw ExpressionException("Ultramat: Tried to construct/assign array-like object of shape " + array_shape 
                        + "with expression of shape " + expression_shape);
            }
        }
        if( size() != expression.size()){
            // Note: throw runtime_error instead, as this should not be caught under any circumstances
            throw std::runtime_error("Ultramat: Tried to construct/assign array-like object of size " + std::to_string(size()) 
                    + " with expression of size " + std::to_string(expression.size())
                    + ". This should not be possible, as they have the same dimensions and shapes, so something has gone dreadfully wrong.");
        }
    }

    // set_stride
    // Use shape to populate stride via a cumulative product, starting with 1.

    void set_stride() noexcept requires (Order == RCOrder::row_major) {
        static_cast<T&>(*this)._stride[dims()] = 1;
        for( std::size_t ii=dims(); ii!=0; --ii){
            static_cast<T&>(*this)._stride[ii-1] = static_cast<T&>(*this)._stride[ii] * static_cast<T&>(*this)._shape[ii-1];            
        }
    }

    void set_stride() noexcept requires (Order == RCOrder::col_major) {
        static_cast<T&>(*this)._stride[0] = 1;
        for( std::size_t ii=0; ii!=dims(); ++ii){
            static_cast<T&>(*this)._stride[ii+1] = static_cast<T&>(*this)._stride[ii] * static_cast<T&>(*this)._shape[ii];
        }
    }

    // variadic_memjump
    // used with round-bracket indexing.
    // requires stride of length dims+1, with the largest stride iterating all the way to the location of end().
    // If row_major, the largest stride is stride[0]. If col_major, the largest stride is stride[dims].

    // Base case
    template<std::size_t N, std::integral Int>
    constexpr decltype(auto) variadic_memjump_impl( Int coord) const noexcept {
        return static_cast<const T&>(*this)._stride[N] * coord; 
    }

    // Recursive step
    template<std::size_t N, std::integral Int, std::integral... Ints>
    constexpr decltype(auto) variadic_memjump_impl( Int coord, Ints... coords) const noexcept {
        return (static_cast<const T&>(*this)._stride[N] * coord) + variadic_memjump_impl<N+1,Ints...>(coords...);
    }

    template<std::integral... Ints>
    constexpr decltype(auto) variadic_memjump( Ints... coords) const noexcept {
        // if row major, must skip first element of stride
        return variadic_memjump_impl<(Order==RCOrder::row_major?1:0),Ints...>(coords...);
    }

};

// ==========================
// Fixed-Size Dense Utils
// ==========================

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
    static constexpr decltype(auto) col_major = index_sequence_to_array(
        typename reverse_index_sequence<typename variadic_stride_impl<std::index_sequence<1,ShapeInts...>,std::index_sequence<>>::type>::type()
    );
    static constexpr decltype(auto) row_major = index_sequence_to_array(
        typename variadic_stride_impl<typename reverse_index_sequence<std::index_sequence<ShapeInts...,1>>::type,std::index_sequence<>>::type()
    );
};

} //namespace
#endif