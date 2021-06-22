#ifndef __ULTRA_DENSE_BASE_HPP
#define __ULTRA_DENSE_BASE_HPP

// DenseBase.hpp
//
// Defines a base class for all dense array-like objects, including Arrays, Vectors, Matrices, and their fixed-size counterparts.

#include "DenseExpression.hpp"

namespace ultra {

// Declare DenseView class and Stripe

template<class T, ReadWrite rw=ReadWrite::writeable> class DenseView;
template<class T, ReadWrite rw=ReadWrite::writeable> class DenseStripe;

// CRTP Base Class
template<class T,RCOrder Order>
class DenseBase : public DenseTag {
    static_assert(Order != RCOrder::mixed_order);

    protected:

    // Helper functions

    constexpr T& derived() noexcept { return static_cast<T&>(*this); }
    constexpr const T& derived() const noexcept { return static_cast<const T&>(*this); }

    // ===============================================
    // Attributes

    constexpr auto dims() const noexcept { return derived()._shape.size(); }
    constexpr auto size() const noexcept { return derived()._data.size(); }
    constexpr auto shape( std::size_t dim) const noexcept { return derived()._shape[dim]; }
    constexpr auto stride( std::size_t dim) const noexcept { return derived()._stride[dim]; }
    constexpr const auto& shape() const noexcept { return derived()._shape; }
    constexpr const auto& stride() const noexcept { return derived()._stride; }
    constexpr const RCOrder order() const noexcept { return Order; }

    // ===============================================
    // Data access

    constexpr auto* data() noexcept { return derived()._data.data(); }
    constexpr const auto* data() const noexcept { return derived()._data.data(); }

    // Fill
    
    template<class U>
    constexpr void fill( const U& u) {
        std::ranges::fill(derived(),u);
    }

    // Access via unsigned ints.
    // Warning: no check that the correct number of ints has been provided.
    
    template<std::integral... Ints> 
    auto operator()( Ints... coords ) const noexcept {
        return derived()._data[variadic_memjump(coords...)];
    }
    
    template<std::integral... Ints> 
    auto& operator()( Ints... coords ) noexcept {
        return derived()._data[variadic_memjump(coords...)];
    }

    // Access via range
    
    template<std::ranges::range Coords> requires std::integral<typename Coords::value_type>
    auto operator()( const Coords& coords) const {
        return derived()._data[
            std::inner_product(coords.begin(),coords.end(),derived()._stride.begin()+(Order==RCOrder::row_major),0)
        ];
    }

    template<std::ranges::range Coords> requires std::integral<typename Coords::value_type>
    auto& operator()( const Coords& coords) {
        return derived()._data[
            std::inner_product(coords.begin(),coords.end(),derived()._stride.begin()+(Order==RCOrder::row_major),0)
        ];
    }

    // ===============================================
    //  Expressions

    struct Equal{ template<class U, class V> void operator()( U& u, const V& v) const { u = v; }};
    struct AddEqual{ template<class U, class V> void operator()( U& u, const V& v) const { u += v; }};
    struct SubEqual{ template<class U, class V> void operator()( U& u, const V& v) const { u -= v; }};
    struct MulEqual{ template<class U, class V> void operator()( U& u, const V& v) const { u *= v; }};
    struct DivEqual{ template<class U, class V> void operator()( U& u, const V& v) const { u /= v; }};

    template<class F, class E>
    void accept_expression( const E& expression ){
        F f{};
        if( derived().is_contiguous() && expression.is_contiguous() && expression.order() == Order ){
            auto expr_it = expression.begin();
            for(auto it=begin(), it_end=end(); it != it_end; ++it, ++expr_it) f(*it,*expr_it);
        } else {
            std::size_t stripe_dim = expression.required_stripe_dim();
            if( stripe_dim == dims() ) stripe_dim = ( Order == RCOrder::row_major ? dims()-1 : 0 );
            std::size_t stripes = num_stripes(stripe_dim);
            for( std::size_t stripe_num=0; stripe_num != stripes; ++stripe_num){
                auto stripe = get_stripe(stripe_num,stripe_dim,Order);
                auto expr_stripe = expression.get_stripe(stripe_num,stripe_dim,Order);
                auto expr_it = expr_stripe.begin();
                for(auto it=stripe.begin(), it_end=stripe.end(); it != it_end; ++it, ++expr_it) f(*it,*expr_it);
            }
        }
    }

    template<class U>
    decltype(auto) operator=( const DenseExpression<U>& expression) {
        check_expression(expression);
        accept_expression<Equal>(expression);
        return derived();
    }

    // Version without checking
    // Useful for non-fixed arrays.
    template<class U>
    decltype(auto) equal_expression( const DenseExpression<U>& expression) {
        accept_expression<Equal>(expression);
        return derived();
    }

    template<class U>
    decltype(auto) operator+=( const DenseExpression<U>& expression ){
        check_expression(expression);
        accept_expression<AddEqual>(expression);
        return derived();
    }

    template<class U>
    decltype(auto) operator-=( const DenseExpression<U>& expression ){
        check_expression(expression);
        accept_expression<SubEqual>(expression);
        return derived();
    }

    template<class U>
    decltype(auto) operator*=( const DenseExpression<U>& expression ){
        check_expression(expression);
        accept_expression<MulEqual>(expression);
        return derived();
    }
    
    template<class U>
    decltype(auto) operator/=( const DenseExpression<U>& expression ){
        check_expression(expression);
        accept_expression<DivEqual>(expression);
        return derived();
    }

    // ===============================================
    // View creation

    DenseView<T> view() { return DenseView<T>(derived());}
    DenseView<T,ReadWrite::read_only> view() const { return DenseView<T,ReadWrite::read_only>(derived());}

    template<class... Slices> requires ( std::is_same<Slice,Slices>::value && ... )
    DenseView<T> view(const Slices&... slices) {
        return DenseView<T>(derived()).slice(slices...);
    }

    template<class... Slices> requires ( std::is_same<Slice,Slices>::value && ... )
    DenseView<T,ReadWrite::read_only> view(const Slices&... slices) const {
        return DenseView<T,ReadWrite::read_only>(derived()).slice(slices...);
    }

    template<std::ranges::range Slices> requires ( std::is_same<typename Slices::value_type,Slice>::value )
    DenseView<T> view(const Slices& slices) {
        return DenseView<T>(derived()).slice(slices);
    }

    template<std::ranges::range Slices> requires ( std::is_same<typename Slices::value_type,Slice>::value )
    DenseView<T,ReadWrite::read_only> view(const Slices& slices) const {
        return DenseView<T,ReadWrite::read_only>(derived()).slice(slices);
    }

    template<bool constness>
    struct SquBrktSlicer {
        using reference = std::conditional_t<constness,const T&,T&>;
        reference _ref;
        std::vector<Slice> _slices;
        SquBrktSlicer() = delete;
        SquBrktSlicer( reference ref, const Slice& slice) : _ref(ref), _slices{slice} {}
        SquBrktSlicer& operator[](const Slice& slice){ _slices.push_back(slice); return *this;}
        SquBrktSlicer& operator[](std::size_t ii){ _slices.push_back(ii==Slice::all ? Slice{Slice::all,Slice::all} : Slice{ii,ii+1}); return *this;}
        auto operator()() { return _ref.view(_slices);}
        auto operator()() const { return _ref.view(_slices);}
    };

    SquBrktSlicer<false> operator[](const Slice& slice) { return SquBrktSlicer<false>(derived(),slice);}
    SquBrktSlicer<false> operator[](std::size_t ii) {
        return SquBrktSlicer<false>(derived(),ii==Slice::all ? Slice{Slice::all,Slice::all} : Slice{ii,ii+1});
    }
    SquBrktSlicer<true> operator[](const Slice& slice) const { return SquBrktSlicer<true>(derived(),slice);}
    SquBrktSlicer<true> operator[](std::size_t ii) const {
        return SquBrktSlicer<true>(derived(),ii==Slice::all ? Slice{Slice::all,Slice::all} : Slice{ii,ii+1});
    }

    // ===============================================
    // Reshaping

    template<std::ranges::range Shape>
    requires std::integral<typename Shape::value_type>
    T& reshape( const Shape& shape ){
        // Ensure this is contiguous
        if( !derived().is_contiguous() ) throw std::runtime_error("Ultramat: Cannot reshape a non-contiguous array");
        // Ensure that the new shape has the correct size.
        auto size = std::accumulate(shape.begin(),shape.end(),1,std::multiplies<typename Shape::value_type>{});
        if( size != derived().size() ) throw std::runtime_error("Ultramat: Cannot reshape, result would have incorrect size");
        // Set new shape
        derived()._shape.resize(shape.size());
        std::ranges::copy( shape, derived()._shape.begin());
        // reset stride and return
        derived()._stride.resize(shape.size()+1);
        set_stride();
        return derived(); 
    }

    template<std::integral... Ints>
    T& reshape( Ints... shape){
        return reshape(std::array<std::size_t,sizeof...(Ints)>{{shape...}});
    }

    // ===============================================
    // Broadcasting

    template<std::ranges::range... Shapes>
    requires (( !std::is_base_of<DenseTag,Shapes>::value &&  std::integral<typename Shapes::value_type>) && ... )
    auto broadcast( const Shapes&... shapes) const {
        return derived().view().broadcast(shapes...);
    }

    template<class... Denses> requires ( std::is_base_of<DenseTag,Denses>::value && ... )
    auto broadcast( const Denses&... denses) const {
        return derived().view().broadcast(denses...);
    }

    // ===============================================
    // Permuting/Transposing

    template<std::ranges::range Perm> requires std::integral<typename Perm::value_type>
    auto permute( const Perm& permutations) {
        return derived().view().permute(permutations);
    }

    template<std::ranges::range Perm> requires std::integral<typename Perm::value_type>
    auto permute( const Perm& permutations) const {
        return derived().view().permute(permutations);
    }

    template<std::integral... Perm>
    auto permute( Perm... permutations) {
        return derived().view().permute(permutations...);
    }

    template<std::integral... Perm>
    auto permute( Perm... permutations) const {
        return derived().view().permute(permutations...);
    }

    auto transpose() {
        if( derived().dims() != 2 ) throw std::runtime_error("Ultramat: transpose() requires dims() == 2. Perhaps you wanted permute()?");
        return permute(1,0);
    }

    auto transpose() const {
        if( derived().dims() != 2 ) throw std::runtime_error("Ultramat: transpose() requires dims() == 2. Perhaps you wanted permute()?");
        return permute(1,0);
    }

    auto t() { return transpose(); }
    auto t() const { return transpose(); }

    // ===============================================
    // Iteration

    constexpr auto begin() noexcept { return derived()._data.begin(); }
    constexpr auto begin() const noexcept { return derived()._data.cbegin(); }
    constexpr auto cbegin() const noexcept { return derived()._data.cbegin(); }
    constexpr auto end() noexcept { return derived()._data.end(); }
    constexpr auto end() const noexcept { return derived()._data.cend(); }
    constexpr auto cend() const noexcept { return derived()._data.cend(); }

    // ===============================================
    // Striped Iteration
    
    std::size_t num_stripes( std::size_t dim) const {
        return std::accumulate(derived()._shape.begin(),derived()._shape.end(),1,std::multiplies<std::size_t>{}) / derived()._shape[dim];
    }

    std::size_t num_stripes() const { return num_stripes((derived().dims()-1)*(Order == RCOrder::row_major)); }

    std::ptrdiff_t jump_to_stripe( std::size_t stripe, std::size_t dim, RCOrder order) const {
        std::ptrdiff_t result=0;
        if( order == RCOrder::col_major){
            for(std::size_t ii=0; ii!=dims(); ++ii){
                if( ii == dim ) continue;
                if( !stripe ) break;
                result += ( stripe % derived()._shape[ii]) * derived()._stride[ii+(Order==RCOrder::row_major)];
                stripe /= derived()._shape[ii];
            }
        } else {
            for(std::size_t ii=dims(); ii!=0; --ii){
                if( ii-1 == dim ) continue;
                if( !stripe ) break;
                result += ( stripe % derived()._shape[ii-1]) * derived()._stride[ii-1+(Order==RCOrder::row_major)];
                stripe /= derived()._shape[ii-1];
            }
        }
        return result;
    }

    auto get_stripe( std::size_t stripe_num, std::size_t dim, RCOrder order){
        return DenseStripe<T,ReadWrite::writeable>( derived().data()+jump_to_stripe(stripe_num,dim,order), stride(dim+(Order==RCOrder::row_major)), shape(dim));
    }

    auto get_stripe( std::size_t stripe_num, std::size_t dim, RCOrder order) const {
        return DenseStripe<T,ReadWrite::read_only>( derived().data()+jump_to_stripe(stripe_num,dim,order), stride(dim+(Order==RCOrder::row_major)), shape(dim));
    }

    decltype(auto) required_stripe_dim() const { return dims(); }

    // ===============================================
    // Utils
    
    // check_expression
    // Tests whether an expression is compatible.

    template<class U>
    void check_expression( const DenseExpression<U>& expression){
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
        derived()._stride[dims()] = 1;
        for( std::size_t ii=dims(); ii!=0; --ii){
            derived()._stride[ii-1] = derived()._stride[ii] * derived()._shape[ii-1];            
        }
    }

    void set_stride() noexcept requires (Order == RCOrder::col_major) {
        derived()._stride[0] = 1;
        for( std::size_t ii=0; ii!=dims(); ++ii){
            derived()._stride[ii+1] = derived()._stride[ii] * derived()._shape[ii];
        }
    }

    // is_contiguous
    // Determine if a container is contiguous. As this is guaranteed for everything except a DenseView, this function
    // is trivial. DenseView shadows it with a much more interesting function.
    constexpr bool is_contiguous() const noexcept { return true;}

    // is_omp_parallelisable
    // Determine if iterator is OpenMP compatible.
    // Is true for all dense containers, but must be defined for compatibility with DenseExpressions, some of which
    // are not trivially parallelisable.
    constexpr bool is_omp_parallelisable() const noexcept { return true; }

    // variadic_memjump
    // used with round-bracket indexing.
    // requires stride of length dims+1, with the largest stride iterating all the way to the location of end().
    // If row_major, the largest stride is stride[0]. If col_major, the largest stride is stride[dims].

    // Base case
    template<std::size_t N, std::integral Int>
    constexpr decltype(auto) variadic_memjump_impl( Int coord) const noexcept {
        return derived()._stride[N] * coord; 
    }

    // Recursive step
    template<std::size_t N, std::integral Int, std::integral... Ints>
    constexpr decltype(auto) variadic_memjump_impl( Int coord, Ints... coords) const noexcept {
        return (derived()._stride[N] * coord) + variadic_memjump_impl<N+1,Ints...>(coords...);
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
    static constexpr auto col_major = index_sequence_to_array(
        typename reverse_index_sequence<typename variadic_stride_impl<std::index_sequence<1,ShapeInts...>,std::index_sequence<>>::type>::type()
    );
    static constexpr auto row_major = index_sequence_to_array(
        typename variadic_stride_impl<typename reverse_index_sequence<std::index_sequence<ShapeInts...,1>>::type,std::index_sequence<>>::type()
    );
};

} //namespace
#endif
