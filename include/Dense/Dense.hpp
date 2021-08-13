#ifndef __ULTRA_DENSE_HPP
#define __ULTRA_DENSE_HPP

// Dense.hpp
//
// Defines components used to build N-dimensional arrays.
//
// * DenseImpl   -- CRTP base class, defines most common functions
// * DenseView   -- A dense object that references data existing elsewhere. Views can be transformed in a number of ways.
// * DenseStripe -- A 1D view along a given dimension. Used extensively for iteration over dense objects, including expressions.
// * Dense       -- Dynamically allocated N-dimensional array
// * DenseFixed  -- Fixed-size statically allocated N-dimensional array

#include "DenseExpression.hpp"

namespace ultra {

// ===============================================
// DenseImpl
//
// CRTP Base Class
// Defines a base class for all dense array-like objects, including Arrays, Vectors, Matrices, and their fixed-size counterparts.

template<class T>
class DenseImpl {
    constexpr T& derived() noexcept { return static_cast<T&>(*this); }
    constexpr const T& derived() const noexcept { return static_cast<const T&>(*this); }

    protected:

    // ===============================================
    // Attributes

    constexpr auto dims() const noexcept { return derived()._shape.size(); }
    constexpr auto size() const noexcept { return derived()._data.size(); }
    constexpr auto shape( std::size_t dim) const noexcept { return derived()._shape[dim]; }
    constexpr auto stride( std::size_t dim) const noexcept { return derived()._stride[dim]; }
    constexpr const auto& shape() const noexcept { return derived()._shape; }
    constexpr const auto& stride() const noexcept { return derived()._stride; }
    static constexpr DenseOrder order() { return T::order(); }
    static constexpr DenseOrder Order = order();

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
    
    template<shapelike Coords>
    auto operator()( const Coords& coords) const {
        return derived()._data[
            std::inner_product(coords.begin(),coords.end(),derived()._stride.begin()+(Order==DenseOrder::row_major),0)
        ];
    }

    template<shapelike Coords>
    auto& operator()( const Coords& coords) {
        return derived()._data[
            std::inner_product(coords.begin(),coords.end(),derived()._stride.begin()+(Order==DenseOrder::row_major),0)
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
    decltype(auto) accept_expression( E&& expression ){
        F f{};

        // broadcasting bits
        bool is_broadcasting = expression.is_broadcasting();
        if( dims() > expression.dims()){
            is_broadcasting = true;
        }
        if( dims() < expression.dims()){
            throw ExpressionException("Ultramat: Tried to construct/assign array-like object of dims " + std::to_string(dims()) 
                    + " with expression of dims " + std::to_string(expression.dims()) +
                    ". Under broadcasting rules, you cannot write to an object with fewer dimensions.");
        }
        bool broadcasting_failed = false;
        if( order() == DenseOrder::col_major) {
            for( std::size_t ii=0; ii<expression.dims(); ++ii){
                if( shape(ii) != expression.shape(ii)){
                    if ( expression.shape(ii) == 1 ){
                       is_broadcasting = true;
                    } else {
                        broadcasting_failed = true;
                    }
                }
            }
        } else {
            for( std::size_t ii=0; ii<expression.dims(); ++ii){
                std::size_t jj= ii +(dims() - expression.dims());
                if( expression.shape(ii) != expression.shape(jj)){
                    if( expression.shape(jj) == 1 ){
                        is_broadcasting = true;
                    } else {
                        broadcasting_failed = true;
                    }
                }
            }
        }
        if(broadcasting_failed){
            std::string array_shape("( ");
            std::string expression_shape("( ");
            for( std::size_t ii=0; ii<dims(); ++ii){
                array_shape += std::to_string(shape(ii)) + ' ';
            }
            for( std::size_t ii=0; ii<expression.dims(); ++ii){
                expression_shape += std::to_string(expression.shape(ii)) + ' ';
            }
            array_shape += ')';
            expression_shape += ')';
            throw ExpressionException("Ultramat: Tried to construct/assign array-like object of shape " + array_shape 
                            + "with expression of shape " + expression_shape +
                            ". This is not permitted under broadcasting rules.");
        }

        if( derived().is_contiguous() && expression.is_contiguous() && expression.order() == Order && !is_broadcasting){
            // simple linear update
            auto it_end = end();
#pragma omp parallel for schedule(guided,16) default(none) shared(f,expression) firstprivate(it_end)
            for(auto it=IteratorTuple(begin(),expression.begin()); it != it_end; ++it){
                f(*it,*get<1>(it));
            }
        } else {
            // general 'striped' update
            std::size_t stripe_dim = expression.required_stripe_dim();
            if( stripe_dim == dims() ) stripe_dim = ( Order == DenseOrder::row_major ? dims()-1 : 0 );
            auto s = shape();
            DenseStriper end( stripe_dim, Order, s, 1);
#pragma omp parallel for schedule(dynamic,1) default(none) shared(f,expression,stripe_dim,s) firstprivate(end)
            for( auto striper = DenseStriper(stripe_dim,Order,s,0); striper != end; ++striper){
                auto stripe = get_stripe(striper);
                auto expr_stripe = expression.get_stripe(striper);
                auto expr_it = expr_stripe.begin();
                for(auto it=stripe.begin(), it_end=stripe.end(); it != it_end; ++it, ++expr_it) f(*it,*expr_it);
            }
        }

        return derived();
    }

    template<class U>
    decltype(auto) equal_expression( const DenseExpression<U>& expression){
        return accept_expression<Equal>(expression);
    }

    template<class U>
    decltype(auto) equal_expression( DenseExpression<U>&& expression){
        return accept_expression<Equal>(std::move(expression));
    }

    // In-place operators

    template<class U>
    decltype(auto) operator=( const DenseExpression<U>& expression) {
        return accept_expression<Equal>(expression);
    }

    template<class U>
    decltype(auto) operator=( DenseExpression<U>&& expression) {
        return accept_expression<Equal>(std::move(expression));
    }

    template<class U>
    decltype(auto) operator+=( const DenseExpression<U>& expression ){
        return accept_expression<AddEqual>(expression);
    }

    template<class U>
    decltype(auto) operator+=( DenseExpression<U>&& expression ){
        return accept_expression<AddEqual>(std::move(expression));
    }

    template<class U>
    decltype(auto) operator-=( const DenseExpression<U>& expression ){
        return accept_expression<SubEqual>(expression);
    }

    template<class U>
    decltype(auto) operator-=( DenseExpression<U>&& expression ){
        return accept_expression<SubEqual>(std::move(expression));
    }

    template<class U>
    decltype(auto) operator*=( const DenseExpression<U>& expression ){
        return accept_expression<MulEqual>(expression);
    }

    template<class U>
    decltype(auto) operator*=( DenseExpression<U>&& expression ){
        return accept_expression<MulEqual>(std::move(expression));
    }

    template<class U>
    decltype(auto) operator/=( const DenseExpression<U>& expression ){
        return accept_expression<DivEqual>(expression);
    }
    
    template<class U>
    decltype(auto) operator/=( DenseExpression<U>&& expression ){
        return accept_expression<DivEqual>(std::move(expression));
    }

    // Scalar in-place updates

    template<class U> requires number<U>
    decltype(auto) operator=( U u) {
        return operator=(DenseFixed<U,Order,1>(u));
    }

    template<class U> requires number<U>
    decltype(auto) operator+=( U u) {
        return operator+=(DenseFixed<U,Order,1>(u));
    }

    template<class U> requires number<U>
    decltype(auto) operator-=( U u) {
        return operator-=(DenseFixed<U,Order,1>(u));
    }

    template<class U> requires number<U>
    decltype(auto) operator*=( U u) {
        return operator*=(DenseFixed<U,Order,1>(u));
    }

    template<class U> requires number<U>
    decltype(auto) operator/=( U u) {
        return operator/=(DenseFixed<U,Order,1>(u));
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

    template<shapelike Shape>
    decltype(auto) reshape( const Shape& shape ){
        // Ensure this is contiguous
        if( !derived().is_contiguous() ) throw std::runtime_error("Ultramat: Cannot reshape a non-contiguous array");
        // Ensure that the new shape has the correct size.
        auto size = std::accumulate(shape.begin(),shape.end(),1,std::multiplies<typename Shape::value_type>{});
        if( size != derived().size() ) throw std::runtime_error("Ultramat: Cannot reshape, result would have incorrect size");
        // Set new shape and stride
        derived().resize_shape_and_stride(shape.size());
        std::ranges::copy( shape, derived()._shape.begin());
        set_stride();
        return derived(); 
    }

    template<std::integral... Ints>
    decltype(auto) reshape( Ints... shape){
        return reshape(std::array<std::size_t,sizeof...(Ints)>{{shape...}});
    }

    // ===============================================
    // Permuting/Transposing

    template<shapelike Perm>
    auto permute( const Perm& permutations) {
        return derived().view().permute(permutations);
    }

    template<shapelike Perm>
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

    auto hermitian() const {
        return hermitian(*this);
    }

    auto h() const {
        return hermitian(*this);
    }

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

    private:

    auto _get_stripe_helper( const DenseStriper& striper) const {
        std::ptrdiff_t stripe_jump=0, stripe_stride=0;
        bool broadcasting = ( dims() < striper.dims());
        bool mixed_order = ( order() != striper.order());
        if( broadcasting && mixed_order ) throw std::runtime_error("Ultramat: Can't broadcast row-major and col-major objects at once.");
        if( striper.order() == DenseOrder::col_major ){
            for( std::size_t ii=0; ii<dims(); ++ii){
                if( ii==striper.stripe_dim() ){
                    /* ptr stays the same, do nothing */ 
                } else if( shape(ii)==1 ){
                    /* ptr stays the same, but note if we're broadcasting */ 
                    if( striper.shape(ii) != 1 ) broadcasting=true;
                } else {
                    stripe_jump += striper.index(ii) * stride(ii+(Order==DenseOrder::row_major));
                }
            }
            if( striper.stripe_dim() >= dims() || (shape(striper.stripe_dim()) == 1 && striper.shape(striper.stripe_dim()) != 1 )) {
                stripe_stride = 0;
                broadcasting = true;
            } else {
                stripe_stride = stride(striper.stripe_dim()+(Order==DenseOrder::row_major));
            }
        } else {
            std::size_t extra_dims = striper.dims() - dims();
            for( std::size_t ii=striper.dims(); ii!=extra_dims; --ii){
                std::size_t jj = ii - extra_dims;
                if( ii==striper.stripe_dim()+1 ){
                    /* ptr stays the same, do nothing */ 
                } else if ( shape(jj-1)==1 ){
                    /* ptr stays the same, but note if we're broadcasting */ 
                    if( striper.shape(ii-1) != 1 ) broadcasting=true;
                } else {
                    stripe_jump += striper.index(ii) * stride(jj-(Order==DenseOrder::col_major));
                }
            }
            if( striper.stripe_dim() < extra_dims || (shape(striper.stripe_dim()-extra_dims) == 1 && striper.shape(striper.stripe_dim()) != 1)) {
                stripe_stride = 0;
                broadcasting = true;
            } else {
                stripe_stride = stride(striper.stripe_dim()-extra_dims+(Order==DenseOrder::row_major));
            }
        }
        if( broadcasting && mixed_order) throw std::runtime_error("Ultramat: Can't broadcast row-major and col-major objects at once.");
        return std::make_pair( stripe_jump, stripe_stride);
    }
    
    public:

    auto get_stripe( const DenseStriper& striper){
        auto* stripe_ptr = derived().data();
        std::ptrdiff_t stripe_jump, stripe_stride;
        std::tie(stripe_jump,stripe_stride) = _get_stripe_helper(striper);
        return DenseStripe<T,ReadWrite::writeable>( stripe_ptr + stripe_jump, stripe_stride, striper.stripe_size());
    }

    auto get_stripe( const DenseStriper& striper) const {
        auto* stripe_ptr = derived().data();
        std::ptrdiff_t stripe_jump, stripe_stride;
        std::tie(stripe_jump,stripe_stride) = _get_stripe_helper(striper);
        return DenseStripe<T,ReadWrite::read_only>( stripe_ptr + stripe_jump, stripe_stride, striper.stripe_size());
    }

    constexpr std::size_t required_stripe_dim() const { return dims();}

    // ===============================================
    // Utils
    
    // check_expression
    // Tests whether an expression is exactly compatible -- same dims and shape

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

    void set_stride() noexcept requires (Order == DenseOrder::row_major) {
        derived()._stride[dims()] = 1;
        for( std::size_t ii=dims(); ii!=0; --ii){
            derived()._stride[ii-1] = derived()._stride[ii] * derived()._shape[ii-1];            
        }
    }

    void set_stride() noexcept requires (Order == DenseOrder::col_major) {
        derived()._stride[0] = 1;
        for( std::size_t ii=0; ii!=dims(); ++ii){
            derived()._stride[ii+1] = derived()._stride[ii] * derived()._shape[ii];
        }
    }

    // is_contiguous
    // Determine if a container is contiguous. As this is guaranteed for everything except a DenseView, this function
    // is trivial. DenseView shadows it with a much more interesting function.
    constexpr bool is_contiguous() const noexcept { return true;}

    // is_broadcasting
    // Always false for dense objects, though expressions of dense objects might be.
    constexpr bool is_broadcasting() const noexcept { return false;}

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
        return variadic_memjump_impl<(Order==DenseOrder::row_major?1:0),Ints...>(coords...);
    }

};

// ===============================================
// DenseView
//
// A container that refers to data belonging to some other dense container.
// A view does not control the lifetime of its contents. It may be
// writeable or read-only, contiguous or non-contiguous.
// Copies/moves in constant time.

template<class T, ReadWrite rw>
class DenseView : public DenseExpression<DenseView<T,rw>>, public DenseImpl<DenseView<T,rw>> {

    friend DenseImpl<DenseView<T,rw>>;

    static constexpr ReadWrite other_rw = (rw == ReadWrite::read_only ? ReadWrite::writeable : ReadWrite::read_only);
    friend DenseView<T,other_rw>;

    void resize_shape_and_stride( std::size_t size){
        _shape.resize(size);
        _stride.resize(size+1);
    }

public:

    using value_type = typename T::value_type;
    using difference_type = std::ptrdiff_t;
    using pointer = std::conditional_t<rw==ReadWrite::writeable,value_type*, const value_type*>;
    using reference = std::conditional_t<rw==ReadWrite::writeable,value_type&,const value_type&>;
    using shape_type = std::vector<std::size_t>;
    using stride_type = std::vector<std::ptrdiff_t>;
    static constexpr DenseOrder Order = T::order();
    static constexpr DenseOrder order() { return Order; }

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

    using DenseImpl<DenseView<T,rw>>::dims;
    using DenseImpl<DenseView<T,rw>>::shape;
    using DenseImpl<DenseView<T,rw>>::stride;
    using DenseImpl<DenseView<T,rw>>::fill;
    using DenseImpl<DenseView<T,rw>>::get_stripe;
    using DenseImpl<DenseView<T,rw>>::reshape;
    using DenseImpl<DenseView<T,rw>>::required_stripe_dim;
    using DenseImpl<DenseView<T,rw>>::operator();
    using DenseImpl<DenseView<T,rw>>::operator[];
    using DenseImpl<DenseView<T,rw>>::operator=;
    using DenseImpl<DenseView<T,rw>>::operator+=;
    using DenseImpl<DenseView<T,rw>>::operator-=;
    using DenseImpl<DenseView<T,rw>>::operator*=;
    using DenseImpl<DenseView<T,rw>>::operator/=;
    using DenseImpl<DenseView<T,rw>>::is_omp_parallelisable;
    using DenseImpl<DenseView<T,rw>>::is_broadcasting;

    std::size_t size() const noexcept { return _size;}
    pointer data() const noexcept { return _data; }
    pointer data() noexcept requires ( rw == ReadWrite::writeable ){ return _data; }

    bool is_contiguous() const noexcept { return _is_contiguous; }

    bool test_contiguous() const noexcept requires (Order==DenseOrder::row_major) {
        ptrdiff_t stride = 1;
        if( stride != _stride[dims()]) return false;
        for( std::size_t ii=dims(); ii != 0; --ii){
            stride *= _shape[ii-1];
            if( stride != _stride[ii-1]) return false;
        }
        return true;
    }

    bool test_contiguous() const noexcept requires (Order==DenseOrder::col_major) {
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
        return iterator(data() + _stride[Order==DenseOrder::col_major? dims() : 0],_shape,_stride,true);
    }
    
    const_iterator end() const {
        return const_iterator(data() + _stride[Order==DenseOrder::col_major? dims() : 0],_shape,_stride,true);
    }

    // ===============================================
    // Special view methods:
    // - Slicing
    // - Broadcasting
    // - Permuting/Transposing

    template<std::ranges::sized_range Slices> requires ( std::is_same<typename Slices::value_type,Slice>::value )
    DenseView slice( const Slices& slices ) const {
        // Create copy to work with
        DenseView result(*this);
        std::size_t stride_offset = ( Order == DenseOrder::row_major );
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
        if( Order == DenseOrder::row_major ){
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

    template<shapelike Perm>
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
            copy._stride[ii + (Order==DenseOrder::row_major)] = _stride[permutations[ii] + (Order==DenseOrder::row_major)];
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
    static constexpr DenseOrder Order = DenseView<T,rw>::Order;

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
        _pos[Order==DenseOrder::row_major ? stride.size()-1 : 0] = end;
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
    iterator_impl<constness>& operator++() requires ( Order == DenseOrder::col_major ) {
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

    iterator_impl<constness>& operator++() requires ( Order == DenseOrder::row_major ) {
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
    
    iterator_impl<constness>& operator--() requires ( Order == DenseOrder::col_major ) {
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

    iterator_impl<constness>& operator--() requires ( Order == DenseOrder::row_major ) {
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
    iterator_impl<constness>& operator+=( difference_type diff) requires ( Order == DenseOrder::col_major ) {
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

    iterator_impl<constness>& operator+=( difference_type diff) requires ( Order == DenseOrder::row_major ) {
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

    iterator_impl<constness>& operator-=( difference_type diff) requires (Order == DenseOrder::col_major ) {
        // If diff is less than 0, call the in-place add method instead
        if( diff < 0){
            return (*this += (-diff));
        } else {
            std::size_t idx = 0;
            while( diff != 0 && idx != _shape.size() ) {
                // Go to end of current dimension, add the difference onto diff
                _ptr += (_shape[idx]-1 - _pos[idx]) * _stride[idx];
                diff += (_shape[idx]-1 - _pos[idx]);
                _pos[idx] = _shape[idx]-1;
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

    iterator_impl<constness>& operator-=( difference_type diff) requires (Order == DenseOrder::row_major ) {
        // If diff is less than 0, call the in-place add method instead
        if( diff < 0){
            return (*this += (-diff));
        } else {
            std::size_t idx = _shape.size();
            while( diff != 0 && idx != 0 ) {
                // Go to end of current dimension, add the difference onto diff
                _ptr += (_shape[idx-1]-1 - _pos[idx]) * _stride[idx];
                diff += (_shape[idx-1]-1 - _pos[idx]);
                _pos[idx] = _shape[idx-1]-1;
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
        if( Order == DenseOrder::col_major ){
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
    auto operator<=>( const iterator_impl<constness_r>& it_r) const requires ( Order == DenseOrder::row_major ) {
        for( std::size_t ii=0; ii<_pos.size(); ++ii ){
            if( _pos[ii] == it_r._pos[ii] ) continue;
            return ( _pos[ii] < it_r._pos[ii] ? std::strong_ordering::less : std::strong_ordering::greater );
        }
        return std::strong_ordering::equal;
    }

    template<bool constness_r>
    auto operator<=>( const iterator_impl<constness_r>& it_r) const requires ( Order == DenseOrder::col_major ) {
        for( int ii=_pos.size()-1; ii>=0; --ii ){
            if( _pos[ii] == it_r._pos[ii] ) continue;
            return ( _pos[ii] < it_r._pos[ii] ? std::strong_ordering::less : std::strong_ordering::greater );
        }
        return std::strong_ordering::equal;
    }
};

// ===============================================
// DenseStripe
//
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

// ===============================================
// Dense
//
// Defines generic dense array-like containers, including Array, Matrix, Vector, and their fixed-size counterparts.
// Preferred interface is via the Array alias.

template<class T, DenseType Type, DenseOrder Order>
class Dense : public DenseExpression<Dense<T,Type,Order>>, public DenseImpl<Dense<T,Type,Order>> {

    friend DenseImpl<Dense<T,Type,Order>>;

    static constexpr bool is_nd = ( Type == DenseType::nd );
    static constexpr std::size_t fixed_dims = static_cast<std::size_t>(Type);

public:

    using value_type = std::conditional_t<std::is_same<T,bool>::value,Bool,T>;
    using shape_type = std::conditional_t<is_nd,std::vector<std::size_t>,std::array<std::size_t,fixed_dims>>;
    using stride_type = std::conditional_t<is_nd,std::vector<std::size_t>,std::array<std::size_t,fixed_dims+1>>;
    using data_type = std::vector<value_type>;
    using iterator = data_type::iterator;
    using const_iterator = data_type::const_iterator;
    static constexpr DenseOrder order() { return Order; }

    // For convenience, specify row/col major via Array<T>::row/col_major
    using row_major = Dense<T,Type,DenseOrder::row_major>;
    using col_major = Dense<T,Type,DenseOrder::col_major>;

    // View of self
    using View = DenseView<Dense<T,Type,Order>>;

private: 

    shape_type  _shape;
    stride_type _stride;
    data_type   _data;

public:

    // ===============================================
    // Constructors

    Dense() = default;
    Dense( const Dense& other) = default;
    Dense( Dense&& other) = default;
    Dense& operator=( const Dense& other) = default;
    Dense& operator=( Dense&& other) = default;
    
    // Swap
    void swap( Dense& other) noexcept { 
        _shape.swap(other._shape);
        _stride.swap(other._stride);
        _data.swap(other._data);
    }

    friend void swap( Dense& a, Dense& b) noexcept { a.swap(b); }

    // Construct from shape
    template<shapelike Shape> requires ( is_nd )
    Dense( const Shape& shape ) :
        _shape(shape.size()),
        _stride(shape.size()+1),
        _data(std::accumulate(shape.begin(),shape.end(),1,std::multiplies<typename Shape::value_type>{}))
    {
        std::ranges::copy( shape, _shape.begin());
        set_stride();
    }

    template<shapelike Shape> requires ( is_nd )
    Dense( const Shape& shape, const value_type& fill) :
        _shape(shape.size()),
        _stride(shape.size()+1),
        _data(std::accumulate(shape.begin(),shape.end(),1,std::multiplies<typename Shape::value_type>{}),fill)
    {
        std::ranges::copy( shape, _shape.begin());
        set_stride();
    }

    template<shapelike Shape> requires ( !is_nd )
    Dense( const Shape& shape ) :
        _data(std::accumulate(shape.begin(),shape.end(),1,std::multiplies<typename Shape::value_type>{}))
    {
        test_fixed_dims(shape.size());
        std::ranges::copy( shape, _shape.begin());
        set_stride();
    }

    template<shapelike Shape> requires ( !is_nd )
    Dense( const Shape& shape, const value_type& fill) :
        _data(std::accumulate(shape.begin(),shape.end(),1,std::multiplies<typename Shape::value_type>{}),fill)
    {
        test_fixed_dims(shape.size());
        std::ranges::copy( shape, _shape.begin());
        set_stride();
    }

    Dense( std::size_t size ) requires ( !is_nd && fixed_dims == 1 ) : _shape{size}, _data(size) { set_stride(); }
    
    Dense( std::size_t size, const value_type& fill ) requires ( !is_nd && fixed_dims == 1 ) : _shape{size}, _data(size,fill) { set_stride(); }

    Dense( std::size_t rows, std::size_t cols ) requires ( !is_nd && fixed_dims == 2 ) : _shape{rows,cols}, _data(rows*cols) { set_stride(); }
    
    Dense( std::size_t rows, std::size_t cols, const value_type& fill ) requires ( !is_nd && fixed_dims == 2 ) :
        _shape{rows,cols},
        _data(rows*cols,fill) 
    {
        set_stride();
    }

    // Construct from an expression

    template<class U> requires ( is_nd )
    Dense( const DenseExpression<U>& expression) :
        _shape(expression.dims()),
        _stride(expression.dims()+1),
        _data(expression.size())
    {
        for( std::size_t ii = 0; ii < dims(); ++ii) _shape[ii] = expression.shape(ii);
        set_stride();
        equal_expression(expression);
    }

    template<class U> requires ( is_nd )
    Dense( DenseExpression<U>&& expression) :
        _shape(expression.dims()),
        _stride(expression.dims()+1),
        _data(expression.size())
    {
        for( std::size_t ii = 0; ii < dims(); ++ii) _shape[ii] = expression.shape(ii);
        set_stride();
        equal_expression(std::move(expression));
    }

    template<class U> requires ( !is_nd )
    Dense( const DenseExpression<U>& expression) :
        _data(expression.size())
    {
        test_fixed_dims(shape.size());
        for( std::size_t ii = 0; ii < dims(); ++ii) _shape[ii] = expression.shape(ii);
        set_stride();
        equal_expression(expression);
    }

    template<class U> requires ( !is_nd )
    Dense( DenseExpression<U>&& expression) :
        _data(expression.size())
    {
        test_fixed_dims(shape.size());
        for( std::size_t ii = 0; ii < dims(); ++ii) _shape[ii] = expression.shape(ii);
        set_stride();
        equal_expression(std::move(expression));
    }

    template<class U> requires ( is_nd )
    Dense& operator=( const DenseExpression<U>& expression) {
        // check expression shape matches self. If not, resize and reshape in place
        try {
            check_expression(expression);
        } catch(const ExpressionException&) {
            _shape.resize(expression.dims());
            _stride.resize(expression.dims()+1);
            _data.resize(expression.size());
            for( std::size_t ii = 0; ii < dims(); ++ii) _shape[ii] = expression.shape(ii);
            set_stride();
        }
        return equal_expression(expression);
    }

    template<class U> requires ( is_nd )
    Dense& operator=( DenseExpression<U>&& expression) {
        // check expression shape matches self. If not, resize and reshape in place
        try {
            check_expression(expression);
        } catch(const ExpressionException&) {
            _shape.resize(expression.dims());
            _stride.resize(expression.dims()+1);
            _data.resize(expression.size());
            for( std::size_t ii = 0; ii < dims(); ++ii) _shape[ii] = expression.shape(ii);
            set_stride();
        }
        return equal_expression(std::move(expression));
    }

    template<class U> requires ( !is_nd )
    Dense& operator=( const DenseExpression<U>& expression) {
        // check expression shape matches self. If not, resize and reshape in place
        try {
            check_expression(expression);
        } catch(const ExpressionException&) {
            test_fixed_dims(expression.dims());
            _data.resize(expression.size());
            for( std::size_t ii = 0; ii < dims(); ++ii) _shape[ii] = expression.shape(ii);
            set_stride();
        }
        return equal_expression(expression);
    }

    template<class U> requires ( !is_nd )
    Dense& operator=( DenseExpression<U>&& expression) {
        // check expression shape matches self. If not, resize and reshape in place
        try {
            check_expression(expression);
        } catch(const ExpressionException&) {
            test_fixed_dims(expression.dims());
            _data.resize(expression.size());
            for( std::size_t ii = 0; ii < dims(); ++ii) _shape[ii] = expression.shape(ii);
            set_stride();
        }
        return equal_expression(std::move(expression));
    }

    // ===============================================
    // Pull in methods from base class

    using DenseImpl<Dense<T,Type,Order>>::dims;
    using DenseImpl<Dense<T,Type,Order>>::size;
    using DenseImpl<Dense<T,Type,Order>>::shape;
    using DenseImpl<Dense<T,Type,Order>>::stride;
    using DenseImpl<Dense<T,Type,Order>>::data;
    using DenseImpl<Dense<T,Type,Order>>::fill;
    using DenseImpl<Dense<T,Type,Order>>::view;
    using DenseImpl<Dense<T,Type,Order>>::reshape;
    using DenseImpl<Dense<T,Type,Order>>::permute;
    using DenseImpl<Dense<T,Type,Order>>::transpose;
    using DenseImpl<Dense<T,Type,Order>>::t;
    using DenseImpl<Dense<T,Type,Order>>::begin;
    using DenseImpl<Dense<T,Type,Order>>::end;
    using DenseImpl<Dense<T,Type,Order>>::get_stripe;
    using DenseImpl<Dense<T,Type,Order>>::required_stripe_dim;
    using DenseImpl<Dense<T,Type,Order>>::operator();
    using DenseImpl<Dense<T,Type,Order>>::operator[];
    using DenseImpl<Dense<T,Type,Order>>::operator=;
    using DenseImpl<Dense<T,Type,Order>>::operator+=;
    using DenseImpl<Dense<T,Type,Order>>::operator-=;
    using DenseImpl<Dense<T,Type,Order>>::operator*=;
    using DenseImpl<Dense<T,Type,Order>>::operator/=;
    using DenseImpl<Dense<T,Type,Order>>::equal_expression;
    using DenseImpl<Dense<T,Type,Order>>::check_expression;
    using DenseImpl<Dense<T,Type,Order>>::set_stride;
    using DenseImpl<Dense<T,Type,Order>>::is_contiguous;
    using DenseImpl<Dense<T,Type,Order>>::is_omp_parallelisable;
    using DenseImpl<Dense<T,Type,Order>>::is_broadcasting;

private:

    void test_fixed_dims( std::size_t d ) const {
        if( d != dims() ){
            if( fixed_dims==1 ) throw std::runtime_error("Ultra: Tried to construct Vector with shape of size " + std::to_string(d) + '.');
            if( fixed_dims==2 ) throw std::runtime_error("Ultra: Tried to construct Matrix with shape of size " + std::to_string(d) + '.');
        }
    }

    void resize_shape_and_stride( std::size_t size) requires ( is_nd ) {
        _shape.resize(size);
        _stride.resize(size+1);
    }

    void resize_shape_and_stride( std::size_t size) requires ( !is_nd ) {
        test_fixed_dims(size);
    }

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

} // namespace
#endif
