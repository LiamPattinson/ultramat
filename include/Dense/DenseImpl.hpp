#ifndef __ULTRA_DENSE_IMPL_HPP
#define __ULTRA_DENSE_IMPL_HPP

/*! \file DenseImpl.hpp
 *  \brief Defines CRTP base class for dense objects.
 */

#include "ultramat/include/Utils/Utils.hpp"
#include "DenseUtils.hpp"
#include "DenseStripe.hpp"
#include "Expressions/DenseExpression.hpp"

namespace ultra {

// ===============================================
// DenseImpl


/*! \brief CRTP base class for all dense containers.
 *
 *  Defines generic methods for #ultra::Dense, #ultra::DenseFixed, and #ultra::DenseView.
 */
template<class T>
class DenseImpl {

    //! Convenience function for accessing methods from the derived class
    constexpr T& derived() noexcept { return static_cast<T&>(*this); }

    //! Convenience function for accessing const methods from the derived class
    constexpr const T& derived() const noexcept { return static_cast<const T&>(*this); }

    protected:

    // ===============================================
    // Attributes

    //! Get the number of dimensions of the derived class.
    constexpr auto dims() const noexcept { return derived()._shape.size(); }

    //! Get the total number of elements contained in a derived object.
    constexpr auto size() const noexcept { return derived()._data.size(); }

    //! Get the size of the `dim`'th dimension of the derived object -- see \ref dense_shape.
    constexpr auto shape( std::size_t dim) const noexcept { return derived()._shape[dim]; }

    //! Get the `dim`'th \ref dense_stride of the derived object.
    constexpr auto stride( std::size_t dim) const noexcept { return derived()._stride[dim]; }

    //! Get a const reference to the derived object's \ref dense_shape
    constexpr const auto& shape() const noexcept { return derived()._shape; }

    //! Get a const reference to the derived object's \ref dense_stride
    constexpr const auto& stride() const noexcept { return derived()._stride; }

    //! Get the \ref dense_order of the derived class.
    static constexpr DenseOrder order() { return T::order(); }

    //! The \ref dense_order of the derived class.
    static constexpr DenseOrder Order = order();

    // ===============================================
    // Data access

    /*! @name data
     *  Advanced users -- get a raw pointer to the contained data of the derived object. 
     */
    ///@{

    //! Non-const (writeable) version
    constexpr auto* data() noexcept { return derived()._data.data(); }

    //! Const (read-only) version
    constexpr const auto* data() const noexcept { return derived()._data.data(); }

    ///@}

    //! Set all elements of the derived object to the value of `u`.
    template<class U>
    constexpr void fill( const U& u) {
        std::ranges::fill(derived(),u);
    }

    /*! @name operator()
     *  Access individual elements using integer coordinates, or take a view using slices
     */
    ///@{
    
    /*! \brief Access using a series of integers, by value
     *  Note that there is no check to ensure the correct number of integers has been provided!
     */
    template<std::integral... Ints> 
    auto operator()( Ints... coords ) const noexcept {
        return derived()._data[variadic_memjump(coords...)];
    }
    
    /*! \brief Access using a series of integers, by reference
     *  This permits individual elements to be updated on the left hand side of an expression.
     *  Note that there is no check to ensure the correct number of integers has been provided!
     */
    template<std::integral... Ints> 
    auto& operator()( Ints... coords ) noexcept {
        return derived()._data[variadic_memjump(coords...)];
    }

    /*! \brief Create sliced view
     *  If at least one argument provided to `operator()` is a Slice, instead returns a sliced view.
     *  Note that there is no check to ensure the correct number of slices/integers has been provided!
     */
    template<class... Ts> requires variadic_contains<Slice,Ts...>::value 
    auto operator()( Ts... ts ) noexcept {
        return DenseView<T>(derived()).slice(to_slice(ts)...);
    }

    /*! \brief Create sliced read-only view
     *  If at least one argument provided to `operator()` is a Slice, instead returns a sliced view.
     *  Note that there is no check to ensure the correct number of slices/integers has been provided!
     */
    template<class... Ts> requires variadic_contains<Slice,Ts...>::value 
    auto operator()( Ts... ts ) const noexcept {
        return DenseView<T,ReadWrite::read_only>(derived()).slice(to_slice(ts)...);
    }
    
    /*! \brief Access using a #ultra::shapelike, returns by value
     *  #ultra::shapelike is used to represent any non-Ultramat array-like structure of integers.
     *  Note that there is no check to ensure the correct number of integers has been provided!
     */
    template<shapelike Coords>
    auto operator()( const Coords& coords) const {
        return derived()._data[
            std::inner_product(coords.begin(),coords.end(),derived()._stride.begin()+(Order==DenseOrder::row_major),0)
        ];
    }

    /*! \brief Access using a #ultra::shapelike, returns by reference
     *  This permits individual elements to be updated on the left hand side of an expression.
     *  #ultra::shapelike is used to represent any non-Ultramat array-like structure of integers.
     *  Note that there is no check to ensure the correct number of integers has been provided!
     */
    template<shapelike Coords>
    auto& operator()( const Coords& coords) {
        return derived()._data[
            std::inner_product(coords.begin(),coords.end(),derived()._stride.begin()+(Order==DenseOrder::row_major),0)
        ];
    }

    ///@}

    // ===============================================
    //  Expressions

    struct _Equal{ template<class U, class V> void operator()( U& u, const V& v) const { u = v; }};
    struct _AddEqual{ template<class U, class V> void operator()( U& u, const V& v) const { u += v; }};
    struct _SubEqual{ template<class U, class V> void operator()( U& u, const V& v) const { u -= v; }};
    struct _MulEqual{ template<class U, class V> void operator()( U& u, const V& v) const { u *= v; }};
    struct _DivEqual{ template<class U, class V> void operator()( U& u, const V& v) const { u /= v; }};

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
            DenseStripeIndex end( stripe_dim, Order, s, 1);
#pragma omp parallel for schedule(dynamic,1) default(none) shared(f,expression,stripe_dim,s) firstprivate(end)
            for( auto striper = DenseStripeIndex(stripe_dim,Order,s,0); striper != end; ++striper){
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
        return accept_expression<_Equal>(expression);
    }

    template<class U>
    decltype(auto) equal_expression( DenseExpression<U>&& expression){
        return accept_expression<_Equal>(std::move(expression));
    }

    // In-place operators

    template<class U>
    decltype(auto) operator=( const DenseExpression<U>& expression) {
        return accept_expression<_Equal>(expression);
    }

    template<class U>
    decltype(auto) operator=( DenseExpression<U>&& expression) {
        return accept_expression<_Equal>(std::move(expression));
    }

    template<class U>
    decltype(auto) operator+=( const DenseExpression<U>& expression ){
        return accept_expression<_AddEqual>(expression);
    }

    template<class U>
    decltype(auto) operator+=( DenseExpression<U>&& expression ){
        return accept_expression<_AddEqual>(std::move(expression));
    }

    template<class U>
    decltype(auto) operator-=( const DenseExpression<U>& expression ){
        return accept_expression<_SubEqual>(expression);
    }

    template<class U>
    decltype(auto) operator-=( DenseExpression<U>&& expression ){
        return accept_expression<_SubEqual>(std::move(expression));
    }

    template<class U>
    decltype(auto) operator*=( const DenseExpression<U>& expression ){
        return accept_expression<_MulEqual>(expression);
    }

    template<class U>
    decltype(auto) operator*=( DenseExpression<U>&& expression ){
        return accept_expression<_MulEqual>(std::move(expression));
    }

    template<class U>
    decltype(auto) operator/=( const DenseExpression<U>& expression ){
        return accept_expression<_DivEqual>(expression);
    }
    
    template<class U>
    decltype(auto) operator/=( DenseExpression<U>&& expression ){
        return accept_expression<_DivEqual>(std::move(expression));
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

    template<class... Slices> requires ( (std::is_same<Slice,Slices>::value || std::is_integral<Slices>::value) && ... )
    DenseView<T> view(const Slices&... slices) {
        return DenseView<T>(derived()).slice(to_slice(slices)...);
    }

    template<class... Slices> requires ( (std::is_same<Slice,Slices>::value || std::is_integral<Slices>::value) && ... )
    DenseView<T,ReadWrite::read_only> view(const Slices&... slices) const {
        return DenseView<T,ReadWrite::read_only>(derived()).slice(to_slice(slices)...);
    }

    template<std::ranges::range Slices> requires ( std::is_same<typename Slices::value_type,Slice>::value )
    DenseView<T> view(const Slices& slices) {
        return DenseView<T>(derived()).slice(slices);
    }

    template<std::ranges::range Slices> requires ( std::is_same<typename Slices::value_type,Slice>::value )
    DenseView<T,ReadWrite::read_only> view(const Slices& slices) const {
        return DenseView<T,ReadWrite::read_only>(derived()).slice(slices);
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
        derived()._shape.resize(shape.size());
        derived()._stride.resize(shape.size()+1);
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

    auto _get_stripe_helper( const DenseStripeIndex& striper) const {
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

    auto get_stripe( const DenseStripeIndex& striper){
        auto* stripe_ptr = derived().data();
        std::ptrdiff_t stripe_jump, stripe_stride;
        std::tie(stripe_jump,stripe_stride) = _get_stripe_helper(striper);
        return DenseStripe<T,ReadWrite::writeable>( stripe_ptr + stripe_jump, stripe_stride, striper.stripe_size());
    }

    auto get_stripe( const DenseStripeIndex& striper) const {
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

} // namespace ultra
#endif
