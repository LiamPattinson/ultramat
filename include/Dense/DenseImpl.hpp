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

    private:

    /*! @name Assignment Structs
     *  A collection of utility structs used alongside #accept_expression() to allow the same code to be reused
     *  for assignment and in-place operators.
     */
    ///@{

    //! Utility struct representing assignment operation
    struct _Equal{ template<class U, class V> void operator()( U& u, const V& v) const { u = v; }};
    //! Utility struct representing in-place addition
    struct _AddEqual{ template<class U, class V> void operator()( U& u, const V& v) const { u += v; }};
    //! Utility struct representing in-place subtraction
    struct _SubEqual{ template<class U, class V> void operator()( U& u, const V& v) const { u -= v; }};
    //! Utility struct representing in-place multiplication
    struct _MulEqual{ template<class U, class V> void operator()( U& u, const V& v) const { u *= v; }};
    //! Utility struct representing in-place division
    struct _DivEqual{ template<class U, class V> void operator()( U& u, const V& v) const { u /= v; }};

    ///@}

    public:

    /*! \brief Function defining how an expression is evaluated to fill a \ref DenseObject,
     *  \tparam F Utility struct defining assignment or an in-place operator. See #_Equal.
     *  \tparam E A generic expression.
     *  This function takes a generic expression, which may be a \ref DenseObject, and evaluates
     *  each element of the derived object.
     *  The expression must be broadcastable to the size of the derived object.
     */
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

    /*! @name equal_expression
     *  Shorthand to using #accept_expression() using an assignment operator.
     */
    ///@{

    //! Equate to a const reference expression.
    template<class U>
    decltype(auto) equal_expression( const DenseExpression<U>& expression){
        return accept_expression<_Equal>(expression);
    }

    //! Equate to an rvalue expression.
    template<class U>
    decltype(auto) equal_expression( DenseExpression<U>&& expression){
        return accept_expression<_Equal>(std::move(expression));
    }

    ///@}

    /*! @name In-place expression operators
     *  Allows the derived class to be updated in-place using expressions.
     */
    ///@{

    //! Assign from const-ref expression
    template<class U>
    decltype(auto) operator=( const DenseExpression<U>& expression) {
        return accept_expression<_Equal>(expression);
    }

    //! Assign from rvalue expression
    template<class U>
    decltype(auto) operator=( DenseExpression<U>&& expression) {
        return accept_expression<_Equal>(std::move(expression));
    }

    //! Add in place from const-ref expression
    template<class U>
    decltype(auto) operator+=( const DenseExpression<U>& expression ){
        return accept_expression<_AddEqual>(expression);
    }

    //! Add in place from rvalue expression
    template<class U>
    decltype(auto) operator+=( DenseExpression<U>&& expression ){
        return accept_expression<_AddEqual>(std::move(expression));
    }

    //! Subtract in place from const-ref expression
    template<class U>
    decltype(auto) operator-=( const DenseExpression<U>& expression ){
        return accept_expression<_SubEqual>(expression);
    }

    //! Subtract in place from rvalue expression
    template<class U>
    decltype(auto) operator-=( DenseExpression<U>&& expression ){
        return accept_expression<_SubEqual>(std::move(expression));
    }

    //! Multiply in place from const-ref expression
    template<class U>
    decltype(auto) operator*=( const DenseExpression<U>& expression ){
        return accept_expression<_MulEqual>(expression);
    }

    //! Multiply in place from rvalue expression
    template<class U>
    decltype(auto) operator*=( DenseExpression<U>&& expression ){
        return accept_expression<_MulEqual>(std::move(expression));
    }

    //! Divide in place from const-ref expression
    template<class U>
    decltype(auto) operator/=( const DenseExpression<U>& expression ){
        return accept_expression<_DivEqual>(expression);
    }
    
    //! Divide in place from rvalue expression
    template<class U>
    decltype(auto) operator/=( DenseExpression<U>&& expression ){
        return accept_expression<_DivEqual>(std::move(expression));
    }
    ///@}

    /*! @name Scalar in-place updates
     *  Broadcast a scalar to the size of the derived object, and operate in place.
     */
    ///@{

    //! Assign to scalar
    template<class U> requires number<U>
    decltype(auto) operator=( U u) {
        return operator=(DenseFixed<U,Order,1>(u));
    }

    //! Add scalar in place
    template<class U> requires number<U>
    decltype(auto) operator+=( U u) {
        return operator+=(DenseFixed<U,Order,1>(u));
    }

    //! Subtract scalar in place
    template<class U> requires number<U>
    decltype(auto) operator-=( U u) {
        return operator-=(DenseFixed<U,Order,1>(u));
    }

    //! Multiply by scalar in place
    template<class U> requires number<U>
    decltype(auto) operator*=( U u) {
        return operator*=(DenseFixed<U,Order,1>(u));
    }

    //! Divide by scalar in place
    template<class U> requires number<U>
    decltype(auto) operator/=( U u) {
        return operator/=(DenseFixed<U,Order,1>(u));
    }

    ///@}

    // ===============================================
    // View creation

    /*! @name View Creation 
     * \brief Methods for generating #ultra::DenseView from other \ref DenseObject%s.
     */
    ///@{

    //! Create a full view over the derived object.
    DenseView<T> view() {
        return DenseView<T>(derived());
    }
    
    //! Create a read-only full view over the derived object.
    DenseView<T,ReadWrite::read_only> view() const {
        return DenseView<T,ReadWrite::read_only>(derived());
    }

    //! Create a partial view over the derived object using slices or integers
    template<class... Slices> requires ( (std::is_same<Slice,Slices>::value || std::is_integral<Slices>::value) && ... )
    DenseView<T> view(const Slices&... slices) {
        return DenseView<T>(derived()).slice(to_slice(slices)...);
    }

    //! Create a read-only partial view over the derived object using slices or integers
    template<class... Slices> requires ( (std::is_same<Slice,Slices>::value || std::is_integral<Slices>::value) && ... )
    DenseView<T,ReadWrite::read_only> view(const Slices&... slices) const {
        return DenseView<T,ReadWrite::read_only>(derived()).slice(to_slice(slices)...);
    }

    //! Create a partial view over the derived object using a container of slices
    template<std::ranges::range Slices> requires ( std::is_same<typename Slices::value_type,Slice>::value )
    DenseView<T> view(const Slices& slices) {
        return DenseView<T>(derived()).slice(slices);
    }

    //! Create a read-only partial view over the derived object using a container of slices
    template<std::ranges::range Slices> requires ( std::is_same<typename Slices::value_type,Slice>::value )
    DenseView<T,ReadWrite::read_only> view(const Slices& slices) const {
        return DenseView<T,ReadWrite::read_only>(derived()).slice(slices);
    }

    ///@}

    // ===============================================
    // Reshaping
    
    /*! @name Reshaping
     * \brief Methods for modifying the shape of contiguous and dynamic \ref DenseObject%s
     */
    ///@{

    /*! \brief Set the shape of an existing \ref DenseObject using a #ultra::shapelike.
     *  The total number of elements in the reshaped object must match the original number of elements. This method
     *  is not available for #ultra::DenseFixed.
     */
    template<shapelike Shape>
    decltype(auto) reshape( const Shape& shape ){
        // Ensure this is contiguous
        if( !derived().is_contiguous() ) throw std::runtime_error("Ultramat: Cannot reshape a non-contiguous array");
        // Ensure that the new shape makes sense
        for( auto&& x : shape ){
            if( x <= 0 ) throw std::runtime_error("Ultramat: Cannot have zero or negative shape.");
        }
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

    /*! \brief Set the shape of an existing \ref DenseObject using a series of integers.
     */
    template<std::integral... Ints>
    decltype(auto) reshape( Ints... shape){
        return reshape(std::array<std::size_t,sizeof...(Ints)>{{shape...}});
    }

    ///@}

    // ===============================================
    // Permuting/Transposing

    /*! @name Permuting and Transposing
     * \brief Create a view with a different order of dimensions for \ref DenseObject%s.
     *
     *  When permuting, the number of provided dimensions must match the current number of dimensions, and must be a permutation of
     *  [0,1,2,3,...,N-1] for an ND object. For example, a matrix transpose is achieved by calling `permute(1,0)`. Swapping the
     *  last two dimensions of a 4D object would be `permute(0,1,3,2)`, and completely reversing the order of dimensions of a
     *  5D object would be `permute(4,3,2,1,0)`. The `transpose()` or `t()` functions are simple shorthand for `permute(1,0)`,
     *  while `hermitian()` will transpose and take the complex conjugate. Note that `hermitian()` will create a non-view, as
     *  otherwise it would require modifying the original object.
     */
    ///@{

    //! Create a view and swap the order of dimensions using a #ultra::shapelike
    template<shapelike Perm>
    auto permute( const Perm& permutations) {
        return derived().view().permute(permutations);
    }

    //! Create a read-only view and swap the order of dimensions using a #ultra::shapelike
    template<shapelike Perm>
    auto permute( const Perm& permutations) const {
        return derived().view().permute(permutations);
    }

    //! Create a view and swap the order of dimensions using a series of integers
    template<std::integral... Perm>
    auto permute( Perm... permutations) {
        return derived().view().permute(permutations...);
    }

    //! Create a read-only view and swap the order of dimensions using a series of integers
    template<std::integral... Perm>
    auto permute( Perm... permutations) const {
        return derived().view().permute(permutations...);
    }

    //! Create a view and swap the dimensions of a 2D object.
    auto transpose() {
        if( derived().dims() != 2 ) throw std::runtime_error("Ultramat: transpose() requires dims() == 2. Perhaps you wanted permute()?");
        return permute(1,0);
    }

    //! Create a read-only view and swap the dimensions of a 2D object.
    auto transpose() const {
        if( derived().dims() != 2 ) throw std::runtime_error("Ultramat: transpose() requires dims() == 2. Perhaps you wanted permute()?");
        return permute(1,0);
    }

    //! Create a view and swap the dimensions of a 2D object.
    auto t() { return transpose(); }

    //! Create a read-only view and swap the dimensions of a 2D object.
    auto t() const { return transpose(); }

    //! Create a 2D #ultra::Dense, taking the complex conjugate and swapping the dimensions.
    auto hermitian() const {
        return hermitian(*this);
    }

    //! Create a 2D #ultra::Dense, taking the complex conjugate and swapping the dimensions.
    auto h() const {
        return hermitian(*this);
    }

    ///@}

    // ===============================================
    // Iteration

    /*! @name Iteration
     * \brief Get iterators to the start and end of the derived object. Not used by #ultra::DenseView.
     */
    ///@{
    
    constexpr auto begin() noexcept { return derived()._data.begin(); }         //!< Get iterator to the start of the \ref DenseObject.
    constexpr auto begin() const noexcept { return derived()._data.cbegin(); }  //!< Get const-iterator to the start of the \ref DenseObject.
    constexpr auto cbegin() const noexcept { return derived()._data.cbegin(); } //!< Get const-iterator to the start of the \ref DenseObject.
    constexpr auto end() noexcept { return derived()._data.end(); }             //!< Get iterator to the end of the \ref DenseObject.
    constexpr auto end() const noexcept { return derived()._data.cend(); }      //!< Get const-iterator to the end of the \ref DenseObject.
    constexpr auto cend() const noexcept { return derived()._data.cend(); }     //!< Get const-iterator to the end of the \ref DenseObject.

    ///@}

    // ===============================================
    // Striped Iteration

    /*! @name Striped Iteration
     * \brief Methods for creating #ultra::DenseStripe, mostly used internally.
     */

    ///@{

    private:

    //! Get memory position at which a stripe should begin and the stride of the stripe.
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

    //! Get #ultra::DenseStripe given a #ultra::DenseStripeIndex.
    auto get_stripe( const DenseStripeIndex& striper){
        auto* stripe_ptr = derived().data();
        std::ptrdiff_t stripe_jump, stripe_stride;
        std::tie(stripe_jump,stripe_stride) = _get_stripe_helper(striper);
        return DenseStripe<T,ReadWrite::writeable>( stripe_ptr + stripe_jump, stripe_stride, striper.stripe_size());
    }

    //! Get read-only #ultra::DenseStripe given a #ultra::DenseStripeIndex
    auto get_stripe( const DenseStripeIndex& striper) const {
        auto* stripe_ptr = derived().data();
        std::ptrdiff_t stripe_jump, stripe_stride;
        std::tie(stripe_jump,stripe_stride) = _get_stripe_helper(striper);
        return DenseStripe<T,ReadWrite::read_only>( stripe_ptr + stripe_jump, stripe_stride, striper.stripe_size());
    }

    //! Must stripes be over a particular dimension? In general, no, so return dims().
    constexpr std::size_t required_stripe_dim() const { return dims();}

    ///@}

    // ===============================================
    // Utils
    
    //! Test whether an expression is exactly compatible -- same dims and shape
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

    //! Use shape to populate stride via a cumulative product, starting with 1. Row major version.
    void set_stride() noexcept requires (Order == DenseOrder::row_major) {
        derived()._stride[dims()] = 1;
        for( std::size_t ii=dims(); ii!=0; --ii){
            derived()._stride[ii-1] = derived()._stride[ii] * derived()._shape[ii-1];            
        }
    }

    //! Use shape to populate stride via a cumulative product, starting with 1. Column major version.
    void set_stride() noexcept requires (Order == DenseOrder::col_major) {
        derived()._stride[0] = 1;
        for( std::size_t ii=0; ii!=dims(); ++ii){
            derived()._stride[ii+1] = derived()._stride[ii] * derived()._shape[ii];
        }
    }

    //! Determine if a container is contiguous. True for everything except a #ultra::DenseView, which may be either true or false.
    constexpr bool is_contiguous() const noexcept { return true;}

    //! Determine if this object is broadcasted. Always false for \ref DenseObject%s, though expressions might be.
    constexpr bool is_broadcasting() const noexcept { return false;}

    // Determine if iterator is OpenMP compatible. Is true for all \ref DenseObject%s.
    constexpr bool is_omp_parallelisable() const noexcept { return true; }

    private:

    /*! @name variadic_memjump
     *  \brief Utility function used to enable round-bracket indexing.
     *  Uses stride to convert a series of integers to the required jump in memory to access a particular element. This function
     *  does no error checking, so negative integers, out-of-bounds, or just providing too many integers can cause undefined behaviour.
     */

    ///@{

    //! Base case
    template<std::size_t N, std::integral Int>
    constexpr decltype(auto) variadic_memjump_impl( Int coord) const noexcept {
        return derived()._stride[N] * coord; 
    }

    //! Recursive step
    template<std::size_t N, std::integral Int, std::integral... Ints>
    constexpr decltype(auto) variadic_memjump_impl( Int coord, Ints... coords) const noexcept {
        return (derived()._stride[N] * coord) + variadic_memjump_impl<N+1,Ints...>(coords...);
    }

    //! Interface
    template<std::integral... Ints>
    constexpr decltype(auto) variadic_memjump( Ints... coords) const noexcept {
        // if row major, must skip first element of stride
        return variadic_memjump_impl<(Order==DenseOrder::row_major?1:0),Ints...>(coords...);
    }

    ///@}
};

} // namespace ultra
#endif
