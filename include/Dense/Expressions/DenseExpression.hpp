#ifndef __ULTRA_DENSE_EXPRESSION_HPP
#define __ULTRA_DENSE_EXPRESSION_HPP

/*! \file DenseExpression.hpp
 *  \brief Defines expression templates for `Dense` objects.
 *
 *  Expressions are used to represent complex computations and to avoid unnecessary
 *  allocations and copies. The use of expressions permits lazy evaluation over each
 *  element of a `Dense` object, such that no intermediate arrays are built.
 *
 *  See https://en.wikipedia.org/wiki/Expression_templates for more information.
 */

#include "ultramat/include/Dense/DenseUtils.hpp"
#include "ultramat/include/Dense/DenseStripe.hpp"

namespace ultra {

// ==============================================
// DenseExpression

/*! \brief A CRTP base class for all types of `Dense` 
 *  \tparam T The class inheriting from `DenseExpression`, using CRTP.
 *
 *  `DenseExpression` is the CRTP base class for all expressions involving
 *  `Dense` objects. This includes arithmetic methods, `cmath` overloads, 
 *  more complex manipulations, and even `Dense` objects themselves. This permits
 *  any `Dense` object to be involved in an expression with any other `Dense` object.
 *  The public functions defined within must be present in any class deriving from it.
 */
template<class T>
class DenseExpression {
    //! Utility function casting `*this` to `T&`.
    constexpr T& derived() noexcept { return static_cast<T&>(*this); }
    //! Utility function casting `*this` to `const T&`.
    constexpr const T& derived() const noexcept { return static_cast<const T&>(*this); }
    
    public:

    //! Returns the total number of elements in the derived object.
    constexpr decltype(auto) size() const { return derived().size(); }
    //! Returns number of dimensions of derived object.
    constexpr decltype(auto) dims() const { return derived().dims(); }
    //! Returns size of derived object in each dimension (usually represented by `std::vector<std::size_t>` or `std::array<std::size_t,N>`)
    constexpr decltype(auto) shape() const { return derived().shape(); }
    //! Returns size of derived object in the given dimension.
    constexpr decltype(auto) shape(std::size_t ii) const { return derived().shape(ii); }
    //! Returns the row/column-major ordering of the derived object.
    static constexpr decltype(auto) order() { return T::order(); }

    //! Returns true when the derived object is contiguous.
    constexpr bool is_contiguous() const noexcept { return derived().is_contiguous(); }
    //! Returns true when the derived object is broadcasted.
    constexpr bool is_broadcasting() const noexcept { return derived().is_broadcasting(); }
    //! Returns true when the derived object may be parallelised within each stripe.
    constexpr bool is_omp_parallelisable() const noexcept { return derived().is_omp_parallelisable(); }


    /*! @name begin
     *  Returns iterator pointing to the beginning of the derived class.
     */
    ///@{

    //! Non-const version
    constexpr decltype(auto) begin() { return derived().begin(); }
    //! Const version
    constexpr decltype(auto) begin() const { return derived().begin(); }
    ///@}
    
    /*! @name end
     *  Returns iterator pointing to the end of the derived class (one element after the last valid element)
     */
    ///@{

    //! Non-const version
    constexpr decltype(auto) end() { return derived().end(); }
    //! Const version
    constexpr decltype(auto) end() const { return derived().end(); }
    ///@}

    /*! @name get_stripe
     *  Returns stripe from derived object.
     */
    ///@{
    
    //! Read/write version
    decltype(auto) get_stripe( const DenseStripeIndex& striper) { return derived().get_stripe(striper); }
    //! Read-only version
    decltype(auto) get_stripe( const DenseStripeIndex& striper) const { return derived().get_stripe(striper); }
    ///@}
 
    //! Returns the mandatory striping dimension of the derived object. Should return dims() if it may be striped in any dimension.
    decltype(auto) required_stripe_dim() const { return derived().required_stripe_dim(); }
};

// ==============================================
// ExpressionException

//! Generic exception for when expressions go wrong. Behaves like `std::runtime_error`.
class ExpressionException : public std::runtime_error {
    using std::runtime_error::runtime_error;
};

// ==============================================
// eval and friends

//! Helper alias which gives an appropriate intermediate type to be generated by an expression.
template<class T>
using eval_result = Dense<
    typename std::remove_cvref_t<T>::value_type,
    (std::remove_cvref_t<T>::order() == DenseOrder::mixed ? default_order : std::remove_cvref_t<T>::order())
>;

/*! @name eval
 *  Forces evaluation of an expression to a temporary, and returns it.
 */
///@{

//! Operates on const reference
template<class T>
decltype(auto) eval( const DenseExpression<T>& t){
    return eval_result<T>(static_cast<const T&>(t));
}

//! Operates on non-const (forwarding) reference
template<class T>
decltype(auto) eval( DenseExpression<T>&& t){
    return eval_result<T>(static_cast<T&&>(t));
}
///@}

/*! @name view
 * Forces evaluation of an expression to a temporary, and returns a view to that temporary. Take care to avoid dangling references!
 */
///@{

//! Operates on const reference and takes arbitrary number of slices
template<class T, class... Slices> requires ( std::is_same<Slice,Slices>::value && ...)
eval_result<T> view( const DenseExpression<T>& t, const Slices&... slices){
    return eval(eval(static_cast<const T&>(t)).view(slices...));
}

//! Operates on non-const (forwarding) reference and takes arbitrary number of slices
template<class T, class... Slices> requires ( std::is_same<Slice,Slices>::value && ...)
eval_result<T> view( DenseExpression<T>&& t, const Slices&... slices){
    return eval(eval(static_cast<T&&>(t)).view(slices...));
}

//! Operates on const reference and takes a container of slices
template<class T, std::ranges::range Slices> requires ( std::is_same<Slice,typename Slices::value_type>::value )
eval_result<T> view( const DenseExpression<T>& t, const Slices& slices){
    return eval(eval(static_cast<const T&>(t)).view(slices));
}

//! Operates on non-const (forwarding) reference and takes a container of slices
template<class T, std::ranges::range Slices> requires ( std::is_same<Slice,typename Slices::value_type>::value )
eval_result<T> view( DenseExpression<T>&& t, const Slices& slices){
    return eval(eval(static_cast<T&&>(t)).view(slices));
}
///@}

/*! @name reshape
 *  Forces evaluation of an expression to a temporary, reshapes it, and returns the result.
 */
///@{

//! Operates on const reference with a given shape (usually `std::vector<std::size_t>` or `std::array<std::size_t,N>`)
template<class T, shapelike Shape>
eval_result<T> reshape( const DenseExpression<T>& t, const Shape& shape){
    return eval(eval(static_cast<const T&>(t)).reshape(shape));
}

//! Operates on non-const (forwarding) reference with a given shape (usually `std::vector<std::size_t>` or `std::array<std::size_t,N>`)
template<class T, shapelike Shape>
eval_result<T> reshape( DenseExpression<T>&& t, const Shape& shape){
    return eval(eval(static_cast<T&&>(t)).reshape(shape));
}

//! Operates on const reference with shape given by a series of integers
template<class T, std::integral... Ints>
eval_result<T> reshape( const DenseExpression<T>& t, Ints... ints){
    return eval(eval(static_cast<const T&>(t)).reshape(ints...));
}

//! Operates on non-const (forwarding) reference with shape given by a series of integers
template<class T, std::integral... Ints>
eval_result<T> reshape( DenseExpression<T>&& t, Ints... ints){
    return eval(eval(static_cast<T&&>(t)).reshape(ints...));
}
///@}

/*! @name permute
 *  Forces evaluation of an expression to a temporary, permutes the dimensions, and returns the result.
 */
///@{

//! Operates on const reference with a permutation given by a container of unique integers, i.e. `std::vector<std::size_t>{1,0}` is the matrix transpose.
template<class T, shapelike Shape>
eval_result<T> permute( const DenseExpression<T>& t, const Shape& shape){
    return eval(eval(static_cast<const T&>(t)).permute(shape));
}

//! Operates on non-const (forwarding) reference with a permutation given by a container of unique integers, i.e. `std::vector<std::size_t>{1,0}` is the matrix transpose.
template<class T, shapelike Shape>
eval_result<T> permute( DenseExpression<T>&& t, const Shape& shape){
    return eval(eval(static_cast<T&&>(t)).permute(shape));
}

//! Operates on const reference with a permutation given by a series of unique integers
template<class T, std::integral... Perm>
eval_result<T> permute( const DenseExpression<T>& t, Perm... permutations){
    return eval(eval(static_cast<const T&>(t)).permute(permutations...));
}

//! Operates on non-const (forwarding) reference with a permutation given by a series of unique integers
template<class T, std::integral... Perm>
eval_result<T> permute( DenseExpression<T>&& t, Perm... permutations){
    return eval(eval(static_cast<const T&&>(t)).permute(permutations...));
}
///@}

/*! @name transpose
 *  Forces evaluation of an expression to a temporary, performs the transpose, and returns the result. Valid only for 2D `Dense` objects.
 */
///@{

//! Operates on const reference
template<class T>
decltype(auto) transpose( const DenseExpression<T>& t) {
    return eval(eval(static_cast<const T&>(t)).transpose());
}

//! Operates on non-const (forwarding) reference
template<class T>
decltype(auto) transpose( DenseExpression<T>&& t) {
    return eval(eval(static_cast<T&&>(t)).transpose());
}
///@}

/*! @name hermitian
 *  Forces evaluation of an expression to a temporary, performs the Hermitian transpose, and returns the result. Valid only for 2D `Dense` objects.
 */
///@{

//! Operates on const reference
template<class T>
decltype(auto) hermitian( const DenseExpression<T>& t) {
    return eval(conj(eval(static_cast<const T&>(t)).transpose()));
}

//! Operates on non-const (forwarding) reference
template<class T>
decltype(auto) hermitian( DenseExpression<T>&& t) {
    return eval(conj(eval(static_cast<T&&>(t)).transpose()));
}
///@}

} // namespace ultra
#endif
