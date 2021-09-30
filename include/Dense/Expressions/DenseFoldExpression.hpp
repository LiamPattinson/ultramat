#ifndef __ULTRA_DENSE_FOLD_EXPRESSION_HPP
#define __ULTRA_DENSE_FOLD_EXPRESSION_HPP

/*! \file DenseFoldExpression.hpp
 *  \brief Defines varieties of fold/reduction operations over `Dense` objects (e.g. min, max, sum)
 *
 * Defines `DenseGeneralFoldExpression`, `DenseLinearFoldExpression`, `DenseAccumulateExpresson`, and `DenseBooleanFoldExpresson`.
 * Each perform a similar role of reducing the dimension of a `Dense` object by one, by means such as calculating the sum along one dimension.
 * - `DenseGeneralFoldExpression` applies an arbitrary function over a whole dimension using begin and end iterators.
 * - `DenseLinearFoldExpression` applies an arbitrary function which can be applied to each element one-by-one.
 * - `DenseAccumulateExpression` is similar to `DenseLinearFoldExpression`, though the starting value is taken as the first element.
 * - `DenseBooleanFoldExpression` is optimised to early-exit as soon as the result of a fold is known.
 */

#include "DenseExpression.hpp"

namespace ultra {

// ==============================================
// Fold Policies

// Inherits from one of four 'policy' classes
// - GeneralFold: start_val is provided of type StartT, takes function of form `StartT f(T.begin(),T.end(),StartT) `
// - LinearFold: start_val is provided of type StartT, takes function of form `StartT f(StartT,T) `
// - Accumulate: start_val is not provided, takes function of form `T f(T,T)`. Includes min, max, sum, prod
// - BooleanFold: all_of, any_of, none_of, all may stop early under the right conditions

template<class F, class StartType, class T>
class GeneralFoldPolicy {
protected:
    using input_type = std::remove_cvref_t<T>;
    using input_value_type = typename input_type::value_type;
    using start_type = StartType;
    using value_type = decltype(std::declval<F>()(std::declval<input_type>().begin(),std::declval<input_type>().end(),std::declval<start_type>()));
    static_assert( std::is_convertible<start_type,std::remove_cvref_t<value_type>>::value );
    static constexpr bool is_general_fold = true;
    static constexpr bool is_linear_fold = false;
    static constexpr bool is_accumulate = false;
    static constexpr bool is_boolean_fold = false;
    start_type _start_val;
public:
    GeneralFoldPolicy( const start_type& start_val) : _start_val(start_val) {}
    start_type get() const { return _start_val; }

    value_type exec( const F& f, const input_type& t, const DenseStriper& striper) const {
        F f_copy(f);
        value_type val = _start_val;
        auto stripe = t.get_stripe(striper);
        val = f_copy( stripe.begin(), stripe.end(), val);
        return val;
    }
};

template<class F, class StartType, class T>
class LinearFoldPolicy {
protected:
    using input_type = std::remove_cvref_t<T>;
    using input_value_type = typename std::remove_cvref_t<T>::value_type;
    using start_type = StartType;
    using value_type = decltype(std::declval<F>()(std::declval<start_type>(),std::declval<input_value_type>()));
    static_assert( std::is_convertible<start_type,std::remove_cvref_t<value_type>>::value );
    static constexpr bool is_general_fold = false;
    static constexpr bool is_linear_fold = true;
    static constexpr bool is_accumulate = false;
    static constexpr bool is_boolean_fold = false;
    start_type _start_val;
public:
    LinearFoldPolicy( const start_type& start_val) : _start_val(start_val) {}
    start_type get() const { return _start_val; }

    value_type exec( const F& f, const input_type& t, const DenseStriper& striper) const {
        F f_copy(f);
        value_type val = _start_val;
        auto stripe = t.get_stripe(striper);
        for( auto&& x : stripe ) val = f_copy(val,x);
        return val;
    }
};

template<class F, class ValueType, class T>
class AccumulatePolicy {
protected:
    using input_type = std::remove_cvref_t<T>;
    using input_value_type = typename std::remove_cvref_t<T>::value_type;
    using value_type = decltype(std::declval<F>()(std::declval<input_value_type>(),std::declval<input_value_type>()));
    static_assert( std::is_convertible<ValueType,input_value_type>::value );
    static constexpr bool is_general_fold = false;
    static constexpr bool is_linear_fold = false;
    static constexpr bool is_accumulate = true;
    static constexpr bool is_boolean_fold = false;
public:
    value_type exec( const F& f, const input_type& t, const DenseStriper& striper) const {
        F f_copy(f);
        auto stripe = t.get_stripe(striper);
        auto it = stripe.begin();
        auto end = stripe.end();
        std::remove_cvref_t<value_type> val = *it;
        ++it;
        for(; it != end; ++it ) val = f_copy(val,*it);
        return val;
    }
};

template<class F, class ValueType, class T>
class BooleanFoldPolicy {
    static_assert( std::is_same<ValueType,bool>::value );
protected:
    using value_type = ValueType;
    using input_type = std::remove_cvref_t<T>;
    using input_value_type = typename std::remove_cvref_t<T>::value_type;
    using result_type = decltype(std::declval<F>()(std::declval<ValueType>(),std::declval<input_value_type>()));
    static_assert( std::is_same<result_type,value_type>::value );
    static constexpr bool is_general_fold = false;
    static constexpr bool is_linear_fold = false;
    static constexpr bool is_accumulate = false;
    static constexpr bool is_boolean_fold = true;
public:
    value_type exec( const F&, const input_type& t, const DenseStriper& striper) const {
        /* Rather than using F directly, we will instead make use of start_bool and early_exit_bool */ \
        bool result = F::start_bool;
        auto stripe = t.get_stripe(striper);
        for( auto&& x : stripe){
            if( x == F::early_exit_bool ) return !result;
        }
        return result;
    }
};

template<class F, class ValueType, class T, template<class,class,class> class FoldPolicy>
class DenseFoldExpressionImpl : public DenseExpression<DenseFoldExpressionImpl<F,ValueType,T,FoldPolicy>>, public FoldPolicy<F,ValueType,T> {
    
public:

    using input_value_type = FoldPolicy<F,ValueType,T>::input_value_type;
    using value_type = FoldPolicy<F,ValueType,T>::value_type;

    static constexpr bool is_general_fold = FoldPolicy<F,ValueType,T>::is_general_fold;
    static constexpr bool is_linear_fold = FoldPolicy<F,ValueType,T>::is_linear_fold;
    static constexpr bool is_accumulate = FoldPolicy<F,ValueType,T>::is_accumulate;
    static constexpr bool is_boolean_fold = FoldPolicy<F,ValueType,T>::is_boolean_fold;
    static constexpr bool requires_start_val = is_general_fold || is_linear_fold;

private:

    using arg_t = std::conditional_t< std::is_lvalue_reference<T>::value, T, std::remove_cvref_t<T>>;
    using ref_t = const std::remove_cvref_t<arg_t>&;
    using Stripe_t = decltype(std::declval<std::add_const_t<std::remove_cvref_t<T>>>().get_stripe(std::declval<DenseStriper>()));

    F           _f;
    arg_t       _t;
    std::size_t _fold_dim;
    std::size_t _fold_size;

public:

    DenseFoldExpressionImpl() = delete;
    DenseFoldExpressionImpl( const DenseFoldExpressionImpl& ) = delete;
    DenseFoldExpressionImpl( DenseFoldExpressionImpl&& ) = default;
    DenseFoldExpressionImpl& operator=( const DenseFoldExpressionImpl& ) = delete;
    DenseFoldExpressionImpl& operator=( DenseFoldExpressionImpl&& ) = default;

    DenseFoldExpressionImpl( const F& f, T&& t, const ValueType& start_val, std::size_t fold_dim ) : 
        FoldPolicy<F,ValueType,T>(start_val),
        _f(f),
        _t(std::forward<T>(t)),
        _fold_dim(fold_dim),
        _fold_size(_t.shape(fold_dim))
    {
        if( _fold_dim >= _t.dims() ) throw ExpressionException("Ultramat: Fold dimension must be smaller than dims()");
    }

    DenseFoldExpressionImpl( const F& f, T&& t, std::size_t fold_dim ) : 
        _f(f),
        _t(std::forward<T>(t)),
        _fold_dim(fold_dim),
        _fold_size(_t.shape(fold_dim))
    {
        if( _fold_dim >= _t.dims() ) throw ExpressionException("Ultramat: Fold dimension must be smaller than dims()");
    }

    std::size_t dims() const { return std::max((std::size_t)1,_t.dims()-1); }
    decltype(auto) shape(std::size_t ii) const { return (_t.dims()==1 ? 1 : _t.shape(ii < _fold_dim ? ii : ii+1)); }
    constexpr bool is_contiguous() const noexcept { return _t.is_contiguous(); }
    constexpr bool is_broadcasting() const noexcept { return _t.is_broadcasting(); }
    constexpr bool is_omp_parallelisable() const noexcept { return _t.is_omp_parallelisable(); }
    decltype(auto) required_stripe_dim() const { return dims(); }

    static constexpr DenseOrder order() { return std::remove_cvref_t<T>::order(); }

    std::size_t size() const { 
        std::size_t result=1;
        for(std::size_t ii=0; ii<dims(); ++ii) result*=shape(ii);
        return result;
    }

    std::vector<std::size_t> shape() const {
        std::vector<std::size_t> result(dims());
        for( std::size_t ii=0; ii<dims(); ++ii) result[ii] = shape(ii);
        return result;
    }

    // Define const_iterator class
 
    // Notes:
    // begin() should return a const_iterator containing a stripe generator over _t, passing on the stripe dim
    // Dereferencing this will generate a stripe, perform the fold operation, and return the result.
    // Incrementing this will increment the stripe generator.

    class const_iterator : public FoldPolicy<F,ValueType,T> {

        F            _f;
        ref_t        _t;
        DenseStriper _striper;
        
        public:

        const_iterator() = delete;
        const_iterator( const const_iterator& ) = default;
        const_iterator( const_iterator&& ) = default;
        const_iterator& operator=( const const_iterator& ) = default;
        const_iterator& operator=( const_iterator&& ) = default;
        
        template<shapelike Shape> 
        const_iterator( const F& f, ref_t t, std::size_t fold_dim, DenseOrder order, const Shape& shape, bool end, value_type start_val) :
            FoldPolicy<F,ValueType,T>(start_val),
            _f(f),
            _t(t),
            _striper(fold_dim,order,shape,end)
        {}

        template<shapelike Shape> 
        const_iterator( const F& f, ref_t t, std::size_t fold_dim, DenseOrder order, const Shape& shape, bool end) :
            _f(f),
            _t(t),
            _striper(fold_dim,order,shape,end)
        {}

        template<shapelike Shape> 
        const_iterator( const F& f, ref_t t, std::size_t fold_dim, const Shape& shape, bool end, value_type start_val) :
            const_iterator(f,t,fold_dim,t.order(),shape,end,start_val)
        {}

        template<shapelike Shape> 
        const_iterator( const F& f, ref_t t, std::size_t fold_dim, const Shape& shape, bool end) :
            const_iterator(f,t,fold_dim,t.order(),shape,end)
        {}

        decltype(auto) operator*(){
            return FoldPolicy<F,ValueType,T>::exec( _f, _t, _striper);
        }

        const_iterator& operator++() { ++_striper; return *this; }
        const_iterator& operator--() { --_striper; return *this; }
        template<std::integral I> const_iterator& operator+=( const I& ii) { _striper+=ii; return *this; }
        template<std::integral I> const_iterator& operator-=( const I& ii) { _striper-=ii; return *this; }
        template<std::integral I> const_iterator operator+( const I& ii) { auto result(*this); result+=ii; return result; }
        template<std::integral I> const_iterator operator-( const I& ii) { auto result(*this); result-=ii; return result; }
        bool operator==(const const_iterator& it) const { return _striper == it._striper; }
        auto operator<=>(const const_iterator& it) const { return _striper <=> it._striper; }
        std::ptrdiff_t operator-(const const_iterator& it) const { return _striper - it._striper; }
    };

    const_iterator begin() const requires (requires_start_val) {
        return const_iterator(_f,_t,_fold_dim,_t.shape(),0,FoldPolicy<F,ValueType,T>::get());
    }
    
    const_iterator end() const requires (requires_start_val) {
        return const_iterator(_f,_t,_fold_dim,_t.shape(),1,FoldPolicy<F,ValueType,T>::get());
    }

    const_iterator begin() const requires (!requires_start_val) {
        return const_iterator(_f,_t,_fold_dim,_t.shape(),0);
    }

    const_iterator end() const requires (!requires_start_val) {
        return const_iterator(_f,_t,_fold_dim,_t.shape(),1);
    }

    // Define stripe class

    class Stripe : public FoldPolicy<F,ValueType,T> {

        F            _f;
        ref_t        _t;
        std::size_t  _fold_dim;
        std::size_t  _fold_size;
        bool         _fold_1d;
        DenseStriper _striper;

        public:
        
        Stripe( const F& f, ref_t t, std::size_t fold_dim, std::size_t fold_size, const DenseStriper& striper, bool fold_1d, const value_type& val) :
            FoldPolicy<F,ValueType,T>(val),
            _f(f),
            _t(t),
            _fold_dim(fold_dim),
            _fold_size(fold_size),
            _fold_1d(fold_1d),
            _striper(striper)
        {}

        Stripe( const F& f, ref_t t, std::size_t fold_dim, std::size_t fold_size, const DenseStriper& striper, bool fold_1d) :
            _f(f),
            _t(t),
            _fold_dim(fold_dim),
            _fold_size(fold_size),
            _fold_1d(fold_1d),
            _striper(striper)
        {}

        // Define iterator type
        class Iterator : public FoldPolicy<F,ValueType,T> {

            F            _f;
            ref_t        _t;
            std::size_t  _stripe_dim;
            std::size_t  _fold_dim;
            DenseStriper _striper;

            public: 

            Iterator() = delete;
            Iterator( const Iterator& ) = default;
            Iterator( Iterator&& ) = default;
            Iterator& operator=( const Iterator& ) = default;
            Iterator& operator=( Iterator&& ) = default;
            
            Iterator( const F& f, ref_t t, std::size_t stripe_dim, std::size_t fold_dim, const DenseStriper& striper, value_type start_val) :
                FoldPolicy<F,ValueType,T>(start_val),
                _f(f),
                _t(t),
                _stripe_dim(stripe_dim + (striper.order()==DenseOrder::row_major) + (stripe_dim >= fold_dim)),
                _fold_dim(fold_dim + (striper.order()==DenseOrder::row_major)),
                _striper(striper)
            {}

            Iterator( const F& f, ref_t t, std::size_t stripe_dim, std::size_t fold_dim, const DenseStriper& striper) :
                _f(f),
                _t(t),
                _stripe_dim(stripe_dim + (striper.order()==DenseOrder::row_major) + (stripe_dim >= fold_dim)),
                _fold_dim(fold_dim + (striper.order()==DenseOrder::row_major)),
                _striper(striper)
            {}

            decltype(auto) operator*(){
                return FoldPolicy<F,ValueType,T>::exec( _f, _t, _striper);
            }

            Iterator& operator++() { ++_striper.index(_stripe_dim); return *this; }
            Iterator& operator--() { --_striper.index(_stripe_dim); return *this; }
            template<std::integral I> Iterator& operator+=( const I& ii) { _striper.index(_stripe_dim) += ii; return *this; }
            template<std::integral I> Iterator& operator-=( const I& ii) { _striper.index(_stripe_dim) -= ii; return *this; }
            template<std::integral I> Iterator operator+( const I& ii) { auto result(*this); result+=ii; return result; }
            template<std::integral I> Iterator operator-( const I& ii) { auto result(*this); result-=ii; return result; }
            bool operator==( const Iterator& other) const { return _striper.index(_stripe_dim) == other._striper.index(_stripe_dim);}
            auto operator<=>( const Iterator& other) const { return _striper.index(_stripe_dim) <=> other._striper.index(_stripe_dim);}
            std::ptrdiff_t operator-( const Iterator& other) const { return _striper.index(_stripe_dim) - other._striper.index(_stripe_dim);}

        };

        DenseStriper get_inner_striper( bool end) const {
            std::vector<std::size_t> inner_shape(_striper.dims()+(!_fold_1d),0);
            if( _fold_1d){
                inner_shape[0] = _fold_size; 
            } else {
                for( std::size_t ii=0; ii<_striper.dims(); ++ii){
                    inner_shape[ii + (ii>=_fold_dim)] = _striper.shape(ii); 
                }
                inner_shape[_fold_dim] = _fold_size;
            }
            DenseStriper inner_striper( _fold_dim, _striper.order(), inner_shape, 0);
            if( _fold_1d){
                inner_striper.index(0) = _striper.index(0); 
            } else {
                for( std::size_t ii=0; ii<=_striper.dims(); ++ii){
                    inner_striper.index(ii + (ii>=(_fold_dim+(_striper.order()==DenseOrder::row_major)))) = _striper.index(ii); 
                }
            }
            if( end ){
                std::size_t end_dim = _striper.stripe_dim() + (_striper.stripe_dim() >= _fold_dim);
                if( _fold_1d) {
                    inner_striper.index(0) = inner_striper.shape(0);
                } else {
                    inner_striper.index(end_dim + (_striper.order() == DenseOrder::row_major)) = inner_striper.shape(end_dim);
                }
            }
            return inner_striper;
        }

        Iterator begin() const requires (requires_start_val) {
            auto inner_striper = get_inner_striper(0);
            return Iterator( _f, _t, _striper.stripe_dim(), _fold_dim, inner_striper, FoldPolicy<F,ValueType,T>::get());
        }

        Iterator begin() const requires (!requires_start_val) {
            auto inner_striper = get_inner_striper(0);
            return Iterator( _f, _t, _striper.stripe_dim() ,_fold_dim, inner_striper);
        }

        Iterator end() const requires (requires_start_val) {
            auto inner_striper = get_inner_striper(1);
            return Iterator( _f, _t, _striper.stripe_dim(), _fold_dim, inner_striper, FoldPolicy<F,ValueType,T>::get());
        }

        Iterator end() const requires (!requires_start_val) {
            auto inner_striper = get_inner_striper(1);
            return Iterator( _f, _t, _striper.stripe_dim(), _fold_dim, inner_striper);
        }
    };

    // Get stripe from _t
    decltype(auto) get_stripe( const DenseStriper& striper) const requires (requires_start_val) {
        return Stripe(_f,_t,_fold_dim,_fold_size,striper,_t.dims()==1,FoldPolicy<F,ValueType,T>::get());
    }

    decltype(auto) get_stripe( const DenseStriper& striper) const requires (!requires_start_val) {
        return Stripe(_f,_t,_fold_dim,_fold_size,striper,_t.dims()==1);
    }
};

template<class F, class ValueType, class T> using DenseGeneralFoldExpression = DenseFoldExpressionImpl<F,ValueType,T,GeneralFoldPolicy>;
template<class F, class ValueType, class T> using DenseLinearFoldExpression = DenseFoldExpressionImpl<F,ValueType,T,LinearFoldPolicy>;
template<class F, class T> using DenseAccumulateExpression = DenseFoldExpressionImpl<F,typename std::remove_cvref_t<T>::value_type,T,AccumulatePolicy>;
template<class F, class T> using DenseBooleanFoldExpression = DenseFoldExpressionImpl<F,bool,T,BooleanFoldPolicy>;

} // namespace ultra
#endif
