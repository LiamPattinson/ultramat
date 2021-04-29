#ifndef __ULTRA_DENSE_EXPRESSION_HPP
#define __ULTRA_DENSE_EXPRESSION_HPP

// DenseExpression.hpp
//
// See https://en.wikipedia.org/wiki/Expression_templates
// 
// Expressions are used to represent complex computations and to
// avoid unnecessary allocations and copies.

#include "Utils.hpp"

namespace ultra {

// Base DenseExpression
// Requires that class T has a size, dims, shape, and stride.
// Additionally requires that T is iterable and 'stripeable'.

template<class T>
struct DenseExpression {

    constexpr T& derived() noexcept { return static_cast<T&>(*this); }
    constexpr const T& derived() const noexcept { return static_cast<const T&>(*this); }

    decltype(auto) size() const { return derived().size(); }
    decltype(auto) dims() const { return derived().dims(); }
    decltype(auto) shape() const { return derived().shape(); }
    decltype(auto) shape(std::size_t ii) const { return derived().shape(ii); }
    decltype(auto) order() const { return derived().order(); }

    constexpr bool is_contiguous() const noexcept { return derived().is_contiguous(); }
    constexpr bool is_omp_parallelisable() const noexcept { return derived().is_omp_parallelisable(); }

    decltype(auto) begin() { return derived().begin(); }
    decltype(auto) begin() const { return derived().begin(); }

    decltype(auto) get_stripe(std::size_t stripe, std::size_t dim, RCOrder order) { return derived().get_stripe(stripe,dim,order); }
    decltype(auto) get_stripe(std::size_t stripe, std::size_t dim, RCOrder order) const { return derived().get_stripe(stripe,dim,order); }
    decltype(auto) num_stripes(std::size_t dim) const { return derived().num_stripes(dim); }
};

// Generic exception for when expressions go wrong.

class ExpressionException : public std::runtime_error {
    using std::runtime_error::runtime_error;
};

// eval
// Forces evaluation of an expession to a temporary, and returns it.
// When applied to a container such as an array, it does not store a temporary, but rather provides
// perfect forwarding of that container.

template<class T>
using eval_result = std::conditional_t< 
    types<std::remove_cvref_t<T>>::is_dense,
    decltype(std::forward<T>(std::declval<T>())),
    Array<typename std::remove_cvref_t<T>::value_type>
>;

template<class T>
decltype(auto) eval( DenseExpression<T>& t){
    return eval_result<T>(static_cast<T&>(t));
}

template<class T>
decltype(auto) eval( DenseExpression<T>&& t){
    return eval_result<T>(static_cast<T&&>(t));
}

// Expressions utils

struct Begin { template<class T> decltype(auto) operator()( T&& t) { return t.begin(); }};
struct End { template<class T> decltype(auto) operator()( T&& t) { return t.end(); }};
struct Deref { template<class T> decltype(auto) operator()( T&& t) { return *t; }};
struct PrefixInc { template<class T> decltype(auto) operator()( T&& t) { return ++t; }};
struct Order{ template<class T> bool operator()( T&& t) { return t.order(); }};
struct IsContiguous { template<class T> bool operator()( T&& t) { return t.is_contiguous(); }};
struct IsOmpParallelisable { template<class T> bool operator()( T&& t) { return t.is_omp_parallelisable(); }};

struct GetStripe {
    std::size_t _stripe, _dim;
    RCOrder _order;
    template<class T>
    decltype(auto) operator()( T&& t) { return t.get_stripe(_stripe,_dim,_order); }
};

// apply_to_each
// std::apply(f,tuple) calls a function f with args given by the tuple.
// apply_to_each returns a tuple given by (f(tuple_args[0]),f(tuple_args[1]),...) where f is unary.
// Similar to the possible implementation of std::apply from https://en.cppreference.com/w/cpp/utility/apply

template<class F,class Tuple, std::size_t... I>
constexpr decltype(auto) apply_to_each_impl( F&& f, Tuple&& t, std::index_sequence<I...>){
    return std::make_tuple( std::invoke(std::forward<F>(f), std::get<I>(std::forward<Tuple>(t)))...);
}

template<class F, class Tuple>
constexpr decltype(auto) apply_to_each( F&& f, Tuple&& t){
    return apply_to_each_impl( std::forward<F>(f), std::forward<Tuple>(t),
            std::make_index_sequence<std::tuple_size<std::remove_reference_t<Tuple>>::value>{});
}

// all_of_tuple/any_of_tuple
// For a tuple of bools, determine all_of/any_of

template<class Tuple, std::size_t... I>
constexpr bool all_of_tuple_impl(Tuple&& t, std::index_sequence<I...>){
    return (std::get<I>(std::forward<Tuple>(t)) & ...);
}

template<class Tuple, std::size_t... I>
constexpr bool any_of_tuple_impl(Tuple&& t, std::index_sequence<I...>){
    return (std::get<I>(std::forward<Tuple>(t)) | ...);
}

template<class Tuple>
constexpr bool all_of_tuple(Tuple&& t){
    return all_of_tuple_impl(std::forward<Tuple>(t), std::make_index_sequence<std::tuple_size<std::remove_reference_t<Tuple>>::value>{});
}

template<class Tuple>
constexpr bool any_of_tuple(Tuple&& t){
    return any_of_tuple_impl(std::forward<Tuple>(t), std::make_index_sequence<std::tuple_size<std::remove_reference_t<Tuple>>::value>{});
}

// get_common_order
// for a tuple of RCOrder, return row_major if all row_major, col_major if all col_major, and mixed_order otherwise.

template<std::size_t N, class Tuple>
struct GetCommonOrderImpl {
    RCOrder operator()( Tuple&& t) const noexcept {
        RCOrder order = std::get<N>(std::forward<Tuple>(t)).order();
        return (order  == GetCommonOrderImpl<N-1,Tuple>{}(std::forward<Tuple>(t)) ? order : RCOrder::mixed_order);
    }
};

template<class Tuple> struct GetCommonOrderImpl<0,Tuple>{
    RCOrder operator()( Tuple&& t) const noexcept{ return std::get<0>(std::forward<Tuple>(t)).order();}
};

template<class Tuple>
RCOrder get_common_order(Tuple&& t){
    return GetCommonOrderImpl<std::tuple_size<std::remove_cvref_t<Tuple>>::value-1,Tuple>{}(std::forward<Tuple>(t));
}

// ElementWiseDenseExpression
// Performs an operation taking an arbitray number of args. Operation acts independently over each arg.
// All inputs and outputs must be of the same shape.

template<class F, class... Args>
class ElementWiseDenseExpression : public DenseExpression<ElementWiseDenseExpression<F,Args...>> {

public:

    using value_type = decltype( std::apply( F{}, std::tuple<typename std::remove_cvref_t<Args>::value_type...>()));

private:
    
    using tuple_t = decltype(std::forward_as_tuple(std::declval<Args>()...));
    tuple_t _args;

    template<std::size_t... I>
    decltype(auto) test_dims_impl(std::index_sequence<I...>) const {
        return std::array<bool,sizeof...(Args)>{{std::get<0>(_args).dims() != std::get<I>(_args).dims()...}};
    }
    decltype(auto) test_dims() const {
        return test_dims_impl(std::make_index_sequence<sizeof...(Args)>{});
    }

    template<std::size_t... I>
    decltype(auto) test_shape_impl(std::size_t ii,std::index_sequence<I...>) const {
        return std::array<bool,sizeof...(Args)>{{std::get<0>(_args).shape(ii) != std::get<I>(_args).shape(ii)...}};
    }
    decltype(auto) test_shape(std::size_t ii) const {
        return test_shape_impl(ii,std::make_index_sequence<sizeof...(Args)>{});
    }


public:

    ElementWiseDenseExpression() = delete;
    ElementWiseDenseExpression( const ElementWiseDenseExpression& ) = delete;
    ElementWiseDenseExpression( ElementWiseDenseExpression&& ) = default;
    ElementWiseDenseExpression& operator=( const ElementWiseDenseExpression& ) = delete;
    ElementWiseDenseExpression& operator=( ElementWiseDenseExpression&& ) = default;

    ElementWiseDenseExpression( Args&&... args) : _args(std::forward<Args>(args)...) {
        if( std::ranges::any_of(test_dims(),[](bool b){return b;}) ){
            throw std::runtime_error("ElementWiseDenseExpression: args have incompatible dimensions.");
        }
        for( std::size_t ii=0, end=std::get<0>(_args).dims(); ii<end; ++ii){
            if( std::ranges::any_of(test_shape(ii),[](bool b){return b;}) ){
                throw std::runtime_error("ElementWiseDenseExpression: args have incompatible shapes.");
            }
        }
        // No need to test size, this requirement is satisfied implicitly.
    }

    decltype(auto) size() const { return std::get<0>(_args).size(); }
    decltype(auto) dims() const { return std::get<0>(_args).dims(); }
    decltype(auto) shape() const { return std::get<0>(_args).shape(); }
    decltype(auto) shape(std::size_t ii) const { return std::get<0>(_args).shape(ii); }
    decltype(auto) order() const noexcept { return get_common_order(_args); }
    decltype(auto) num_stripes(std::size_t dim) const { return std::get<0>(_args).num_stripes(dim); }

    constexpr bool is_contiguous() const noexcept { return all_of_tuple(apply_to_each(IsContiguous{},_args)); }
    constexpr bool is_omp_parallelisable() const noexcept { return all_of_tuple(apply_to_each(IsOmpParallelisable{},_args)); }

    // Define const_iterator class
 
    // Notes:
    // begin() should return a compound iterator over Args, containing _its.
    // Dereferencing this will return F(*_its[0],*_its[1],...).
    // Incrementing this will increment _its.

    class const_iterator {
        using ItTuple = std::tuple< typename std::remove_cvref_t<Args>::const_iterator ...>;
        
        F f;
        ItTuple _its;
        
        public:
        
        const_iterator( ItTuple&& its) : f{}, _its(std::move(its)) {}
        decltype(auto) operator*() { return std::apply(f,apply_to_each(Deref{},_its)); }
        const_iterator& operator++() { apply_to_each(PrefixInc{},_its); return *this; }
    };

    const_iterator begin() const { return const_iterator(apply_to_each(Begin{},_args)); }

    // Define stripe class
    // As element-wise operations are strictly non-modifying, only read_only stripes are permitted.
 
    class Stripe {
        
        using StripeTuple = std::tuple< decltype(std::declval<std::add_const_t<std::remove_cvref_t<Args>>>().get_stripe(0,0,RCOrder::col_major)) ...>;

        StripeTuple _stripes;

        public:

        Stripe( StripeTuple&& stripes) : _stripes(std::move(stripes)) {}
        
        // Define iterator type
        // Notes:
        // begin() should return a compound iterator over Args, containing _its.
        // Dereferencing this will return F(*_its[0],*_its[1],...).
        // Incrementing this will increment _its.

        class Iterator {
            
            using ItTuple = std::tuple<typename decltype(std::declval<std::add_const_t<std::remove_cvref_t<Args>>>().get_stripe(0,0,RCOrder::col_major))::Iterator ...>;
            
            F       _f;
            ItTuple _its;
            
            public:
            
            Iterator( ItTuple&& its) : _f{}, _its(std::move(its)) {}
            decltype(auto) operator*() { return std::apply(_f,apply_to_each(Deref{},_its)); }
            Iterator& operator++() { apply_to_each(PrefixInc{},_its); return *this; }
        };
        
        Iterator begin() const { return Iterator(apply_to_each(Begin{},_stripes)); }
    };

    // Get stripes from each Arg
    decltype(auto) get_stripe( std::size_t stripe_num, std::size_t dim, RCOrder order) const {
        return Stripe(std::move(apply_to_each(GetStripe{stripe_num,dim,order},_args)));
    }

};

// CumulativeDenseExpression
// Binary operation over a single arg. Returns something of the same shape.
// For multi-dimensional arrays, sums over the given dimension only. Defaults to zero.

template<class F, class T>
class CumulativeDenseExpression : public DenseExpression<CumulativeDenseExpression<F,T>> {
    
public:

    using value_type = typename std::remove_cvref_t<T>::value_type;

private:

    using ref_t = decltype(std::forward<T>(std::declval<T>()));

    ref_t _t;
    value_type _start_val;

public:

    CumulativeDenseExpression() = delete;
    CumulativeDenseExpression( const CumulativeDenseExpression& ) = delete;
    CumulativeDenseExpression( CumulativeDenseExpression&& ) = default;
    CumulativeDenseExpression& operator=( const CumulativeDenseExpression& ) = delete;
    CumulativeDenseExpression& operator=( CumulativeDenseExpression&& ) = default;

    CumulativeDenseExpression( T&& t, const value_type& start_val) : _t(std::forward<T>(t)) , _start_val(start_val) {}

    decltype(auto) size() const { return _t.size(); }
    decltype(auto) dims() const { return _t.dims(); }
    decltype(auto) shape() const { return _t.shape(); }
    decltype(auto) shape(std::size_t ii) const { return _t.shape(ii); }
    decltype(auto) order() const noexcept { return _t.order(); }
    decltype(auto) num_stripes(std::size_t dim) const { return _t.num_stripes(dim); }

    // CumulativeDenseExpressions cannot be performed in parallel and must make use of striped iteration, hence will
    // appear as non-contiguous and non-omp-parallel. Each stripe may still be determined in parallel however.
    constexpr bool is_contiguous() const noexcept { return false; }
    constexpr bool is_omp_parallelisable() const noexcept { return false; }

    // Define stripe class

    class Stripe {
        
        using Stripe_t = decltype(std::declval<std::add_const_t<std::remove_cvref_t<T>>>().get_stripe(0,0,RCOrder::col_major));

        Stripe_t          _stripe;
        const value_type& _val;

        public:

        Stripe( Stripe_t&& stripe, const value_type& val) : _stripe(std::move(stripe)), _val(val) {}
        
        // Define iterator type

        class Iterator {
            
            using It_t = typename decltype(std::declval<std::add_const_t<std::remove_cvref_t<T>>>().get_stripe(0,0,RCOrder::col_major))::Iterator;
            
            F           _f;
            It_t        _it;
            value_type  _val;
            
            public:
           
            Iterator( It_t&& it, const value_type& val) : _f{}, _it(std::move(it)), _val(val)  {}
            decltype(auto) operator*() { _val = _f(*_it,_val); return _val; }
            Iterator& operator++() { ++_it; return *this; }
        };
        
        Iterator begin() const { return Iterator(_stripe.begin(),_val); }
    };

    // Get stripe from _t
    decltype(auto) get_stripe( std::size_t stripe_num, std::size_t dim, RCOrder order) const {
        return Stripe(std::move(_t.get_stripe(stripe_num,dim,order)),_start_val);
    }

    // Define const_iterator dummy class

    struct const_iterator {
        const_iterator() { throw std::runtime_error("Ultramat CumulativeDenseExpression: Must use striped iteration!");}
        decltype(auto) operator*() { return 0; }
        const_iterator& operator++() { return *this; }
    };
    const_iterator begin() const { return const_iterator(); }
};

// FoldDenseExpression
// Binary operation over a single arg. Performs a 'fold' over a single dimension.
// Full reductions are performed by iteratively reducing until only 1 dimension is left.

template<class F, class T>
class FoldDenseExpression : public DenseExpression<FoldDenseExpression<F,T>> {
    
public:

    using value_type = typename std::remove_cvref_t<T>::value_type;

private:

    using ref_t = decltype(std::forward<T>(std::declval<T>()));
    using Stripe_t = decltype(std::declval<std::add_const_t<std::remove_cvref_t<T>>>().get_stripe(0,0,RCOrder::col_major));

    ref_t                       _t;
    std::size_t                 _fold_dim;
    std::size_t                 _fold_size;
    value_type                  _start_val;

public:

    FoldDenseExpression() = delete;
    FoldDenseExpression( const FoldDenseExpression& ) = delete;
    FoldDenseExpression( FoldDenseExpression&& ) = default;
    FoldDenseExpression& operator=( const FoldDenseExpression& ) = delete;
    FoldDenseExpression& operator=( FoldDenseExpression&& ) = default;

    FoldDenseExpression( T&& t, std::size_t fold_dim, const value_type& start_val) : 
        _t(std::forward<T>(t)),
        _fold_dim(fold_dim),
        _fold_size(t.shape(fold_dim)),
        _start_val(start_val)
    {}

    std::size_t size() const { 
        std::size_t result=1;
        for(std::size_t ii=0; ii<dims(); ++ii) result*=shape(ii);
        return result;
    }

    std::size_t dims() const { return _t.dims()-1; }
    decltype(auto) shape(std::size_t ii) const { return _t.shape(ii < _fold_dim ? ii : ii+1); }
    decltype(auto) order() const noexcept { return _t.order(); }
    decltype(auto) num_stripes(std::size_t dim) const { return size()/shape(dim); }
    constexpr bool is_contiguous() const noexcept { return _t.is_contiguous(); }
    constexpr bool is_omp_parallelisable() const noexcept { return _t.is_omp_parallelisable(); }

    // Define const_iterator class
 
    // Notes:
    // begin() should return a const_iterator containing a stripe generator over _t, passing on the stripe dim
    // Dereferencing this will generate a stripe, perform the fold operation, and return the result.
    // Incrementing this will increment the stripe generator.

    class const_iterator {

        F           _f;
        ref_t       _t;
        std::size_t _stripe_dim;
        RCOrder     _order;
        std::size_t _stripe_num;
        std::size_t _stripe_inc;
        value_type  _start_val;
        
        public:
        
        const_iterator( ref_t t, std::size_t stripe_dim, RCOrder order, std::size_t stripe_num, std::size_t stripe_inc, value_type start_val) :
            _f{},
            _t(std::forward<T>(t)),
            _stripe_dim(stripe_dim),
            _order(order),
            _stripe_num(stripe_num),
            _stripe_inc(stripe_inc),
            _start_val(start_val)
        {}

        const_iterator( ref_t t, std::size_t stripe_dim, std::size_t stripe_num, std::size_t stripe_inc, value_type start_val) :
            const_iterator(std::forward<T>(t),stripe_dim,t.order(),stripe_num,stripe_inc,start_val)
        {}

        decltype(auto) operator*() {
            value_type val = _start_val;
            auto stripe = _t.get_stripe(_stripe_num,_stripe_dim,_order);
            for( auto&& x : stripe ) val = _f(x,val);
            return val;
        }

        const_iterator& operator++() { _stripe_num+=_stripe_inc; return *this; }
        bool operator==(const const_iterator& it) { return _stripe_num == it._stripe_num; }
    };

    const_iterator begin() const { return const_iterator(std::forward<T>(_t),_fold_dim,0,1,_start_val); }
    const_iterator end() const { return const_iterator(std::forward<T>(_t),_fold_dim,num_stripes(),1,_start_val); }

    // Define stripe class
    //
    // Stripe number 'stripe_num' along dimension 'dim' of the fictitious result, following stripe numbering convention given by 'order'.
    // Fold along 'fold_dim' of the arg.
    // 
    // Say we have an array of shape (Nx,Ny,Nz)
    // Start at stripe 0, as always. Say we're using col_major stripe numbering, and also folding dimension 0.
    // This should result in an array of shape (Ny,Nz)
    // Stripe 0 should begin by generating stripe 0 over dim 0 of the arg. When calling operator*, it should perform a fold over this whole inner stirpe
    // We then call ++, which moves us to location (1,0) in the result. We then generate stripe 1 in the arg, and fold again.
    // This continues until reaching end location (Ny,0), at which point we stop and generate a new stripe in the result.
    // Stripe 1 starts at location (0,1), and ends at location (Ny,1). The inner stripe number is now Ny, and goes to 2Ny.
    // 
    // What if instead we fold dimension 1?
    // This should result in an array of shape (Nx,Nz)
    // Stripe 0 should begin by generating stripe 0 over dim 1 of the arg.
    // When we call ++, we go to location (1,0) in the result, and then generate stripe 1 in the arg. 
    // We continue until reaching end location (Nx,0), and then generate a new result stripe.
    // Stripe 1 again starts at location (0,1), and the inner stripe number starts as Nx, and goes to 2Nx.
    //
    // The schema is therefore: increment inner stripe number by 1 each time.
    // Begin at stripe 'stripe_num*shape(0)', end at stripe '(stripe_num+1)*shape(0).
    // The same applies for row_major, only using 'stripe_num*shape(dims()-1)' and '(stripe_num+1)*shape(dims()-1)'
    //
    // Things get trickier if the result stripe dim is non-zero/non-max for col_major/row_major. 
    // Say it's 1 instead, and we're back to col_major and folding over 0.
    // The first iteration is the same, but when we call ++ we go to (0,1).
    // Rather than incrementing the inner stripe to 1, we instead have to increment Ny.

    class Stripe {

        ref_t       _t;
        std::size_t _fold_dim;
        RCOrder     _order;
        value_type  _val;

        std::size_t _start_stripe_num;
        std::size_t _end_stripe_num;
        std::size_t _stripe_num_inc;

        public:
        
        Stripe( ref_t t, std::size_t fold_dim, RCOrder order,  const value_type& val,
                std::size_t start_stripe_num, std::size_t end_stripe_num, std::size_t stripe_num_inc):
            _t(std::forward<T>(t)),
            _fold_dim(fold_dim),
            _order(order),
            _val(val),
            _start_stripe_num(start_stripe_num),
            _end_stripe_num(end_stripe_num),
            _stripe_num_inc(stripe_num_inc)
        {}
 
        // Define iterator type
        using Iterator = const_iterator;

        Iterator begin() const {
            return Iterator( std::forward<T>(_t), _fold_dim, _order, _start_stripe_num, _stripe_num_inc, _val);
        }

        Iterator end() const {
            return Iterator( std::forward<T>(_t), _fold_dim, _order, _end_stripe_num, _stripe_num_inc, _val);
        }
    };

    // Get stripe from _t
    decltype(auto) get_stripe( std::size_t stripe_num, std::size_t dim, RCOrder order) const {
        std::size_t start_stripe_num, end_stripe_num, stripe_num_inc;
        std::size_t first_dim = ( order == RCOrder::col_major ? 0 : dims()-1);
        if( dim==first_dim ){
            start_stripe_num = stripe_num*shape(first_dim);
            stripe_num_inc = 1;
        } else {
            start_stripe_num = stripe_num;
            stripe_num_inc = shape(first_dim);
        }
        end_stripe_num = start_stripe_num+stripe_num_inc*shape(dim);
        return Stripe(std::forward<T>(_t),_fold_dim,order,_start_val,start_stripe_num,end_stripe_num,stripe_num_inc);
    }
};

} // namespace
#endif
