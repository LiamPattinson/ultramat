#ifndef __ULTRA_DENSE_LINEAR_ALGEBRA_HPP
#define __ULTRA_DENSE_LINEAR_ALGEBRA_HPP

// DenseLinearAlgebra
//
// Defines expressions for linear algebra problems

#include "Dense.hpp"
#include "DenseMath.hpp"

namespace ultra {

// =========================
// Matrix Generators
// * eye -- Produces NxM array with ones down the kth diagonal
// * identity -- Produces NxN array with ones down the main diagonal

template<class T>
class DenseEyeGenerator : public DenseExpression<DenseEyeGenerator<T>> {
    
public:

    using value_type = T;
    using array_type = Array<std::ptrdiff_t>;
    using view_type  = decltype(Array<std::ptrdiff_t>().broadcast(std::array<std::size_t,2>()));

private:

    array_type _rows;
    array_type _cols;
    view_type  _bcast_rows;
    view_type  _bcast_cols;

public:

    DenseEyeGenerator() = delete;
    DenseEyeGenerator( const DenseEyeGenerator& ) = delete;
    DenseEyeGenerator& operator=( const DenseEyeGenerator& ) = delete;
    
    DenseEyeGenerator( DenseEyeGenerator&& other) :
        _rows(std::move(other._rows)),
        _cols(std::move(other._cols)),
        _bcast_rows(_rows.broadcast(_rows,_cols)),
        _bcast_cols(_cols.broadcast(_rows,_cols))
    {}

    DenseEyeGenerator& operator=( DenseEyeGenerator&& other){
        _rows = std::move(other._rows);
        _cols = std::move(other._cols);
        _bcast_rows = _rows.broadcast(_rows,_cols);
        _bcast_cols = _cols.broadcast(_rows,_cols);
    }

    DenseEyeGenerator( std::size_t rows, std::size_t cols, std::ptrdiff_t k) : 
        _rows(arange<std::ptrdiff_t>(0,rows,1)),
        _cols(reshape(arange<std::ptrdiff_t>(-k,static_cast<std::ptrdiff_t>(cols)-k,1),1,cols)),
        _bcast_rows(_rows.broadcast(_rows,_cols)),
        _bcast_cols(_cols.broadcast(_rows,_cols))
    {}

    decltype(auto) size() const { return _bcast_rows.size(); }
    decltype(auto) dims() const { return _bcast_rows.dims(); }
    decltype(auto) shape() const { return _bcast_rows.shape(); }
    decltype(auto) shape(std::size_t ii) const { return _bcast_rows.shape(ii); }
    static constexpr DenseOrder order() { return default_order; }
    decltype(auto) num_stripes(std::size_t dim) const { return _bcast_rows.num_stripes(dim); }
    decltype(auto) required_stripe_dim() const { return dims(); }

    constexpr bool is_contiguous() const noexcept { return true; }
    constexpr bool is_omp_parallelisable() const noexcept { return true; }

    // Define iterator class
    
    template<class U>
    class iterator_impl {

        using it_t = decltype(std::declval<const U>().begin());

        it_t _row_it;
        it_t _col_it;

        public:

        iterator_impl() = delete;
        iterator_impl( const iterator_impl& ) = default;
        iterator_impl( iterator_impl&& ) = default;
        iterator_impl& operator=( const iterator_impl& ) = default;
        iterator_impl& operator=( iterator_impl&& ) = default;
        
        iterator_impl( const it_t& row_it, const it_t& col_it) : 
            _row_it(row_it),
            _col_it(col_it)
        {}

        T operator*() { return *_row_it == *_col_it; }

        iterator_impl& operator++(){ ++_row_it; ++_col_it; return *this; }
        iterator_impl& operator--(){ --_row_it; --_col_it; return *this; }
        template<std::integral I> iterator_impl& operator+=( const I& ii) { _row_it += ii; _col_it += ii; return *this; }
        template<std::integral I> iterator_impl& operator-=( const I& ii) { _row_it -= ii; _col_it -= ii; return *this; }
        template<std::integral I> iterator_impl operator+( const I& ii) { auto result(*this); result+=ii; return result; }
        template<std::integral I> iterator_impl operator-( const I& ii) { auto result(*this); result-=ii; return result; }
        bool operator==( const iterator_impl& other) const { return _row_it == other._row_it;}
        auto operator<=>( const iterator_impl& other) const { return _row_it <=> other._row_it;}
        std::ptrdiff_t operator-( const iterator_impl& other) const { other._row_it - _row_it;}
    };

    using const_iterator = iterator_impl<view_type>;
    const_iterator begin() const { return const_iterator(_bcast_rows.begin(),_bcast_cols.begin()); }
    const_iterator end() const { return const_iterator(_bcast_rows.end(),_bcast_cols.end()); }

    // Define stripe class
 
    class Stripe {

        using stripe_t = decltype(std::declval<const view_type>().get_stripe(0,0,DenseOrder::col_major));

        stripe_t _row_stripe;
        stripe_t _col_stripe;

        public:

        Stripe( const stripe_t& row_stripe, const stripe_t& col_stripe) :
            _row_stripe(row_stripe),
            _col_stripe(col_stripe)
        {}

        using Iterator = iterator_impl<stripe_t>;
        Iterator begin() const { return Iterator(_row_stripe.begin(),_col_stripe.begin()); }
        Iterator end() const { return Iterator(_row_stripe.end(),_col_stripe.end()); }
    };

    decltype(auto) get_stripe( std::size_t stripe_num, std::size_t dim, DenseOrder order) const {
        return Stripe(_bcast_rows.get_stripe(stripe_num,dim,order),_bcast_cols.get_stripe(stripe_num,dim,order));
    }
};

template<class T=double>
decltype(auto) eye( std::size_t rows, std::size_t cols, std::ptrdiff_t k=0){
    return DenseEyeGenerator<T>(rows,cols,k);
}

template<class T=double>
decltype(auto) identity( std::size_t size){
    return eye<T>(size,size,0);
}

// =========================
// Vector Operations
// * dot

// =========================
// Matrix Operations
// * trace
// * matmul

// =========================
// Matrix solvers, Ax=b
// * gaussian_elimination
// * lu_factorisation, lu_solve

} // namespace
#endif
