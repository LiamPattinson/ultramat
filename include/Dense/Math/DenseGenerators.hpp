#ifndef __ULTRA_DENSE_GENERATORS_HPP
#define __ULTRA_DENSE_GENERATORS_HPP

#include "ultramat/include/Dense/Expressions/DenseGeneratorExpression.hpp"

namespace ultra {

// ============================
// Zeros/Ones

template<class T>
class ConstantFunctor {
    T _t;
public:
    ConstantFunctor( const T& t) : _t(t){}
    T operator()( std::size_t) const {
        return _t;
    }
};

template<class T=double, shapelike Shape>
decltype(auto) zeros( const Shape& shape) {
    return DenseGeneratorExpression( ConstantFunctor<T>(0), shape);
}

template<class T=double>
decltype(auto) zeros( std::size_t N) {
    return zeros(std::array<std::size_t,1>{N});
}

template<class T=double, shapelike Shape>
decltype(auto) ones( const Shape& shape) {
    return DenseGeneratorExpression( ConstantFunctor<T>(1), shape);
}

template<class T=double>
decltype(auto) ones( std::size_t N) {
    return ones(std::array<std::size_t,1>{N});
}

// ============================
// linspace and logspace
// Generate an evenly distributed set of values in the range [start,stop]
// logspace calculates pow(10,linspace(start,stop,N))

template<class T>
class LinspaceFunctor {
    T _start;
    T _stop;
    std::size_t _N_minus_1;
public:
    LinspaceFunctor( const T& start, const T& stop, std::size_t N ) : _start(start), _stop(stop), _N_minus_1(N-1) {}
    T operator()( std::size_t idx) {
        // Not the most efficient, but should be robust.
        return _start*(static_cast<T>(_N_minus_1-idx)/_N_minus_1) + _stop*(static_cast<T>(idx)/_N_minus_1);
    }
};

template<class T1, class T2> requires (!std::is_integral<decltype(T1()*T2())>::value)
decltype(auto) linspace( const T1& start, const T2& stop, std::size_t N) {
    return DenseGeneratorExpression( LinspaceFunctor<decltype(T1()*T2())>(start,stop,N), std::array<std::size_t,1>{N});
}

template<class T1, class T2> requires std::integral<decltype(T1()*T2())>
decltype(auto) linspace( const T1& start, const T2& stop, std::size_t N) {
    return linspace<double>(start,stop,N);
}

template<class T1,class T2>
decltype(auto) logspace( const T1& start, const T2& stop, std::size_t N) {
    return pow(10,linspace(start,stop,N));
}

// ============================
// arange/regspace
// Generate points distributed set of values in the range [start,stop) with a given step size

template<class T>
class ArangeFunctor {
    T _start;
    T _step;
public:
    ArangeFunctor( const T& start, const T& step ) : _start(start), _step(step) {}
    T operator()( std::size_t idx) {
        return _start + idx*_step;
    }
};

template<class T>
decltype(auto) arange( const T& start, const T& stop, const T& step) {
    double size = (0.+stop-start)/step;
    std::size_t num_vals = ( std::fabs(size - std::round(size)) < 1e-5*std::fabs(stop)  ? std::round(size) : std::ceil(size));
    if( num_vals <= 0 ){
        throw std::runtime_error("Ultra: arange, stop must be greater than start for positive step, or less than start for negative step");
    }
    return DenseGeneratorExpression( ArangeFunctor<T>(start,step), std::array<std::size_t,1>{num_vals});
}

template<class T1,class T2,class T3>
decltype(auto) arange( const T1& start, const T2& stop, const T3& step) {
    using common_t = decltype(T1()*T2()*T3());
    return arange( (common_t)start, (common_t)stop, (common_t)step);
}

template<class T1, class T2, class T3>
decltype(auto) regspace( const T1& start, const T2& stop, const T3& step) {
    return arange(start,stop,step);
}

// ============================
// Random number generation
// 
// Takes as first template parameter a random number distribution. See <random>.
// There is no standard concept for this, but at a minimum it must define a
// result_type, and must have the function `result_type operator()(std::size_t)`.
// It must also be copyable.
//
// A random number generator may be provided as a second argument. By default, we use
// the slow but high-quality mt19937_64. 
//
// RandomFunctors must be initialised with a distribution functor and a seed (typically
// std::size_t or unsigned). If no seed is provided, it will make use of the random
// number generator's default seed, plus 1 for each instance.

template<class Dist, std::uniform_random_bit_generator RNG = std::mt19937_64>
class RandomFunctor {

public:

    using dist_result_type = typename Dist::result_type;
    using rng_result_type = typename RNG::result_type;

private:

    Dist            _dist;
    RNG             _rng;
    static rng_result_type _static_seed;

    rng_result_type _get_seed() const {
        rng_result_type seed;
        #pragma omp atomic capture
        seed = _static_seed++;
        return seed;
    }

public:
    
    RandomFunctor( const Dist& dist ) : 
        _dist(dist),
        _rng(_get_seed())
    {}

    RandomFunctor( const RandomFunctor& other ) :
        _dist(other._dist),
        _rng(_get_seed())
    {}

    RandomFunctor( RandomFunctor&& other ) :
        _dist(std::move(other._dist)),
        _rng(std::move(other._rng))
    {}

    dist_result_type operator()( std::size_t ) {
        return _dist(_rng);
    }
};

template<class Dist, std::uniform_random_bit_generator RNG>
typename RandomFunctor<Dist,RNG>::rng_result_type RandomFunctor<Dist,RNG>::_static_seed = RNG::default_seed;

template<class Dist, std::uniform_random_bit_generator RNG = std::mt19937_64, shapelike Range>
decltype(auto) random( const Dist& dist, const Range& range) {
    return DenseGeneratorExpression( RandomFunctor<Dist,RNG>(dist), range);
}

template<class Dist, std::uniform_random_bit_generator RNG = std::mt19937_64>
decltype(auto) random( const Dist& dist, std::size_t N) {
    return DenseGeneratorExpression( RandomFunctor<Dist,RNG>(dist), std::array<std::size_t,1>{N});
}

// random_uniform
// Floating point: produces values in the range [min,max)
// Integrer: produces values in the range [min,max] (inclusive of max)

template<class T1, class T2, shapelike Range>
requires std::floating_point<decltype(T1()*T2())>
decltype(auto) random_uniform( T1 min, T2 max, const Range& range) {
    return random( std::uniform_real_distribution<decltype(T1()*T2())>(min,max), range);
}

template<class T1, class T2, shapelike Range>
requires std::integral<T1> && std::integral<T2>
decltype(auto) random_uniform( T1 min, T2 max, const Range& range) {
    return random( std::uniform_int_distribution<decltype(T1()*T2())>(min,max), range);
}

template<class T1, class T2> requires number<T1> && number<T2>
decltype(auto) random_uniform( T1 min, T2 max, std::size_t N) {
    return random_uniform( min, max, std::array<std::size_t,1>{N});
}

// random_normal/random_gaussian (identical functions)
// Produces random numbers according to:
// f(x;\mu,\sigma) = \frac{1}{2\pi\sigma}\exp\left(-\frac{(x-\mu)^2}{2\sigma^2}\right)
// where \mu is mean and \sigma is stddev

template<class T1, class T2, shapelike Range>
requires number<T1> && number<T2>
decltype(auto) random_normal( T1 mean, T2 stddev, const Range& range) {
    using real_t = std::conditional_t< std::is_floating_point<decltype(T1()*T2())>::value, decltype(T1()*T2()), double>;
    return random( std::normal_distribution<real_t>(mean,stddev), range);
}

template<class T1, class T2, shapelike Range>
requires number<T1> && number<T2>
decltype(auto) random_gaussian( T1 mean, T2 stddev, const Range& range) {
    return random_normal( mean, stddev, range);
}

template<class T1, class T2> requires number<T1> && number<T2>
decltype(auto) random_normal( T1 mean, T2 stddev, std::size_t N) {
    return random_normal( mean, stddev, std::array<std::size_t,1>{N});
}

template<class T1, class T2> requires number<T1> && number<T2>
decltype(auto) random_gaussian( T1 mean, T2 stddev, std::size_t N) {
    return random_gaussian( mean, stddev, std::array<std::size_t,1>{N});
}

} // namespace ultra
#endif
