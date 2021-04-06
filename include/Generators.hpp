#ifndef __ULTRA_GENERATORS_HPP
#define __ULTRA_GENERATORS_HPP

// Generators
//
// Defines expressions that generate elements on demand.

#include "Math.hpp"

namespace ultra {

template<class F>
class GeneratorExpression : public Expression<GeneratorExpression<F>> {
    
public:

    using contains = decltype(std::declval<F>()());

private:

    F _generator;
    std::vector<std::size_t> _shape;
    std::size_t _size;

public:

    template< std::ranges::sized_range Range>
    GeneratorExpression(const F& generator, const Range& range) : 
        _generator(generator), 
        _shape(std::ranges::size(range)),
        _size(std::accumulate(range.begin(),range.end(),1,std::multiplies<std::size_t>{})) 
    {
        std::copy(range.begin(),range.end(),_shape.begin());
    }


    std::size_t size() const { return _size; }
    std::size_t shape(std::size_t ii) const { return _shape[ii]; }
    std::size_t dims() const { return _shape.size(); }

    // Define iterator class

    class const_iterator {
        
        F _generator;
        
        public:
        
        const_iterator( const F& generator) : _generator(generator) {}
        decltype(auto) operator*() { return _generator(); }
        const_iterator& operator++() { /* do nothing! */ return *this; }
    };

    const_iterator begin() const { return const_iterator(_generator); }
};

// Zeros/Ones

template<class T> struct ZeroFunctor { constexpr T operator()() const { return 0; } };
template<class T> struct OneFunctor { constexpr T operator()() const { return 1; } };

template<std::ranges::sized_range Range, class T=std::size_t>
decltype(auto) zeros( const Range& range) {
    return GeneratorExpression<ZeroFunctor<T>>( ZeroFunctor<T>(), range);
}

template<class T=std::size_t>
decltype(auto) zeros( std::size_t N) {
    return GeneratorExpression<ZeroFunctor<T>>( ZeroFunctor<T>(), std::array<std::size_t,1>{N});
}

template<std::ranges::sized_range Range, class T=std::size_t>
decltype(auto) ones( const Range& range) {
    return GeneratorExpression<OneFunctor<T>>( OneFunctor<T>(), range);
}

template<class T=std::size_t>
decltype(auto) ones( std::size_t N) {
    return GeneratorExpression<OneFunctor<T>>( OneFunctor<T>(), std::array<std::size_t,1>{N});
}

// Linspace

template<class T>
class LinspaceFunctor {
    T _start;
    T _stop;
    std::size_t _idx;
    std::size_t _N_minus_1;
public:
    LinspaceFunctor( const T& start, const T& stop, std::size_t N ) : _start(start), _stop(stop), _idx(0), _N_minus_1(N-1) {}
    T operator()() {
        // Not the most efficient, but should be robust.
        T result = _start*(static_cast<T>(_N_minus_1-_idx)/_N_minus_1) + _stop*(static_cast<T>(_idx)/_N_minus_1);
        ++_idx;
        return result;
    }
};

template<std::floating_point T>
decltype(auto) linspace( const T& start, const T& stop, std::size_t N) {
    return GeneratorExpression<LinspaceFunctor<T>>( LinspaceFunctor<T>(start,stop,N), std::array<std::size_t,1>{N});
}

template<std::integral T>
decltype(auto) linspace( const T& start, const T& stop, std::size_t N) {
    return linspace<double>(start,stop,N);
}

template<std::ranges::sized_range Range, std::floating_point T>
decltype(auto) linspace( const T& start, const T& stop, const Range& range) {
    std::size_t N = std::accumulate(range.begin(),range.end(),1,std::multiplies<std::size_t>{});
    return GeneratorExpression<LinspaceFunctor<T>>( LinspaceFunctor<T>(start,stop,N), range);
}

template<std::ranges::sized_range Range, std::integral T>
decltype(auto) linspace( const T& start, const T& stop, const Range& range) {
    return linspace<double>(start,stop,range);
}

// Note: logspace could be constructed using LinspaceFunctor and pow from Math.hpp
//       Encountered an issue with dangling references, so until that can be resolved,
//       it will be implemented the naive (and boring) way.
//       Essentially, linspace is used as a 'leaf' in the expression tree, but when the
//       function returns, the GeneratorExpression will have dangling references to
//       a LinspaceFunctor object and a std::vector used to denote its size.
//       It's not clear how this problem could be resolved without excessive over-
//       engineering.

/*
template<std::floating_point T>
decltype(auto) logspace( const T& start, const T& stop, std::size_t N) {
    return pow(tens(N),linspace<T>(start,stop,N));
}

template<std::integral T>
decltype(auto) logspace( const T& start, const T& stop, std::size_t N) {
    return logspace<double>(start,stop,N);
}

template<std::ranges::sized_range Range, std::floating_point T>
decltype(auto) logspace( const T& start, const T& stop, const Range& range) {
    return pow(tens(range),linspace<T>(start,stop,range));
}

template<std::ranges::sized_range Range, std::integral T>
decltype(auto) logspace( const T& start, const T& stop, const Range& range) {
    return logspace<double>(start,stop,range);
}
*/

template<class T>
class LogspaceFunctor {
    LinspaceFunctor<T> lin;
public:
    LogspaceFunctor( const T& start, const T& stop, std::size_t N ) : lin(start,stop,N) {}
    T operator()() {
        // Not the most efficient, but should be robust.
        T result = std::pow(10,lin());
        return result;
    }
};

template<std::floating_point T>
decltype(auto) logspace( const T& start, const T& stop, std::size_t N) {
    return GeneratorExpression<LogspaceFunctor<T>>( LogspaceFunctor<T>(start,stop,N), std::array<std::size_t,1>{N});
}

template<std::integral T>
decltype(auto) logspace( const T& start, const T& stop, std::size_t N) {
    return logspace<double>(start,stop,N);
}

template<std::ranges::sized_range Range, std::floating_point T>
decltype(auto) logspace( const T& start, const T& stop, const Range& range) {
    std::size_t N = std::accumulate(range.begin(),range.end(),1,std::multiplies<std::size_t>{});
    return GeneratorExpression<LogspaceFunctor<T>>( LogspaceFunctor<T>(start,stop,N), range);
}

template<std::ranges::sized_range Range, std::integral T>
decltype(auto) logspace( const T& start, const T& stop, const Range& range) {
    return logspace<double>(start,stop,range);
}

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

    using dist_result_type = Dist::result_type;
    using rng_result_type = RNG::result_type;
    static const rng_result_type default_seed = RNG::default_seed;

private:

    Dist _dist;
    RNG _rng;
    static rng_result_type seed_increment;

public:
    
    RandomFunctor( const Dist& dist, rng_result_type seed ) : _dist(dist), _rng( seed==default_seed ? seed + seed_increment++ : seed ) {}

    dist_result_type operator()() {
        return _dist(_rng);
    }
};
template<class Dist, std::uniform_random_bit_generator RNG>
RandomFunctor<Dist,RNG>::rng_result_type RandomFunctor<Dist,RNG>::seed_increment = 0;

template<class Dist, std::uniform_random_bit_generator RNG = std::mt19937_64>
decltype(auto) random( const Dist& dist, std::size_t N, typename RNG::result_type seed = RNG::default_seed) {
    return GeneratorExpression<RandomFunctor<Dist,RNG>>( RandomFunctor<Dist,RNG>(dist,seed), std::array<std::size_t,1>{N});
}

template<class Dist, std::uniform_random_bit_generator RNG = std::mt19937_64, std::ranges::sized_range Range>
decltype(auto) random( const Dist& dist, const Range& range, typename RNG::result_type seed = RNG::default_seed) {
    return GeneratorExpression<RandomFunctor<Dist,RNG>>( RandomFunctor<Dist,RNG>(dist,seed), range);
}

} // namespace
#endif

