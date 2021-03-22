#include "ultramat/include/Array.hpp"
#include <gtest/gtest.h>

using namespace ultra;
using shape_vec = std::vector<std::size_t>;

TEST(ArrayMathTest,Arithmetic){
    auto shape = shape_vec{5,10,20};
    Array<int>         a(shape);
    Array<float>       b(shape);
    Array<unsigned>    c(shape);
    Array<double>      d(shape);
    Array<std::size_t> e(shape);
    for( auto&& x : a) x=1;
    for( auto&& x : b) x=2;
    for( auto&& x : c) x=3;
    for( auto&& x : d) x=4;
    for( auto&& x : e) x=2;
    // Combine negation, addition, multiplication, subtraction, and dividing, all between different types.
    Array<double> f = -a + b * c - d / e;
    bool correct = true;
    for( auto&& x : f ) if( x != 3) correct=false;
    EXPECT_TRUE(correct);
}

TEST(ArrayMathTest,Math){
    auto shape = shape_vec{5,10,20};
    Array<int>         a(shape);
    Array<float>       b(shape);
    Array<double>      c(shape);
    Array<std::size_t> d(shape);
    for( auto&& x : a) x=10;
    for( auto&& x : b) x=2.5;
    for( auto&& x : c) x=0.5;
    
    // Some easy ones to start...
    d = ultra::floor(b); // expression assignment
    Array<unsigned> e = ultra::ceil(c); // expression copy 
    bool floor_correct = true, ceil_correct=true;
    for( auto&& x : d ) if( x != 2) floor_correct=false;
    for( auto&& x : e ) if( x != 1) ceil_correct=false;
    EXPECT_TRUE(floor_correct);
    EXPECT_TRUE(ceil_correct);

    // And now something a bit mad...
    Array<double> f =(ultra::exp(c) + ultra::pow(b,ultra::sin(e)) * ultra::log(a) ) / d;
    double answer = (std::exp(0.5) + std::pow(2.5,std::sin(1)) * std::log(10))/2;
    bool lots_of_math_correct = true;
    for( auto&& x : f ) if( std::abs(x-answer) > 1e-5) lots_of_math_correct=false;
    EXPECT_TRUE(lots_of_math_correct);

    // Cumulative expressions
    Array<std::size_t> g = ultra::cumsum(ultra::round(b+c));
    {
        auto it = g.begin(), end = g.end();
        std::size_t idx = 1;
        bool cumsum_correct = true;
        for(;it!=end; ++it, ++idx){
            if ( *it != idx*3 ) cumsum_correct = false;
        } 
        EXPECT_TRUE(cumsum_correct);
        EXPECT_TRUE(g.shape(0) == 5);
        EXPECT_TRUE(g.shape(1) == 10);
        EXPECT_TRUE(g.shape(2) == 20);
    }

    auto shape2 = shape_vec{3,3};
    Array<double> h(shape2);
    for( auto&& x : h) x=10;
    Array<double> i = ultra::cumprod(ultra::log(h));
    {
        auto it = i.begin(), end = i.end();
        std::size_t idx = 1;
        bool cumprod_correct = true;
        for(;it!=end; ++it, ++idx){
            if ( std::abs(*it - std::pow(std::log(10),idx)) > 1e-5 ) cumprod_correct = false;
        } 
        EXPECT_TRUE(cumprod_correct);
        EXPECT_TRUE(i.shape(0) == 3);
        EXPECT_TRUE(i.shape(1) == 3);
    }

    // Set things to zeros/ones
    Array<int> j = zeros(shape);
    a = zeros(shape);
    Array<int> k = ones(shape);
    b = ones(shape);
    bool zero_correct=true, one_correct=true;
    for( auto&& x : j ) if( x ) zero_correct=false;
    for( auto&& x : a ) if( x ) zero_correct=false;
    for( auto&& x : k ) if( x!=1 ) one_correct=false;
    for( auto&& x : b ) if( x!=1 ) one_correct=false;
    EXPECT_TRUE(zero_correct);
    EXPECT_TRUE(one_correct);
    EXPECT_TRUE(j.shape(0) == 5);
    EXPECT_TRUE(j.shape(1) == 10);
    EXPECT_TRUE(j.shape(2) == 20);
    EXPECT_TRUE(k.shape(0) == 5);
    EXPECT_TRUE(k.shape(1) == 10);
    EXPECT_TRUE(k.shape(2) == 20);
    EXPECT_TRUE(a.shape(0) == 5);
    EXPECT_TRUE(a.shape(1) == 10);
    EXPECT_TRUE(a.shape(2) == 20);
    EXPECT_TRUE(b.shape(0) == 5);
    EXPECT_TRUE(b.shape(1) == 10);
    EXPECT_TRUE(b.shape(2) == 20);

    // linspace
    {
        Array<double> l = linspace(0,1,101);
        EXPECT_TRUE(l.dims() == 1);
        EXPECT_TRUE(l.size() == 101);
        EXPECT_TRUE(l.shape(0) == 101);
        bool linear_linspace_correct = true;
        std::size_t idx=0;
        for(auto&& x : l) if( std::abs(x - (idx++)*0.01) > 1e-5 ) linear_linspace_correct = false;
        EXPECT_TRUE(linear_linspace_correct);
    }

    Array<float> m = linspace(1.f,1000.f,shape);
    {
        EXPECT_TRUE(m.dims() == 3);
        EXPECT_TRUE(m.size() == 1000);
        EXPECT_TRUE(m.shape(0) == 5);
        EXPECT_TRUE(m.shape(1) == 10);
        EXPECT_TRUE(m.shape(2) == 20);
        bool ranged_linspace_correct = true;
        std::size_t idx=1;
        for(auto&& x : m) if( std::abs(x - (idx++)) > 1e-2 ) ranged_linspace_correct = false;
        EXPECT_TRUE(ranged_linspace_correct);
    }
}
