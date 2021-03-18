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
    bool correct = true;
    for( auto&& x : f ) if( std::abs(x-answer) > 1e-5) correct=false;
    EXPECT_TRUE(correct);
}
