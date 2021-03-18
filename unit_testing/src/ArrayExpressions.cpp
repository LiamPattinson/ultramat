#include "ultramat/include/Array.hpp"
#include <gtest/gtest.h>

using namespace ultra;
using shape_vec = std::vector<std::size_t>;

TEST(ArrayExpressionTest,Arithmetic){

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
