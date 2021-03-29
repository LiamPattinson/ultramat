#include "ultramat/include/Array.hpp"
#include "ultramat/include/Arithmetic.hpp"
#include "ultramat/include/Generators.hpp"
#include <gtest/gtest.h>

// NOTE:
// Depends on arithmetic, as it is necessary to test whether generators
// play nicely with other expressions.

using namespace ultra;
using shape_vec = std::vector<std::size_t>;

TEST(ArrayGeneratorsTest,OnesAndZeros){
    auto shape = shape_vec{5,10,20};

    // Set things to zeros/ones
    Array<int>    a(shape); a = zeros(shape);
    Array<float>  b = ones(shape);

    bool zero_correct=true, one_correct=true;
    for( auto&& x : a ) if( x )    zero_correct=false;
    for( auto&& x : b ) if( x!=1 ) one_correct=false;

    EXPECT_TRUE(zero_correct);
    EXPECT_TRUE(one_correct);
    EXPECT_TRUE(a.shape(0) == 5);
    EXPECT_TRUE(a.shape(1) == 10);
    EXPECT_TRUE(a.shape(2) == 20);
    EXPECT_TRUE(b.shape(0) == 5);
    EXPECT_TRUE(b.shape(1) == 10);
    EXPECT_TRUE(b.shape(2) == 20);

    // Combine ones with arithmetic
    Array<float> c = b + b + ones(shape) + b;
    bool four_correct = true;
    for( auto&& x : c ) if( x!=4 ) four_correct=false;
    EXPECT_TRUE(four_correct);
}

TEST(ArrayGeneratorsTest,Linspace){
    auto shape = shape_vec{5,10,20};
    std::size_t idx;

    // linear
    Array<double> a = linspace(0,1,101);
    EXPECT_TRUE(a.dims() == 1);
    EXPECT_TRUE(a.size() == 101);
    EXPECT_TRUE(a.shape(0) == 101);
    bool linear_linspace_correct = true;
    idx=0;
    for(auto&& x : a) if( std::abs(x - (idx++)*0.01) > 1e-5 ) linear_linspace_correct = false;
    EXPECT_TRUE(linear_linspace_correct);

    // Shaped
    Array<float> b = linspace(1.f,1000.f,shape);
    EXPECT_TRUE(b.dims() == 3);
    EXPECT_TRUE(b.size() == 1000);
    EXPECT_TRUE(b.shape(0) == 5);
    EXPECT_TRUE(b.shape(1) == 10);
    EXPECT_TRUE(b.shape(2) == 20);
    bool shaped_linspace_correct = true;
    idx=1;
    for(auto&& x : b) if( std::abs(x - (idx++)) > 1e-3 ) shaped_linspace_correct = false;
    EXPECT_TRUE(shaped_linspace_correct);
}

TEST(ArrayGeneratorsTest,Logspace){
    Array<double> a = logspace(-2,2,5);
    // Note:
    // Early versions of this code had a dangling reference bug. Using assertions here
    // to halt testing if anything weird happens.
    ASSERT_TRUE(a.dims() == 1);
    ASSERT_TRUE(a.size() == 5);
    ASSERT_TRUE(a.shape(0) == 5);
    bool logspace_correct = true;
    int idx=-2;
    for(auto&& x : a) if( std::fabs(x - std::pow(10,idx++)) > 1e-5 ) logspace_correct = false;
    EXPECT_TRUE(logspace_correct);
}
