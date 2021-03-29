#include "ultramat/include/Array.hpp"
#include "ultramat/include/Arithmetic.hpp"
#include "ultramat/include/Cumulative.hpp"
#include <gtest/gtest.h>

using namespace ultra;
using shape_vec = std::vector<std::size_t>;

// NOTE:
// Depends on arithmetic, as it is necessary to test cumulative functions
// both as 'root' expressions and as 'leaf' expressions.

TEST(ArrayCumulativeTest,CumulativeSum){
    auto shape = shape_vec{5,10,20};
    Array<float>       a(shape);
    Array<double>      b(shape);
    for( auto&& x : a) x=2.5;
    for( auto&& x : b) x=0.5;

    Array<std::size_t> c = ultra::cumsum(a+b);
    bool cumsum_correct = true;
    std::size_t idx = 1;
    for( auto&& x : c ) if( x != 3*(idx++) ) cumsum_correct = false;
    EXPECT_TRUE(cumsum_correct);
    EXPECT_TRUE(c.shape(0) == 5);
    EXPECT_TRUE(c.shape(1) == 10);
    EXPECT_TRUE(c.shape(2) == 20);

}

TEST(ArrayCumulativeTest,CumulativeProduct){
    auto shape = shape_vec{3,3};
    Array<double> a(shape);
    Array<double> b(shape);
    for( auto&& x : a) x=1;
    for( auto&& x : b) x=2;

    Array<double> c = a + ultra::cumprod(b);
    bool cumprod_correct = true;
    std::size_t idx = 1;
    for( auto&& x : c ) if( std::fabs( x - (1 + std::pow(2,idx++))) > 1e-5) cumprod_correct=false;
    EXPECT_TRUE(cumprod_correct);
    EXPECT_TRUE(c.shape(0) == 3);
    EXPECT_TRUE(c.shape(1) == 3);
}
