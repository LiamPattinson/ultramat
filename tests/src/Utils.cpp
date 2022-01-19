#include "ultramat/include/Utils/Utils.hpp"
#include <gtest/gtest.h>

using namespace ultra;

TEST(UtilsTest,Constants){
    float x = 1;
    for( std::size_t ii=0; ii<100; ++ii) x *= 10;
    ASSERT_TRUE(std::isinf(x));
    EXPECT_EQ( x, Inf);
    EXPECT_EQ( x, Infinity);
    EXPECT_EQ( x, infty);
    float y = -1;
    for( std::size_t ii=0; ii<100; ++ii) y *= 10;
    ASSERT_TRUE(std::isinf(y));
    EXPECT_EQ( y, ninf);
    EXPECT_EQ( y, Ninf);
    // Can't compare nan -- it the only thing for which x == x is false.
    EXPECT_TRUE(std::isnan(NaN));
}

