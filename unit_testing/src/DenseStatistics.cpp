#include "ultramat/include/Array.hpp"
#include "ultramat/include/Dense/Math/DenseFolds.hpp"
#include "ultramat/include/Dense/Math/DenseGenerators.hpp"
#include <gtest/gtest.h>

using namespace ultra;

TEST(DenseStatisticsTest,MeanAndStddev){
    Array<double> m = mean(random_normal(10,2,10000));
    Array<double> v = var(random_normal(10,2,10000));
    Array<double> s = stddev(random_normal(10,2,10000));
    EXPECT_LT( std::abs(m(0) - 10), 1e-1 );
    EXPECT_LT( std::abs(v(0) - 4), 2e-1 );
    EXPECT_LT( std::abs(s(0) - 2), 1e-1 );
}

