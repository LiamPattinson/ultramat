#include "ultramat/include/Array.hpp"
#include "ultramat/include/Dense/Math/DenseArithmetic.hpp"
#include "ultramat/include/Dense/Math/DenseCumulative.hpp"
#include <gtest/gtest.h>

using namespace ultra;

TEST(DenseCumulativeTest,CumulativeSum){
    auto shape = Shape{5,10,20};
    Array<float>::col_major  a(shape);
    Array<double>::row_major b(shape);
    int count;
    count=0; for( auto&& x : a) x=2.5*count++;
    count=0; for( auto&& x : b) x=0.3*count++;

    // cumsum is evaluating: can use auto type deduction
    auto cumsum_0 = cumsum(a+b,0);
    auto cumsum_1 = cumsum(a+b,1);
    auto cumsum_2 = cumsum(a+b,2);

    EXPECT_TRUE(cumsum_0.shape(0) == 5);
    EXPECT_TRUE(cumsum_1.shape(0) == 5);
    EXPECT_TRUE(cumsum_2.shape(0) == 5);
    EXPECT_TRUE(cumsum_0.shape(1) == 10);
    EXPECT_TRUE(cumsum_1.shape(1) == 10);
    EXPECT_TRUE(cumsum_2.shape(1) == 10);
    EXPECT_TRUE(cumsum_0.shape(2) == 20);
    EXPECT_TRUE(cumsum_1.shape(2) == 20);
    EXPECT_TRUE(cumsum_2.shape(2) == 20);

    bool cumsum_0_correct = true;
    for( std::size_t ii=0; ii<10; ++ii){
        for( std::size_t jj=0; jj<20; ++jj){
            double result = 0;
            for( std::size_t kk=0; kk<5; ++kk){
                result += a(kk,ii,jj) + b(kk,ii,jj);
                if( std::fabs(result - cumsum_0(kk,ii,jj)) > 1e-5 ) cumsum_0_correct = false;
            }
        }
    }
    EXPECT_TRUE(cumsum_0_correct);

    bool cumsum_1_correct = true;
    for( std::size_t ii=0; ii<5; ++ii){
        for( std::size_t jj=0; jj<20; ++jj){
            double result = 0;
            for( std::size_t kk=0; kk<10; ++kk){
                result += a(ii,kk,jj) + b(ii,kk,jj);
                if( std::fabs(result - cumsum_1(ii,kk,jj)) > 1e-5 ) cumsum_1_correct = false;
            }
        }
    }
    EXPECT_TRUE(cumsum_1_correct);
    
    bool cumsum_2_correct = true;
    for( std::size_t ii=0; ii<5; ++ii){
        for( std::size_t jj=0; jj<10; ++jj){
            double result = 0;
            for( std::size_t kk=0; kk<20; ++kk){
                result += a(ii,jj,kk) + b(ii,jj,kk);
                if( std::fabs(result - cumsum_2(ii,jj,kk)) > 1e-5 ) cumsum_2_correct = false;
            }
        }
    }
    EXPECT_TRUE(cumsum_2_correct);

}
