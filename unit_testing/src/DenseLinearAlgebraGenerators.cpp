#include "ultramat/include/Dense/LinearAlgebra/DenseLinearAlgebraGenerators.hpp"
#include <gtest/gtest.h>

using namespace ultra;

TEST(DenseLinearAlgebraGeneratorsTest,Eye){
    Array<double> a1 = eye(30,7);
    Array<double> a2 = eye(30,7,2);
    Array<double> a3 = eye(30,7,-1);
    EXPECT_EQ( a1.shape(0), 30);
    EXPECT_EQ( a2.shape(0), 30);
    EXPECT_EQ( a3.shape(0), 30);
    EXPECT_EQ( a1.shape(1), 7);
    EXPECT_EQ( a2.shape(1), 7);
    EXPECT_EQ( a3.shape(1), 7);
    bool a1_correct=true, a2_correct=true, a3_correct=true;
    for( int ii=0; ii<30; ++ii){
        for( int jj=0; jj<7; ++jj){
            if( (a1(ii,jj) && ii!=jj) || (!a1(ii,jj) && ii==jj)) a1_correct=false;
            if( (a2(ii,jj) && ii!=jj-2) || (!a2(ii,jj) && ii==jj-2)) a2_correct=false;
            if( (a3(ii,jj) && ii!=jj+1) || (!a3(ii,jj) && ii==jj+1)) a3_correct=false;
        }
    }
    EXPECT_TRUE( a1_correct);
    EXPECT_TRUE( a2_correct);
    EXPECT_TRUE( a3_correct);
}

TEST(DenseLinearAlgebraGeneratorsTest,Identity){
    Array<double> a = identity(20);
    EXPECT_EQ( a.shape(0), 20);
    EXPECT_EQ( a.shape(1), 20);
    bool correct=true;
    for( int ii=0; ii<20; ++ii){
        for( int jj=0; jj<20; ++jj){
            if( (a(ii,jj) && ii!=jj) || (!a(ii,jj) && ii==jj)) correct=false;
        }
    }
    EXPECT_TRUE( correct);
}
