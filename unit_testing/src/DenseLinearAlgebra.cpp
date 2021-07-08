#include "ultramat/include/DenseLinearAlgebra.hpp"
#include <gtest/gtest.h>
#include <iostream>

using namespace ultra;
using shape_vec = std::vector<std::size_t>;

TEST(LinearAlgebraTest,Generator){
    Array<double> a = eye(3,3);
    Array<double> b = identity(3);
    EXPECT_TRUE( a.dims() == 2);
    EXPECT_TRUE( b.dims() == 2);
    EXPECT_TRUE( a.shape(0) == 3);
    EXPECT_TRUE( b.shape(0) == 3);
    EXPECT_TRUE( a.shape(1) == 3);
    EXPECT_TRUE( b.shape(1) == 3);
    bool simple_eye_correct=true, simple_identity_correct=true;
    for( std::size_t ii=0; ii<3; ++ii){
        for( std::size_t jj=0; jj<3; ++jj){
            if( a(ii,jj) != (ii==jj) ) simple_eye_correct=false;
            if( b(ii,jj) != (ii==jj) ) simple_identity_correct=false;
        }
    }
    EXPECT_TRUE(simple_eye_correct);
    EXPECT_TRUE(simple_identity_correct);

    Array<double>::row_major c = eye(31,43,3);
    Array<double>::col_major d = 3*eye(31,43,3);
    Array<double>::row_major e = eye(104,12,-2);
    Array<double>::col_major f = eye(104,12,-2)-10;
    bool complex_eye_k_pos_correct=true, complex_eye_k_neg_correct=true;
    for( std::size_t ii=0; ii<31; ++ii){
        for( std::size_t jj=0; jj<43; ++jj){
            if( c(ii,jj) != (ii==(jj-3)) ) complex_eye_k_pos_correct=false;
            if( d(ii,jj) != 3*(ii==(jj-3)) ) complex_eye_k_pos_correct=false;
            
        }
    }
    for( std::size_t ii=0; ii<104; ++ii){
        for( std::size_t jj=0; jj<12; ++jj){
            if( e(ii,jj) != (ii==(jj+2)) ) complex_eye_k_neg_correct=false;
            if( f(ii,jj) != (ii==(jj+2))-10 ) complex_eye_k_neg_correct=false;
        }
    }
    EXPECT_TRUE(complex_eye_k_pos_correct);
    EXPECT_TRUE(complex_eye_k_neg_correct);
}

