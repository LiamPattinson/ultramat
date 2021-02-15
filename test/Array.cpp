#include "Array.hpp"
#include <gtest/gtest.h>
#include <vector>

using namespace ultra;
using size_vec = std::vector<std::size_t>;

TEST(ArrayTest,Constructors){
    size_vec shape_1 = {25};
    size_vec shape_2 = {50,30};
    size_vec shape_3 = {12,15,90};

    Array<int>    array_1(shape_1);
    Array<float>  array_2(shape_2);
    Array<double> array_3(shape_3, Array<double>::col_major);

    // Test status

    EXPECT_TRUE( array_1.is_initialised() && array_1.owns_data() && array_1.is_contiguous() && array_1.is_semicontiguous());
    EXPECT_TRUE( array_2.is_initialised() && array_2.owns_data() && array_2.is_contiguous() && array_2.is_semicontiguous());
    EXPECT_TRUE( array_3.is_initialised() && array_3.owns_data() && array_3.is_contiguous() && array_3.is_semicontiguous());

    EXPECT_TRUE( array_1.is_row_major() && array_1.is_col_major());
    EXPECT_TRUE( array_2.is_row_major() && !array_2.is_col_major());
    EXPECT_TRUE( !array_3.is_row_major() && array_3.is_col_major());

    // Test attributes

    EXPECT_TRUE(array_1.dims() == 1);  
    EXPECT_TRUE(array_2.dims() == 2);  
    EXPECT_TRUE(array_3.dims() == 3);  

    EXPECT_TRUE(array_1.size() == 25);
    EXPECT_TRUE(array_2.size() == 50*30);
    EXPECT_TRUE(array_3.size() == 12*15*90);
    EXPECT_TRUE(array_1.size(0) == 25);
    for(unsigned ii=0; ii<2; ++ii) EXPECT_TRUE(array_2.size(ii) == shape_2[ii]);
    for(unsigned ii=0; ii<3; ++ii) EXPECT_TRUE(array_3.size(ii) == shape_3[ii]);

    // Test copy

    Array<float>  array_4(array_2);

    EXPECT_TRUE( array_2.is_initialised() && array_2.owns_data() && array_2.is_contiguous() && array_2.is_semicontiguous());
    EXPECT_TRUE( array_2.is_row_major() && !array_2.is_col_major());
    EXPECT_TRUE(array_2.dims() == 2);  
    EXPECT_TRUE(array_2.size() == 50*30);
    for(unsigned ii=0; ii<2; ++ii) EXPECT_TRUE(array_2.size(ii) == shape_2[ii]);

    EXPECT_TRUE( array_4.is_initialised() && array_4.owns_data() && array_4.is_contiguous() && array_4.is_semicontiguous());
    EXPECT_TRUE( array_4.is_row_major() && !array_4.is_col_major());
    EXPECT_TRUE(array_4.dims() == 2);  
    EXPECT_TRUE(array_4.size() == 50*30);
    for(unsigned ii=0; ii<2; ++ii) EXPECT_TRUE(array_4.size(ii) == shape_2[ii]);

    // Test move

    Array<double> array_5(std::move(array_3));

    EXPECT_FALSE( array_3.is_initialised() || array_3.owns_data());
    // All other attributes of array_3 are undefined

    EXPECT_TRUE( array_5.is_initialised() && array_5.owns_data() && array_5.is_contiguous() && array_5.is_semicontiguous());
    EXPECT_TRUE( !array_5.is_row_major() && array_5.is_col_major());
    EXPECT_TRUE(array_5.dims() == 3);  
    EXPECT_TRUE(array_5.size() == 12*15*90);
    for(unsigned ii=0; ii<3; ++ii) EXPECT_TRUE(array_5.size(ii) == shape_3[ii]);
}

/*
TEST(ArrayTest,ElementAccess){
    arma::ivec shape = arma::ivec{50,30};
    Array<float> array(shape);

    array(arma::ivec{21,0}) = 42.42;
    array(arma::ivec{0,10}) = 3.14159;
    array(arma::ivec{5,5}) = 64.32;

    // test that direct access does in fact write, and does in fact return what we expect
    EXPECT_TRUE(fabs(array(arma::ivec{21,0}) - 42.42) < 1e-5);
    EXPECT_TRUE(fabs(array(arma::ivec{0,10}) - 3.14159) < 1e-5);
    EXPECT_TRUE(fabs(array(arma::ivec{5,5}) - 64.32) < 1e-5);

    // test that direct access writes where we expect it to
    auto it = array.begin();

    EXPECT_TRUE( fabs(*(it+21) - 42.42) < 1e-5 );
    EXPECT_TRUE( fabs(*(it+500) - 3.14159) < 1e-5 );
    EXPECT_TRUE( fabs(*(it+255) - 64.32) < 1e-5 );

    // repeat for the harder case of a 3D array
    arma::ivec shape_2 = arma::ivec{50,30,10};
    Array<float> array_2(shape_2);
    array_2(arma::ivec{5,5,5}) = 420.69;
    it = array_2.begin();
    EXPECT_TRUE( fabs(array_2(arma::ivec{5,5,5}) - 420.69) < 1e-5);
    EXPECT_TRUE( fabs(*(it+ 5+50*(5+30*5)) - 420.69) < 1e-5);
}
*/
