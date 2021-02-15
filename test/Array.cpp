#include "Array.hpp"
#include <gtest/gtest.h>

using namespace ultra;
using size_vec = std::vector<std::size_t>;

TEST(ArrayTest,Constructors){
    std::size_t shape_1 = 25;
    std::size_t shape_2[2] = {50,30};
    size_vec shape_3 = {12,15,90};

    // Note: All of these array building methods result in a simple call to the same generic method.
    //  `    There is no need to test passing a size_vec directly, as the same method is called implicity by the pointer method.
    Array<int>    array_1(shape_1); // test single int 1D array build
    Array<float>  array_2(shape_2); // test c-array build
    Array<double> array_3(shape_3.data(), 3, Array<double>::col_major); // test dynamic c_array build

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


TEST(ArrayTest,ElementAccess){
    size_vec shape = {50,30,10};
    Array<float> array(shape);

    array(21,0,0) = 42.42;
    array(0,10,5) = 3.14159;
    array(5,5,3) = 64.32;

    // test that direct access does in fact write, and does in fact return what we expect
    EXPECT_TRUE(fabs(array(21,0,0) - 42.42) < 1e-5);
    EXPECT_TRUE(fabs(array(0,10,5) - 3.14159) < 1e-5);
    EXPECT_TRUE(fabs(array(5,5,3) - 64.32) < 1e-5);

    // Repeat for a column major array
    Array<float> f_array(shape);

    f_array(21,0,0) = 42.42;
    f_array(0,10,5) = 3.14159;
    f_array(5,5,3) = 64.32;

    EXPECT_TRUE(fabs(f_array(21,0,0) - 42.42) < 1e-5);
    EXPECT_TRUE(fabs(f_array(0,10,5) - 3.14159) < 1e-5);
    EXPECT_TRUE(fabs(f_array(5,5,3) - 64.32) < 1e-5);
}
