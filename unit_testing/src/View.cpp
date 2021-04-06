#include "ultramat/include/View.hpp"
#include "ultramat/include/FixedArray.hpp"
#include <gtest/gtest.h>
#include <cmath>

using namespace ultra;

// Requires FixedArray to test.

TEST(ViewTest,FullViewConstructor){

    FixedArray<float,50,30>::row_major row_array(2.0);
    FixedArray<double,12,15,90>::col_major col_array(5.0);

    auto row_view = row_array.view();
    auto col_view = col_array.view();

    // Test attributes

    EXPECT_TRUE(row_view.dims() == 2);  
    EXPECT_TRUE(col_view.dims() == 3);  

    EXPECT_TRUE(row_view.size() == 50*30);
    EXPECT_TRUE(col_view.size() == 12*15*90);

    EXPECT_TRUE(row_view.shape(0) == 50);
    EXPECT_TRUE(row_view.shape(1) == 30);
    EXPECT_TRUE(col_view.shape(0) == 12);
    EXPECT_TRUE(col_view.shape(1) == 15);
    EXPECT_TRUE(col_view.shape(2) == 90);

    EXPECT_TRUE(row_view.stride(0) == 30*50);
    EXPECT_TRUE(row_view.stride(1) == 30);
    EXPECT_TRUE(row_view.stride(2) == 1);
    EXPECT_TRUE(col_view.stride(0) == 1);
    EXPECT_TRUE(col_view.stride(1) == 12);
    EXPECT_TRUE(col_view.stride(2) == 12*15);
    EXPECT_TRUE(col_view.stride(3) == 12*15*90);
    
    // Test copy

    auto row_view_copy(row_view);

    EXPECT_TRUE(row_view_copy.dims() == 2);  
    EXPECT_TRUE(row_view_copy.size() == 50*30);
    EXPECT_TRUE(row_view_copy.shape(0) == 50);
    EXPECT_TRUE(row_view_copy.shape(1) == 30);
    EXPECT_TRUE(row_view_copy.stride(0) == 30*50);
    EXPECT_TRUE(row_view_copy.stride(1) == 30);
    EXPECT_TRUE(row_view_copy.stride(2) == 1);

    // Test move

    auto col_view_move(std::move(col_view));

    EXPECT_TRUE(col_view_move.dims() == 3);  
    EXPECT_TRUE(col_view_move.size() == 12*15*90);
    EXPECT_TRUE(col_view_move.shape(0) == 12);
    EXPECT_TRUE(col_view_move.shape(1) == 15);
    EXPECT_TRUE(col_view_move.shape(2) == 90);
    EXPECT_TRUE(col_view_move.stride(0) == 1);
    EXPECT_TRUE(col_view_move.stride(1) == 12);
    EXPECT_TRUE(col_view_move.stride(2) == 12*15);
    EXPECT_TRUE(col_view_move.stride(3) == 12*15*90);

    // Test full shape and stride reading
    auto shape = col_view_move.shape();
    auto stride = row_view_copy.stride();

    EXPECT_TRUE(shape[0] == 12);
    EXPECT_TRUE(shape[1] == 15);
    EXPECT_TRUE(shape[2] == 90);
    EXPECT_TRUE(stride[0] == 30*50);
    EXPECT_TRUE(stride[1] == 30);
    EXPECT_TRUE(stride[2] == 1);
}

TEST(ViewTest,FullViewElementAccess){
    FixedArray<float,30,20,10> array(17.);
    auto view = array.view();

    // Test fill
    EXPECT_TRUE(fabs(view(0,0,0) - 17.) < 1e-5);
    EXPECT_TRUE(fabs(view(8,2,1) - 17.) < 1e-5);
    EXPECT_TRUE(fabs(view(2,14,5) - 17.) < 1e-5);
    EXPECT_TRUE(fabs(view(29,19,9) - 17.) < 1e-5);

    // Set a few values
    view(21,0,0) = 42.42;
    view(0,10,5) = 3.14159;
    view(5,5,3) = 64.32;

    // Test that getting and setting access the same values
    EXPECT_TRUE(fabs(view(21,0,0) - 42.42) < 1e-5);
    EXPECT_TRUE(fabs(view(0,10,5) - 3.14159) < 1e-5);
    EXPECT_TRUE(fabs(view(5,5,3) - 64.32) < 1e-5);

    // Test that these are at the correct locations in memory
    EXPECT_TRUE(fabs(*(view.data() + 21*200) - 42.42) < 1e-5);
    EXPECT_TRUE(fabs(*(view.data() + 5 + 10*10) - 3.14159) < 1e-5);
    EXPECT_TRUE(fabs(*(view.data() + 3 + 5*10 + 5*200) - 64.32) < 1e-5);

    // Repeat for a column major array
    FixedArray<float,30,20,10>::col_major col_array(17.);
    auto col_view = col_array.view();

    EXPECT_TRUE(fabs(col_view(0,0,0) - 17.) < 1e-5);
    EXPECT_TRUE(fabs(col_view(8,2,1) - 17.) < 1e-5);
    EXPECT_TRUE(fabs(col_view(2,14,5) - 17.) < 1e-5);
    EXPECT_TRUE(fabs(col_view(29,19,9) - 17.) < 1e-5);

    col_view(21,0,0) = 42.42;
    col_view(0,10,5) = 3.14159;
    col_view(5,5,3) = 64.32;

    EXPECT_TRUE(fabs(col_view(21,0,0) - 42.42) < 1e-5);
    EXPECT_TRUE(fabs(col_view(0,10,5) - 3.14159) < 1e-5);
    EXPECT_TRUE(fabs(col_view(5,5,3) - 64.32) < 1e-5);

    EXPECT_TRUE(fabs(*(col_view.data() + 21) - 42.42) < 1e-5);
    EXPECT_TRUE(fabs(*(col_view.data() + 10*30 + 5*600) - 3.14159) < 1e-5);
    EXPECT_TRUE(fabs(*(col_view.data() + 5 + 5*30 + 3*600) - 64.32) < 1e-5);

    // Test again with std::vector-like coordinates
    EXPECT_TRUE(fabs(array(std::vector{21,0,0}) - 42.42) < 1e-5);
    EXPECT_TRUE(fabs(array(std::vector{0,10,5}) - 3.14159) < 1e-5);
    EXPECT_TRUE(fabs(array(std::vector{5,5,3}) - 64.32) < 1e-5);
    EXPECT_TRUE(fabs(col_view(std::vector{21,0,0}) - 42.42) < 1e-5);
    EXPECT_TRUE(fabs(col_view(std::vector{0,10,5}) - 3.14159) < 1e-5);
    EXPECT_TRUE(fabs(col_view(std::vector{5,5,3}) - 64.32) < 1e-5);
}

TEST(ViewTest,Slicing) {
    FixedArray<float,30,20,10>::row_major row_array(17.);
    FixedArray<float,30,20,10>::col_major col_array(5.);
    
    row_array(10,10,4) = 36.;
    col_array(10,10,4) = 36.;

    // Create view excluding boundary elements
    Slice interior(1,-1);
    auto interior_row_view = row_array.view(interior,interior,interior);
    auto interior_col_view = col_array.view(interior,interior,interior);

    EXPECT_TRUE( interior_row_view.shape(0) == 28 );
    EXPECT_TRUE( interior_col_view.shape(0) == 28 );
    EXPECT_TRUE( interior_row_view.shape(1) == 18 );
    EXPECT_TRUE( interior_col_view.shape(1) == 18 );
    EXPECT_TRUE( interior_row_view.shape(2) ==  8 );
    EXPECT_TRUE( interior_col_view.shape(2) ==  8 );
    EXPECT_TRUE( interior_row_view.data() == row_array.data() + 1 + 10 + 200);
    EXPECT_TRUE( interior_col_view.data() == col_array.data() + 1 + 30 + 600);
    EXPECT_TRUE( interior_row_view(9,9,3) == 36. );
    EXPECT_TRUE( interior_col_view(9,9,3) == 36. );

    // Create partial view excluding boundary elements in 0 and 1 dimensions, but not in 2 dimension
    auto partial_interior_row_view = row_array.view(interior,interior);
    auto partial_interior_col_view = col_array.view(interior,interior);

    EXPECT_TRUE( partial_interior_row_view.shape(0) == 28 );
    EXPECT_TRUE( partial_interior_col_view.shape(0) == 28 );
    EXPECT_TRUE( partial_interior_row_view.shape(1) == 18 );
    EXPECT_TRUE( partial_interior_col_view.shape(1) == 18 );
    EXPECT_TRUE( partial_interior_row_view.shape(2) == 10 );
    EXPECT_TRUE( partial_interior_col_view.shape(2) == 10 );
    EXPECT_TRUE( partial_interior_row_view.data() == row_array.data() + 10 + 200);
    EXPECT_TRUE( partial_interior_col_view.data() == col_array.data() + 1 + 30);
    EXPECT_TRUE( partial_interior_row_view(9,9,4) == 36. );
    EXPECT_TRUE( partial_interior_col_view(9,9,4) == 36. );

    auto stepped_row_view = row_array.view(Slice{2,-4,2},Slice{2,-1},Slice{1,Slice::all,3});
    auto stepped_col_view = col_array.view(Slice{2,-4,2},Slice{2,-1},Slice{1,Slice::all,3});

    EXPECT_TRUE( stepped_row_view.shape(0) == 12 );
    EXPECT_TRUE( stepped_col_view.shape(0) == 12 );
    EXPECT_TRUE( stepped_row_view.shape(1) == 17 );
    EXPECT_TRUE( stepped_col_view.shape(1) == 17 );
    EXPECT_TRUE( stepped_row_view.shape(2) == 3 );
    EXPECT_TRUE( stepped_col_view.shape(2) == 3 );
    EXPECT_TRUE( stepped_row_view.data() == row_array.data() + 1 + 2*10 + 2*200);
    EXPECT_TRUE( stepped_col_view.data() == col_array.data() + 2 + 2*30 + 600);
    EXPECT_TRUE( stepped_row_view(4,8,1) == 36. );
    EXPECT_TRUE( stepped_col_view(4,8,1) == 36. );

    // Create reverse view
    Slice reverse{Slice::all,Slice::all,-1};
    auto reverse_row_view = row_array.view(reverse,reverse,reverse);
    auto reverse_col_view = col_array.view(reverse,reverse,reverse);

    EXPECT_TRUE( reverse_row_view.shape(0) == 30 );
    EXPECT_TRUE( reverse_col_view.shape(0) == 30 );
    EXPECT_TRUE( reverse_row_view.shape(1) == 20 );
    EXPECT_TRUE( reverse_col_view.shape(1) == 20 );
    EXPECT_TRUE( reverse_row_view.shape(2) == 10 );
    EXPECT_TRUE( reverse_col_view.shape(2) == 10 );
    EXPECT_TRUE( reverse_row_view.data() == row_array.data() + row_array.size() -1);
    EXPECT_TRUE( reverse_col_view.data() == col_array.data() + col_array.size() -1);
    EXPECT_TRUE( reverse_row_view(19,9,5) == 36. );
    EXPECT_TRUE( reverse_col_view(19,9,5) == 36. );
}
