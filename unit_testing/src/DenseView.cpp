#include "ultramat/include/Array.hpp"
#include <gtest/gtest.h>
#include <cmath>

using namespace ultra;

// Requires Array to test.

TEST(ViewTest,FullViewConstructor){

    Array<float,50,30>::row_major row_array(2.0);
    Array<double,12,15,90>::col_major col_array(5.0);

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
    Array<float,30,20,10> array(17.);
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
    Array<float,30,20,10>::col_major col_array(17.);
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
    Array<float,30,20,10>::row_major row_array(17.);
    Array<float,30,20,10>::col_major col_array(5.);
    
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

    // Create view of a view
    auto double_interior_row_view = interior_row_view.view(interior,interior,interior);
    auto double_interior_col_view = interior_col_view.view(interior,interior,interior);

    EXPECT_TRUE( double_interior_row_view.shape(0) == 26 );
    EXPECT_TRUE( double_interior_col_view.shape(0) == 26 );
    EXPECT_TRUE( double_interior_row_view.shape(1) == 16 );
    EXPECT_TRUE( double_interior_col_view.shape(1) == 16 );
    EXPECT_TRUE( double_interior_row_view.shape(2) ==  6 );
    EXPECT_TRUE( double_interior_col_view.shape(2) ==  6 );
    EXPECT_TRUE( double_interior_row_view.data() == row_array.data() + 2 + 20 + 400);
    EXPECT_TRUE( double_interior_col_view.data() == col_array.data() + 2 + 60 + 1200);
    EXPECT_TRUE( double_interior_row_view(8,8,2) == 36. );
    EXPECT_TRUE( double_interior_col_view(8,8,2) == 36. );

    // create view with step greater than 1
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

TEST(ViewTest,Iteration){
    auto shape = std::vector{10,5,20};
    Array<double,10,5,20>::row_major row_array;
    Array<double,10,5,20>::col_major col_array;
    for( auto it=row_array.begin(), end = row_array.end(); it != end; ++it) *it = it - row_array.begin();
    for( auto it=col_array.begin(), end = col_array.end(); it != end; ++it) *it = it - col_array.begin();
    std::size_t count;

    // Full view
    auto full_row_view = row_array.view();
    auto full_col_view = col_array.view();
    bool full_row_view_correct = true, full_col_view_correct = true;
    count=0;
    for( auto&& x : full_row_view) full_row_view_correct &= (x == count++);
    EXPECT_TRUE(full_row_view_correct);
    count=0;
    for( auto&& x : full_col_view) full_col_view_correct &= (x == count++);
    EXPECT_TRUE(full_col_view_correct);

    // Interior view
    Slice interior(1,-1);
    auto interior_row_view = row_array.view(interior,interior,interior);
    auto interior_col_view = col_array.view(interior,interior,interior);
    bool interior_row_view_correct = true, interior_col_view_correct = true;
    {
        auto it=interior_row_view.begin();
        auto end=interior_row_view.end();
        for( std::size_t ii=1; ii<9; ++ii){
            for( std::size_t jj=1; jj<4; ++jj){
                for( std::size_t kk=1; kk<19; ++kk){
                    interior_row_view_correct &= ( *it == ii*100 + jj*20 + kk );
                    interior_row_view_correct &= ( it != end );
                    ++it;
                }
            }
        }
    }
    EXPECT_TRUE(interior_row_view_correct);
    {
        auto it=interior_col_view.begin();
        auto end=interior_col_view.end();
        for( std::size_t kk=1; kk<19; ++kk){
            for( std::size_t jj=1; jj<4; ++jj){
                for( std::size_t ii=1; ii<9; ++ii){
                    interior_col_view_correct &= ( *it == ii + jj*10 + kk*50 );
                    interior_col_view_correct &= ( it != end );
                    ++it;
                }
            }
        }
    }
    EXPECT_TRUE(interior_col_view_correct);

    // Stepped view
    Slice stepped(1,-1,3);
    auto stepped_row_view = row_array.view(stepped,stepped,stepped);
    auto stepped_col_view = col_array.view(stepped,stepped,stepped);
    bool stepped_row_view_correct = true, stepped_col_view_correct = true;
    {
        auto it=stepped_row_view.begin();
        auto end=stepped_row_view.end();
        for( std::size_t ii=1; ii<9; ii+=3){
            for( std::size_t jj=1; jj<4; jj+=3){
                for( std::size_t kk=1; kk<19; kk+=3){
                    stepped_row_view_correct &= ( *it == ii*100 + jj*20 + kk );
                    stepped_row_view_correct &= ( it != end );
                    ++it;
                }
            }
        }
    }
    EXPECT_TRUE(stepped_row_view_correct);
    {
        auto it=stepped_col_view.begin();
        auto end=stepped_col_view.end();
        for( std::size_t kk=1; kk<19; kk+=3){
            for( std::size_t jj=1; jj<4; jj+=3){
                for( std::size_t ii=1; ii<9; ii+=3){
                    stepped_col_view_correct &= ( *it == ii + jj*10 + kk*50 );
                    stepped_col_view_correct &= ( it != end );
                    ++it;
                }
            }
        }
    }
    EXPECT_TRUE(stepped_col_view_correct);

    // reverse view
    Slice reverse(Slice::all,Slice::all,-1);
    auto reverse_row_view = row_array.view(reverse,reverse,reverse);
    auto reverse_col_view = col_array.view(reverse,reverse,reverse);
    bool reverse_row_view_correct = true, reverse_col_view_correct = true;
    {
        auto it=reverse_row_view.begin();
        auto end=reverse_row_view.end();
        for( std::ptrdiff_t ii=9; ii>=0; --ii){
            for( std::ptrdiff_t jj=4; jj>=0; --jj){
                for( std::ptrdiff_t kk=19; kk>=0; --kk){
                    reverse_row_view_correct &= ( *it == ii*100 + jj*20 + kk );
                    reverse_row_view_correct &= ( it != end );
                    ++it;
                }
            }
        }
    }
    EXPECT_TRUE(reverse_row_view_correct);
    {
        auto it=reverse_col_view.begin();
        auto end=reverse_col_view.end();
        for( std::ptrdiff_t kk=19; kk>=0; --kk){
            for( std::ptrdiff_t jj=4; jj>=0; --jj){
                for( std::ptrdiff_t ii=9; ii>=0; --ii){
                    reverse_col_view_correct &= ( *it == ii + jj*10 + kk*50 );
                    reverse_col_view_correct &= ( it != end );
                    ++it;
                }
            }
        }
    }
    EXPECT_TRUE(reverse_col_view_correct);

    // partially reverse view
    Slice reverse2(Slice::all,Slice::all,-2);
    auto partial_reverse_row_view = row_array.view(stepped,reverse2,stepped);
    auto partial_reverse_col_view = col_array.view(stepped,reverse2,stepped);
    bool partial_reverse_row_view_correct = true, partial_reverse_col_view_correct = true;
    {
        auto it=partial_reverse_row_view.begin();
        auto end=partial_reverse_row_view.end();
        for( std::size_t ii=1; ii<9; ii+=3){
            for( std::ptrdiff_t jj=4; jj>=0; jj-=2){
                for( std::size_t kk=1; kk<19; kk+=3){
                    partial_reverse_row_view_correct &= ( *it == ii*100 + jj*20 + kk );
                    partial_reverse_row_view_correct &= ( it != end );
                    ++it;
                }
            }
        }
    }
    EXPECT_TRUE(partial_reverse_row_view_correct);
    {
        auto it=partial_reverse_col_view.begin();
        auto end=partial_reverse_col_view.end();
        for( std::size_t kk=1; kk<19; kk+=3){
            for( std::ptrdiff_t jj=4; jj>=0; jj-=2){
                for( std::size_t ii=1; ii<9; ii+=3){
                    partial_reverse_col_view_correct &= ( *it == ii + jj*10 + kk*50 );
                    partial_reverse_col_view_correct &= ( it != end );
                    ++it;
                }
            }
        }
    }
    EXPECT_TRUE(partial_reverse_col_view_correct);
}

TEST(ViewTest,Reshape){
    using shape = std::vector<std::size_t>;
    Array<double>::row_major array(shape{8,4,2});
    auto view1 = array.view().reshape(8,8);
    auto view2 = array.view().reshape(std::vector{2,4,4,2});
    
    EXPECT_TRUE( array.dims() == 3 );
    EXPECT_TRUE( array.size() == 64 );
    EXPECT_TRUE( array.shape(0) == 8 );
    EXPECT_TRUE( array.shape(1) == 4 );
    EXPECT_TRUE( array.shape(2) == 2 );
    EXPECT_TRUE( array.stride(0) == 64 );
    EXPECT_TRUE( array.stride(1) == 8 );
    EXPECT_TRUE( array.stride(2) == 2 );
    EXPECT_TRUE( array.stride(3) == 1 );

    EXPECT_TRUE( view1.dims() == 2 );
    EXPECT_TRUE( view1.size() == 64 );
    EXPECT_TRUE( view1.shape(0) == 8 );
    EXPECT_TRUE( view1.shape(1) == 8 );
    EXPECT_TRUE( view1.stride(0) == 64 );
    EXPECT_TRUE( view1.stride(1) == 8 );
    EXPECT_TRUE( view1.stride(2) == 1 );

    EXPECT_TRUE( view2.dims() == 4 );
    EXPECT_TRUE( view2.size() == 64 );
    EXPECT_TRUE( view2.shape(0) == 2 );
    EXPECT_TRUE( view2.shape(1) == 4 );
    EXPECT_TRUE( view2.shape(2) == 4 );
    EXPECT_TRUE( view2.shape(3) == 2 );
    EXPECT_TRUE( view2.stride(0) == 64 );
    EXPECT_TRUE( view2.stride(1) == 32 );
    EXPECT_TRUE( view2.stride(2) == 8 );
    EXPECT_TRUE( view2.stride(3) == 2 );
    EXPECT_TRUE( view2.stride(4) == 1 );

    array(6,3,1) = 15;
    EXPECT_TRUE( view1(6,7) == 15 );
    EXPECT_TRUE( view2(1,2,3,1) == 15 );

    view1(3,5) = 37;
    EXPECT_TRUE( array(3,2,1) == 37 );
    EXPECT_TRUE( view2(0,3,2,1) == 37 );

    view2(0,3,0,1) = 42;
    EXPECT_TRUE( array(3,0,1) == 42 );
    EXPECT_TRUE( view1(3,1) == 42 );

    double count=0;
    for( auto&& x : array) x = count++;
    auto array_it = array.begin();
    auto end = array.end();
    auto view1_it = view1.begin();
    auto view2_it = view2.begin();
    bool iteration_correct=true;
    for(;array_it != end; ++array_it, ++view1_it, ++view2_it){
        iteration_correct &= (*array_it == *view1_it);
        iteration_correct &= (*array_it == *view2_it);
    }
    EXPECT_TRUE(iteration_correct);
    EXPECT_TRUE( view1_it == view1.end() );
    EXPECT_TRUE( view2_it == view2.end() );


    // Repeat for col major
    Array<double>::row_major col_array(shape{8,4,2});
    auto col_view1 = col_array.view().reshape(8,8);
    auto col_view2 = col_array.view().reshape(std::vector{2,4,4,2});
    
    EXPECT_TRUE( col_array.dims() == 3 );
    EXPECT_TRUE( col_array.size() == 64 );
    EXPECT_TRUE( col_array.shape(0) == 8 );
    EXPECT_TRUE( col_array.shape(1) == 4 );
    EXPECT_TRUE( col_array.shape(2) == 2 );
    EXPECT_TRUE( col_array.stride(0) == 64 );
    EXPECT_TRUE( col_array.stride(1) == 8 );
    EXPECT_TRUE( col_array.stride(2) == 2 );
    EXPECT_TRUE( col_array.stride(3) == 1 );

    EXPECT_TRUE( col_view1.dims() == 2 );
    EXPECT_TRUE( col_view1.size() == 64 );
    EXPECT_TRUE( col_view1.shape(0) == 8 );
    EXPECT_TRUE( col_view1.shape(1) == 8 );
    EXPECT_TRUE( col_view1.stride(0) == 64 );
    EXPECT_TRUE( col_view1.stride(1) == 8 );
    EXPECT_TRUE( col_view1.stride(2) == 1 );

    EXPECT_TRUE( col_view2.dims() == 4 );
    EXPECT_TRUE( col_view2.size() == 64 );
    EXPECT_TRUE( col_view2.shape(0) == 2 );
    EXPECT_TRUE( col_view2.shape(1) == 4 );
    EXPECT_TRUE( col_view2.shape(2) == 4 );
    EXPECT_TRUE( col_view2.shape(3) == 2 );
    EXPECT_TRUE( col_view2.stride(0) == 64 );
    EXPECT_TRUE( col_view2.stride(1) == 32 );
    EXPECT_TRUE( col_view2.stride(2) == 8 );
    EXPECT_TRUE( col_view2.stride(3) == 2 );
    EXPECT_TRUE( col_view2.stride(4) == 1 );

    col_array(6,3,1) = 15;
    EXPECT_TRUE( col_view1(6,7) == 15 );
    EXPECT_TRUE( col_view2(1,2,3,1) == 15 );

    col_view1(3,5) = 37;
    EXPECT_TRUE( col_array(3,2,1) == 37 );
    EXPECT_TRUE( col_view2(0,3,2,1) == 37 );

    col_view2(0,3,0,1) = 42;
    EXPECT_TRUE( col_array(3,0,1) == 42 );
    EXPECT_TRUE( col_view1(3,1) == 42 );

    count=0;
    for( auto&& x : col_array) x = count++;
    auto col_array_it = col_array.begin();
    auto col_end = col_array.end();
    auto col_view1_it = col_view1.begin();
    auto col_view2_it = col_view2.begin();
    bool col_iteration_correct=true;
    for(;col_array_it != col_end; ++col_array_it, ++col_view1_it, ++col_view2_it){
        iteration_correct &= (*col_array_it == *col_view1_it);
        iteration_correct &= (*col_array_it == *col_view2_it);
    }
    EXPECT_TRUE(col_iteration_correct);
    EXPECT_TRUE( col_view1_it == col_view1.end() );
    EXPECT_TRUE( col_view2_it == col_view2.end() );

}

TEST(ViewTest,Permuting){
    using shape = std::vector<std::size_t>;
    Array<std::size_t>::row_major row_array(shape{5,6,7,8});
    Array<std::size_t>::col_major col_array(shape{5,6,7,8});
    auto row_view = row_array.permute(2,1,0,3);
    auto col_view = col_array.permute(2,1,0,3);
    std::size_t count;
    count=0; for( auto && x : row_view ) x = count++;
    count=0; for( auto && x : col_view ) x = count++;

    bool row_major_permute_correct = true;
    count = 0;
    for(std::size_t kk=0; kk<7; ++kk){
        for(std::size_t jj=0; jj<6; ++jj){
            for(std::size_t ii=0; ii<5; ++ii){
                for(std::size_t ll=0; ll<8; ++ll){
                    row_major_permute_correct &= (row_array(ii,jj,kk,ll) == count++);
                }
            }
        }
    }
    EXPECT_TRUE(row_major_permute_correct);

    bool col_major_permute_correct = true;
    count = 0;
    for(std::size_t ll=0; ll<8; ++ll){
        for(std::size_t ii=0; ii<5; ++ii){
            for(std::size_t jj=0; jj<6; ++jj){
                for(std::size_t kk=0; kk<7; ++kk){
                    col_major_permute_correct &= (col_array(ii,jj,kk,ll) == count++);
                }
            }
        }
    }
    EXPECT_TRUE(col_major_permute_correct);

    Array<double>::row_major row_matrix(shape{3,4});
    Array<double>::col_major col_matrix(shape{3,4});
    auto row_matrix_t = row_matrix.transpose();
    auto col_matrix_t = col_matrix.t();
    count=0; for( auto && x : row_matrix_t ) x = count++;
    count=0; for( auto && x : col_matrix_t ) x = count++;

    bool row_major_transpose_correct = true;
    count = 0;
    for(std::size_t jj=0; jj<4; ++jj){
        for(std::size_t ii=0; ii<3; ++ii){
            row_major_transpose_correct &= (row_matrix(ii,jj) == count++);
        }
    }
    EXPECT_TRUE(row_major_transpose_correct);

    bool col_major_transpose_correct = true;
    count = 0;
    for(std::size_t ii=0; ii<3; ++ii){
        for(std::size_t jj=0; jj<4; ++jj){
            col_major_transpose_correct &= (col_matrix(ii,jj) == count++);
        }
    }
    EXPECT_TRUE(col_major_transpose_correct);

    // check that double transpose does what you'd expect
    auto double_transpose = row_matrix.transpose().transpose();
    bool double_transpose_correct = true;
    for(std::size_t ii=0; ii<3; ++ii){
        for(std::size_t jj=0; jj<4; ++jj){
            double_transpose_correct &= (row_matrix(ii,jj) == double_transpose(ii,jj));
        }
    }
    EXPECT_TRUE(double_transpose_correct);
}
