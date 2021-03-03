#include "Array.hpp"
#include <gtest/gtest.h>

using namespace ultra;
using shape_vec = std::vector<std::size_t>;

TEST(ArrayTest,Constructors){
    std::size_t shape_1 = 25;
    std::size_t shape_2[2] = {50,30};
    shape_vec shape_3 = {12,15,90};

    // Note: All of these array building methods result in a simple call to the same generic method.
    //  `    There is no need to test passing a shape_vec directly, as the same method is called implicity by the pointer method.
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

    // Test copy assignment to uninitialised
    Array<double> array_6;
    array_6 = array_5;

    EXPECT_TRUE( array_5.is_initialised() && array_5.owns_data() && array_5.is_contiguous() && array_5.is_semicontiguous());
    EXPECT_TRUE( !array_5.is_row_major() && array_5.is_col_major());
    EXPECT_TRUE(array_5.dims() == 3);  
    EXPECT_TRUE(array_5.size() == 12*15*90);
    for(unsigned ii=0; ii<3; ++ii) EXPECT_TRUE(array_5.size(ii) == shape_3[ii]);
    
    EXPECT_TRUE( array_6.is_initialised() && array_6.owns_data() && array_6.is_contiguous() && array_6.is_semicontiguous());
    EXPECT_TRUE( !array_6.is_row_major() && array_6.is_col_major());
    EXPECT_TRUE(array_6.dims() == 3);  
    EXPECT_TRUE(array_6.size() == 12*15*90);
    for(unsigned ii=0; ii<3; ++ii) EXPECT_TRUE(array_6.size(ii) == shape_3[ii]);

    // Test copy assignment to initialised
 
    Array<float> array_7(shape_2); // Same size and shape
    Array<float> array_8(5); // Different size and shape
    array_7 = array_4;
    array_8 = array_4;

    EXPECT_TRUE( array_4.is_initialised() && array_4.owns_data() && array_4.is_contiguous() && array_4.is_semicontiguous());
    EXPECT_TRUE( array_4.is_row_major() && !array_4.is_col_major());
    EXPECT_TRUE(array_4.dims() == 2);  
    EXPECT_TRUE(array_4.size() == 50*30);
    for(unsigned ii=0; ii<2; ++ii) EXPECT_TRUE(array_4.size(ii) == shape_2[ii]);

    EXPECT_TRUE( array_7.is_initialised() && array_7.owns_data() && array_7.is_contiguous() && array_7.is_semicontiguous());
    EXPECT_TRUE( array_7.is_row_major() && !array_7.is_col_major());
    EXPECT_TRUE(array_7.dims() == 2);  
    EXPECT_TRUE(array_7.size() == 50*30);
    for(unsigned ii=0; ii<2; ++ii) EXPECT_TRUE(array_7.size(ii) == shape_2[ii]);

    EXPECT_TRUE( array_8.is_initialised() && array_8.owns_data() && array_8.is_contiguous() && array_8.is_semicontiguous());
    EXPECT_TRUE( array_8.is_row_major() && !array_8.is_col_major());
    EXPECT_TRUE(array_8.dims() == 2);  
    EXPECT_TRUE(array_8.size() == 50*30);
    for(unsigned ii=0; ii<2; ++ii) EXPECT_TRUE(array_8.size(ii) == shape_2[ii]);

    // Test copy assignment from uninitialised
    
    array_6 = array_3;
    EXPECT_FALSE( array_3.is_initialised() || array_3.owns_data());
    EXPECT_FALSE( array_6.is_initialised() || array_6.owns_data());

    // Test move assignment

    Array<double> array_9;
    array_9 = std::move(array_5);

    EXPECT_FALSE( array_5.is_initialised() || array_5.owns_data());

    EXPECT_TRUE( array_9.is_initialised() && array_9.owns_data() && array_9.is_contiguous() && array_9.is_semicontiguous());
    EXPECT_TRUE( !array_9.is_row_major() && array_9.is_col_major());
    EXPECT_TRUE(array_9.dims() == 3);  
    EXPECT_TRUE(array_9.size() == 12*15*90);
    for(unsigned ii=0; ii<3; ++ii) EXPECT_TRUE(array_9.size(ii) == shape_3[ii]);
}


TEST(ArrayTest,ElementAccess){
    shape_vec shape = {50,30,10};
    Array<float> array(shape);

    // Set a few values
    array(21,0,0) = 42.42;
    array(0,10,5) = 3.14159;
    array(5,5,3) = 64.32;

    // Test that direct access does in fact write, and does in fact return what we expect
    EXPECT_TRUE(fabs(array(21,0,0) - 42.42) < 1e-5);
    EXPECT_TRUE(fabs(array(0,10,5) - 3.14159) < 1e-5);
    EXPECT_TRUE(fabs(array(5,5,3) - 64.32) < 1e-5);

    // Test that these are at the correct locations in memory
    EXPECT_TRUE(fabs(*(array.data() + 21*300) - 42.42) < 1e-5);
    EXPECT_TRUE(fabs(*(array.data() + 5 + 10*10) - 3.14159) < 1e-5);
    EXPECT_TRUE(fabs(*(array.data() + 3 + 5*10 + 5*300) - 64.32) < 1e-5);

    // Repeat for a column major array
    Array<float> f_array(shape, Array<float>::col_major);

    f_array(21,0,0) = 42.42;
    f_array(0,10,5) = 3.14159;
    f_array(5,5,3) = 64.32;

    EXPECT_TRUE(fabs(f_array(21,0,0) - 42.42) < 1e-5);
    EXPECT_TRUE(fabs(f_array(0,10,5) - 3.14159) < 1e-5);
    EXPECT_TRUE(fabs(f_array(5,5,3) - 64.32) < 1e-5);

    EXPECT_TRUE(fabs(*(f_array.data() + 21) - 42.42) < 1e-5);
    EXPECT_TRUE(fabs(*(f_array.data() + 10*50 + 5*1500) - 3.14159) < 1e-5);
    EXPECT_TRUE(fabs(*(f_array.data() + 5 + 5*50 + 3*1500) - 64.32) < 1e-5);

    // Test again with std::vector-like coordinates
    EXPECT_TRUE(fabs(array(std::vector{21,0,0}) - 42.42) < 1e-5);
    EXPECT_TRUE(fabs(array(std::vector{0,10,5}) - 3.14159) < 1e-5);
    EXPECT_TRUE(fabs(array(std::vector{5,5,3}) - 64.32) < 1e-5);
    EXPECT_TRUE(fabs(f_array(std::vector{21,0,0}) - 42.42) < 1e-5);
    EXPECT_TRUE(fabs(f_array(std::vector{0,10,5}) - 3.14159) < 1e-5);
    EXPECT_TRUE(fabs(f_array(std::vector{5,5,3}) - 64.32) < 1e-5);

    // Test again with C array coordinates
    int coord_1[3] = {21,0,0};
    unsigned coord_2[3] = {0,10,5};
    std::ptrdiff_t coord_3[3] = {5,5,3};

    EXPECT_TRUE(fabs(array(coord_1) - 42.42) < 1e-5);
    EXPECT_TRUE(fabs(array(coord_2) - 3.14159) < 1e-5);
    EXPECT_TRUE(fabs(array(coord_3) - 64.32) < 1e-5);
    EXPECT_TRUE(fabs(f_array(coord_1) - 42.42) < 1e-5);
    EXPECT_TRUE(fabs(f_array(coord_2) - 3.14159) < 1e-5);
    EXPECT_TRUE(fabs(f_array(coord_3) - 64.32) < 1e-5);

    // Test again with dynamic C array coordinates (uses std::vector for simplicity)
    std::vector<int> v_coord_1 = {21,0,0};
    std::vector<unsigned> v_coord_2 = {0,10,5};
    std::vector<std::ptrdiff_t> v_coord_3 = {5,5,3};

    EXPECT_TRUE(fabs(array(v_coord_1) - 42.42) < 1e-5);
    EXPECT_TRUE(fabs(array(v_coord_2) - 3.14159) < 1e-5);
    EXPECT_TRUE(fabs(array(v_coord_3) - 64.32) < 1e-5);
    EXPECT_TRUE(fabs(f_array(v_coord_1) - 42.42) < 1e-5);
    EXPECT_TRUE(fabs(f_array(v_coord_2) - 3.14159) < 1e-5);
    EXPECT_TRUE(fabs(f_array(v_coord_3) - 64.32) < 1e-5);
}

TEST(ArrayTest,FastIteration){
    shape_vec shape = {10,5,20};
    Array<int> row_array(shape, Array<int>::row_major);
    Array<int> col_array(shape, Array<int>::col_major);

    bool row_major_correct = true;
    {
        int count=0;
        for( auto it = row_array.begin_fast(); it != row_array.end_fast(); ++it) *it = count++;

        for( int ii=0; ii<10; ++ii){
            for( int jj=0; jj<5; ++jj){
                for( int kk=0; kk<20; ++kk){
                    if( row_array(ii,jj,kk) != (20*5)*ii + 20*jj + kk){
                       row_major_correct = false;
                    }
                }
            }
        }
    }
    EXPECT_TRUE(row_major_correct);

    bool col_major_correct = true;
    {
        int count=0;
        for( auto it = col_array.begin_fast(); it != col_array.end_fast(); ++it) *it = count++;

        for( int kk=0; kk<20; ++kk){
            for( int jj=0; jj<5; ++jj){
                for( int ii=0; ii<10; ++ii){
                    if( col_array(ii,jj,kk) != (10*5)*kk + 10*jj + ii){
                       col_major_correct = false;
                    }
                }
            }
        }
    }
    EXPECT_TRUE(col_major_correct);
}

TEST(ArrayTest,StripeIteration){
    shape_vec shape = {10,5,20};
    Array<int> row_array(shape, Array<int>::row_major);
    Array<int> col_array(shape, Array<int>::col_major);

    std::size_t row_stripes = row_array.num_stripes();
    std::size_t col_stripes = col_array.num_stripes();

    EXPECT_TRUE(row_stripes == 50);
    EXPECT_TRUE(col_stripes == 100);

    bool row_major_correct = true;
    {
        int count=0;
        for( std::size_t stripe=0; stripe<row_stripes; ++stripe){
            for( auto it = row_array.begin_stripe(stripe); it != row_array.end_stripe(stripe); ++it){
                *it = count++;
            }
        }

        for( int ii=0; ii<10; ++ii){
            for( int jj=0; jj<5; ++jj){
                for( int kk=0; kk<20; ++kk){
                    if( row_array(ii,jj,kk) != (20*5)*ii + 20*jj + kk){
                       row_major_correct = false;
                    }
                }
            }
        }
    }
    EXPECT_TRUE(row_major_correct);

    bool col_major_correct = true;
    {
        int count=0;
        for( std::size_t stripe=0; stripe<col_stripes; ++stripe){
            for( auto it = col_array.begin_stripe(stripe); it != col_array.end_stripe(stripe); ++it){
                *it = count++;
            }
        }

        for( int kk=0; kk<20; ++kk){
            for( int jj=0; jj<5; ++jj){
                for( int ii=0; ii<10; ++ii){
                    if( col_array(ii,jj,kk) != (10*5)*kk + 10*jj + ii){
                       col_major_correct = false;
                    }
                }
            }
        }
    }
    EXPECT_TRUE(col_major_correct);
}

TEST(ArrayTest,GenericIteration){
    shape_vec shape = {10,5,20};
    Array<int> row_array(shape, Array<int>::row_major);
    Array<int> col_array(shape, Array<int>::col_major);

    bool row_major_correct = true;
    {
        int count=0;
        for( auto it = row_array.begin(); it != row_array.end(); ++it) *it = count++;

        for( int ii=0; ii<10; ++ii){
            for( int jj=0; jj<5; ++jj){
                for( int kk=0; kk<20; ++kk){
                    if( row_array(ii,jj,kk) != (20*5)*ii + 20*jj + kk){
                       row_major_correct = false;
                    }
                }
            }
        }
    }
    EXPECT_TRUE(row_major_correct);

    bool col_major_correct = true;
    {
        int count=0;
        for( auto it = col_array.begin(); it != col_array.end(); ++it) *it = count++;

        for( int kk=0; kk<20; ++kk){
            for( int jj=0; jj<5; ++jj){
                for( int ii=0; ii<10; ++ii){
                    if( col_array(ii,jj,kk) != (10*5)*kk + 10*jj + ii){
                       col_major_correct = false;
                    }
                }
            }
        }
    }
    EXPECT_TRUE(col_major_correct);
}

TEST(ArrayTest,FastRandomAccessIteration){
    shape_vec shape = {10,5,20};
    Array<int> row_array(shape, Array<int>::row_major);
    Array<int> col_array(shape, Array<int>::col_major);
    for( auto it = row_array.begin_fast(); it != row_array.end_fast(); ++it) *it = 0;
    for( auto it = col_array.begin_fast(); it != col_array.end_fast(); ++it) *it = 0;
    for( auto it = row_array.begin_fast(); it != row_array.end_fast(); it+=5) *it = 5;
    for( auto it = col_array.begin_fast(); it != col_array.end_fast(); it+=5) *it = 5;
    for( auto it = row_array.end_fast()-1; it != row_array.begin_fast()-1; it-=10) *it = 10;
    for( auto it = col_array.end_fast()-1; it != col_array.begin_fast()-1; it-=10) *it = 10;

    bool row_major_correct = true;
    for( int ii=0; ii<10; ++ii){
        for( int jj=0; jj<5; ++jj){
            for( int kk=0; kk<20; ++kk){
                if( !(kk%5) && row_array(ii,jj,kk) != 5) row_major_correct = false;
                if( !((kk+1)%10) && row_array(ii,jj,kk) != 10) row_major_correct = false;
            }
        }
    }
    EXPECT_TRUE(row_major_correct);

    bool col_major_correct = true;
    for( int kk=0; kk<20; ++kk){
        for( int jj=0; jj<5; ++jj){
            for( int ii=0; ii<10; ++ii){
                if( !(ii%5) && col_array(ii,jj,kk) != 5) col_major_correct = false;
                if( !((ii+1)%10) && col_array(ii,jj,kk) != 10) col_major_correct = false;
            }
        }
    }
    EXPECT_TRUE(col_major_correct);
}

TEST(ArrayTest,GenericRandomAccessIteration){
    shape_vec shape = {10,5,20};
    Array<int> row_array(shape, Array<int>::row_major);
    Array<int> col_array(shape, Array<int>::col_major);
    for( auto it = row_array.begin(); it != row_array.end(); ++it) *it = 0;
    for( auto it = col_array.begin(); it != col_array.end(); ++it) *it = 0;
    for( auto it = row_array.begin(); it != row_array.end(); it+=5) *it = 5;
    for( auto it = col_array.begin(); it != col_array.end(); it+=5) *it = 5;
    for( auto it = row_array.end()-1; it != row_array.begin()-1; it-=10) *it = 10;
    for( auto it = col_array.end()-1; it != col_array.begin()-1; it-=10) *it = 10;

    bool row_major_correct = true;
    for( int ii=0; ii<10; ++ii){
        for( int jj=0; jj<5; ++jj){
            for( int kk=0; kk<20; ++kk){
                if( !(kk%5) && row_array(ii,jj,kk) != 5) row_major_correct = false;
                if( !((kk+1)%10) && row_array(ii,jj,kk) != 10) row_major_correct = false;
            }
        }
    }
    EXPECT_TRUE(row_major_correct);

    bool col_major_correct = true;
    for( int kk=0; kk<20; ++kk){
        for( int jj=0; jj<5; ++jj){
            for( int ii=0; ii<10; ++ii){
                if( !(ii%5) && col_array(ii,jj,kk) != 5) col_major_correct = false;
                if( !((ii+1)%10) && col_array(ii,jj,kk) != 10) col_major_correct = false;
            }
        }
    }
    EXPECT_TRUE(col_major_correct);

}
