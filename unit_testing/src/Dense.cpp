#include "ultramat/include/Dense.hpp"
#include <gtest/gtest.h>
#include <cmath>

using namespace ultra;
using shape = std::vector<std::size_t>;

TEST(ArrayTest,Constructors){
    Array<int> array_1(shape{25}); // default construct
    Array<float>::row_major array_2(shape{50,30},2.0); // fill construct, specify row major
    Array<double>::col_major array_3(shape{12,15,90},5.0); // fill construct, specify col major

    // Test attributes

    EXPECT_TRUE(array_1.dims() == 1);
    EXPECT_TRUE(array_2.dims() == 2);  
    EXPECT_TRUE(array_3.dims() == 3);  

    EXPECT_TRUE(array_1.size() == 25);
    EXPECT_TRUE(array_2.size() == 50*30);
    EXPECT_TRUE(array_3.size() == 12*15*90);

    EXPECT_TRUE(array_1.shape(0) == 25);
    EXPECT_TRUE(array_2.shape(0) == 50);
    EXPECT_TRUE(array_2.shape(1) == 30);
    EXPECT_TRUE(array_3.shape(0) == 12);
    EXPECT_TRUE(array_3.shape(1) == 15);
    EXPECT_TRUE(array_3.shape(2) == 90);

    EXPECT_TRUE(array_1.stride(0) == 25);
    EXPECT_TRUE(array_1.stride(1) == 1);
    EXPECT_TRUE(array_2.stride(0) == 30*50);
    EXPECT_TRUE(array_2.stride(1) == 30);
    EXPECT_TRUE(array_2.stride(2) == 1);
    EXPECT_TRUE(array_3.stride(0) == 1);
    EXPECT_TRUE(array_3.stride(1) == 12);
    EXPECT_TRUE(array_3.stride(2) == 12*15);
    EXPECT_TRUE(array_3.stride(3) == 12*15*90);
    
    // Test copy

    auto array_4(array_2);

    EXPECT_TRUE(array_4.dims() == 2);  
    EXPECT_TRUE(array_4.size() == 50*30);
    EXPECT_TRUE(array_4.shape(0) == 50);
    EXPECT_TRUE(array_4.shape(1) == 30);
    EXPECT_TRUE(array_4.stride(0) == 30*50);
    EXPECT_TRUE(array_4.stride(1) == 30);
    EXPECT_TRUE(array_4.stride(2) == 1);

    // Test move

    auto array_5(std::move(array_3));

    EXPECT_TRUE(array_5.dims() == 3);  
    EXPECT_TRUE(array_5.size() == 12*15*90);
    EXPECT_TRUE(array_5.shape(0) == 12);
    EXPECT_TRUE(array_5.shape(1) == 15);
    EXPECT_TRUE(array_5.shape(2) == 90);
    EXPECT_TRUE(array_5.stride(0) == 1);
    EXPECT_TRUE(array_5.stride(1) == 12);
    EXPECT_TRUE(array_5.stride(2) == 12*15);
    EXPECT_TRUE(array_5.stride(3) == 12*15*90);

    // Test full shape and stride reading
    auto shape = array_5.shape();
    auto stride = array_4.stride();

    EXPECT_TRUE(shape[0] == 12);
    EXPECT_TRUE(shape[1] == 15);
    EXPECT_TRUE(shape[2] == 90);
    EXPECT_TRUE(stride[0] == 30*50);
    EXPECT_TRUE(stride[1] == 30);
    EXPECT_TRUE(stride[2] == 1);

    // Test reshape

    array_5.reshape(6,2,3,5,9,1,10);

    EXPECT_TRUE(array_5.dims() == 7);  
    EXPECT_TRUE(array_5.size() == 12*15*90);
    EXPECT_TRUE(array_5.shape(0) == 6);
    EXPECT_TRUE(array_5.shape(1) == 2);
    EXPECT_TRUE(array_5.shape(2) == 3);
    EXPECT_TRUE(array_5.shape(3) == 5);
    EXPECT_TRUE(array_5.shape(4) == 9);
    EXPECT_TRUE(array_5.shape(5) == 1);
    EXPECT_TRUE(array_5.shape(6) == 10);
    EXPECT_TRUE(array_5.stride(0) == 1);
    EXPECT_TRUE(array_5.stride(1) == 6);
    EXPECT_TRUE(array_5.stride(2) == 6*2);
    EXPECT_TRUE(array_5.stride(3) == 6*2*3);
    EXPECT_TRUE(array_5.stride(4) == 6*2*3*5);
    EXPECT_TRUE(array_5.stride(5) == 6*2*3*5*9);
    EXPECT_TRUE(array_5.stride(6) == 6*2*3*5*9*1);
    EXPECT_TRUE(array_5.stride(7) == 6*2*3*5*9*1*10);

    array_5.reshape(12*15*90);
    EXPECT_TRUE(array_5.dims() == 1);  
    EXPECT_TRUE(array_5.size() == 12*15*90);
    EXPECT_TRUE(array_5.shape(0) == 12*15*90);
    EXPECT_TRUE(array_5.stride(0) == 1);
    EXPECT_TRUE(array_5.stride(1) == 12*15*90);
}


TEST(ArrayTest,ElementAccess){
    Array<float> array(shape{30,20,10},17.);

    // Test fill
    EXPECT_TRUE(fabs(array(0,0,0) - 17.) < 1e-5);
    EXPECT_TRUE(fabs(array(8,2,1) - 17.) < 1e-5);
    EXPECT_TRUE(fabs(array(2,14,5) - 17.) < 1e-5);
    EXPECT_TRUE(fabs(array(29,19,9) - 17.) < 1e-5);

    // Set a few values
    array(21,0,0) = 42.42;
    array(0,10,5) = 3.14159;
    array(5,5,3) = 64.32;

    // Test that getting and setting access the same values
    EXPECT_TRUE(fabs(array(21,0,0) - 42.42) < 1e-5);
    EXPECT_TRUE(fabs(array(0,10,5) - 3.14159) < 1e-5);
    EXPECT_TRUE(fabs(array(5,5,3) - 64.32) < 1e-5);

    // Test that these are at the correct locations in memory
    EXPECT_TRUE(fabs(*(array.data() + 21*200) - 42.42) < 1e-5);
    EXPECT_TRUE(fabs(*(array.data() + 5 + 10*10) - 3.14159) < 1e-5);
    EXPECT_TRUE(fabs(*(array.data() + 3 + 5*10 + 5*200) - 64.32) < 1e-5);

    // Repeat for a column major array
    Array<float>::col_major col_array(shape{30,20,10});
    col_array.fill(17.);

    EXPECT_TRUE(fabs(col_array(0,0,0) - 17.) < 1e-5);
    EXPECT_TRUE(fabs(col_array(8,2,1) - 17.) < 1e-5);
    EXPECT_TRUE(fabs(col_array(2,14,5) - 17.) < 1e-5);
    EXPECT_TRUE(fabs(col_array(29,19,9) - 17.) < 1e-5);

    col_array(21,0,0) = 42.42;
    col_array(0,10,5) = 3.14159;
    col_array(5,5,3) = 64.32;

    EXPECT_TRUE(fabs(col_array(21,0,0) - 42.42) < 1e-5);
    EXPECT_TRUE(fabs(col_array(0,10,5) - 3.14159) < 1e-5);
    EXPECT_TRUE(fabs(col_array(5,5,3) - 64.32) < 1e-5);

    EXPECT_TRUE(fabs(*(col_array.data() + 21) - 42.42) < 1e-5);
    EXPECT_TRUE(fabs(*(col_array.data() + 10*30 + 5*600) - 3.14159) < 1e-5);
    EXPECT_TRUE(fabs(*(col_array.data() + 5 + 5*30 + 3*600) - 64.32) < 1e-5);

    // Test again with std::vector-like coordinates
    EXPECT_TRUE(fabs(array(std::vector{21,0,0}) - 42.42) < 1e-5);
    EXPECT_TRUE(fabs(array(std::vector{0,10,5}) - 3.14159) < 1e-5);
    EXPECT_TRUE(fabs(array(std::vector{5,5,3}) - 64.32) < 1e-5);
    EXPECT_TRUE(fabs(col_array(std::vector{21,0,0}) - 42.42) < 1e-5);
    EXPECT_TRUE(fabs(col_array(std::vector{0,10,5}) - 3.14159) < 1e-5);
    EXPECT_TRUE(fabs(col_array(std::vector{5,5,3}) - 64.32) < 1e-5);

}

TEST(ArrayTest,Iteration){
    Array<int>::row_major row_array(shape{10,5,20},0);
    Array<int>::col_major col_array(shape{10,5,20},0);
    int count;

    bool row_major_correct = true, col_major_correct=true;
    

    // test begin and end explicitly
    count = 0;
    for( auto it=row_array.begin(), end=row_array.end(); it != end; ++it) *it += count++;

    count = 0;
    for( auto it=col_array.begin(), end=col_array.end(); it != end; ++it) *it += count++;

    
    // test begin and end implicitly
    count = 0;
    for( auto&& x : row_array) x += count++;
    
    count = 0;
    for( auto&& x : col_array) x += count++;

    // check correctness
    for( int ii=0; ii<10; ++ii){
        for( int jj=0; jj<5; ++jj){
            for( int kk=0; kk<20; ++kk){
                if( row_array(ii,jj,kk) != 2*((20*5)*ii + 20*jj + kk)){
                   row_major_correct = false;
                }
            }
        }
    }
    EXPECT_TRUE(row_major_correct);

    for( int kk=0; kk<20; ++kk){
        for( int jj=0; jj<5; ++jj){
            for( int ii=0; ii<10; ++ii){
                if( col_array(ii,jj,kk) != 2*((10*5)*kk + 10*jj + ii)){
                   col_major_correct = false;
                }
            }
        }
    }
    EXPECT_TRUE(col_major_correct);
}

// Repeat these tests for fixed dense arrays

TEST(FixedArrayTest,Constructors){

    Array<int,25> array_1; // default construct
    Array<float,50,30>::row_major array_2(2.0); // fill construct, specify row major
    Array<double,12,15,90>::col_major array_3(5.0); // fill construct, specify col major

    // Test attributes

    EXPECT_TRUE(array_1.dims() == 1);
    EXPECT_TRUE(array_2.dims() == 2);  
    EXPECT_TRUE(array_3.dims() == 3);  

    EXPECT_TRUE(array_1.size() == 25);
    EXPECT_TRUE(array_2.size() == 50*30);
    EXPECT_TRUE(array_3.size() == 12*15*90);

    EXPECT_TRUE(array_1.shape(0) == 25);
    EXPECT_TRUE(array_2.shape(0) == 50);
    EXPECT_TRUE(array_2.shape(1) == 30);
    EXPECT_TRUE(array_3.shape(0) == 12);
    EXPECT_TRUE(array_3.shape(1) == 15);
    EXPECT_TRUE(array_3.shape(2) == 90);

    EXPECT_TRUE(array_1.stride(0) == 25);
    EXPECT_TRUE(array_1.stride(1) == 1);
    EXPECT_TRUE(array_2.stride(0) == 30*50);
    EXPECT_TRUE(array_2.stride(1) == 30);
    EXPECT_TRUE(array_2.stride(2) == 1);
    EXPECT_TRUE(array_3.stride(0) == 1);
    EXPECT_TRUE(array_3.stride(1) == 12);
    EXPECT_TRUE(array_3.stride(2) == 12*15);
    EXPECT_TRUE(array_3.stride(3) == 12*15*90);
    
    // Test copy

    auto array_4(array_2);

    EXPECT_TRUE(array_4.dims() == 2);  
    EXPECT_TRUE(array_4.size() == 50*30);
    EXPECT_TRUE(array_4.shape(0) == 50);
    EXPECT_TRUE(array_4.shape(1) == 30);
    EXPECT_TRUE(array_4.stride(0) == 30*50);
    EXPECT_TRUE(array_4.stride(1) == 30);
    EXPECT_TRUE(array_4.stride(2) == 1);

    // Test move

    auto array_5(std::move(array_3));

    EXPECT_TRUE(array_5.dims() == 3);  
    EXPECT_TRUE(array_5.size() == 12*15*90);
    EXPECT_TRUE(array_5.shape(0) == 12);
    EXPECT_TRUE(array_5.shape(1) == 15);
    EXPECT_TRUE(array_5.shape(2) == 90);
    EXPECT_TRUE(array_5.stride(0) == 1);
    EXPECT_TRUE(array_5.stride(1) == 12);
    EXPECT_TRUE(array_5.stride(2) == 12*15);
    EXPECT_TRUE(array_5.stride(3) == 12*15*90);

    // Test full shape and stride reading
    auto shape = array_5.shape();
    auto stride = array_4.stride();

    EXPECT_TRUE(shape[0] == 12);
    EXPECT_TRUE(shape[1] == 15);
    EXPECT_TRUE(shape[2] == 90);
    EXPECT_TRUE(stride[0] == 30*50);
    EXPECT_TRUE(stride[1] == 30);
    EXPECT_TRUE(stride[2] == 1);
}


TEST(FixedArrayTest,ElementAccess){
    Array<float,30,20,10> array(17.);

    // Test fill
    EXPECT_TRUE(fabs(array(0,0,0) - 17.) < 1e-5);
    EXPECT_TRUE(fabs(array(8,2,1) - 17.) < 1e-5);
    EXPECT_TRUE(fabs(array(2,14,5) - 17.) < 1e-5);
    EXPECT_TRUE(fabs(array(29,19,9) - 17.) < 1e-5);

    // Set a few values
    array(21,0,0) = 42.42;
    array(0,10,5) = 3.14159;
    array(5,5,3) = 64.32;

    // Test that getting and setting access the same values
    EXPECT_TRUE(fabs(array(21,0,0) - 42.42) < 1e-5);
    EXPECT_TRUE(fabs(array(0,10,5) - 3.14159) < 1e-5);
    EXPECT_TRUE(fabs(array(5,5,3) - 64.32) < 1e-5);

    // Test that these are at the correct locations in memory
    EXPECT_TRUE(fabs(*(array.data() + 21*200) - 42.42) < 1e-5);
    EXPECT_TRUE(fabs(*(array.data() + 5 + 10*10) - 3.14159) < 1e-5);
    EXPECT_TRUE(fabs(*(array.data() + 3 + 5*10 + 5*200) - 64.32) < 1e-5);

    // Repeat for a column major array
    Array<float,30,20,10>::col_major col_array(17.);

    EXPECT_TRUE(fabs(col_array(0,0,0) - 17.) < 1e-5);
    EXPECT_TRUE(fabs(col_array(8,2,1) - 17.) < 1e-5);
    EXPECT_TRUE(fabs(col_array(2,14,5) - 17.) < 1e-5);
    EXPECT_TRUE(fabs(col_array(29,19,9) - 17.) < 1e-5);

    col_array(21,0,0) = 42.42;
    col_array(0,10,5) = 3.14159;
    col_array(5,5,3) = 64.32;

    EXPECT_TRUE(fabs(col_array(21,0,0) - 42.42) < 1e-5);
    EXPECT_TRUE(fabs(col_array(0,10,5) - 3.14159) < 1e-5);
    EXPECT_TRUE(fabs(col_array(5,5,3) - 64.32) < 1e-5);

    EXPECT_TRUE(fabs(*(col_array.data() + 21) - 42.42) < 1e-5);
    EXPECT_TRUE(fabs(*(col_array.data() + 10*30 + 5*600) - 3.14159) < 1e-5);
    EXPECT_TRUE(fabs(*(col_array.data() + 5 + 5*30 + 3*600) - 64.32) < 1e-5);

    // Test again with std::vector-like coordinates
    EXPECT_TRUE(fabs(array(std::vector{21,0,0}) - 42.42) < 1e-5);
    EXPECT_TRUE(fabs(array(std::vector{0,10,5}) - 3.14159) < 1e-5);
    EXPECT_TRUE(fabs(array(std::vector{5,5,3}) - 64.32) < 1e-5);
    EXPECT_TRUE(fabs(col_array(std::vector{21,0,0}) - 42.42) < 1e-5);
    EXPECT_TRUE(fabs(col_array(std::vector{0,10,5}) - 3.14159) < 1e-5);
    EXPECT_TRUE(fabs(col_array(std::vector{5,5,3}) - 64.32) < 1e-5);

}

TEST(FixedArrayTest,Iteration){
    Array<int,10,5,20>::row_major row_array(0);
    Array<int,10,5,20>::col_major col_array(0);
    int count;

    bool row_major_correct = true, col_major_correct=true;
    

    // test begin and end explicitly
    count = 0;
    for( auto it=row_array.begin(), end=row_array.end(); it != end; ++it) *it += count++;

    count = 0;
    for( auto it=col_array.begin(), end=col_array.end(); it != end; ++it) *it += count++;

    
    // test begin and end implicitly
    count = 0;
    for( auto&& x : row_array) x += count++;
    
    count = 0;
    for( auto&& x : col_array) x += count++;

    // check correctness
    for( int ii=0; ii<10; ++ii){
        for( int jj=0; jj<5; ++jj){
            for( int kk=0; kk<20; ++kk){
                if( row_array(ii,jj,kk) != 2*((20*5)*ii + 20*jj + kk)){
                   row_major_correct = false;
                }
            }
        }
    }
    EXPECT_TRUE(row_major_correct);

    for( int kk=0; kk<20; ++kk){
        for( int jj=0; jj<5; ++jj){
            for( int ii=0; ii<10; ++ii){
                if( col_array(ii,jj,kk) != 2*((10*5)*kk + 10*jj + ii)){
                   col_major_correct = false;
                }
            }
        }
    }
    EXPECT_TRUE(col_major_correct);
}

TEST(FixedArrayTest,StripedIteration){
    using shape = std::vector<std::size_t>;
    Array<int,10,5,20>::row_major row_array;
    Array<int>::col_major col_array(shape{10,5,20});
    int count;

    bool row_major_correct = true, col_major_correct=true;

    // Fill both row major and col major using stripes
    count = 0;
    for( auto&& stripe : row_array.stripes()){
        for( auto&& x : stripe) x = count++;
    }

    count = 0;
    for( auto&& stripe : col_array.stripes()){
        for( auto&& x : stripe) x = count++;
    }

    // check correctness
    for( int ii=0; ii<10; ++ii){
        for( int jj=0; jj<5; ++jj){
            for( int kk=0; kk<20; ++kk){
                if( row_array(ii,jj,kk) != (20*5)*ii + 20*jj + kk){
                   row_major_correct = false;
                }
            }
        }
    }
    EXPECT_TRUE(row_major_correct);

    for( int kk=0; kk<20; ++kk){
        for( int jj=0; jj<5; ++jj){
            for( int ii=0; ii<10; ++ii){
                if( col_array(ii,jj,kk) != (10*5)*kk + 10*jj + ii){
                   col_major_correct = false;
                }
            }
        }
    }
    EXPECT_TRUE(col_major_correct);

    // Repeat using different dimension of stripes
    bool row_dim_1_correct=true, col_dim_2_correct=true;

    count = 0;
    for( auto&& stripe : row_array.stripes(1)){
        for( auto&& x : stripe) x = count++;
    }

    count = 0;
    for( auto&& stripe : col_array.stripes(2)){
        for( auto&& x : stripe) x = count++;
    }

    // check correctness
    for( int ii=0; ii<10; ++ii){
        for( int kk=0; kk<20; ++kk){
            for( int jj=0; jj<5; ++jj){
                if( row_array(ii,jj,kk) != (20*5)*ii + 5*kk + jj){
                   row_dim_1_correct = false;
                }
            }
        }
    }
    EXPECT_TRUE(row_dim_1_correct);

    for( int jj=0; jj<5; ++jj){
        for( int ii=0; ii<10; ++ii){
            for( int kk=0; kk<20; ++kk){
                if( col_array(ii,jj,kk) != (10*20)*jj + 20*ii + kk){
                   col_dim_2_correct = false;
                }
            }
        }
    }
    EXPECT_TRUE(col_dim_2_correct);
}
