#include "ultramat/include/Array.hpp"
#include "ultramat/include/Dense/Math/DenseArithmetic.hpp"

#include <gtest/gtest.h>

using namespace ultra;

TEST(DenseArithmeticTest,Arithmetic){
    auto shape = Shape{5,10,20};
    Array<int>::col_major         a(shape);
    Array<float>::row_major       b(shape);
    Array<unsigned>::col_major    c(shape);
    Array<double>::col_major      d(shape);
    Array<std::size_t>::row_major e(shape);
    for( auto&& x : a) x=1;
    for( auto&& x : b) x=2;
    for( auto&& x : c) x=3;
    for( auto&& x : d) x=4;
    for( auto&& x : e) x=2;
    // Test row major arithmetic
    Array<double>::row_major f = b+e;
    bool f_correct = true;
    for( auto&& x : f ) if( x != 4) f_correct=false;
    EXPECT_TRUE(f_correct);
    // Test col major arithmetic
    Array<double>::col_major g = a+c;
    bool g_correct = true;
    for( auto&& x : g ) if( x != 4) g_correct=false;
    EXPECT_TRUE(g_correct);
    // Combine negation, addition, multiplication, subtraction, and dividing, all between different types and orders. Store in row major
    Array<double>::row_major h = -a + b * c - d / e;
    bool h_correct = true;
    for( auto&& x : h ) if( x != 3) h_correct=false;
    EXPECT_TRUE(h_correct);
    // Repeat, store in col major
    Array<double>::col_major i = -a + b * c - d / e;
    bool i_correct = true;
    for( auto&& x : i ) if( x != 3) i_correct=false;
    EXPECT_TRUE(i_correct);
    // Ensure the original arrays haven't been mangled
    bool a_correct = true, b_correct = true, c_correct = true, d_correct = true, e_correct = true;
    for( auto&& x : a ) if( x != 1) a_correct=false;
    for( auto&& x : b ) if( x != 2) b_correct=false;
    for( auto&& x : c ) if( x != 3) c_correct=false;
    for( auto&& x : d ) if( x != 4) d_correct=false;
    for( auto&& x : e ) if( x != 2) e_correct=false;
    EXPECT_TRUE( a_correct );
    EXPECT_TRUE( b_correct );
    EXPECT_TRUE( c_correct );
    EXPECT_TRUE( d_correct );
    EXPECT_TRUE( e_correct );
}

TEST(DenseArithmeticTest,Boolean){
    auto shape = Shape{5,7};
    Array<int>::col_major      a(shape);
    Array<bool>::row_major     b(shape);
    Array<unsigned>::col_major c(shape);
    Array<float>::col_major    d(shape);
    // fill arrays
    int count;
    count=0; for( auto&& x : a) x = (count++) % 2;
    count=1; for( auto&& x : b) x = (count++) % 2;
    count=0; for( auto&& x : c) x = 5 + ((count++) % 2);
    count=1; for( auto&& x : d) x = 5 + ((count++) % 2);
    // perform comparisons
    Array<bool> not_a = !a;
    Array<bool> a_and_b = a && b;
    Array<bool> a_or_b = a || b;
    Array<bool> c_lt_d = c < d;
    Array<bool> c_gt_d = c > d;
    // Check results
    bool not_a_correct=true, a_and_b_correct=true, a_or_b_correct=true, c_lt_d_correct=true, c_gt_d_correct=true;
    count=1; for( auto&& x : not_a ) not_a_correct &= ( x == ((count++) % 2));
    EXPECT_TRUE(not_a_correct);
    for( auto&& x : a_and_b ) a_and_b_correct &= !x;
    EXPECT_TRUE(a_and_b_correct);
    for( auto&& x : a_or_b ) a_or_b_correct &= x;
    EXPECT_TRUE(a_or_b_correct);
    count=1; for( auto&& x : c_lt_d ) c_lt_d_correct &= ( x == ((count++) % 2));
    EXPECT_TRUE(c_lt_d_correct);
    count=0; for( auto&& x : c_gt_d ) c_gt_d_correct &= ( x == ((count++) % 2));
    EXPECT_TRUE(c_gt_d_correct);
    // test that results can themselves be used in arithmetic
    Array<int> bool_arithmetic = (!a + (c < d));
    bool bool_arithmetic_correct=true;
    count=1; for( auto&& x : bool_arithmetic ) bool_arithmetic_correct &= ( x == 2*((count++) % 2));
    EXPECT_TRUE(bool_arithmetic_correct);
}

TEST(DenseArithmeticTest,InPlaceArithmetic){
    auto shape = Shape{5,10,20};
    Array<double>::row_major a(shape);
    Array<double>::col_major b(shape);
    Array<float,5,10,20>::col_major c(2);
    int count=1;
    for(std::size_t ii=0; ii<5; ++ii){
        for(std::size_t jj=0; jj<10; ++jj){
            for(std::size_t kk=0; kk<20; ++kk){
                a(ii,jj,kk) = count;
                b(ii,jj,kk) = count;
                ++count;
            }
        }
    }

    bool correct_1=true;
    a += b*c;
    count=1;
    for(std::size_t ii=0; ii<5; ++ii){
        for(std::size_t jj=0; jj<10; ++jj){
            for(std::size_t kk=0; kk<20; ++kk){
                correct_1 &= a(ii,jj,kk) == 3*count++;
            }
        }
    }
    EXPECT_TRUE(correct_1);
    
    bool correct_2=true;
    a *= c;
    count=1;
    for(std::size_t ii=0; ii<5; ++ii){
        for(std::size_t jj=0; jj<10; ++jj){
            for(std::size_t kk=0; kk<20; ++kk){
                correct_2 &= a(ii,jj,kk) == 6*count++;
            }
        }
    }
    EXPECT_TRUE(correct_2);

    bool correct_3=true;
    a /= (b+b+b)*c;
    for(std::size_t ii=0; ii<5; ++ii){
        for(std::size_t jj=0; jj<10; ++jj){
            for(std::size_t kk=0; kk<20; ++kk){
                correct_3 &= a(ii,jj,kk) == 1;
            }
        }
    }
    EXPECT_TRUE(correct_3);

}

TEST(DenseArithmeticTest,ScalarArithmetic){
    auto shape = Shape{5,10,20};
    Array<float>::col_major a(shape);
    Array<long>::row_major b(shape);
    int count;
    count=0; for( auto&& x : a) x=count++;
    count=0; for( auto&& x : b) x=count++;
    count = 0; for( auto&& x : a ){ EXPECT_TRUE( x == count++ ); }
    count = 0; for( auto&& x : b ){ EXPECT_TRUE( x == count++ ); }
    Array<double>::col_major c = 2+a;
    Array<int>::row_major d = b * 3;
    Array<int>::row_major e = 2 * d + b * 3;
    // Check that c and d are correct
    bool c_correct = true, d_correct = true, e_correct = true;
    count = 0; for( auto&& x : c ) if( x != 2+count++) c_correct=false;
    count = 0; for( auto&& x : d ) if( x != 3*count++) d_correct=false;
    count = 0; for( auto&& x : e ) if( x != 9*count++) e_correct=false;
    EXPECT_TRUE(c_correct);
    EXPECT_TRUE(d_correct);
    EXPECT_TRUE(e_correct);

    // Test in-place
    // Assigning to a 
    a = 13;
    c += 3;
    d *= 2;
    e /= 3;
    bool in_place_a_correct = true, in_place_c_correct = true, in_place_d_correct = true, in_place_e_correct = true;
    for( auto&& x : a ) if( x != 13) in_place_a_correct=false;
    count = 0; for( auto&& x : c ) if( x != 5+count++) in_place_c_correct=false;
    count = 0; for( auto&& x : d ) if( x != 6*count++) in_place_d_correct=false;
    count = 0; for( auto&& x : e ) if( x != 3*count++) in_place_e_correct=false;
    EXPECT_TRUE(in_place_a_correct);
    EXPECT_TRUE(in_place_c_correct);
    EXPECT_TRUE(in_place_d_correct);
    EXPECT_TRUE(in_place_e_correct);
}
