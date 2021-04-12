#include "ultramat/include/Dense.hpp"
#include "ultramat/include/Arithmetic.hpp"
#include <gtest/gtest.h>

using namespace ultra;
using shape_vec = std::vector<std::size_t>;

TEST(ArrayArithmeticTest,Arithmetic){
    auto shape = shape_vec{5,10,20};
    Array<int>         a(shape);
    Array<float>       b(shape);
    Array<unsigned>    c(shape);
    Array<double>      d(shape);
    Array<std::size_t> e(shape);
    for( auto&& x : a) x=1;
    for( auto&& x : b) x=2;
    for( auto&& x : c) x=3;
    for( auto&& x : d) x=4;
    for( auto&& x : e) x=2;
    // Combine negation, addition, multiplication, subtraction, and dividing, all between different types.
    Array<double> f = -a + b * c - d / e;
    // Check that f is correct
    bool f_correct = true;
    for( auto&& x : f ) if( x != 3) f_correct=false;
    EXPECT_TRUE(f_correct);
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

TEST(ArrayArithmeticTest,InPlaceArithmetic){
    auto shape = shape_vec{5,10,20};
    Array<double> a(shape);
    Array<double> b(shape);
    Array<float,5,10,20> c;
    int count;
    count=1; for(auto&& x : a ) x = count++;
    count=1; for(auto&& x : b ) x = count++;
    for(auto&& x : c ) x = 2;

    bool correct_1=true;
    a += b*c;
    count=1;
    for(auto&& x : a) correct_1 &= (x == 3*count++);
    EXPECT_TRUE(correct_1);
    
    bool correct_2=true;
    a *= c;
    count=1;
    for(auto&& x : a) correct_2 &= (x == 6*count++);
    EXPECT_TRUE(correct_2);

    bool correct_3=true;
    a /= (b+b+b)*c;
    for(auto&& x : a){
        correct_3 &= (x == 1);
    }
    EXPECT_TRUE(correct_3);

}
