#include "ultramat/include/Array.hpp"
#include "ultramat/include/Dense/Math/DenseArithmetic.hpp"
#include "ultramat/include/Dense/Math/DenseMath.hpp"
#include "ultramat/include/Dense/Math/DenseGenerators.hpp"
#include <gtest/gtest.h>

using namespace ultra;

TEST(DenseGeneratorTest,OnesAndZeros){
    auto shape = Shape{5,10,20};

    // Set things to zeros/ones
    Array<int>    a(shape); a = zeros(shape);
    Array<float>  b = ones(shape);

    bool zero_correct=true, one_correct=true;
    for( auto&& x : a ) if( x )    zero_correct=false;
    for( auto&& x : b ) if( x!=1 ) one_correct=false;

    EXPECT_TRUE(zero_correct);
    EXPECT_TRUE(one_correct);
    EXPECT_TRUE(a.shape(0) == 5);
    EXPECT_TRUE(a.shape(1) == 10);
    EXPECT_TRUE(a.shape(2) == 20);
    EXPECT_TRUE(b.shape(0) == 5);
    EXPECT_TRUE(b.shape(1) == 10);
    EXPECT_TRUE(b.shape(2) == 20);

    // Combine ones with arithmetic
    Array<float> c = b + b + ones(shape) + b;
    bool four_correct = true;
    for( auto&& x : c ) if( x!=4 ) four_correct=false;
    EXPECT_TRUE(four_correct);
}

TEST(DenseGeneratorTest,Linspace){
    std::size_t idx;

    Array<double> a = linspace(0,1,101);
    EXPECT_TRUE(a.dims() == 1);
    EXPECT_TRUE(a.size() == 101);
    EXPECT_TRUE(a.shape(0) == 101);
    bool linear_linspace_correct = true;
    idx=0;
    for(auto&& x : a) if( std::abs(x - (idx++)*0.01) > 1e-5 ) linear_linspace_correct = false;
    EXPECT_TRUE(linear_linspace_correct);

    Array<double> b = linspace(0,1,101) + linspace(0,1,101);
    EXPECT_TRUE(b.dims() == 1);
    EXPECT_TRUE(b.size() == 101);
    EXPECT_TRUE(b.shape(0) == 101);
    bool double_linspace_correct = true;
    idx=0;
    for(auto&& x : b) if( std::abs(x - (idx++)*0.02) > 1e-5 ) double_linspace_correct = false;
    EXPECT_TRUE(double_linspace_correct);

}

TEST(DenseGeneratorTest,Arange){
    int count;
    Array<int> int_a = arange(0,100,3);
    Array<int> int_b = arange(0,100,4);
    EXPECT_TRUE(int_a.dims() == 1);
    EXPECT_TRUE(int_b.dims() == 1);
    EXPECT_TRUE(int_a.size() == 34);
    EXPECT_TRUE(int_b.size() == 25);
    bool int_a_correct = true, int_b_correct = true;
    count=0; for(auto&& x : int_a) if( x != 3*count++ ) int_a_correct = false;
    count=0; for(auto&& x : int_b) if( x != 4*count++ ) int_b_correct = false;
    EXPECT_TRUE(int_a_correct);
    EXPECT_TRUE(int_b_correct);

    // test reverse steps
    Array<int> int_c = arange(100,0,-3);
    Array<int> int_d = arange(100,0,-4);
    EXPECT_TRUE(int_c.dims() == 1);
    EXPECT_TRUE(int_d.dims() == 1);
    EXPECT_TRUE(int_c.size() == 34);
    EXPECT_TRUE(int_d.size() == 25);
    bool int_c_correct = true, int_d_correct = true;
    count=0; for(auto&& x : int_c) if( x != 100-3*count++ ) int_c_correct = false;
    count=0; for(auto&& x : int_d) if( x != 100-4*count++ ) int_d_correct = false;
    EXPECT_TRUE(int_c_correct);
    EXPECT_TRUE(int_d_correct);

    // repeat with floats
    Array<float> float_a = arange(0.,100,3);
    Array<float> float_b = arange(0.,100,4);
    EXPECT_TRUE(float_a.dims() == 1);
    EXPECT_TRUE(float_b.dims() == 1);
    EXPECT_TRUE(float_a.size() == 34);
    EXPECT_TRUE(float_b.size() == 25);
    bool float_a_correct = true, float_b_correct = true;
    count=0; for(auto&& x : float_a) if( x != 3*count++ ) float_a_correct = false;
    count=0; for(auto&& x : float_b) if( x != 4*count++ ) float_b_correct = false;
    EXPECT_TRUE(float_a_correct);
    EXPECT_TRUE(float_b_correct);

    // test reverse steps
    Array<float> float_c = arange(100,0,-3.);
    Array<float> float_d = arange(100,0,-4.);
    EXPECT_TRUE(float_c.dims() == 1);
    EXPECT_TRUE(float_d.dims() == 1);
    EXPECT_TRUE(float_c.size() == 34);
    EXPECT_TRUE(float_d.size() == 25);
    bool float_c_correct = true, float_d_correct = true;
    count=0; for(auto&& x : float_c) if( x != 100-3*count++ ) float_c_correct = false;
    count=0; for(auto&& x : float_d) if( x != 100-4*count++ ) float_d_correct = false;
    EXPECT_TRUE(float_c_correct);
    EXPECT_TRUE(float_d_correct);

}

TEST(DenseGeneratorTest,Logspace){
    Array<double> a = logspace(-2,2,5);
    // Note:
    // Early versions of this code had a dangling reference bug. Using assertions here
    // to halt testing if anything weird happens.
    ASSERT_TRUE(a.dims() == 1);
    ASSERT_TRUE(a.size() == 5);
    ASSERT_TRUE(a.shape(0) == 5);
    bool logspace_correct = true;
    int idx=-2;
    for(auto&& x : a) if( std::fabs(x - std::pow(10,idx++)) > 1e-5 ) logspace_correct = false;
    EXPECT_TRUE(logspace_correct);
}

TEST(DenseGeneratorTest,Random){
    Shape shape{5,10,20}; 
    Array<double> a = random(std::uniform_real_distribution<double>(0,1),100);
    Array<double> b = random(std::uniform_real_distribution<double>(0,1),100);
    Array<int> c = random(std::uniform_int_distribution<int>(-5000000,500000),shape);
    Array<int> d = random(std::uniform_int_distribution<int>(-5000000,500000),shape);
    EXPECT_TRUE(a.dims() == 1);
    EXPECT_TRUE(a.size() == 100);
    EXPECT_TRUE(a.shape(0) == 100);
    EXPECT_TRUE(b.dims() == 1);
    EXPECT_TRUE(b.size() == 100);
    EXPECT_TRUE(b.shape(0) == 100);
    EXPECT_TRUE(c.dims() == 3);
    EXPECT_TRUE(c.size() == 1000);
    EXPECT_TRUE(c.shape(0) == 5);
    EXPECT_TRUE(c.shape(1) == 10);
    EXPECT_TRUE(c.shape(2) == 20);
    EXPECT_TRUE(d.dims() == 3);
    EXPECT_TRUE(d.size() == 1000);
    EXPECT_TRUE(d.shape(0) == 5);
    EXPECT_TRUE(d.shape(1) == 10);
    EXPECT_TRUE(d.shape(2) == 20);
    
    // Test that the numbers generated don't repeat
    // This mostly tests that random numbers generated in parallel are in fact unique across different threads
    // (there is a very slim chance that this will fail even if things are working correctly!)
    bool random_double_correct = true;
    std::sort(b.begin(),b.end());
    for( auto it = b.begin(), end = b.end()-1; it != end; ++it){
        if( *it == *(it+1) ) random_double_correct = false;
    }
    EXPECT_TRUE(random_double_correct);
    
    // Test that they can be used in arithmetic. Using simplified version (first one will be an int distribution, second real).
    Array<double> e = (a + b + random_uniform(0,5,100)) * random_uniform(0.,100,100);
    EXPECT_TRUE(e.dims() == 1);
    EXPECT_TRUE(e.size() == 100);
    EXPECT_TRUE(e.shape(0) == 100);
    bool e_correct = true;
    for( auto&& x : e ) if ( x < 0 || x > 700 ) e_correct = false;
    EXPECT_TRUE( e_correct );

    // Test Gaussian
    std::size_t num_vals = 10000;
    Array<double> normal = random_normal(10,2,num_vals);
    EXPECT_TRUE(normal.dims() == 1);
    EXPECT_TRUE(normal.size() == num_vals);
    EXPECT_TRUE(normal.shape(0) == num_vals);
    int within_1_stddev = 0, within_2_stddev = 0;
    for( auto&& x : normal ){
        if( x > 6 && x < 14 ){
            within_2_stddev++;
            if( x > 8 && x < 12 ){
                within_1_stddev++;
            }
        }
    }
    double percent_within_1_stddev = (100.*within_1_stddev) / num_vals;
    double percent_within_2_stddev = (100.*within_2_stddev) / num_vals;
    EXPECT_LT( std::fabs(percent_within_1_stddev-68.27), 2 );
    EXPECT_LT( std::fabs(percent_within_2_stddev-95.45), 2 );
}
