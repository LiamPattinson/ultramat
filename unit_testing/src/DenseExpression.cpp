#include "ultramat/include/Array.hpp"
#include "ultramat/include/Dense/Math/DenseArithmetic.hpp"
#include "ultramat/include/Dense/Math/DenseMath.hpp"
#include <gtest/gtest.h>

using namespace ultra;

TEST(DenseExpressionTest,Eval){
    Shape shape{5,10,20};
    Array<int> a(shape);
    Array<int> b(shape);
    Array<double> c(shape);
    Array<double> d(shape);
    for( auto&& x : a ) x = 7;
    for( auto&& x : b ) x = -3;
    for( auto&& x : c ) x = 1.5;
    for( auto&& x : d ) x = -0.5;

    // Standard, without eval
    Array<double> e = (a + b) + (c + d);

    // With eval sprinkled throughout
    Array<double> f = eval(eval(eval(a+b) + (eval(eval(c)) + eval(d))));

    EXPECT_TRUE( e.dims() == 3 );
    EXPECT_TRUE( e.size() == 1000 );
    EXPECT_TRUE( e.shape(0) == 5 );
    EXPECT_TRUE( e.shape(1) == 10 );
    EXPECT_TRUE( e.shape(2) == 20 );
    EXPECT_TRUE( f.dims() == 3 );
    EXPECT_TRUE( f.size() == 1000 );
    EXPECT_TRUE( f.shape(0) == 5 );
    EXPECT_TRUE( f.shape(1) == 10 );
    EXPECT_TRUE( f.shape(2) == 20 );
    ASSERT_TRUE( e.size() == f.size() );

    bool correct=true;
    auto e_it = e.begin();
    auto f_it = f.begin();
    auto e_end = e.end();
    for(; e_it != e_end; ++e_it){
        if( *e_it != *f_it) correct=false;
    }
    EXPECT_TRUE(correct);

    // Evaluating reshape
    Array<float> g = reshape( a*b + c, Shape{50,2,5,2});
    EXPECT_TRUE( g.dims() == 4 );
    EXPECT_TRUE( g.size() == 1000 );
    EXPECT_TRUE( g.shape(0) == 50 );
    EXPECT_TRUE( g.shape(1) == 2 );
    EXPECT_TRUE( g.shape(2) == 5 );
    EXPECT_TRUE( g.shape(3) == 2 );
    bool g_correct=true;
    for( auto&& x : g ) if( x != -19.5 ) g_correct = false;
    EXPECT_TRUE(g_correct);

    // Evaluating view
    Array<double> h = view( a*b + c, Slice{0,2}, Slice{0,2},Slice{0,2});
    EXPECT_TRUE( h.dims() == 3 );
    EXPECT_TRUE( h.size() == 8 );
    EXPECT_TRUE( h.shape(0) == 2 );
    EXPECT_TRUE( h.shape(1) == 2 );
    EXPECT_TRUE( h.shape(2) == 2 );
    bool h_correct=true;
    for( auto&& x : h ) if( x != -19.5 ) h_correct = false;
    EXPECT_TRUE(h_correct);

    // Evaluating Hermitian
    Array<std::complex<double>> k(Shape{5,10}); k.fill(std::complex<double>(1,1));
    Array<std::complex<double>> l = hermitian(2*k);
    EXPECT_TRUE( l.dims() == 2 );
    EXPECT_TRUE( l.size() == 50 );
    EXPECT_TRUE( l.shape(0) == 10 );
    EXPECT_TRUE( l.shape(1) == 5 );
    bool l_correct=true;
    for( auto&& x : l ) if( x != std::complex<double>(2,-2) ) l_correct = false;
    EXPECT_TRUE(l_correct);
}
