#include "ultramat/include/Array.hpp"
#include "ultramat/include/Dense/Math/DenseArithmetic.hpp"
#include "ultramat/include/Dense/Math/DenseMath.hpp"
#include <gtest/gtest.h>

using namespace ultra;

TEST(DenseMathTest,Math){
    auto shape = Shape{5,10,20};
    Array<int>         a(shape);
    Array<float>       b(shape);
    Array<double>      c(shape);
    Array<std::size_t> d(shape);
    for( auto&& x : a) x=10;
    for( auto&& x : b) x=2.5;
    for( auto&& x : c) x=0.5;
    
    // Some easy ones to start...
    d = ultra::floor(b); // expression assignment
    Array<unsigned> e = ultra::ceil(c); // expression copy 
    bool floor_correct = true, ceil_correct=true;
    for( auto&& x : d ) if( x != 2) floor_correct=false;
    for( auto&& x : e ) if( x != 1) ceil_correct=false;
    EXPECT_TRUE(floor_correct);
    EXPECT_TRUE(ceil_correct);

    // And now something a bit mad...
    Array<double> f =(ultra::exp(c) + ultra::pow(b,ultra::sin(e)) * ultra::log(a) ) / d;
    double answer = (std::exp(0.5) + std::pow(2.5,std::sin(1)) * std::log(10))/2;
    bool lots_of_math_correct = true;
    for( auto&& x : f ) if( std::abs(x-answer) > 1e-5) lots_of_math_correct=false;
    EXPECT_TRUE(lots_of_math_correct);

    // Check that calling ultra functions on arithmetic types works as expected
    EXPECT_TRUE( std::sin(0.9) == ultra::sin(0.9) );
}

TEST(DenseMathTest,ComplexMath){
    using std::complex;
    using namespace std::complex_literals;

    auto shape = Shape{5,10,20}; int count;
    
    Array<double> a(shape);
    count = -(int)a.size()/2;
    for( auto&& x : a ) x = count++;

    Array<complex<double>> b(shape);
    count = 0;
    for( auto&& x : b){
        x = std::cos((count%4)*pi/2) + std::sin((count%4)*pi/2)*1i;
        ++count;
    }

    Array<complex<float>> c(shape);
    count = 0;
    for( auto&& x : c){
        x = std::cos((count%4)*pi/2) + std::sin((count%4)*pi/2)*1i;
        --count;
    }

    Array<complex<double>> d(shape);
    count=0;
    for( auto&& x : d){
        x = 1i*(pi/2)*count++;
    }
    
    // Simple addition
    Array<complex<double>> addition = b + c;
    bool addition_correct = true;
    count = 0;
    for( auto&& x : addition ){ // expect pattern of 2 0 -2 0 2 etc
        if( std::fabs(std::real(x) - ((count%2) ? 0 : 2 *((count%4)==0 ? 1 : -1))) > 1e-5 ) addition_correct=false;
        count++;
    }
    EXPECT_TRUE(addition_correct);

    // real/imag
    Array<double> re = real(c);
    Array<double> im = imag(c);
    bool real_correct=true, imag_correct=true;
    count=0;
    for( auto&& x: re ){ // expect pattern of 1 0 -1 0 1 etc
        if( std::fabs(x - ((count%2) ? 0 : ((count%4)==0 ? 1 : -1))) > 1e-5 ) real_correct=false;
        count++;
    }
    count=0;
    for( auto&& x: im ){ // expect pattern of 0 -1 0 1 0 etc
        if( std::fabs(x - ((count%2) ? ((count%4)==1 ? -1 : 1) : 0)) > 1e-5 ) imag_correct=false;
        count++;
    }
    EXPECT_TRUE(real_correct);
    EXPECT_TRUE(imag_correct);

    // complex exp
    Array<complex<double>> complex_exp = exp(d);;
    bool complex_exp_correct = true;
    count = 0;
    for( auto&& x : complex_exp ){ // expect pattern of 1 i -1 -i
        if( std::norm(x - ((count%2) ? ((count%4)==1 ? 1i : -1i) : ((count%4)==0 ? 1 : -1))) > 1e-5 ) complex_exp_correct=false;
        count++;
    }
    EXPECT_TRUE(complex_exp_correct);

    // sqrt vs complex sqrt
    Array<double> std_sqrt = sqrt(a);
    Array<std::complex<double>> cx_sqrt = complex_sqrt(a);
    bool std_sqrt_correct=true, cx_sqrt_correct=true;
    count = 0;
    for( auto&& x : std_sqrt ){
        if( count < (int)a.size()/2 ){
            if( !std::isnan(x) ) std_sqrt_correct = false;
        } else {
            if( std::isnan(x) ) std_sqrt_correct = false;
        }
        count++;
    }
    for( auto&& x : cx_sqrt ) if(std::isnan(std::real(x))) cx_sqrt_correct = false;
    EXPECT_TRUE(std_sqrt_correct);
    EXPECT_TRUE(cx_sqrt_correct);

    // complex pow
    Array<double> exponent(shape); exponent.fill(0.5);
    Array<double> std_pow = pow( -1, exponent);
    Array<std::complex<double>> cx_pow = complex_pow(-1,exponent);
    bool std_pow_correct=true, cx_pow_correct=true;
    count=0; for(auto&& x: std_pow) if( !std::isnan(x)) std_pow_correct=false;
    count=0; for(auto&& x: cx_pow)  if( std::norm(x - 1i) > 1e-5) cx_pow_correct=false;
    EXPECT_TRUE(std_pow_correct);
    EXPECT_TRUE(cx_pow_correct);
}

TEST(DenseMathTest,Where){
    Array<std::size_t> a(Shape{3,3});
    Array<double> b(Shape{3,3}); b.fill(pi);
    Array<double> c(Shape{3,3}); c.fill(e);
    int count=0; for(auto&& x : a) x = count++;
    // for every third element, e. Otherwise pi+e.
    Array<double> d = where( !(a%3), c, b+c); 
    bool where_correct = true;
    count=0;
    for( auto&& x : d ) if ( std::abs(x - (e + (((count++)%3) ? pi : 0))) > 1e-2 )  where_correct=false;
    EXPECT_TRUE( where_correct);
}
