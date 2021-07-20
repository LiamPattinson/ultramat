#include "ultramat/include/Dense.hpp"
#include "ultramat/include/DenseMath.hpp"
#include <gtest/gtest.h>
#include <iostream>

using namespace ultra;
using shape_vec = std::vector<std::size_t>;

TEST(ArrayMathTest,Constants){
    float x = 1;
    for( std::size_t ii=0; ii<100; ++ii) x *= 10;
    ASSERT_TRUE(std::isinf(x));
    EXPECT_EQ( x, Inf);
    EXPECT_EQ( x, Infinity);
    EXPECT_EQ( x, infty);
    float y = -1;
    for( std::size_t ii=0; ii<100; ++ii) y *= 10;
    ASSERT_TRUE(std::isinf(y));
    EXPECT_EQ( y, ninf);
    EXPECT_EQ( y, Ninf);
    // Can't compare nan -- it the only thing for which x == x is false.
    EXPECT_TRUE(std::isnan(NaN));
}

TEST(ArrayMathTest,Arithmetic){
    auto shape = shape_vec{5,10,20};
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

TEST(ArrayMathTest,Boolean){
    auto shape = shape_vec{5,7};
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

TEST(ArrayMathTest,InPlaceArithmetic){
    auto shape = shape_vec{5,10,20};
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

TEST(ArrayMathTest,ScalarArithmetic){
    auto shape = shape_vec{5,10,20};
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

TEST(ArrayMathTest,Math){
    auto shape = shape_vec{5,10,20};
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

TEST(ArrayMathTest,ComplexMath){
    using std::complex;
    using namespace std::complex_literals;

    auto shape = shape_vec{5,10,20}; int count;
    
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

TEST(ArrayMathTest,Eval){
    shape_vec shape{5,10,20};
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
    Array<float> g = reshape( a*b + c, shape_vec{50,2,5,2});
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
    Array<double> h = view( a*b + c, Slice{Slice::all,2}, Slice{Slice::all,2},Slice{Slice::all,2});
    EXPECT_TRUE( h.dims() == 3 );
    EXPECT_TRUE( h.size() == 8 );
    EXPECT_TRUE( h.shape(0) == 2 );
    EXPECT_TRUE( h.shape(1) == 2 );
    EXPECT_TRUE( h.shape(2) == 2 );
    bool h_correct=true;
    for( auto&& x : h ) if( x != -19.5 ) h_correct = false;
    EXPECT_TRUE(h_correct);

    // Evaluating Hermitian
    Array<std::complex<double>> k(shape_vec{5,10}); k.fill(std::complex<double>(1,1));
    Array<std::complex<double>> l = hermitian(2*k);
    EXPECT_TRUE( l.dims() == 2 );
    EXPECT_TRUE( l.size() == 50 );
    EXPECT_TRUE( l.shape(0) == 10 );
    EXPECT_TRUE( l.shape(1) == 5 );
    bool l_correct=true;
    for( auto&& x : l ) if( x != std::complex<double>(2,-2) ) l_correct = false;
    EXPECT_TRUE(l_correct);
}

TEST(ArrayMathTest,CumulativeSum){
    auto shape = shape_vec{5,10,20};
    Array<float>::col_major  a(shape);
    Array<double>::row_major b(shape);
    int count;
    count=0; for( auto&& x : a) x=2.5*count++;
    count=0; for( auto&& x : b) x=0.3*count++;

    // cumsum is evaluating: can use auto type deduction
    auto cumsum_0 = ultra::cumsum(a+b,0);
    auto cumsum_1 = ultra::cumsum(a+b,1);
    auto cumsum_2 = ultra::cumsum(a+b,2);

    EXPECT_TRUE(cumsum_0.shape(0) == 5);
    EXPECT_TRUE(cumsum_1.shape(0) == 5);
    EXPECT_TRUE(cumsum_2.shape(0) == 5);
    EXPECT_TRUE(cumsum_0.shape(1) == 10);
    EXPECT_TRUE(cumsum_1.shape(1) == 10);
    EXPECT_TRUE(cumsum_2.shape(1) == 10);
    EXPECT_TRUE(cumsum_0.shape(2) == 20);
    EXPECT_TRUE(cumsum_1.shape(2) == 20);
    EXPECT_TRUE(cumsum_2.shape(2) == 20);

    bool cumsum_0_correct = true;
    for( std::size_t ii=0; ii<10; ++ii){
        for( std::size_t jj=0; jj<20; ++jj){
            double result = 0;
            for( std::size_t kk=0; kk<5; ++kk){
                result += a(kk,ii,jj) + b(kk,ii,jj);
                if( std::fabs(result - cumsum_0(kk,ii,jj)) > 1e-5 ) cumsum_0_correct = false;
            }
        }
    }
    EXPECT_TRUE(cumsum_0_correct);

    bool cumsum_1_correct = true;
    for( std::size_t ii=0; ii<5; ++ii){
        for( std::size_t jj=0; jj<20; ++jj){
            double result = 0;
            for( std::size_t kk=0; kk<10; ++kk){
                result += a(ii,kk,jj) + b(ii,kk,jj);
                if( std::fabs(result - cumsum_1(ii,kk,jj)) > 1e-5 ) cumsum_1_correct = false;
            }
        }
    }
    EXPECT_TRUE(cumsum_1_correct);
    
    bool cumsum_2_correct = true;
    for( std::size_t ii=0; ii<5; ++ii){
        for( std::size_t jj=0; jj<10; ++jj){
            double result = 0;
            for( std::size_t kk=0; kk<20; ++kk){
                result += a(ii,jj,kk) + b(ii,jj,kk);
                if( std::fabs(result - cumsum_2(ii,jj,kk)) > 1e-5 ) cumsum_2_correct = false;
            }
        }
    }
    EXPECT_TRUE(cumsum_2_correct);

}

template<class T1, class T2>
T1 single_sum( const T2& array, std::size_t dim0=0) {
    return fast_sum(array,dim0);
}

template<class T1, class T2>
T1 double_sum( const T2& array, std::size_t dim0, std::size_t dim1) {
    return fast_sum(fast_sum(array,dim0),dim1);
}

template<class T1, class T2>
T1 triple_sum( const T2& array, std::size_t dim0, std::size_t dim1, std::size_t dim2) {
    return fast_sum(fast_sum(fast_sum(array,dim0),dim1),dim2);
}

template<class T1, class T2>
T1 quad_sum( const T2& array, std::size_t dim0, std::size_t dim1, std::size_t dim2, std::size_t dim3) {
    return fast_sum(fast_sum(fast_sum(fast_sum(array,dim0),dim1),dim2),dim3);
}

template<class T1, class T2>
bool test_1D( const T2& array ) {
    bool correct = true;
    T1 result = single_sum<T1,T2>(array);
    if( result.dims() != 1 ){ correct=false; std::cerr << "Incorrect dims" << std::endl;}
    if( result.size() != 1 ){ correct=false; std::cerr << "Incorrect size" << std::endl;}
    if( result.shape(0) != 1 ){ correct=false; std::cerr << "Incorrect shape(0)" << std::endl;}
    std::size_t total=0;
    for( std::size_t ii=0; ii<array.shape(0); ++ii){
        total += array(ii);
    }
    if( total != result(0) ){ correct=false; std::cerr << "Expected: " << total << " Actual: " << result(0) << std::endl;}
    return correct;
}

template<class T1, class T2>
bool test_2D_single( const T2& array, std::size_t dim ) {
    bool correct = true;
    T1 result = single_sum<T1,T2>(array,dim);
    if( result.dims() != 1 ){ correct=false; std::cerr << "Incorrect dims" << std::endl;}
    if( result.shape(0) != array.shape(!dim) ){ correct=false; std::cerr << "Incorrect shape(0)" << std::endl;}
    std::size_t total;
    shape_vec permutation{0,1};
    std::swap( permutation[1], permutation[dim] );
    auto view = array.permute(permutation);
    for( std::size_t ii=0; ii<view.shape(0); ++ii){
        total=0;
        for( std::size_t jj=0; jj<view.shape(1); ++jj){
            total += view(ii,jj);
        }
        if( total != result(ii) ){
            correct=false;
            std::cerr << "[" << ii << "] Expected: " << total << " Actual: " << result(ii) << std::endl;
        }
    }
    return correct;
}

template<class T1, class T2>
bool test_2D_double( const T2& array, std::size_t dim ) {
    bool correct = true;
    T1 result = double_sum<T1,T2>(array,dim,0);
    if( result.dims() != 1 ){ correct=false; std::cerr << "Incorrect dims" << std::endl;}
    if( result.shape(0) != 1 ){ correct=false; std::cerr << "Incorrect shape(0)" << std::endl;}
    std::size_t total = 0;
    for( std::size_t ii=0; ii<array.shape(0); ++ii){
        for( std::size_t jj=0; jj<array.shape(1); ++jj){
            total += array(ii,jj);
        }
    }
    if( total != result(0) ){
        correct=false;
        std::cerr << " Expected: " << total << " Actual: " << result(0) << std::endl;
    }
    return correct;
}

template<class T1, class T2>
bool test_3D_single( const T2& array, std::size_t dim ) {
    bool correct = true;
    T1 result = single_sum<T1,T2>(array,dim);
    if( result.dims() != 2 ){ correct=false; std::cerr << "Incorrect dims" << std::endl;}
    if( result.shape(0) != array.shape(dim==0 ? 1 : 0) ){ correct=false; std::cerr << "Incorrect shape(0)" << std::endl;}
    if( result.shape(1) != array.shape(dim<=1 ? 2 : 1) ){ correct=false; std::cerr << "Incorrect shape(1)" << std::endl;}
    std::size_t total;
    shape_vec permutation{0,1,2};
    std::swap( permutation[2], permutation[dim] );
    std::sort(permutation.begin(), permutation.begin()+2);
    auto view = array.permute(permutation);
    for( std::size_t ii=0; ii<view.shape(0); ++ii){
        for( std::size_t jj=0; jj<view.shape(1); ++jj){
            total=0;
            for( std::size_t kk=0; kk<view.shape(2); ++kk){
                total += view(ii,jj,kk);
            }
            if( total != result(ii,jj) ){
                correct=false;
                std::cerr << "[" << ii << "," << jj << "] Expected: " << total << " Actual: " << result(ii,jj) << std::endl;
            }
        }
    }
    return correct;
}

template<class T1, class T2>
bool test_3D_double( const T2& array, std::size_t dim0 , std::size_t dim1) {
    bool correct = true;
    T1 result = double_sum<T1,T2>(array,dim0,dim1);
    if( result.dims() != 1 ){ correct=false; std::cerr << "Incorrect dims" << std::endl;}
    std::size_t total;
    shape_vec permutation{0,1,2};
    std::swap( permutation[2], permutation[dim0] );
    std::sort(permutation.begin(), permutation.begin()+2);
    std::swap( permutation[1], permutation[dim1] );
    std::sort(permutation.begin(), permutation.begin()+1);
    auto view = array.permute(permutation);
    for( std::size_t ii=0; ii<view.shape(0); ++ii){
        total=0;
        for( std::size_t jj=0; jj<view.shape(1); ++jj){
            for( std::size_t kk=0; kk<view.shape(2); ++kk){
                total += view(ii,jj,kk);
            }
        }
        if( total != result(ii) ){
            correct=false;
            std::cerr << "[" << ii << "] Expected: " << total << " Actual: " << result(ii) << std::endl;
        }
    }
    return correct;
}

template<class T1, class T2>
bool test_3D_triple( const T2& array, std::size_t dim0 , std::size_t dim1) {
    bool correct = true;
    T1 result = triple_sum<T1,T2>(array,dim0,dim1,0);
    if( result.dims() != 1 ){ correct=false; std::cerr << "Incorrect dims" << std::endl;}
    if( result.size() != 1 ){ correct=false; std::cerr << "Incorrect size" << std::endl;}
    if( result.shape(0) != 1 ){ correct=false; std::cerr << "Incorrect shape(0)" << std::endl;}
    std::size_t total=0;
    for( std::size_t ii=0; ii<array.shape(0); ++ii){
        for( std::size_t jj=0; jj<array.shape(1); ++jj){
            for( std::size_t kk=0; kk<array.shape(2); ++kk){
                total += array(ii,jj,kk);
            }
        }
    }
    if( total != result(0) ){
        correct=false;
        std::cerr << "Expected: " << total << " Actual: " << result(0) << std::endl;
    }
    return correct;
}

template<class T1, class T2>
bool test_4D_single( const T2& array, std::size_t dim ) {
    bool correct = true;
    T1 result = single_sum<T1,T2>(array,dim);
    if( result.dims() != 3 ){ correct=false; std::cerr << "Incorrect dims" << std::endl;}
    if( result.shape(0) != array.shape(dim==0 ? 1 : 0) ){ correct=false; std::cerr << "Incorrect shape(0)" << std::endl;}
    if( result.shape(1) != array.shape(dim<=1 ? 2 : 1) ){ correct=false; std::cerr << "Incorrect shape(1)" << std::endl;}
    if( result.shape(2) != array.shape(dim<=2 ? 3 : 2) ){ correct=false; std::cerr << "Incorrect shape(2)" << std::endl;}
    std::size_t total;
    shape_vec permutation{0,1,2,3};
    std::swap( permutation[3], permutation[dim] );
    std::sort(permutation.begin(), permutation.begin()+3);
    auto view = array.permute(permutation);
    for( std::size_t ii=0; ii<view.shape(0); ++ii){
        for( std::size_t jj=0; jj<view.shape(1); ++jj){
            for( std::size_t kk=0; kk<view.shape(2); ++kk){
                total=0;
                for( std::size_t ll=0; ll<view.shape(3); ++ll){
                    total += view(ii,jj,kk,ll);
                }
                if( total != result(ii,jj,kk) ){
                    correct=false;
                    std::cerr << "[" << ii << "," << jj << "," << kk << "] Expected: " << total << " Actual: " << result(ii,jj,kk) << std::endl;
                }
            }
        }
    }
    return correct;
}

template<class T1, class T2>
bool test_4D_double( const T2& array, std::size_t dim0, std::size_t dim1) {
    bool correct = true;
    T1 result = double_sum<T1,T2>(array,dim0,dim1);
    if( result.dims() != 2 ){ correct=false; std::cerr << "Incorrect dims" << std::endl;}
    std::size_t total;
    shape_vec permutation{0,1,2,3};
    std::swap( permutation[3], permutation[dim0] );
    std::sort(permutation.begin(), permutation.begin()+3);
    std::swap( permutation[2], permutation[dim1] );
    std::sort(permutation.begin(), permutation.begin()+2);
    auto view = array.permute(permutation);
    for( std::size_t ii=0; ii<view.shape(0); ++ii){
        for( std::size_t jj=0; jj<view.shape(1); ++jj){
            total=0;
            for( std::size_t kk=0; kk<view.shape(2); ++kk){
                for( std::size_t ll=0; ll<view.shape(3); ++ll){
                    total += view(ii,jj,kk,ll);
                }
            }
            if( total != result(ii,jj) ){
                correct=false;
                std::cerr << "[" << ii << "," << jj << "] Expected: " << total << " Actual: " << result(ii,jj) << std::endl;
            }
        }
    }
    return correct;
}

template<class T1, class T2>
bool test_4D_triple( const T2& array, std::size_t dim0, std::size_t dim1, std::size_t dim2) {
    bool correct = true;
    T1 result = triple_sum<T1,T2>(array,dim0,dim1,dim2);
    if( result.dims() != 1 ){ correct=false; std::cerr << "Incorrect dims" << std::endl;}
    std::size_t total;
    shape_vec permutation{0,1,2,3};
    std::swap( permutation[3], permutation[dim0] );
    std::sort(permutation.begin(), permutation.begin()+3);
    std::swap( permutation[2], permutation[dim1] );
    std::sort(permutation.begin(), permutation.begin()+2);
    std::swap( permutation[1], permutation[dim2] );
    auto view = array.permute(permutation);
    for( std::size_t ii=0; ii<view.shape(0); ++ii){
        total=0;
        for( std::size_t jj=0; jj<view.shape(1); ++jj){
            for( std::size_t kk=0; kk<view.shape(2); ++kk){
                for( std::size_t ll=0; ll<view.shape(3); ++ll){
                    total += view(ii,jj,kk,ll);
                }
            }
        }
        if( total != result(ii) ){
            correct=false;
            std::cerr << "[" << ii << "] Expected: " << total << " Actual: " << result(ii) << std::endl;
        }
    }
    return correct;
}

template<class T1, class T2>
bool test_4D_quad( const T2& array, std::size_t dim0 , std::size_t dim1, std::size_t dim2) {
    bool correct = true;
    T1 result = quad_sum<T1,T2>(array,dim0,dim1,dim2,0);
    if( result.dims() != 1 ){ correct=false; std::cerr << "Incorrect dims" << std::endl;}
    std::size_t total = 0;
    for( std::size_t ii=0; ii<array.shape(0); ++ii){
        for( std::size_t jj=0; jj<array.shape(1); ++jj){
            for( std::size_t kk=0; kk<array.shape(2); ++kk){
                for( std::size_t ll=0; ll<array.shape(3); ++ll){
                    total += array(ii,jj,kk,ll);
                }
            }
        }
    }
    if( total != result(0) ){
        correct=false;
        std::cerr << "Expected: " << total << " Actual: " << result(0) << std::endl;
    }
    return correct;
}

TEST(ArrayMathTest,SumTest){
    // Tests folding in general
    Array<std::size_t>::col_major col_1D(shape_vec{6});
    Array<std::size_t>::col_major col_2D(shape_vec{6,7});
    Array<std::size_t>::col_major col_3D(shape_vec{6,7,8});
    Array<std::size_t>::col_major col_4D(shape_vec{4,5,6,7});
    Array<std::size_t>::row_major row_1D(shape_vec{6});
    Array<std::size_t>::row_major row_2D(shape_vec{6,7});
    Array<std::size_t>::row_major row_3D(shape_vec{6,7,8});
    Array<std::size_t>::row_major row_4D(shape_vec{4,5,6,7});
    std::size_t count;
    count = 0; for( auto&& x : col_1D ) x = count++;
    count = 0; for( auto&& x : col_2D ) x = count++;
    count = 0; for( auto&& x : col_3D ) x = count++;
    count = 0; for( auto&& x : col_4D ) x = count++;
    count = 0; for( auto&& x : row_1D ) x = count++;
    count = 0; for( auto&& x : row_2D ) x = count++;
    count = 0; for( auto&& x : row_3D ) x = count++;
    count = 0; for( auto&& x : row_4D ) x = count++;
    
    // 1D
    //EXPECT_TRUE( test_1D<Array<std::size_t>::col_major>( col_1D));
    //EXPECT_TRUE( test_1D<Array<std::size_t>::row_major>( row_1D));
    //EXPECT_TRUE( test_1D<Array<std::size_t>::row_major>( col_1D));
    //EXPECT_TRUE( test_1D<Array<std::size_t>::col_major>( row_1D));

    // 2D
    for( std::size_t ii=0; ii<2; ++ii){
        //EXPECT_TRUE( test_2D_single<Array<std::size_t>::col_major>( col_2D, ii));
        //EXPECT_TRUE( test_2D_single<Array<std::size_t>::row_major>( row_2D, ii));
        EXPECT_TRUE( test_2D_single<Array<std::size_t>::row_major>( col_2D, ii));
        EXPECT_TRUE( test_2D_single<Array<std::size_t>::col_major>( row_2D, ii));
        /*
        EXPECT_TRUE( test_2D_double<Array<std::size_t>::col_major>( col_2D, ii));
        EXPECT_TRUE( test_2D_double<Array<std::size_t>::row_major>( row_2D, ii));
        EXPECT_TRUE( test_2D_double<Array<std::size_t>::row_major>( col_2D, ii));
        EXPECT_TRUE( test_2D_double<Array<std::size_t>::col_major>( row_2D, ii));
        */
    }

    // 3D
    for( std::size_t ii=0; ii<3; ++ii){
        //EXPECT_TRUE( test_3D_single<Array<std::size_t>::col_major>( col_3D, ii));
        //EXPECT_TRUE( test_3D_single<Array<std::size_t>::row_major>( row_3D, ii));
        /*
        EXPECT_TRUE( test_3D_single<Array<std::size_t>::row_major>( col_3D, ii));
        EXPECT_TRUE( test_3D_single<Array<std::size_t>::col_major>( row_3D, ii));
        */
        for( std::size_t jj=0; jj<2; ++jj){
            /*
            EXPECT_TRUE( test_3D_double<Array<std::size_t>::col_major>( col_3D, ii, jj));
            EXPECT_TRUE( test_3D_double<Array<std::size_t>::row_major>( row_3D, ii, jj));
            EXPECT_TRUE( test_3D_double<Array<std::size_t>::row_major>( col_3D, ii, jj));
            EXPECT_TRUE( test_3D_double<Array<std::size_t>::col_major>( row_3D, ii, jj));
            EXPECT_TRUE( test_3D_triple<Array<std::size_t>::col_major>( col_3D, ii, jj));
            EXPECT_TRUE( test_3D_triple<Array<std::size_t>::row_major>( row_3D, ii, jj));
            EXPECT_TRUE( test_3D_triple<Array<std::size_t>::row_major>( col_3D, ii, jj));
            EXPECT_TRUE( test_3D_triple<Array<std::size_t>::col_major>( row_3D, ii, jj));
            */
        }
    }

    // 4D
    for( std::size_t ii=0; ii<4; ++ii){
        //EXPECT_TRUE( test_4D_single<Array<std::size_t>::col_major>( col_4D, ii));
        //EXPECT_TRUE( test_4D_single<Array<std::size_t>::row_major>( row_4D, ii));
        /*
        EXPECT_TRUE( test_4D_single<Array<std::size_t>::row_major>( col_4D, ii));
        EXPECT_TRUE( test_4D_single<Array<std::size_t>::col_major>( row_4D, ii));
        for( std::size_t jj=0; jj<3; ++jj){
            EXPECT_TRUE( test_4D_double<Array<std::size_t>::col_major>( col_4D, ii, jj));
            EXPECT_TRUE( test_4D_double<Array<std::size_t>::row_major>( row_4D, ii, jj));
            EXPECT_TRUE( test_4D_double<Array<std::size_t>::row_major>( col_4D, ii, jj));
            EXPECT_TRUE( test_4D_double<Array<std::size_t>::col_major>( row_4D, ii, jj));
            for( std::size_t kk=0; kk<2; ++kk){
                EXPECT_TRUE( test_4D_triple<Array<std::size_t>::col_major>( col_4D, ii, jj, kk));
                EXPECT_TRUE( test_4D_triple<Array<std::size_t>::row_major>( row_4D, ii, jj, kk));
                EXPECT_TRUE( test_4D_triple<Array<std::size_t>::row_major>( col_4D, ii, jj, kk));
                EXPECT_TRUE( test_4D_triple<Array<std::size_t>::col_major>( row_4D, ii, jj, kk));
                EXPECT_TRUE( test_4D_quad<Array<std::size_t>::col_major>( col_4D, ii, jj, kk));
                EXPECT_TRUE( test_4D_quad<Array<std::size_t>::row_major>( row_4D, ii, jj, kk));
                EXPECT_TRUE( test_4D_quad<Array<std::size_t>::row_major>( col_4D, ii, jj, kk));
                EXPECT_TRUE( test_4D_quad<Array<std::size_t>::col_major>( row_4D, ii, jj, kk));
            }
        }
        */
    }
}

TEST(ArrayMathTest,PreciseSumTest){
    /* FIXME
    Array<double,4> a;
    a(0) = 1; a(1) = 1e100; a(2) = 1; a(3) = -1e100;
    Array<double> fast = fast_sum(a);
    Array<double> pairwise = pairwise_sum(a);
    Array<double> precise = precise_sum(a);
    EXPECT_NE( fast(0), 2);
    EXPECT_NE( pairwise(0), 2);
    EXPECT_EQ( precise(0), 2);
*/
}

TEST(ArrayMathTest,BooleanFold){
    /* FIXME
    Array<bool> a(shape_vec{4,7});
    for( std::size_t ii=0; ii<a.shape(0); ++ii){
        for( std::size_t jj=0; jj<a.shape(1); ++jj){
            switch(ii){
                case(0): a(ii,jj) = false; break;
                case(1): a(ii,jj) = true; break;
                case(2): a(ii,jj) = (jj%2); break;
                case(3): a(ii,jj) = (jj==4); break;
            }
        }
    }
    Array<bool> all  = all_of(a,1);
    Array<bool> any  = any_of(a,1);
    Array<bool> none = none_of(a,1);
    EXPECT_TRUE(all.dims() == 1);
    EXPECT_TRUE(any.dims() == 1);
    EXPECT_TRUE(none.dims() == 1);
    EXPECT_TRUE(all.shape(0) == 4);
    EXPECT_TRUE(any.shape(0) == 4);
    EXPECT_TRUE(none.shape(0) == 4);
    EXPECT_FALSE(all(0));
    EXPECT_FALSE(any(0));
    EXPECT_TRUE(none(0));
    EXPECT_TRUE(all(1));
    EXPECT_TRUE(any(1));
    EXPECT_FALSE(none(1));
    EXPECT_FALSE(all(2));
    EXPECT_TRUE(any(2));
    EXPECT_FALSE(none(2));
    EXPECT_FALSE(all(3));
    EXPECT_TRUE(any(3));
    EXPECT_FALSE(none(3));
*/
}

TEST(ArrayMathTest,Accumulate){
    /* FIXME
    // test min,max,prod
    Array<std::size_t> a(shape_vec{3,4});
    a(0,0) = 3; a(0,1) = 3; a(0,2) = 7; a(0,3) = 0;
    a(1,0) = 3; a(1,1) = 1; a(1,2) = 4; a(1,3) = 4;
    a(2,0) = 0; a(2,1) = 2; a(2,2) = 1; a(2,3) = 0;
    Array<std::size_t> min_0 = min(a,0);
    Array<std::size_t> min_1 = min(a,1);
    Array<std::size_t> max_0 = max(a,0);
    Array<std::size_t> max_1 = max(a,1);
    Array<std::size_t> prod_0 = prod(a,0);
    Array<std::size_t> prod_1 = prod(a,1);
    Array<std::size_t> minmax_0 = min(max(a,0),0);
    Array<std::size_t> minmax_1 = min(max(a,1),0);
    EXPECT_EQ(min_0(0),0);
    EXPECT_EQ(min_0(1),1);
    EXPECT_EQ(min_0(2),1);
    EXPECT_EQ(min_0(3),0);
    EXPECT_EQ(max_0(0),3);
    EXPECT_EQ(max_0(1),3);
    EXPECT_EQ(max_0(2),7);
    EXPECT_EQ(max_0(3),4);
    EXPECT_EQ(prod_0(0),0);
    EXPECT_EQ(prod_0(1),6);
    EXPECT_EQ(prod_0(2),28);
    EXPECT_EQ(prod_0(3),0);
    EXPECT_EQ(min_1(0),0);
    EXPECT_EQ(min_1(1),1);
    EXPECT_EQ(min_1(2),0);
    EXPECT_EQ(max_1(0),7);
    EXPECT_EQ(max_1(1),4);
    EXPECT_EQ(max_1(2),2);
    EXPECT_EQ(prod_1(0),0);
    EXPECT_EQ(prod_1(1),48);
    EXPECT_EQ(prod_1(2),0);
    EXPECT_EQ(minmax_0(0),3);
    EXPECT_EQ(minmax_1(0),2);
*/
}

TEST(ArrayMathTest,Fold){
    /* FIXME
    // Test with lambda function. Have 2D Array of 2D FixedArrays, get maximum norm in each dimension
    Array<Array<double,2>> vector_field(shape_vec{6,7});
    for(std::size_t ii=0; ii<6; ++ii){
        for(std::size_t jj=0; jj<7; ++jj){
            Array<double,2> vec;
            vec(0) = ii+1;
            vec(1) = (int)ii-(int)jj;
            vector_field(ii,jj) = vec;
        }
    }
    Array<double> max_norm_0 = fold([](double a,const Array<double,2>& v){return std::max(a,std::sqrt(v(0)*v(0)+v(1)*v(1)));},vector_field,0,0);
    Array<double> max_norm_1 = fold([](double a,const Array<double,2>& v){return std::max(a,std::sqrt(v(0)*v(0)+v(1)*v(1)));},vector_field,0,1);
    EXPECT_TRUE( std::fabs(max_norm_0(0) - std::sqrt(6*6+5*5)) < 1e-5);
    EXPECT_TRUE( std::fabs(max_norm_0(1) - std::sqrt(6*6+4*4)) < 1e-5);
    EXPECT_TRUE( std::fabs(max_norm_0(2) - std::sqrt(6*6+3*3)) < 1e-5);
    EXPECT_TRUE( std::fabs(max_norm_0(3) - std::sqrt(6*6+2*2)) < 1e-5);
    EXPECT_TRUE( std::fabs(max_norm_0(4) - std::sqrt(6*6+1*1)) < 1e-5);
    EXPECT_TRUE( std::fabs(max_norm_0(5) - std::sqrt(6*6+0*0)) < 1e-5);
    EXPECT_TRUE( std::fabs(max_norm_0(6) - std::sqrt(6*6+1*1)) < 1e-5);
    EXPECT_TRUE( std::fabs(max_norm_1(0) - std::sqrt(1*1+6*6)) < 1e-5);
    EXPECT_TRUE( std::fabs(max_norm_1(1) - std::sqrt(2*2+5*5)) < 1e-5);
    EXPECT_TRUE( std::fabs(max_norm_1(2) - std::sqrt(3*3+4*4)) < 1e-5);
    EXPECT_TRUE( std::fabs(max_norm_1(3) - std::sqrt(4*4+3*3)) < 1e-5);
    EXPECT_TRUE( std::fabs(max_norm_1(4) - std::sqrt(5*5+4*4)) < 1e-5);
    EXPECT_TRUE( std::fabs(max_norm_1(5) - std::sqrt(6*6+5*5)) < 1e-5);
*/
}

TEST(ArrayMathTest,OnesAndZeros){
    auto shape = shape_vec{5,10,20};

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

TEST(ArrayMathTest,Linspace){
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

TEST(ArrayMathTest,Arange){
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

TEST(ArrayMathTest,Logspace){
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

TEST(ArrayMathTest,Random){
    shape_vec shape{5,10,20}; 
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
    
    // Test that the numbers generated don't match (there is a very slim chance that this will fail!)
    bool random_double_correct = true;
    for( auto c_it = c.begin(), d_it = d.begin(), c_end = c.end(); c_it != c_end; ++c_it, ++d_it){
        if( *c_it == *d_it ) random_double_correct = false;
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
    EXPECT_TRUE( std::fabs(percent_within_1_stddev-68.27) < 0.5 );
    EXPECT_TRUE( std::fabs(percent_within_2_stddev-95.45) < 0.5 );
}

TEST(ArrayMathTest,MeanAndStddev){
    /* FIXME
    Array<double> m = mean(random_normal(10,2,10000));
    Array<double> v = var(random_normal(10,2,10000));
    Array<double> s = stddev(random_normal(10,2,10000));
    EXPECT_LT( std::abs(m(0) - 10), 1e-1 );
    EXPECT_LT( std::abs(v(0) - 4), 1e-1 );
    EXPECT_LT( std::abs(s(0) - 2), 1e-1 );
    */
}
/*
TEST(ArrayMathTest,Where){
    Array<std::size_t> a(shape_vec{3,3});
    Array<double> b(shape_vec{3,3}); b.fill(pi);
    Array<double> c(shape_vec{3,3}); c.fill(e);
    int count=0; for(auto&& x : a) x = count++;
    // for every third element, e. Otherwise pi+e.
    Array<double> d = where( !(a%3), c, b+c); 
    bool where_correct = true;
    count=0;
    for( auto&& x : d ) if ( std::abs(x - (e + (((count++)%3) ? pi : 0))) > 1e-2 )  where_correct=false;
    EXPECT_TRUE( where_correct);
}
*/
