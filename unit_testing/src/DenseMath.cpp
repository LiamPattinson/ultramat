#include "ultramat/include/Dense.hpp"
#include "ultramat/include/DenseMath.hpp"
#include <gtest/gtest.h>
#include <iostream>

using namespace ultra;
using shape_vec = std::vector<std::size_t>;

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
    // Combine negation, addition, multiplication, subtraction, and dividing, all between different types.
    Array<double>::row_major f = -a + b * c - d / e;
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
    return sum(array,dim0);
}

template<class T1, class T2>
T1 double_sum( const T2& array, std::size_t dim0, std::size_t dim1) {
    return sum(sum(array,dim0),dim1);
}

template<class T1, class T2>
T1 triple_sum( const T2& array, std::size_t dim0, std::size_t dim1, std::size_t dim2) {
    return sum(sum(sum(array,dim0),dim1),dim2);
}

template<class T1, class T2>
T1 quad_sum( const T2& array, std::size_t dim0, std::size_t dim1, std::size_t dim2, std::size_t dim3) {
    return sum(sum(sum(sum(array,dim0),dim1),dim2),dim3);
}

template<class T1, class T2>
T1 quin_sum( const T2& array, std::size_t dim0, std::size_t dim1, std::size_t dim2, std::size_t dim3, std::size_t dim4) {
    return sum(sum(sum(sum(sum(array,dim0),dim1),dim2),dim3),dim4);
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

template<class T1, class T2>
bool test_5D_single( const T2& array, std::size_t dim ) {
    bool correct = true;
    T1 result = single_sum<T1,T2>(array,dim);
    if( result.dims() != 4 ){ correct=false; std::cerr << "Incorrect dims" << std::endl;}
    if( result.shape(0) != array.shape(dim==0 ? 1 : 0) ){ correct=false; std::cerr << "Incorrect shape(0)" << std::endl;}
    if( result.shape(1) != array.shape(dim<=1 ? 2 : 1) ){ correct=false; std::cerr << "Incorrect shape(1)" << std::endl;}
    if( result.shape(2) != array.shape(dim<=2 ? 3 : 2) ){ correct=false; std::cerr << "Incorrect shape(2)" << std::endl;}
    if( result.shape(3) != array.shape(dim<=3 ? 4 : 3) ){ correct=false; std::cerr << "Incorrect shape(3)" << std::endl;}
    std::size_t total;
    shape_vec permutation{0,1,2,3,4};
    std::swap( permutation[4], permutation[dim] );
    std::sort(permutation.begin(), permutation.begin()+4);
    auto view = array.permute(permutation);
    for( std::size_t ii=0; ii<view.shape(0); ++ii){
        for( std::size_t jj=0; jj<view.shape(1); ++jj){
            for( std::size_t kk=0; kk<view.shape(2); ++kk){
                for( std::size_t ll=0; ll<view.shape(3); ++ll){
                    total=0;
                    for( std::size_t mm=0; mm<view.shape(4); ++mm){
                        total += view(ii,jj,kk,ll,mm);
                    }
                    if( total != result(ii,jj,kk,ll) ){
                        correct=false;
                        std::cerr << "[" << ii << "," << jj << "," << kk << "," << ll << "] Expected: " << total << " Actual: " << result(ii,jj,kk,ll) << std::endl;
                    }
                }
            }
        }
    }
    return correct;
}

template<class T1, class T2>
bool test_5D_double( const T2& array, std::size_t dim0 , std::size_t dim1) {
    bool correct = true;
    T1 result = double_sum<T1,T2>(array,dim0,dim1);
    if( result.dims() != 3 ){ correct=false; std::cerr << "Incorrect dims" << std::endl;}
    std::size_t total;
    shape_vec permutation{0,1,2,3,4};
    std::swap( permutation[4], permutation[dim0] );
    std::sort(permutation.begin(), permutation.begin()+4);
    std::swap( permutation[3], permutation[dim1] );
    std::sort(permutation.begin(), permutation.begin()+3);
    auto view = array.permute(permutation);
    for( std::size_t ii=0; ii<view.shape(0); ++ii){
        for( std::size_t jj=0; jj<view.shape(1); ++jj){
            for( std::size_t kk=0; kk<view.shape(2); ++kk){
                total=0;
                for( std::size_t ll=0; ll<view.shape(3); ++ll){
                    for( std::size_t mm=0; mm<view.shape(4); ++mm){
                        total += view(ii,jj,kk,ll,mm);
                    }
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
bool test_5D_triple( const T2& array, std::size_t dim0 , std::size_t dim1, std::size_t dim2) {
    bool correct = true;
    T1 result = triple_sum<T1,T2>(array,dim0,dim1,dim2);
    if( result.dims() != 2 ){ correct=false; std::cerr << "Incorrect dims" << std::endl;}
    std::size_t total;
    shape_vec permutation{0,1,2,3,4};
    std::swap( permutation[4], permutation[dim0] );
    std::sort(permutation.begin(), permutation.begin()+4);
    std::swap( permutation[3], permutation[dim1] );
    std::sort(permutation.begin(), permutation.begin()+3);
    std::swap( permutation[2], permutation[dim2] );
    std::sort(permutation.begin(), permutation.begin()+2);
    auto view = array.permute(permutation);
    for( std::size_t ii=0; ii<view.shape(0); ++ii){
        for( std::size_t jj=0; jj<view.shape(1); ++jj){
            total=0;
            for( std::size_t kk=0; kk<view.shape(2); ++kk){
                for( std::size_t ll=0; ll<view.shape(3); ++ll){
                    for( std::size_t mm=0; mm<view.shape(4); ++mm){
                        total += view(ii,jj,kk,ll,mm);
                    }
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
bool test_5D_quad( const T2& array, std::size_t dim0 , std::size_t dim1, std::size_t dim2, std::size_t dim3) {
    bool correct = true;
    T1 result = quad_sum<T1,T2>(array,dim0,dim1,dim2,dim3);
    if( result.dims() != 1 ){ correct=false; std::cerr << "Incorrect dims" << std::endl;}
    std::size_t total;
    shape_vec permutation{0,1,2,3,4};
    std::swap( permutation[4], permutation[dim0] );
    std::sort(permutation.begin(), permutation.begin()+4);
    std::swap( permutation[3], permutation[dim1] );
    std::sort(permutation.begin(), permutation.begin()+3);
    std::swap( permutation[2], permutation[dim2] );
    std::sort(permutation.begin(), permutation.begin()+2);
    std::swap( permutation[1], permutation[dim3] );
    auto view = array.permute(permutation);
    for( std::size_t ii=0; ii<view.shape(0); ++ii){
        total=0;
        for( std::size_t jj=0; jj<view.shape(1); ++jj){
            for( std::size_t kk=0; kk<view.shape(2); ++kk){
                for( std::size_t ll=0; ll<view.shape(3); ++ll){
                    for( std::size_t mm=0; mm<view.shape(4); ++mm){
                        total += view(ii,jj,kk,ll,mm);
                    }
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
bool test_5D_quin( const T2& array, std::size_t dim0 , std::size_t dim1, std::size_t dim2, std::size_t dim3) {
    bool correct = true;
    T1 result = quin_sum<T1,T2>(array,dim0,dim1,dim2,dim3,0);
    if( result.dims() != 1 ){ correct=false; std::cerr << "Incorrect dims" << std::endl;}
    if( result.size() != 1 ){ correct=false; std::cerr << "Incorrect size" << std::endl;}
    std::size_t total=0;
    for( std::size_t ii=0; ii<array.shape(0); ++ii){
        for( std::size_t jj=0; jj<array.shape(1); ++jj){
            for( std::size_t kk=0; kk<array.shape(2); ++kk){
                for( std::size_t ll=0; ll<array.shape(3); ++ll){
                    for( std::size_t mm=0; mm<array.shape(4); ++mm){
                        total += array(ii,jj,kk,ll,mm);
                    }
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
    Array<std::size_t>::col_major col_5D(shape_vec{3,4,2,5,2});
    Array<std::size_t>::row_major row_1D(shape_vec{6});
    Array<std::size_t>::row_major row_2D(shape_vec{6,7});
    Array<std::size_t>::row_major row_3D(shape_vec{6,7,8});
    Array<std::size_t>::row_major row_4D(shape_vec{4,5,6,7});
    Array<std::size_t>::row_major row_5D(shape_vec{3,4,2,5,2});
    std::size_t count;
    count = 0; for( auto&& x : col_1D ) x = count++;
    count = 0; for( auto&& x : col_2D ) x = count++;
    count = 0; for( auto&& x : col_3D ) x = count++;
    count = 0; for( auto&& x : col_4D ) x = count++;
    count = 0; for( auto&& x : col_5D ) x = count++;
    count = 0; for( auto&& x : row_1D ) x = count++;
    count = 0; for( auto&& x : row_2D ) x = count++;
    count = 0; for( auto&& x : row_3D ) x = count++;
    count = 0; for( auto&& x : row_4D ) x = count++;
    count = 0; for( auto&& x : row_5D ) x = count++;
    
    // 1D
    EXPECT_TRUE( test_1D<Array<std::size_t>::col_major>( col_1D));
    EXPECT_TRUE( test_1D<Array<std::size_t>::row_major>( row_1D));
    EXPECT_TRUE( test_1D<Array<std::size_t>::row_major>( col_1D));
    EXPECT_TRUE( test_1D<Array<std::size_t>::col_major>( row_1D));

    // 2D
    for( std::size_t ii=0; ii<2; ++ii){
        EXPECT_TRUE( test_2D_single<Array<std::size_t>::col_major>( col_2D, ii));
        EXPECT_TRUE( test_2D_single<Array<std::size_t>::row_major>( row_2D, ii));
        EXPECT_TRUE( test_2D_single<Array<std::size_t>::row_major>( col_2D, ii));
        EXPECT_TRUE( test_2D_single<Array<std::size_t>::col_major>( row_2D, ii));
        EXPECT_TRUE( test_2D_double<Array<std::size_t>::col_major>( col_2D, ii));
        EXPECT_TRUE( test_2D_double<Array<std::size_t>::row_major>( row_2D, ii));
        EXPECT_TRUE( test_2D_double<Array<std::size_t>::row_major>( col_2D, ii));
        EXPECT_TRUE( test_2D_double<Array<std::size_t>::col_major>( row_2D, ii));
    }

    // 3D
    for( std::size_t ii=0; ii<3; ++ii){
        EXPECT_TRUE( test_3D_single<Array<std::size_t>::col_major>( col_3D, ii));
        EXPECT_TRUE( test_3D_single<Array<std::size_t>::row_major>( row_3D, ii));
        EXPECT_TRUE( test_3D_single<Array<std::size_t>::row_major>( col_3D, ii));
        EXPECT_TRUE( test_3D_single<Array<std::size_t>::col_major>( row_3D, ii));
        for( std::size_t jj=0; jj<2; ++jj){
            EXPECT_TRUE( test_3D_double<Array<std::size_t>::col_major>( col_3D, ii, jj));
            EXPECT_TRUE( test_3D_double<Array<std::size_t>::row_major>( row_3D, ii, jj));
            EXPECT_TRUE( test_3D_double<Array<std::size_t>::row_major>( col_3D, ii, jj));
            EXPECT_TRUE( test_3D_double<Array<std::size_t>::col_major>( row_3D, ii, jj));
            EXPECT_TRUE( test_3D_triple<Array<std::size_t>::col_major>( col_3D, ii, jj));
            EXPECT_TRUE( test_3D_triple<Array<std::size_t>::row_major>( row_3D, ii, jj));
            EXPECT_TRUE( test_3D_triple<Array<std::size_t>::row_major>( col_3D, ii, jj));
            EXPECT_TRUE( test_3D_triple<Array<std::size_t>::col_major>( row_3D, ii, jj));
        }
    }

    // 4D
    for( std::size_t ii=0; ii<4; ++ii){
        EXPECT_TRUE( test_4D_single<Array<std::size_t>::col_major>( col_4D, ii));
        EXPECT_TRUE( test_4D_single<Array<std::size_t>::row_major>( row_4D, ii));
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
    }

    // 5D
    for( std::size_t ii=0; ii<5; ++ii){
        EXPECT_TRUE( test_5D_single<Array<std::size_t>::col_major>( col_5D, ii));
        EXPECT_TRUE( test_5D_single<Array<std::size_t>::row_major>( row_5D, ii));
        EXPECT_TRUE( test_5D_single<Array<std::size_t>::row_major>( col_5D, ii));
        EXPECT_TRUE( test_5D_single<Array<std::size_t>::col_major>( row_5D, ii));
        for( std::size_t jj=0; jj<4; ++jj){
            EXPECT_TRUE( test_5D_double<Array<std::size_t>::col_major>( col_5D, ii, jj));
            EXPECT_TRUE( test_5D_double<Array<std::size_t>::row_major>( row_5D, ii, jj));
            EXPECT_TRUE( test_5D_double<Array<std::size_t>::row_major>( col_5D, ii, jj));
            EXPECT_TRUE( test_5D_double<Array<std::size_t>::col_major>( row_5D, ii, jj));
            for( std::size_t kk=0; kk<3; ++kk){
                EXPECT_TRUE( test_5D_triple<Array<std::size_t>::col_major>( col_5D, ii, jj, kk));
                EXPECT_TRUE( test_5D_triple<Array<std::size_t>::row_major>( row_5D, ii, jj, kk));
                EXPECT_TRUE( test_5D_triple<Array<std::size_t>::row_major>( col_5D, ii, jj, kk));
                EXPECT_TRUE( test_5D_triple<Array<std::size_t>::col_major>( row_5D, ii, jj, kk));
                for( std::size_t ll=0; ll<2; ++ll){
                    EXPECT_TRUE( test_5D_quad<Array<std::size_t>::col_major>( col_5D, ii, jj, kk, ll));
                    EXPECT_TRUE( test_5D_quad<Array<std::size_t>::row_major>( row_5D, ii, jj, kk, ll));
                    EXPECT_TRUE( test_5D_quad<Array<std::size_t>::row_major>( col_5D, ii, jj, kk, ll));
                    EXPECT_TRUE( test_5D_quad<Array<std::size_t>::col_major>( row_5D, ii, jj, kk, ll));
                    EXPECT_TRUE( test_5D_quin<Array<std::size_t>::col_major>( col_5D, ii, jj, kk, ll));
                    EXPECT_TRUE( test_5D_quin<Array<std::size_t>::row_major>( row_5D, ii, jj, kk, ll));
                    EXPECT_TRUE( test_5D_quin<Array<std::size_t>::row_major>( col_5D, ii, jj, kk, ll));
                    EXPECT_TRUE( test_5D_quin<Array<std::size_t>::col_major>( row_5D, ii, jj, kk, ll));
                }
            }
        }
    }
}

TEST(ArrayMathTest,BooleanFold){
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
}

TEST(ArrayMathTest,Accumulate){
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
}

TEST(ArrayMathTest,Fold){
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
}
