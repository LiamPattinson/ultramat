#include "ultramat/include/Dense.hpp"
#include "ultramat/include/DenseMath.hpp"
#include <gtest/gtest.h>
#include <iostream>

using namespace ultra;
using shape_vec = std::vector<std::size_t>;

TEST(ArrayArithmeticTest,Arithmetic){
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

TEST(ArrayArithmeticTest,InPlaceArithmetic){
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

TEST(ArrayCumulativeTest,CumulativeSum){
    auto shape = shape_vec{5,10,20};
    Array<float>  a(shape);
    Array<double> b(shape);
    for( auto&& x : a) x=2.5;
    for( auto&& x : b) x=0.5;

    Array<std::size_t> c = ultra::cumsum(a+b);
    bool cumsum_correct = true;
    for( std::size_t ii=0; ii<5; ++ii){
        for( std::size_t jj=0; jj<10; ++jj){
            for( std::size_t kk=0; kk<20; ++kk){
                if( c(ii,jj,kk) != 3*(kk+1) ) cumsum_correct = false;
            }
        }
    }
    EXPECT_TRUE(cumsum_correct);
    EXPECT_TRUE(c.shape(0) == 5);
    EXPECT_TRUE(c.shape(1) == 10);
    EXPECT_TRUE(c.shape(2) == 20);

}

TEST(ArrayCumulativeTest,CumulativeProduct){
    auto shape = shape_vec{3,3};
    Array<double>::col_major a(shape);
    Array<double>::col_major b(shape);
    for( auto&& x : a) x=1;
    for( auto&& x : b) x=2;

    Array<double>::col_major c = a + ultra::cumprod(b);
    bool cumprod_correct = true;
    for( std::size_t jj=0; jj<3; ++jj){
        for( std::size_t ii=0; ii<3; ++ii){
            if( std::fabs( c(ii,jj) - (1+std::pow(2,ii+1))) > 1e-5 ) cumprod_correct = false;
        }
    }
    EXPECT_TRUE(cumprod_correct);
    EXPECT_TRUE(c.shape(0) == 3);
    EXPECT_TRUE(c.shape(1) == 3);
}

class ArraySumTest : public ::testing::Test {

    protected:

    Array<std::size_t>::col_major col_1D, col_2D, col_3D, col_4D, col_5D;
    Array<std::size_t>::row_major row_1D, row_2D, row_3D, row_4D, row_5D;

    ArraySumTest() :
        col_1D(shape_vec{6}),
        col_2D(shape_vec{6,7}),
        col_3D(shape_vec{6,7,8}),
        col_4D(shape_vec{4,5,6,7}),
        col_5D(shape_vec{3,4,2,5,2}),
        row_1D(shape_vec{6}),
        row_2D(shape_vec{6,7}),
        row_3D(shape_vec{6,7,8}),
        row_4D(shape_vec{4,5,6,7}),
        row_5D(shape_vec{3,4,2,5,2})
    {
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
};

TEST_F(ArraySumTest,SumTest){
    // Tests folding in general
    
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
