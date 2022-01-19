#include "ultramat/include/Array.hpp"
#include "ultramat/include/Dense/Math/DenseFolds.hpp"
#include <gtest/gtest.h>
#include <iostream>

using namespace ultra;

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
    Shape permutation{0,1};
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
    Shape permutation{0,1,2};
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
    Shape permutation{0,1,2};
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
    Shape permutation{0,1,2,3};
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
    Shape permutation{0,1,2,3};
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
    Shape permutation{0,1,2,3};
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

TEST(DenseFoldTest,SumTest){
    // Tests folding in general
    Array<std::size_t>::col_major col_1D(Shape{6});
    Array<std::size_t>::col_major col_2D(Shape{6,7});
    Array<std::size_t>::col_major col_3D(Shape{6,7,8});
    Array<std::size_t>::col_major col_4D(Shape{4,5,6,7});
    Array<std::size_t>::row_major row_1D(Shape{6});
    Array<std::size_t>::row_major row_2D(Shape{6,7});
    Array<std::size_t>::row_major row_3D(Shape{6,7,8});
    Array<std::size_t>::row_major row_4D(Shape{4,5,6,7});
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
}

TEST(DenseFoldTest,PreciseSumTest){
    Array<double,4> a;
    a(0) = 1; a(1) = 1e100; a(2) = 1; a(3) = -1e100;
    Array<double> fast = fast_sum(a);
    Array<double> pairwise = pairwise_sum(a);
    Array<double> precise = precise_sum(a);
    EXPECT_NE( fast(0), 2);
    EXPECT_NE( pairwise(0), 2);
    EXPECT_EQ( precise(0), 2);
}

TEST(DenseFoldTest,BooleanFold){
    Array<bool> a(Shape{4,7});
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

TEST(DenseFoldTest,Accumulate){
    // test min,max,prod
    Array<std::size_t> a(Shape{3,4});
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

TEST(DenseFoldTest,Fold){
    // Test with lambda function. Have 2D Array of 2D FixedArrays, get maximum norm in each dimension
    Array<Array<double,2>> vector_field(Shape{6,7});
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
