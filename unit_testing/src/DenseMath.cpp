#include "ultramat/include/Dense.hpp"
#include "ultramat/include/DenseMath.hpp"
#include <gtest/gtest.h>

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

TEST(ArrayFoldTest,Sum){
    int count;
    // TODO special case for 1D->1D
    // TODO check given dims is compatible
    // TODO figure out how to handle 4D or greater
    
    // 2D
    Array<double>::col_major col_2D(shape_vec{6,7});
    Array<double>::row_major row_2D(shape_vec{6,7});
    count=0; for( auto&& x : col_2D) x=count++;
    count=0; for( auto&& x : row_2D) x=count++;

    // sum over each direction, col major
    Array<double>::col_major sum_2D_col_0 = sum(col_2D,0);
    Array<double>::col_major sum_2D_col_1 = sum(col_2D,1);
    EXPECT_TRUE( sum_2D_col_0.dims() == 1 );
    EXPECT_TRUE( sum_2D_col_1.dims() == 1 );
    EXPECT_TRUE( sum_2D_col_0.shape(0) == 7 );
    EXPECT_TRUE( sum_2D_col_1.shape(0) == 6 );
    bool sum_2D_col_0_correct=true;
    for( std::size_t ii=0; ii<7; ++ii){
        double sum=0;
        for( std::size_t jj=0; jj<6; ++jj){
            sum += col_2D(jj,ii);
        }
        if(sum!=sum_2D_col_0(ii)) sum_2D_col_0_correct=false;
    }
    EXPECT_TRUE(sum_2D_col_0_correct);
    bool sum_2D_col_1_correct=true;
    for( std::size_t ii=0; ii<6; ++ii){
        double sum=0;
        for( std::size_t jj=0; jj<7; ++jj){
            sum += col_2D(ii,jj);
        }
        if(sum!=sum_2D_col_1(ii)) sum_2D_col_1_correct=false;
    }
    EXPECT_TRUE(sum_2D_col_1_correct);

    // sum over each direction, row major
    Array<double>::row_major sum_2D_row_0 = sum(row_2D,0);
    Array<double>::row_major sum_2D_row_1 = sum(row_2D,1);
    EXPECT_TRUE( sum_2D_row_0.dims() == 1 );
    EXPECT_TRUE( sum_2D_row_1.dims() == 1 );
    EXPECT_TRUE( sum_2D_row_0.shape(0) == 7 );
    EXPECT_TRUE( sum_2D_row_1.shape(0) == 6 );
    bool sum_2D_row_0_correct=true;
    for( std::size_t ii=0; ii<7; ++ii){
        double sum=0;
        for( std::size_t jj=0; jj<6; ++jj){
            sum += row_2D(jj,ii);
        }
        if(sum!=sum_2D_row_0(ii)) sum_2D_row_0_correct=false;
    }
    EXPECT_TRUE(sum_2D_row_0_correct);
    bool sum_2D_row_1_correct=true;
    for( std::size_t ii=0; ii<6; ++ii){
        double sum=0;
        for( std::size_t jj=0; jj<7; ++jj){
            sum += row_2D(ii,jj);
        }
        if(sum!=sum_2D_row_1(ii)) sum_2D_row_1_correct=false;
    }
    EXPECT_TRUE(sum_2D_row_1_correct);

    // sum over each direction, mixed col to row
    Array<double>::row_major sum_2D_col_to_row_0 = sum(col_2D,0);
    Array<double>::row_major sum_2D_col_to_row_1 = sum(col_2D,1);
    EXPECT_TRUE( sum_2D_col_to_row_0.dims() == 1 );
    EXPECT_TRUE( sum_2D_col_to_row_1.dims() == 1 );
    EXPECT_TRUE( sum_2D_col_to_row_0.shape(0) == 7 );
    EXPECT_TRUE( sum_2D_col_to_row_1.shape(0) == 6 );
    bool sum_2D_col_to_row_0_correct=true;
    for( std::size_t ii=0; ii<7; ++ii){
        double sum=0;
        for( std::size_t jj=0; jj<6; ++jj){
            sum += col_2D(jj,ii);
        }
        if(sum!=sum_2D_col_to_row_0(ii)) sum_2D_col_to_row_0_correct=false;
    }
    EXPECT_TRUE(sum_2D_col_to_row_0_correct);
    bool sum_2D_col_to_row_1_correct=true;
    for( std::size_t ii=0; ii<6; ++ii){
        double sum=0;
        for( std::size_t jj=0; jj<7; ++jj){
            sum += col_2D(ii,jj);
        }
        if(sum!=sum_2D_col_to_row_1(ii)) sum_2D_col_to_row_1_correct=false;
    }
    EXPECT_TRUE(sum_2D_col_to_row_1_correct);

    // sum over each direction, mixed row to col
    Array<double>::col_major sum_2D_row_to_col_0 = sum(row_2D,0);
    Array<double>::col_major sum_2D_row_to_col_1 = sum(row_2D,1);
    EXPECT_TRUE( sum_2D_row_to_col_0.dims() == 1 );
    EXPECT_TRUE( sum_2D_row_to_col_1.dims() == 1 );
    EXPECT_TRUE( sum_2D_row_to_col_0.shape(0) == 7 );
    EXPECT_TRUE( sum_2D_row_to_col_1.shape(0) == 6 );
    bool sum_2D_row_to_col_0_correct=true;
    for( std::size_t ii=0; ii<7; ++ii){
        double sum=0;
        for( std::size_t jj=0; jj<6; ++jj){
            sum += row_2D(jj,ii);
        }
        if(sum!=sum_2D_row_to_col_0(ii)) sum_2D_row_to_col_0_correct=false;
    }
    EXPECT_TRUE(sum_2D_row_to_col_0_correct);
    bool sum_2D_row_to_col_1_correct=true;
    for( std::size_t ii=0; ii<6; ++ii){
        double sum=0;
        for( std::size_t jj=0; jj<7; ++jj){
            sum += row_2D(ii,jj);
        }
        if(sum!=sum_2D_row_to_col_1(ii)) sum_2D_row_to_col_1_correct=false;
    }
    EXPECT_TRUE(sum_2D_row_to_col_1_correct);

    // 3D
    Array<double>::col_major col_3D(shape_vec{6,7,8});
    Array<double>::row_major row_3D(shape_vec{6,7,8});
    count=0; for( auto&& x : col_3D) x=count++;
    count=0; for( auto&& x : row_3D) x=count++;

    // sum over each direction, col major
    Array<double>::col_major sum_3D_col_0 = sum(col_3D,0);
    Array<double>::col_major sum_3D_col_1 = sum(col_3D,1);
    Array<double>::col_major sum_3D_col_2 = sum(col_3D,2);
    EXPECT_TRUE( sum_3D_col_0.dims() == 2 );
    EXPECT_TRUE( sum_3D_col_1.dims() == 2 );
    EXPECT_TRUE( sum_3D_col_2.dims() == 2 );
    EXPECT_TRUE( sum_3D_col_0.shape(0) == 7 );
    EXPECT_TRUE( sum_3D_col_1.shape(0) == 6 );
    EXPECT_TRUE( sum_3D_col_2.shape(0) == 6 );
    EXPECT_TRUE( sum_3D_col_0.shape(1) == 8 );
    EXPECT_TRUE( sum_3D_col_1.shape(1) == 8 );
    EXPECT_TRUE( sum_3D_col_2.shape(1) == 7 );
    bool sum_3D_col_0_correct=true;
    for( std::size_t ii=0; ii<7; ++ii){
        for( std::size_t jj=0; jj<8; ++jj){
            double sum=0;
            for( std::size_t kk=0; kk<6; ++kk){
                sum += col_3D(kk,ii,jj);
            }
            if(sum!=sum_3D_col_0(ii,jj)) sum_3D_col_0_correct=false;
        }
    }
    EXPECT_TRUE(sum_3D_col_0_correct);
    bool sum_3D_col_1_correct=true;
    for( std::size_t ii=0; ii<6; ++ii){
        for( std::size_t jj=0; jj<8; ++jj){
            double sum=0;
            for( std::size_t kk=0; kk<7; ++kk){
                sum += col_3D(ii,kk,jj);
            }
            if(sum!=sum_3D_col_1(ii,jj)) sum_3D_col_1_correct=false;
        }
    }
    EXPECT_TRUE(sum_3D_col_1_correct);
    bool sum_3D_col_2_correct=true;
    for( std::size_t ii=0; ii<6; ++ii){
        for( std::size_t jj=0; jj<7; ++jj){
            double sum=0;
            for( std::size_t kk=0; kk<8; ++kk){
                sum += col_3D(ii,jj,kk);
            }
            if(sum!=sum_3D_col_2(ii,jj)) sum_3D_col_2_correct=false;
        }
    }
    EXPECT_TRUE(sum_3D_col_2_correct);

    // sum over each direction, row major
    Array<double>::row_major sum_3D_row_0 = sum(row_3D,0);
    Array<double>::row_major sum_3D_row_1 = sum(row_3D,1);
    Array<double>::row_major sum_3D_row_2 = sum(row_3D,2);
    EXPECT_TRUE( sum_3D_row_0.dims() == 2 );
    EXPECT_TRUE( sum_3D_row_1.dims() == 2 );
    EXPECT_TRUE( sum_3D_row_2.dims() == 2 );
    EXPECT_TRUE( sum_3D_row_0.shape(0) == 7 );
    EXPECT_TRUE( sum_3D_row_1.shape(0) == 6 );
    EXPECT_TRUE( sum_3D_row_2.shape(0) == 6 );
    EXPECT_TRUE( sum_3D_row_0.shape(1) == 8 );
    EXPECT_TRUE( sum_3D_row_1.shape(1) == 8 );
    EXPECT_TRUE( sum_3D_row_2.shape(1) == 7 );
    bool sum_3D_row_0_correct=true;
    for( std::size_t ii=0; ii<7; ++ii){
        for( std::size_t jj=0; jj<8; ++jj){
            double sum=0;
            for( std::size_t kk=0; kk<6; ++kk){
                sum += row_3D(kk,ii,jj);
            }
            if(sum!=sum_3D_row_0(ii,jj)) sum_3D_row_0_correct=false;
        }
    }
    EXPECT_TRUE(sum_3D_row_0_correct);
    bool sum_3D_row_1_correct=true;
    for( std::size_t ii=0; ii<6; ++ii){
        for( std::size_t jj=0; jj<8; ++jj){
            double sum=0;
            for( std::size_t kk=0; kk<7; ++kk){
                sum += row_3D(ii,kk,jj);
            }
            if(sum!=sum_3D_row_1(ii,jj)) sum_3D_row_1_correct=false;
        }
    }
    EXPECT_TRUE(sum_3D_row_1_correct);
    bool sum_3D_row_2_correct=true;
    for( std::size_t ii=0; ii<6; ++ii){
        for( std::size_t jj=0; jj<7; ++jj){
            double sum=0;
            for( std::size_t kk=0; kk<8; ++kk){
                sum += row_3D(ii,jj,kk);
            }
            if(sum!=sum_3D_row_2(ii,jj)) sum_3D_row_2_correct=false;
        }
    }
    EXPECT_TRUE(sum_3D_row_2_correct);

    // sum over each direction, mixed col to row 
    Array<double>::row_major sum_3D_col_to_row_0 = sum(col_3D,0);
    Array<double>::row_major sum_3D_col_to_row_1 = sum(col_3D,1);
    Array<double>::row_major sum_3D_col_to_row_2 = sum(col_3D,2);
    EXPECT_TRUE( sum_3D_col_to_row_0.dims() == 2 );
    EXPECT_TRUE( sum_3D_col_to_row_1.dims() == 2 );
    EXPECT_TRUE( sum_3D_col_to_row_2.dims() == 2 );
    EXPECT_TRUE( sum_3D_col_to_row_0.shape(0) == 7 );
    EXPECT_TRUE( sum_3D_col_to_row_1.shape(0) == 6 );
    EXPECT_TRUE( sum_3D_col_to_row_2.shape(0) == 6 );
    EXPECT_TRUE( sum_3D_col_to_row_0.shape(1) == 8 );
    EXPECT_TRUE( sum_3D_col_to_row_1.shape(1) == 8 );
    EXPECT_TRUE( sum_3D_col_to_row_2.shape(1) == 7 );
    bool sum_3D_col_to_row_0_correct=true;
    for( std::size_t ii=0; ii<7; ++ii){
        for( std::size_t jj=0; jj<8; ++jj){
            double sum=0;
            for( std::size_t kk=0; kk<6; ++kk){
                sum += col_3D(kk,ii,jj);
            }
            if(sum!=sum_3D_col_to_row_0(ii,jj)) sum_3D_col_to_row_0_correct=false;
        }
    }
    EXPECT_TRUE(sum_3D_col_to_row_0_correct);
    bool sum_3D_col_to_row_1_correct=true;
    for( std::size_t ii=0; ii<6; ++ii){
        for( std::size_t jj=0; jj<8; ++jj){
            double sum=0;
            for( std::size_t kk=0; kk<7; ++kk){
                sum += col_3D(ii,kk,jj);
            }
            if(sum!=sum_3D_col_to_row_1(ii,jj)) sum_3D_col_to_row_1_correct=false;
        }
    }
    EXPECT_TRUE(sum_3D_col_to_row_1_correct);
    bool sum_3D_col_to_row_2_correct=true;
    for( std::size_t ii=0; ii<6; ++ii){
        for( std::size_t jj=0; jj<7; ++jj){
            double sum=0;
            for( std::size_t kk=0; kk<8; ++kk){
                sum += col_3D(ii,jj,kk);
            }
            if(sum!=sum_3D_col_to_row_2(ii,jj)) sum_3D_col_to_row_2_correct=false;
        }
    }
    EXPECT_TRUE(sum_3D_col_to_row_2_correct);

    // sum over each direction, mixed row to col
    Array<double>::col_major sum_3D_row_to_col_0 = sum(row_3D,0);
    Array<double>::col_major sum_3D_row_to_col_1 = sum(row_3D,1);
    Array<double>::col_major sum_3D_row_to_col_2 = sum(row_3D,2);
    EXPECT_TRUE( sum_3D_row_to_col_0.dims() == 2 );
    EXPECT_TRUE( sum_3D_row_to_col_1.dims() == 2 );
    EXPECT_TRUE( sum_3D_row_to_col_2.dims() == 2 );
    EXPECT_TRUE( sum_3D_row_to_col_0.shape(0) == 7 );
    EXPECT_TRUE( sum_3D_row_to_col_1.shape(0) == 6 );
    EXPECT_TRUE( sum_3D_row_to_col_2.shape(0) == 6 );
    EXPECT_TRUE( sum_3D_row_to_col_0.shape(1) == 8 );
    EXPECT_TRUE( sum_3D_row_to_col_1.shape(1) == 8 );
    EXPECT_TRUE( sum_3D_row_to_col_2.shape(1) == 7 );
    bool sum_3D_row_to_col_0_correct=true;
    for( std::size_t ii=0; ii<7; ++ii){
        for( std::size_t jj=0; jj<8; ++jj){
            double sum=0;
            for( std::size_t kk=0; kk<6; ++kk){
                sum += row_3D(kk,ii,jj);
            }
            if(sum!=sum_3D_row_to_col_0(ii,jj)) sum_3D_row_to_col_0_correct=false;
        }
    }
    EXPECT_TRUE(sum_3D_row_to_col_0_correct);
    bool sum_3D_row_to_col_1_correct=true;
    for( std::size_t ii=0; ii<6; ++ii){
        for( std::size_t jj=0; jj<8; ++jj){
            double sum=0;
            for( std::size_t kk=0; kk<7; ++kk){
                sum += row_3D(ii,kk,jj);
            }
            if(sum!=sum_3D_row_to_col_1(ii,jj)) sum_3D_row_to_col_1_correct=false;
        }
    }
    EXPECT_TRUE(sum_3D_row_to_col_1_correct);
    bool sum_3D_row_to_col_2_correct=true;
    for( std::size_t ii=0; ii<6; ++ii){
        for( std::size_t jj=0; jj<7; ++jj){
            double sum=0;
            for( std::size_t kk=0; kk<8; ++kk){
                sum += row_3D(ii,jj,kk);
            }
            if(sum!=sum_3D_row_to_col_2(ii,jj)) sum_3D_row_to_col_2_correct=false;
        }
    }
    EXPECT_TRUE(sum_3D_row_to_col_2_correct);
    
    // double sum over col
    Array<double>::col_major sum_3D_col_00 = sum(sum(col_3D,0),0);
    Array<double>::col_major sum_3D_col_01 = sum(sum(col_3D,1),0);
    Array<double>::col_major sum_3D_col_02 = sum(sum(col_3D,2),0);
    Array<double>::col_major sum_3D_col_10 = sum(sum(col_3D,0),1);
    Array<double>::col_major sum_3D_col_11 = sum(sum(col_3D,1),1);
    Array<double>::col_major sum_3D_col_12 = sum(sum(col_3D,2),1);
    EXPECT_TRUE( sum_3D_col_00.dims() == 1 );
    EXPECT_TRUE( sum_3D_col_01.dims() == 1 );
    EXPECT_TRUE( sum_3D_col_02.dims() == 1 );
    EXPECT_TRUE( sum_3D_col_10.dims() == 1 );
    EXPECT_TRUE( sum_3D_col_11.dims() == 1 );
    EXPECT_TRUE( sum_3D_col_12.dims() == 1 );
    EXPECT_TRUE( sum_3D_col_00.shape(0) == 8 );
    EXPECT_TRUE( sum_3D_col_01.shape(0) == 8 );
    EXPECT_TRUE( sum_3D_col_02.shape(0) == 7 );
    EXPECT_TRUE( sum_3D_col_10.shape(0) == 7 );
    EXPECT_TRUE( sum_3D_col_11.shape(0) == 6 );
    EXPECT_TRUE( sum_3D_col_12.shape(0) == 6 );
    bool sum_3D_col_00_correct=true, sum_3D_col_01_correct=true;
    for( std::size_t ii=0; ii<8; ++ii){
        double sum=0;
        for( std::size_t jj=0; jj<7; ++jj){
            for( std::size_t kk=0; kk<6; ++kk){
                sum += col_3D(kk,jj,ii);
            }
        }
        if(sum!=sum_3D_col_00(ii)) sum_3D_col_00_correct=false;
        if(sum!=sum_3D_col_01(ii)) sum_3D_col_01_correct=false;
    }
    EXPECT_TRUE(sum_3D_col_00_correct);
    EXPECT_TRUE(sum_3D_col_01_correct);
    bool sum_3D_col_02_correct=true, sum_3D_col_10_correct=true;
    for( std::size_t ii=0; ii<7; ++ii){
        double sum=0;
        for( std::size_t jj=0; jj<6; ++jj){
            for( std::size_t kk=0; kk<8; ++kk){
                sum += col_3D(jj,ii,kk);
            }
        }
        if(sum!=sum_3D_col_02(ii)) sum_3D_col_02_correct=false;
        if(sum!=sum_3D_col_10(ii)) sum_3D_col_10_correct=false;
    }
    EXPECT_TRUE(sum_3D_col_02_correct);
    EXPECT_TRUE(sum_3D_col_10_correct);
    bool sum_3D_col_11_correct=true, sum_3D_col_12_correct=true;
    for( std::size_t ii=0; ii<6; ++ii){
        double sum=0;
        for( std::size_t jj=0; jj<7; ++jj){
            for( std::size_t kk=0; kk<8; ++kk){
                sum += col_3D(ii,jj,kk);
            }
        }
        if(sum!=sum_3D_col_11(ii)) sum_3D_col_11_correct=false;
        if(sum!=sum_3D_col_12(ii)) sum_3D_col_12_correct=false;
    }
    EXPECT_TRUE(sum_3D_col_11_correct);
    EXPECT_TRUE(sum_3D_col_12_correct);

    // double sum over row
    Array<double>::row_major sum_3D_row_00 = sum(sum(row_3D,0),0);
    Array<double>::row_major sum_3D_row_01 = sum(sum(row_3D,1),0);
    Array<double>::row_major sum_3D_row_02 = sum(sum(row_3D,2),0);
    Array<double>::row_major sum_3D_row_10 = sum(sum(row_3D,0),1);
    Array<double>::row_major sum_3D_row_11 = sum(sum(row_3D,1),1);
    Array<double>::row_major sum_3D_row_12 = sum(sum(row_3D,2),1);
    EXPECT_TRUE( sum_3D_row_00.dims() == 1 );
    EXPECT_TRUE( sum_3D_row_01.dims() == 1 );
    EXPECT_TRUE( sum_3D_row_02.dims() == 1 );
    EXPECT_TRUE( sum_3D_row_10.dims() == 1 );
    EXPECT_TRUE( sum_3D_row_11.dims() == 1 );
    EXPECT_TRUE( sum_3D_row_12.dims() == 1 );
    EXPECT_TRUE( sum_3D_row_00.shape(0) == 8 );
    EXPECT_TRUE( sum_3D_row_01.shape(0) == 8 );
    EXPECT_TRUE( sum_3D_row_02.shape(0) == 7 );
    EXPECT_TRUE( sum_3D_row_10.shape(0) == 7 );
    EXPECT_TRUE( sum_3D_row_11.shape(0) == 6 );
    EXPECT_TRUE( sum_3D_row_12.shape(0) == 6 );
    bool sum_3D_row_00_correct=true, sum_3D_row_01_correct=true;
    for( std::size_t ii=0; ii<8; ++ii){
        double sum=0;
        for( std::size_t jj=0; jj<7; ++jj){
            for( std::size_t kk=0; kk<6; ++kk){
                sum += row_3D(kk,jj,ii);
            }
        }
        if(sum!=sum_3D_row_00(ii)) sum_3D_row_00_correct=false;
        if(sum!=sum_3D_row_01(ii)) sum_3D_row_01_correct=false;
    }
    EXPECT_TRUE(sum_3D_row_00_correct);
    EXPECT_TRUE(sum_3D_row_01_correct);
    bool sum_3D_row_02_correct=true, sum_3D_row_10_correct=true;
    for( std::size_t ii=0; ii<7; ++ii){
        double sum=0;
        for( std::size_t jj=0; jj<6; ++jj){
            for( std::size_t kk=0; kk<8; ++kk){
                sum += row_3D(jj,ii,kk);
            }
        }
        if(sum!=sum_3D_row_02(ii)) sum_3D_row_02_correct=false;
        if(sum!=sum_3D_row_10(ii)) sum_3D_row_10_correct=false;
    }
    EXPECT_TRUE(sum_3D_row_02_correct);
    EXPECT_TRUE(sum_3D_row_10_correct);
    bool sum_3D_row_11_correct=true, sum_3D_row_12_correct=true;
    for( std::size_t ii=0; ii<6; ++ii){
        double sum=0;
        for( std::size_t jj=0; jj<7; ++jj){
            for( std::size_t kk=0; kk<8; ++kk){
                sum += row_3D(ii,jj,kk);
            }
        }
        if(sum!=sum_3D_row_11(ii)) sum_3D_row_11_correct=false;
        if(sum!=sum_3D_row_12(ii)) sum_3D_row_12_correct=false;
    }
    EXPECT_TRUE(sum_3D_row_11_correct);
    EXPECT_TRUE(sum_3D_row_12_correct);

    // double sum, col to row
    Array<double>::row_major sum_3D_col_to_row_00 = sum(sum(col_3D,0),0);
    Array<double>::row_major sum_3D_col_to_row_01 = sum(sum(col_3D,1),0);
    Array<double>::row_major sum_3D_col_to_row_02 = sum(sum(col_3D,2),0);
    Array<double>::row_major sum_3D_col_to_row_10 = sum(sum(col_3D,0),1);
    Array<double>::row_major sum_3D_col_to_row_11 = sum(sum(col_3D,1),1);
    Array<double>::row_major sum_3D_col_to_row_12 = sum(sum(col_3D,2),1);
    EXPECT_TRUE( sum_3D_col_to_row_00.dims() == 1 );
    EXPECT_TRUE( sum_3D_col_to_row_01.dims() == 1 );
    EXPECT_TRUE( sum_3D_col_to_row_02.dims() == 1 );
    EXPECT_TRUE( sum_3D_col_to_row_10.dims() == 1 );
    EXPECT_TRUE( sum_3D_col_to_row_11.dims() == 1 );
    EXPECT_TRUE( sum_3D_col_to_row_12.dims() == 1 );
    EXPECT_TRUE( sum_3D_col_to_row_00.shape(0) == 8 );
    EXPECT_TRUE( sum_3D_col_to_row_01.shape(0) == 8 );
    EXPECT_TRUE( sum_3D_col_to_row_02.shape(0) == 7 );
    EXPECT_TRUE( sum_3D_col_to_row_10.shape(0) == 7 );
    EXPECT_TRUE( sum_3D_col_to_row_11.shape(0) == 6 );
    EXPECT_TRUE( sum_3D_col_to_row_12.shape(0) == 6 );
    bool sum_3D_col_to_row_00_correct=true, sum_3D_col_to_row_01_correct=true;
    for( std::size_t ii=0; ii<8; ++ii){
        double sum=0;
        for( std::size_t jj=0; jj<7; ++jj){
            for( std::size_t kk=0; kk<6; ++kk){
                sum += col_3D(kk,jj,ii);
            }
        }
        if(sum!=sum_3D_col_to_row_00(ii)) sum_3D_col_to_row_00_correct=false;
        if(sum!=sum_3D_col_to_row_01(ii)) sum_3D_col_to_row_01_correct=false;
    }
    EXPECT_TRUE(sum_3D_col_to_row_00_correct);
    EXPECT_TRUE(sum_3D_col_to_row_01_correct);
    bool sum_3D_col_to_row_02_correct=true, sum_3D_col_to_row_10_correct=true;
    for( std::size_t ii=0; ii<7; ++ii){
        double sum=0;
        for( std::size_t jj=0; jj<6; ++jj){
            for( std::size_t kk=0; kk<8; ++kk){
                sum += col_3D(jj,ii,kk);
            }
        }
        if(sum!=sum_3D_col_to_row_02(ii)) sum_3D_col_to_row_02_correct=false;
        if(sum!=sum_3D_col_to_row_10(ii)) sum_3D_col_to_row_10_correct=false;
    }
    EXPECT_TRUE(sum_3D_col_to_row_02_correct);
    EXPECT_TRUE(sum_3D_col_to_row_10_correct);
    bool sum_3D_col_to_row_11_correct=true, sum_3D_col_to_row_12_correct=true;
    for( std::size_t ii=0; ii<6; ++ii){
        double sum=0;
        for( std::size_t jj=0; jj<7; ++jj){
            for( std::size_t kk=0; kk<8; ++kk){
                sum += col_3D(ii,jj,kk);
            }
        }
        if(sum!=sum_3D_col_to_row_11(ii)) sum_3D_col_to_row_11_correct=false;
        if(sum!=sum_3D_col_to_row_12(ii)) sum_3D_col_to_row_12_correct=false;
    }
    EXPECT_TRUE(sum_3D_col_to_row_11_correct);
    EXPECT_TRUE(sum_3D_col_to_row_12_correct);
    
    // double sum, row_to_col
    Array<double>::col_major sum_3D_row_to_col_00 = sum(sum(row_3D,0),0);
    Array<double>::col_major sum_3D_row_to_col_01 = sum(sum(row_3D,1),0);
    Array<double>::col_major sum_3D_row_to_col_02 = sum(sum(row_3D,2),0);
    Array<double>::col_major sum_3D_row_to_col_10 = sum(sum(row_3D,0),1);
    Array<double>::col_major sum_3D_row_to_col_11 = sum(sum(row_3D,1),1);
    Array<double>::col_major sum_3D_row_to_col_12 = sum(sum(row_3D,2),1);
    EXPECT_TRUE( sum_3D_row_to_col_00.dims() == 1 );
    EXPECT_TRUE( sum_3D_row_to_col_01.dims() == 1 );
    EXPECT_TRUE( sum_3D_row_to_col_02.dims() == 1 );
    EXPECT_TRUE( sum_3D_row_to_col_10.dims() == 1 );
    EXPECT_TRUE( sum_3D_row_to_col_11.dims() == 1 );
    EXPECT_TRUE( sum_3D_row_to_col_12.dims() == 1 );
    EXPECT_TRUE( sum_3D_row_to_col_00.shape(0) == 8 );
    EXPECT_TRUE( sum_3D_row_to_col_01.shape(0) == 8 );
    EXPECT_TRUE( sum_3D_row_to_col_02.shape(0) == 7 );
    EXPECT_TRUE( sum_3D_row_to_col_10.shape(0) == 7 );
    EXPECT_TRUE( sum_3D_row_to_col_11.shape(0) == 6 );
    EXPECT_TRUE( sum_3D_row_to_col_12.shape(0) == 6 );
    bool sum_3D_row_to_col_00_correct=true, sum_3D_row_to_col_01_correct=true;
    for( std::size_t ii=0; ii<8; ++ii){
        double sum=0;
        for( std::size_t jj=0; jj<7; ++jj){
            for( std::size_t kk=0; kk<6; ++kk){
                sum += row_3D(kk,jj,ii);
            }
        }
        if(sum!=sum_3D_row_to_col_00(ii)) sum_3D_row_to_col_00_correct=false;
        if(sum!=sum_3D_row_to_col_01(ii)) sum_3D_row_to_col_01_correct=false;
    }
    EXPECT_TRUE(sum_3D_row_to_col_00_correct);
    EXPECT_TRUE(sum_3D_row_to_col_01_correct);
    bool sum_3D_row_to_col_02_correct=true, sum_3D_row_to_col_10_correct=true;
    for( std::size_t ii=0; ii<7; ++ii){
        double sum=0;
        for( std::size_t jj=0; jj<6; ++jj){
            for( std::size_t kk=0; kk<8; ++kk){
                sum += row_3D(jj,ii,kk);
            }
        }
        if(sum!=sum_3D_row_to_col_02(ii)) sum_3D_row_to_col_02_correct=false;
        if(sum!=sum_3D_row_to_col_10(ii)) sum_3D_row_to_col_10_correct=false;
    }
    EXPECT_TRUE(sum_3D_row_to_col_02_correct);
    EXPECT_TRUE(sum_3D_row_to_col_10_correct);
    bool sum_3D_row_to_col_11_correct=true, sum_3D_row_to_col_12_correct=true;
    for( std::size_t ii=0; ii<6; ++ii){
        double sum=0;
        for( std::size_t jj=0; jj<7; ++jj){
            for( std::size_t kk=0; kk<8; ++kk){
                sum += row_3D(ii,jj,kk);
            }
        }
        if(sum!=sum_3D_row_to_col_11(ii)) sum_3D_row_to_col_11_correct=false;
        if(sum!=sum_3D_row_to_col_12(ii)) sum_3D_row_to_col_12_correct=false;
    }
    EXPECT_TRUE(sum_3D_row_to_col_11_correct);
    EXPECT_TRUE(sum_3D_row_to_col_12_correct);

    // Striping over something 4D
    Array<double>::col_major col_4D(shape_vec{6,7,8,9});
    count=0; for( auto&& x : col_4D) x=count++;
// Fails!
/*
    Array<double>::col_major sum_4D_00 = sum(sum(c,0),0);
    EXPECT_TRUE( sum_4D_00.dims() == 2 );
    EXPECT_TRUE( sum_4D_00.shape(0) == 8 );
    EXPECT_TRUE( sum_4D_00.shape(1) == 9 );
    bool sum_4D_00_correct=true;
    for( std::size_t ii=0; ii<8; ++ii){
        for( std::size_t jj=0; jj<9; ++ii){
            double sum=0;
            for( std::size_t kk=0; kk<6; ++kk){
                for( std::size_t ll=0; ll<7; ++ll){
                    sum += c(kk,ll,ii,jj);
                }
            }
            if(sum!=sum_4D_00(ii,jj)) sum_4D_00_correct=false;
        }
    }
    EXPECT_TRUE(sum_4D_00_correct);
    Array<double>::col_major sum_4D_11 = sum(sum(c,1),1);
    EXPECT_TRUE( sum_4D_11.dims() == 2 );
    EXPECT_TRUE( sum_4D_11.shape(0) == 6 );
    EXPECT_TRUE( sum_4D_11.shape(1) == 9 );
    bool sum_4D_11_correct=true;
    for( std::size_t ii=0; ii<6; ++ii){
        for( std::size_t jj=0; jj<9; ++ii){
            double sum=0;
            for( std::size_t kk=0; kk<7; ++kk){
                for( std::size_t ll=0; ll<8; ++ll){
                    sum += c(ii,kk,ll,jj);
                }
            }
            if(sum!=sum_4D_11(ii,jj)) sum_4D_11_correct=false;
        }
    }
    EXPECT_TRUE(sum_4D_11_correct);
*/
}
