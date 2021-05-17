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
    double total;
    
    // 1D
    Array<double>::col_major col_1D(shape_vec{6});
    Array<double>::row_major row_1D(shape_vec{6});
    count=0; for( auto&& x : col_1D) x=count++;
    count=0; for( auto&& x : row_1D) x=count++;

    // sum, col major
    Array<double>::col_major sum_1D_col = sum(col_1D,0);
    EXPECT_TRUE( sum_1D_col.dims() == 1 );
    EXPECT_TRUE( sum_1D_col.size() == 1 );
    EXPECT_TRUE( sum_1D_col.shape(0) == 1 );
    total=0;
    for( std::size_t ii=0; ii<6; ++ii){
        total += col_1D(ii);
    }
    EXPECT_TRUE(total == sum_1D_col(0));

    // sum, row major
    Array<double>::row_major sum_1D_row = sum(row_1D,0);
    EXPECT_TRUE( sum_1D_row.dims() == 1 );
    EXPECT_TRUE( sum_1D_row.size() == 1 );
    EXPECT_TRUE( sum_1D_row.shape(0) == 1 );
    EXPECT_TRUE(total == sum_1D_row(0));

    // sum, mixed col to row
    Array<double>::row_major sum_1D_col_to_row = sum(col_1D,0);
    EXPECT_TRUE( sum_1D_col_to_row.dims() == 1 );
    EXPECT_TRUE( sum_1D_col_to_row.size() == 1 );
    EXPECT_TRUE( sum_1D_col_to_row.shape(0) == 1 );
    EXPECT_TRUE(total == sum_1D_col_to_row(0));

    // sum over each direction, mixed row to col
    Array<double>::col_major sum_1D_row_to_col = sum(row_1D,0);
    EXPECT_TRUE( sum_1D_row_to_col.dims() == 1 );
    EXPECT_TRUE( sum_1D_row_to_col.size() == 1 );
    EXPECT_TRUE( sum_1D_row_to_col.shape(0) == 1 );
    EXPECT_TRUE(total == sum_1D_row_to_col(0));
    
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
        total=0;
        for( std::size_t jj=0; jj<6; ++jj){
            total += col_2D(jj,ii);
        }
        if(total!=sum_2D_col_0(ii)) sum_2D_col_0_correct=false;
    }
    EXPECT_TRUE(sum_2D_col_0_correct);
    bool sum_2D_col_1_correct=true;
    for( std::size_t ii=0; ii<6; ++ii){
        total=0;
        for( std::size_t jj=0; jj<7; ++jj){
            total += col_2D(ii,jj);
        }
        if(total!=sum_2D_col_1(ii)) sum_2D_col_1_correct=false;
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
        total=0;
        for( std::size_t jj=0; jj<6; ++jj){
            total += row_2D(jj,ii);
        }
        if(total!=sum_2D_row_0(ii)) sum_2D_row_0_correct=false;
    }
    EXPECT_TRUE(sum_2D_row_0_correct);
    bool sum_2D_row_1_correct=true;
    for( std::size_t ii=0; ii<6; ++ii){
        total=0;
        for( std::size_t jj=0; jj<7; ++jj){
            total += row_2D(ii,jj);
        }
        if(total!=sum_2D_row_1(ii)) sum_2D_row_1_correct=false;
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
        total=0;
        for( std::size_t jj=0; jj<6; ++jj){
            total += col_2D(jj,ii);
        }
        if(total!=sum_2D_col_to_row_0(ii)) sum_2D_col_to_row_0_correct=false;
    }
    EXPECT_TRUE(sum_2D_col_to_row_0_correct);
    bool sum_2D_col_to_row_1_correct=true;
    for( std::size_t ii=0; ii<6; ++ii){
        total=0;
        for( std::size_t jj=0; jj<7; ++jj){
            total += col_2D(ii,jj);
        }
        if(total!=sum_2D_col_to_row_1(ii)) sum_2D_col_to_row_1_correct=false;
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
        total=0;
        for( std::size_t jj=0; jj<6; ++jj){
            total += row_2D(jj,ii);
        }
        if(total!=sum_2D_row_to_col_0(ii)) sum_2D_row_to_col_0_correct=false;
    }
    EXPECT_TRUE(sum_2D_row_to_col_0_correct);
    bool sum_2D_row_to_col_1_correct=true;
    for( std::size_t ii=0; ii<6; ++ii){
        total=0;
        for( std::size_t jj=0; jj<7; ++jj){
            total += row_2D(ii,jj);
        }
        if(total!=sum_2D_row_to_col_1(ii)) sum_2D_row_to_col_1_correct=false;
    }
    EXPECT_TRUE(sum_2D_row_to_col_1_correct);

    // double sum over col
    Array<double>::col_major sum_2D_col_00 = sum(sum(col_2D,0),0);
    Array<double>::col_major sum_2D_col_01 = sum(sum(col_2D,1),0);
    EXPECT_TRUE( sum_2D_col_00.dims() == 1 );
    EXPECT_TRUE( sum_2D_col_01.dims() == 1 );
    EXPECT_TRUE( sum_2D_col_00.size() == 1 );
    EXPECT_TRUE( sum_2D_col_01.size() == 1 );
    EXPECT_TRUE( sum_2D_col_00.shape(0) == 1 );
    EXPECT_TRUE( sum_2D_col_01.shape(0) == 1 );
    total=0;
    for( std::size_t ii=0; ii<6; ++ii){
        for( std::size_t jj=0; jj<7; ++jj){
            total += col_2D(ii,jj);
        }
    }
    EXPECT_TRUE(total==sum_2D_col_00(0));
    EXPECT_TRUE(total==sum_2D_col_01(0));

    // double sum over row
    Array<double>::row_major sum_2D_row_00 = sum(sum(row_2D,0),0);
    Array<double>::row_major sum_2D_row_01 = sum(sum(row_2D,1),0);
    EXPECT_TRUE( sum_2D_row_00.dims() == 1 );
    EXPECT_TRUE( sum_2D_row_01.dims() == 1 );
    EXPECT_TRUE( sum_2D_row_00.size() == 1 );
    EXPECT_TRUE( sum_2D_row_01.size() == 1 );
    EXPECT_TRUE( sum_2D_row_00.shape(0) == 1 );
    EXPECT_TRUE( sum_2D_row_01.shape(0) == 1 );
    EXPECT_TRUE(total==sum_2D_row_00(0));
    EXPECT_TRUE(total==sum_2D_row_01(0));

    // double sum, row to col
    Array<double>::col_major sum_2D_row_to_col_00 = sum(sum(row_2D,0),0);
    Array<double>::col_major sum_2D_row_to_col_01 = sum(sum(row_2D,1),0);
    EXPECT_TRUE( sum_2D_row_to_col_00.dims() == 1 );
    EXPECT_TRUE( sum_2D_row_to_col_01.dims() == 1 );
    EXPECT_TRUE( sum_2D_row_to_col_00.size() == 1 );
    EXPECT_TRUE( sum_2D_row_to_col_01.size() == 1 );
    EXPECT_TRUE( sum_2D_row_to_col_00.shape(0) == 1 );
    EXPECT_TRUE( sum_2D_row_to_col_01.shape(0) == 1 );
    EXPECT_TRUE(total==sum_2D_row_to_col_00(0));
    EXPECT_TRUE(total==sum_2D_row_to_col_01(0));

    // double sum, col to row
    Array<double>::row_major sum_2D_col_to_row_00 = sum(sum(col_2D,0),0);
    Array<double>::row_major sum_2D_col_to_row_01 = sum(sum(col_2D,1),0);
    EXPECT_TRUE( sum_2D_col_to_row_00.dims() == 1 );
    EXPECT_TRUE( sum_2D_col_to_row_01.dims() == 1 );
    EXPECT_TRUE( sum_2D_col_to_row_00.size() == 1 );
    EXPECT_TRUE( sum_2D_col_to_row_01.size() == 1 );
    EXPECT_TRUE( sum_2D_col_to_row_00.shape(0) == 1 );
    EXPECT_TRUE( sum_2D_col_to_row_01.shape(0) == 1 );
    EXPECT_TRUE(total==sum_2D_col_to_row_00(0));
    EXPECT_TRUE(total==sum_2D_col_to_row_01(0));

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
            total=0;
            for( std::size_t kk=0; kk<6; ++kk){
                total += col_3D(kk,ii,jj);
            }
            if(total!=sum_3D_col_0(ii,jj)) sum_3D_col_0_correct=false;
        }
    }
    EXPECT_TRUE(sum_3D_col_0_correct);
    bool sum_3D_col_1_correct=true;
    for( std::size_t ii=0; ii<6; ++ii){
        for( std::size_t jj=0; jj<8; ++jj){
            total=0;
            for( std::size_t kk=0; kk<7; ++kk){
                total += col_3D(ii,kk,jj);
            }
            if(total!=sum_3D_col_1(ii,jj)) sum_3D_col_1_correct=false;
        }
    }
    EXPECT_TRUE(sum_3D_col_1_correct);
    bool sum_3D_col_2_correct=true;
    for( std::size_t ii=0; ii<6; ++ii){
        for( std::size_t jj=0; jj<7; ++jj){
            total=0;
            for( std::size_t kk=0; kk<8; ++kk){
                total += col_3D(ii,jj,kk);
            }
            if(total!=sum_3D_col_2(ii,jj)) sum_3D_col_2_correct=false;
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
            total=0;
            for( std::size_t kk=0; kk<6; ++kk){
                total += row_3D(kk,ii,jj);
            }
            if(total!=sum_3D_row_0(ii,jj)) sum_3D_row_0_correct=false;
        }
    }
    EXPECT_TRUE(sum_3D_row_0_correct);
    bool sum_3D_row_1_correct=true;
    for( std::size_t ii=0; ii<6; ++ii){
        for( std::size_t jj=0; jj<8; ++jj){
            total=0;
            for( std::size_t kk=0; kk<7; ++kk){
                total += row_3D(ii,kk,jj);
            }
            if(total!=sum_3D_row_1(ii,jj)) sum_3D_row_1_correct=false;
        }
    }
    EXPECT_TRUE(sum_3D_row_1_correct);
    bool sum_3D_row_2_correct=true;
    for( std::size_t ii=0; ii<6; ++ii){
        for( std::size_t jj=0; jj<7; ++jj){
            total=0;
            for( std::size_t kk=0; kk<8; ++kk){
                total += row_3D(ii,jj,kk);
            }
            if(total!=sum_3D_row_2(ii,jj)) sum_3D_row_2_correct=false;
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
            total=0;
            for( std::size_t kk=0; kk<6; ++kk){
                total += col_3D(kk,ii,jj);
            }
            if(total!=sum_3D_col_to_row_0(ii,jj)) sum_3D_col_to_row_0_correct=false;
        }
    }
    EXPECT_TRUE(sum_3D_col_to_row_0_correct);
    bool sum_3D_col_to_row_1_correct=true;
    for( std::size_t ii=0; ii<6; ++ii){
        for( std::size_t jj=0; jj<8; ++jj){
            total=0;
            for( std::size_t kk=0; kk<7; ++kk){
                total += col_3D(ii,kk,jj);
            }
            if(total!=sum_3D_col_to_row_1(ii,jj)) sum_3D_col_to_row_1_correct=false;
        }
    }
    EXPECT_TRUE(sum_3D_col_to_row_1_correct);
    bool sum_3D_col_to_row_2_correct=true;
    for( std::size_t ii=0; ii<6; ++ii){
        for( std::size_t jj=0; jj<7; ++jj){
            total=0;
            for( std::size_t kk=0; kk<8; ++kk){
                total += col_3D(ii,jj,kk);
            }
            if(total!=sum_3D_col_to_row_2(ii,jj)) sum_3D_col_to_row_2_correct=false;
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
            total=0;
            for( std::size_t kk=0; kk<6; ++kk){
                total += row_3D(kk,ii,jj);
            }
            if(total!=sum_3D_row_to_col_0(ii,jj)) sum_3D_row_to_col_0_correct=false;
        }
    }
    EXPECT_TRUE(sum_3D_row_to_col_0_correct);
    bool sum_3D_row_to_col_1_correct=true;
    for( std::size_t ii=0; ii<6; ++ii){
        for( std::size_t jj=0; jj<8; ++jj){
            total=0;
            for( std::size_t kk=0; kk<7; ++kk){
                total += row_3D(ii,kk,jj);
            }
            if(total!=sum_3D_row_to_col_1(ii,jj)) sum_3D_row_to_col_1_correct=false;
        }
    }
    EXPECT_TRUE(sum_3D_row_to_col_1_correct);
    bool sum_3D_row_to_col_2_correct=true;
    for( std::size_t ii=0; ii<6; ++ii){
        for( std::size_t jj=0; jj<7; ++jj){
            total=0;
            for( std::size_t kk=0; kk<8; ++kk){
                total += row_3D(ii,jj,kk);
            }
            if(total!=sum_3D_row_to_col_2(ii,jj)) sum_3D_row_to_col_2_correct=false;
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
        total=0;
        for( std::size_t jj=0; jj<7; ++jj){
            for( std::size_t kk=0; kk<6; ++kk){
                total += col_3D(kk,jj,ii);
            }
        }
        if(total!=sum_3D_col_00(ii)) sum_3D_col_00_correct=false;
        if(total!=sum_3D_col_01(ii)) sum_3D_col_01_correct=false;
    }
    EXPECT_TRUE(sum_3D_col_00_correct);
    EXPECT_TRUE(sum_3D_col_01_correct);
    bool sum_3D_col_02_correct=true, sum_3D_col_10_correct=true;
    for( std::size_t ii=0; ii<7; ++ii){
        total=0;
        for( std::size_t jj=0; jj<6; ++jj){
            for( std::size_t kk=0; kk<8; ++kk){
                total += col_3D(jj,ii,kk);
            }
        }
        if(total!=sum_3D_col_02(ii)) sum_3D_col_02_correct=false;
        if(total!=sum_3D_col_10(ii)) sum_3D_col_10_correct=false;
    }
    EXPECT_TRUE(sum_3D_col_02_correct);
    EXPECT_TRUE(sum_3D_col_10_correct);
    bool sum_3D_col_11_correct=true, sum_3D_col_12_correct=true;
    for( std::size_t ii=0; ii<6; ++ii){
        total=0;
        for( std::size_t jj=0; jj<7; ++jj){
            for( std::size_t kk=0; kk<8; ++kk){
                total += col_3D(ii,jj,kk);
            }
        }
        if(total!=sum_3D_col_11(ii)) sum_3D_col_11_correct=false;
        if(total!=sum_3D_col_12(ii)) sum_3D_col_12_correct=false;
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
        total=0;
        for( std::size_t jj=0; jj<7; ++jj){
            for( std::size_t kk=0; kk<6; ++kk){
                total += row_3D(kk,jj,ii);
            }
        }
        if(total!=sum_3D_row_00(ii)) sum_3D_row_00_correct=false;
        if(total!=sum_3D_row_01(ii)) sum_3D_row_01_correct=false;
    }
    EXPECT_TRUE(sum_3D_row_00_correct);
    EXPECT_TRUE(sum_3D_row_01_correct);
    bool sum_3D_row_02_correct=true, sum_3D_row_10_correct=true;
    for( std::size_t ii=0; ii<7; ++ii){
        total=0;
        for( std::size_t jj=0; jj<6; ++jj){
            for( std::size_t kk=0; kk<8; ++kk){
                total += row_3D(jj,ii,kk);
            }
        }
        if(total!=sum_3D_row_02(ii)) sum_3D_row_02_correct=false;
        if(total!=sum_3D_row_10(ii)) sum_3D_row_10_correct=false;
    }
    EXPECT_TRUE(sum_3D_row_02_correct);
    EXPECT_TRUE(sum_3D_row_10_correct);
    bool sum_3D_row_11_correct=true, sum_3D_row_12_correct=true;
    for( std::size_t ii=0; ii<6; ++ii){
        total=0;
        for( std::size_t jj=0; jj<7; ++jj){
            for( std::size_t kk=0; kk<8; ++kk){
                total += row_3D(ii,jj,kk);
            }
        }
        if(total!=sum_3D_row_11(ii)) sum_3D_row_11_correct=false;
        if(total!=sum_3D_row_12(ii)) sum_3D_row_12_correct=false;
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
        total=0;
        for( std::size_t jj=0; jj<7; ++jj){
            for( std::size_t kk=0; kk<6; ++kk){
                total += col_3D(kk,jj,ii);
            }
        }
        if(total!=sum_3D_col_to_row_00(ii)) sum_3D_col_to_row_00_correct=false;
        if(total!=sum_3D_col_to_row_01(ii)) sum_3D_col_to_row_01_correct=false;
    }
    EXPECT_TRUE(sum_3D_col_to_row_00_correct);
    EXPECT_TRUE(sum_3D_col_to_row_01_correct);
    bool sum_3D_col_to_row_02_correct=true, sum_3D_col_to_row_10_correct=true;
    for( std::size_t ii=0; ii<7; ++ii){
        total=0;
        for( std::size_t jj=0; jj<6; ++jj){
            for( std::size_t kk=0; kk<8; ++kk){
                total += col_3D(jj,ii,kk);
            }
        }
        if(total!=sum_3D_col_to_row_02(ii)) sum_3D_col_to_row_02_correct=false;
        if(total!=sum_3D_col_to_row_10(ii)) sum_3D_col_to_row_10_correct=false;
    }
    EXPECT_TRUE(sum_3D_col_to_row_02_correct);
    EXPECT_TRUE(sum_3D_col_to_row_10_correct);
    bool sum_3D_col_to_row_11_correct=true, sum_3D_col_to_row_12_correct=true;
    for( std::size_t ii=0; ii<6; ++ii){
        total=0;
        for( std::size_t jj=0; jj<7; ++jj){
            for( std::size_t kk=0; kk<8; ++kk){
                total += col_3D(ii,jj,kk);
            }
        }
        if(total!=sum_3D_col_to_row_11(ii)) sum_3D_col_to_row_11_correct=false;
        if(total!=sum_3D_col_to_row_12(ii)) sum_3D_col_to_row_12_correct=false;
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
        total=0;
        for( std::size_t jj=0; jj<7; ++jj){
            for( std::size_t kk=0; kk<6; ++kk){
                total += row_3D(kk,jj,ii);
            }
        }
        if(total!=sum_3D_row_to_col_00(ii)) sum_3D_row_to_col_00_correct=false;
        if(total!=sum_3D_row_to_col_01(ii)) sum_3D_row_to_col_01_correct=false;
    }
    EXPECT_TRUE(sum_3D_row_to_col_00_correct);
    EXPECT_TRUE(sum_3D_row_to_col_01_correct);
    bool sum_3D_row_to_col_02_correct=true, sum_3D_row_to_col_10_correct=true;
    for( std::size_t ii=0; ii<7; ++ii){
        total=0;
        for( std::size_t jj=0; jj<6; ++jj){
            for( std::size_t kk=0; kk<8; ++kk){
                total += row_3D(jj,ii,kk);
            }
        }
        if(total!=sum_3D_row_to_col_02(ii)) sum_3D_row_to_col_02_correct=false;
        if(total!=sum_3D_row_to_col_10(ii)) sum_3D_row_to_col_10_correct=false;
    }
    EXPECT_TRUE(sum_3D_row_to_col_02_correct);
    EXPECT_TRUE(sum_3D_row_to_col_10_correct);
    bool sum_3D_row_to_col_11_correct=true, sum_3D_row_to_col_12_correct=true;
    for( std::size_t ii=0; ii<6; ++ii){
        total=0;
        for( std::size_t jj=0; jj<7; ++jj){
            for( std::size_t kk=0; kk<8; ++kk){
                total += row_3D(ii,jj,kk);
            }
        }
        if(total!=sum_3D_row_to_col_11(ii)) sum_3D_row_to_col_11_correct=false;
        if(total!=sum_3D_row_to_col_12(ii)) sum_3D_row_to_col_12_correct=false;
    }
    EXPECT_TRUE(sum_3D_row_to_col_11_correct);
    EXPECT_TRUE(sum_3D_row_to_col_12_correct);

    // 4D
    Array<double>::col_major col_4D(shape_vec{6,7,8,9});
    Array<double>::row_major row_4D(shape_vec{6,7,8,9});
    count=0; for( auto&& x : col_4D) x=count++;
    count=0; for( auto&& x : row_4D) x=count++;

    // sum over each direction, col major
    Array<double>::col_major sum_4D_col_0 = sum(col_4D,0);
    Array<double>::col_major sum_4D_col_1 = sum(col_4D,1);
    Array<double>::col_major sum_4D_col_2 = sum(col_4D,2);
    Array<double>::col_major sum_4D_col_3 = sum(col_4D,3);
    EXPECT_TRUE( sum_4D_col_0.dims() == 3 );
    EXPECT_TRUE( sum_4D_col_1.dims() == 3 );
    EXPECT_TRUE( sum_4D_col_2.dims() == 3 );
    EXPECT_TRUE( sum_4D_col_3.dims() == 3 );
    EXPECT_TRUE( sum_4D_col_0.shape(0) == 7 );
    EXPECT_TRUE( sum_4D_col_1.shape(0) == 6 );
    EXPECT_TRUE( sum_4D_col_2.shape(0) == 6 );
    EXPECT_TRUE( sum_4D_col_3.shape(0) == 6 );
    EXPECT_TRUE( sum_4D_col_0.shape(1) == 8 );
    EXPECT_TRUE( sum_4D_col_1.shape(1) == 8 );
    EXPECT_TRUE( sum_4D_col_2.shape(1) == 7 );
    EXPECT_TRUE( sum_4D_col_3.shape(1) == 7 );
    EXPECT_TRUE( sum_4D_col_0.shape(2) == 9 );
    EXPECT_TRUE( sum_4D_col_1.shape(2) == 9 );
    EXPECT_TRUE( sum_4D_col_2.shape(2) == 9 );
    EXPECT_TRUE( sum_4D_col_3.shape(2) == 8 );
    bool sum_4D_col_0_correct=true;
    for( std::size_t ii=0; ii<7; ++ii){
        for( std::size_t jj=0; jj<8; ++jj){
            for( std::size_t kk=0; kk<9; ++kk){
                total=0;
                for( std::size_t ll=0; ll<6; ++ll){
                    total += col_4D(ll,ii,jj,kk);
                }
                if(total!=sum_4D_col_0(ii,jj,kk)) sum_4D_col_0_correct=false;
            }
        }
    }
    EXPECT_TRUE(sum_4D_col_0_correct);
    bool sum_4D_col_1_correct=true;
    for( std::size_t ii=0; ii<6; ++ii){
        for( std::size_t jj=0; jj<8; ++jj){
            for( std::size_t kk=0; kk<9; ++kk){
                total=0;
                for( std::size_t ll=0; ll<7; ++ll){
                    total += col_4D(ii,ll,jj,kk);
                }
                if(total!=sum_4D_col_1(ii,jj,kk)) sum_4D_col_1_correct=false;
            }
        }
    }
    EXPECT_TRUE(sum_4D_col_1_correct);
    bool sum_4D_col_2_correct=true;
    for( std::size_t ii=0; ii<6; ++ii){
        for( std::size_t jj=0; jj<7; ++jj){
            for( std::size_t kk=0; kk<9; ++kk){
                total=0;
                for( std::size_t ll=0; ll<8; ++ll){
                    total += col_4D(ii,jj,ll,kk);
                }
                if(total!=sum_4D_col_2(ii,jj,kk)) sum_4D_col_2_correct=false;
            }
        }
    }
    EXPECT_TRUE(sum_4D_col_2_correct);
    bool sum_4D_col_3_correct=true;
    for( std::size_t ii=0; ii<6; ++ii){
        for( std::size_t jj=0; jj<7; ++jj){
            for( std::size_t kk=0; kk<8; ++kk){
                total=0;
                for( std::size_t ll=0; ll<9; ++ll){
                    total += col_4D(ii,jj,kk,ll);
                }
                if(total!=sum_4D_col_3(ii,jj,kk)) sum_4D_col_3_correct=false;
            }
        }
    }
    EXPECT_TRUE(sum_4D_col_3_correct);

    // sum over each direction, col major to row major
    Array<double>::row_major sum_4D_col_to_row_0 = sum(col_4D,0);
    Array<double>::row_major sum_4D_col_to_row_1 = sum(col_4D,1);
    Array<double>::row_major sum_4D_col_to_row_2 = sum(col_4D,2);
    Array<double>::row_major sum_4D_col_to_row_3 = sum(col_4D,3);
    EXPECT_TRUE( sum_4D_col_to_row_0.dims() == 3 );
    EXPECT_TRUE( sum_4D_col_to_row_1.dims() == 3 );
    EXPECT_TRUE( sum_4D_col_to_row_2.dims() == 3 );
    EXPECT_TRUE( sum_4D_col_to_row_3.dims() == 3 );
    EXPECT_TRUE( sum_4D_col_to_row_0.shape(0) == 7 );
    EXPECT_TRUE( sum_4D_col_to_row_1.shape(0) == 6 );
    EXPECT_TRUE( sum_4D_col_to_row_2.shape(0) == 6 );
    EXPECT_TRUE( sum_4D_col_to_row_3.shape(0) == 6 );
    EXPECT_TRUE( sum_4D_col_to_row_0.shape(1) == 8 );
    EXPECT_TRUE( sum_4D_col_to_row_1.shape(1) == 8 );
    EXPECT_TRUE( sum_4D_col_to_row_2.shape(1) == 7 );
    EXPECT_TRUE( sum_4D_col_to_row_3.shape(1) == 7 );
    EXPECT_TRUE( sum_4D_col_to_row_0.shape(2) == 9 );
    EXPECT_TRUE( sum_4D_col_to_row_1.shape(2) == 9 );
    EXPECT_TRUE( sum_4D_col_to_row_2.shape(2) == 9 );
    EXPECT_TRUE( sum_4D_col_to_row_3.shape(2) == 8 );
    bool sum_4D_col_to_row_0_correct=true;
    for( std::size_t ii=0; ii<7; ++ii){
        for( std::size_t jj=0; jj<8; ++jj){
            for( std::size_t kk=0; kk<9; ++kk){
                total=0;
                for( std::size_t ll=0; ll<6; ++ll){
                    total += col_4D(ll,ii,jj,kk);
                }
                if(total!=sum_4D_col_to_row_0(ii,jj,kk)) sum_4D_col_to_row_0_correct=false;
            }
        }
    }
    EXPECT_TRUE(sum_4D_col_to_row_0_correct);
    bool sum_4D_col_to_row_1_correct=true;
    for( std::size_t ii=0; ii<6; ++ii){
        for( std::size_t jj=0; jj<8; ++jj){
            for( std::size_t kk=0; kk<9; ++kk){
                total=0;
                for( std::size_t ll=0; ll<7; ++ll){
                    total += col_4D(ii,ll,jj,kk);
                }
                if(total!=sum_4D_col_to_row_1(ii,jj,kk)) sum_4D_col_to_row_1_correct=false;
            }
        }
    }
    EXPECT_TRUE(sum_4D_col_to_row_1_correct);
    bool sum_4D_col_to_row_2_correct=true;
    for( std::size_t ii=0; ii<6; ++ii){
        for( std::size_t jj=0; jj<7; ++jj){
            for( std::size_t kk=0; kk<9; ++kk){
                total=0;
                for( std::size_t ll=0; ll<8; ++ll){
                    total += col_4D(ii,jj,ll,kk);
                }
                if(total!=sum_4D_col_to_row_2(ii,jj,kk)) sum_4D_col_to_row_2_correct=false;
            }
        }
    }
    EXPECT_TRUE(sum_4D_col_to_row_2_correct);
    bool sum_4D_col_to_row_3_correct=true;
    for( std::size_t ii=0; ii<6; ++ii){
        for( std::size_t jj=0; jj<7; ++jj){
            for( std::size_t kk=0; kk<8; ++kk){
                total=0;
                for( std::size_t ll=0; ll<9; ++ll){
                    total += col_4D(ii,jj,kk,ll);
                }
                if(total!=sum_4D_col_to_row_3(ii,jj,kk)) sum_4D_col_to_row_3_correct=false;
            }
        }
    }
    EXPECT_TRUE(sum_4D_col_to_row_3_correct);

    // double sum, col
    Array<double>::col_major sum_4D_col_00 = sum(sum(col_4D,0),0);
    EXPECT_TRUE( sum_4D_col_00.dims() == 2 );
    EXPECT_TRUE( sum_4D_col_00.shape(0) == 8 );
    EXPECT_TRUE( sum_4D_col_00.shape(1) == 9 );
    bool sum_4D_col_00_correct=true;
    for( std::size_t ii=0; ii<8; ++ii){
        for( std::size_t jj=0; jj<9; ++jj){
            total=0;
            for( std::size_t kk=0; kk<6; ++kk){
                for( std::size_t ll=0; ll<7; ++ll){
                    total += col_4D(kk,ll,ii,jj);
                }
            }
            if(total!=sum_4D_col_00(ii,jj)) sum_4D_col_00_correct=false;
        }
    }
    EXPECT_TRUE(sum_4D_col_00_correct);

    Array<double>::col_major sum_4D_col_10 = sum(sum(col_4D,0),1);
    EXPECT_TRUE( sum_4D_col_10.dims() == 2 );
    EXPECT_TRUE( sum_4D_col_10.shape(0) == 7 );
    EXPECT_TRUE( sum_4D_col_10.shape(1) == 9 );
    bool sum_4D_col_10_correct=true;
    for( std::size_t jj=0; jj<9; ++jj){
        for( std::size_t ii=0; ii<7; ++ii){
            total=0;
            for( std::size_t kk=0; kk<6; ++kk){
                for( std::size_t ll=0; ll<8; ++ll){
                    total += col_4D(kk,ii,ll,jj);
                }
            }
            if(total!=sum_4D_col_10(ii,jj)) sum_4D_col_10_correct=false;
        }
    }
    EXPECT_TRUE(sum_4D_col_10_correct);

    Array<double>::col_major sum_4D_col_01 = sum(sum(col_4D,1),0);
    EXPECT_TRUE( sum_4D_col_01.dims() == 2 );
    EXPECT_TRUE( sum_4D_col_01.shape(0) == 8 );
    EXPECT_TRUE( sum_4D_col_01.shape(1) == 9 );
    bool sum_4D_col_01_correct=true;
    for( std::size_t ii=0; ii<8; ++ii){
        for( std::size_t jj=0; jj<9; ++jj){
            total=0;
            for( std::size_t kk=0; kk<6; ++kk){
                for( std::size_t ll=0; ll<7; ++ll){
                    total += col_4D(kk,ll,ii,jj);
                }
            }
            if(total!=sum_4D_col_01(ii,jj)) sum_4D_col_01_correct=false;
        }
    }
    EXPECT_TRUE(sum_4D_col_01_correct);

    Array<double>::col_major sum_4D_col_02 = sum(sum(col_4D,2),0);
    EXPECT_TRUE( sum_4D_col_02.dims() == 2 );
    EXPECT_TRUE( sum_4D_col_02.shape(0) == 7 );
    EXPECT_TRUE( sum_4D_col_02.shape(1) == 9 );
    bool sum_4D_col_02_correct=true;
    for( std::size_t ii=0; ii<7; ++ii){
        for( std::size_t jj=0; jj<9; ++jj){
            total=0;
            for( std::size_t kk=0; kk<6; ++kk){
                for( std::size_t ll=0; ll<8; ++ll){
                    total += col_4D(kk,ii,ll,jj);
                }
            }
            if(total!=sum_4D_col_02(ii,jj)) sum_4D_col_02_correct=false;
        }
    }
    EXPECT_TRUE(sum_4D_col_02_correct);

    Array<double>::col_major sum_4D_col_12 = sum(sum(col_4D,2),1);
    EXPECT_TRUE( sum_4D_col_12.dims() == 2 );
    EXPECT_TRUE( sum_4D_col_12.shape(0) == 6 );
    EXPECT_TRUE( sum_4D_col_12.shape(1) == 9 );
    bool sum_4D_col_12_correct=true;
    for( std::size_t jj=0; jj<9; ++jj){
        for( std::size_t ii=0; ii<6; ++ii){
            total=0;
            for( std::size_t kk=0; kk<7; ++kk){
                for( std::size_t ll=0; ll<8; ++ll){
                    total += col_4D(ii,kk,ll,jj);
                }
            }
            if(total!=sum_4D_col_12(ii,jj)) sum_4D_col_12_correct=false;
        }
    }
    EXPECT_TRUE(sum_4D_col_12_correct);

    Array<double>::col_major sum_4D_col_11 = sum(sum(col_4D,1),1);
    EXPECT_TRUE( sum_4D_col_11.dims() == 2 );
    EXPECT_TRUE( sum_4D_col_11.shape(0) == 6 );
    EXPECT_TRUE( sum_4D_col_11.shape(1) == 9 );
    bool sum_4D_col_11_correct=true;
    for( std::size_t jj=0; jj<9; ++jj){
        for( std::size_t ii=0; ii<6; ++ii){
            total=0;
            for( std::size_t kk=0; kk<7; ++kk){
                for( std::size_t ll=0; ll<8; ++ll){
                    total += col_4D(ii,kk,ll,jj);
                }
            }
            if(total!=sum_4D_col_11(ii,jj)) sum_4D_col_11_correct=false;
        }
    }
    EXPECT_TRUE(sum_4D_col_11_correct);

    Array<double>::col_major sum_4D_col_13 = sum(sum(col_4D,3),1);
    EXPECT_TRUE( sum_4D_col_13.dims() == 2 );
    EXPECT_TRUE( sum_4D_col_13.shape(0) == 6 );
    EXPECT_TRUE( sum_4D_col_13.shape(1) == 8 );
    bool sum_4D_col_13_correct=true;
    for( std::size_t jj=0; jj<8; ++jj){
        for( std::size_t ii=0; ii<6; ++ii){
            total=0;
            for( std::size_t kk=0; kk<7; ++kk){
                for( std::size_t ll=0; ll<9; ++ll){
                    total += col_4D(ii,kk,jj,ll);
                }
            }
            if(total!=sum_4D_col_13(ii,jj)) sum_4D_col_13_correct=false;
        }
    }
    EXPECT_TRUE(sum_4D_col_13_correct);

    Array<double>::col_major sum_4D_col_21 = sum(sum(col_4D,1),2);
    EXPECT_TRUE( sum_4D_col_21.dims() == 2 );
    EXPECT_TRUE( sum_4D_col_21.shape(0) == 6 );
    EXPECT_TRUE( sum_4D_col_21.shape(1) == 8 );
    bool sum_4D_col_21_correct=true;
    for( std::size_t jj=0; jj<8; ++jj){
        for( std::size_t ii=0; ii<6; ++ii){
            total=0;
            for( std::size_t kk=0; kk<7; ++kk){
                for( std::size_t ll=0; ll<9; ++ll){
                    total += col_4D(ii,kk,jj,ll);
                }
            }
            if(total!=sum_4D_col_21(ii,jj)) sum_4D_col_21_correct=false;
        }
    }
    EXPECT_TRUE(sum_4D_col_21_correct);

    Array<double>::col_major sum_4D_col_23 = sum(sum(col_4D,3),2);
    EXPECT_TRUE( sum_4D_col_23.dims() == 2 );
    EXPECT_TRUE( sum_4D_col_23.shape(0) == 6 );
    EXPECT_TRUE( sum_4D_col_23.shape(1) == 7 );
    bool sum_4D_col_23_correct=true;
    for( std::size_t jj=0; jj<7; ++jj){
        for( std::size_t ii=0; ii<6; ++ii){
            total=0;
            for( std::size_t kk=0; kk<8; ++kk){
                for( std::size_t ll=0; ll<9; ++ll){
                    total += col_4D(ii,jj,kk,ll);
                }
            }
            if(total!=sum_4D_col_23(ii,jj)) sum_4D_col_23_correct=false;
        }
    }
    EXPECT_TRUE(sum_4D_col_23_correct);

    // double sum, row
    Array<double>::row_major sum_4D_row_23 = sum(sum(row_4D,3),2);
    EXPECT_TRUE( sum_4D_row_23.dims() == 2 );
    EXPECT_TRUE( sum_4D_row_23.shape(0) == 6 );
    EXPECT_TRUE( sum_4D_row_23.shape(1) == 7 );
    bool sum_4D_row_23_correct=true;
    for( std::size_t ii=0; ii<6; ++ii){
        for( std::size_t jj=0; jj<7; ++jj){
            total=0;
            for( std::size_t kk=0; kk<8; ++kk){
                for( std::size_t ll=0; ll<9; ++ll){
                    total += row_4D(ii,jj,kk,ll);
                }
            }
            if(total!=sum_4D_row_23(ii,jj)) sum_4D_row_23_correct=false;
        }
    }
    EXPECT_TRUE(sum_4D_row_23_correct);

    Array<double>::row_major sum_4D_row_00 = sum(sum(row_4D,0),0);
    EXPECT_TRUE( sum_4D_row_00.dims() == 2 );
    EXPECT_TRUE( sum_4D_row_00.shape(0) == 8 );
    EXPECT_TRUE( sum_4D_row_00.shape(1) == 9 );
    bool sum_4D_row_00_correct=true;
    count = 0;
    for( std::size_t ii=0; ii<8; ++ii){
        for( std::size_t jj=0; jj<9; ++jj){
            total=0;
            for( std::size_t kk=0; kk<6; ++kk){
                for( std::size_t ll=0; ll<7; ++ll){
                    total += row_4D(kk,ll,ii,jj);
                }
            }
            if(total!=sum_4D_row_00(ii,jj)) sum_4D_row_00_correct=false;
            ++count;
        }
    }
    EXPECT_TRUE(sum_4D_row_00_correct);

    Array<double>::row_major sum_4D_row_10 = sum(sum(row_4D,0),1);
    EXPECT_TRUE( sum_4D_row_10.dims() == 2 );
    EXPECT_TRUE( sum_4D_row_10.shape(0) == 7 );
    EXPECT_TRUE( sum_4D_row_10.shape(1) == 9 );
    bool sum_4D_row_10_correct=true;
    for( std::size_t jj=0; jj<9; ++jj){
        for( std::size_t ii=0; ii<7; ++ii){
            total=0;
            for( std::size_t kk=0; kk<6; ++kk){
                for( std::size_t ll=0; ll<8; ++ll){
                    total += row_4D(kk,ii,ll,jj);
                }
            }
            if(total!=sum_4D_row_10(ii,jj)) sum_4D_row_10_correct=false;
        }
    }
    EXPECT_TRUE(sum_4D_row_10_correct);

    Array<double>::row_major sum_4D_row_01 = sum(sum(row_4D,1),0);
    EXPECT_TRUE( sum_4D_row_01.dims() == 2 );
    EXPECT_TRUE( sum_4D_row_01.shape(0) == 8 );
    EXPECT_TRUE( sum_4D_row_01.shape(1) == 9 );
    bool sum_4D_row_01_correct=true;
    for( std::size_t ii=0; ii<8; ++ii){
        for( std::size_t jj=0; jj<9; ++jj){
            total=0;
            for( std::size_t kk=0; kk<6; ++kk){
                for( std::size_t ll=0; ll<7; ++ll){
                    total += row_4D(kk,ll,ii,jj);
                }
            }
            if(total!=sum_4D_row_01(ii,jj)) sum_4D_row_01_correct=false;
        }
    }
    EXPECT_TRUE(sum_4D_row_01_correct);

    Array<double>::row_major sum_4D_row_02 = sum(sum(row_4D,2),0);
    EXPECT_TRUE( sum_4D_row_02.dims() == 2 );
    EXPECT_TRUE( sum_4D_row_02.shape(0) == 7 );
    EXPECT_TRUE( sum_4D_row_02.shape(1) == 9 );
    bool sum_4D_row_02_correct=true;
    for( std::size_t ii=0; ii<7; ++ii){
        for( std::size_t jj=0; jj<9; ++jj){
            total=0;
            for( std::size_t kk=0; kk<6; ++kk){
                for( std::size_t ll=0; ll<8; ++ll){
                    total += row_4D(kk,ii,ll,jj);
                }
            }
            if(total!=sum_4D_row_02(ii,jj)) sum_4D_row_02_correct=false;
        }
    }
    EXPECT_TRUE(sum_4D_row_02_correct);

    Array<double>::row_major sum_4D_row_12 = sum(sum(row_4D,2),1);
    EXPECT_TRUE( sum_4D_row_12.dims() == 2 );
    EXPECT_TRUE( sum_4D_row_12.shape(0) == 6 );
    EXPECT_TRUE( sum_4D_row_12.shape(1) == 9 );
    bool sum_4D_row_12_correct=true;
    for( std::size_t jj=0; jj<9; ++jj){
        for( std::size_t ii=0; ii<6; ++ii){
            total=0;
            for( std::size_t kk=0; kk<7; ++kk){
                for( std::size_t ll=0; ll<8; ++ll){
                    total += row_4D(ii,kk,ll,jj);
                }
            }
            if(total!=sum_4D_row_12(ii,jj)) sum_4D_row_12_correct=false;
        }
    }
    EXPECT_TRUE(sum_4D_row_12_correct);

    Array<double>::row_major sum_4D_row_11 = sum(sum(row_4D,1),1);
    EXPECT_TRUE( sum_4D_row_11.dims() == 2 );
    EXPECT_TRUE( sum_4D_row_11.shape(0) == 6 );
    EXPECT_TRUE( sum_4D_row_11.shape(1) == 9 );
    bool sum_4D_row_11_correct=true;
    for( std::size_t jj=0; jj<9; ++jj){
        for( std::size_t ii=0; ii<6; ++ii){
            total=0;
            for( std::size_t kk=0; kk<7; ++kk){
                for( std::size_t ll=0; ll<8; ++ll){
                    total += row_4D(ii,kk,ll,jj);
                }
            }
            if(total!=sum_4D_row_11(ii,jj)) sum_4D_row_11_correct=false;
        }
    }
    EXPECT_TRUE(sum_4D_row_11_correct);

    Array<double>::row_major sum_4D_row_13 = sum(sum(row_4D,3),1);
    EXPECT_TRUE( sum_4D_row_13.dims() == 2 );
    EXPECT_TRUE( sum_4D_row_13.shape(0) == 6 );
    EXPECT_TRUE( sum_4D_row_13.shape(1) == 8 );
    bool sum_4D_row_13_correct=true;
    for( std::size_t jj=0; jj<8; ++jj){
        for( std::size_t ii=0; ii<6; ++ii){
            total=0;
            for( std::size_t kk=0; kk<7; ++kk){
                for( std::size_t ll=0; ll<9; ++ll){
                    total += row_4D(ii,kk,jj,ll);
                }
            }
            if(total!=sum_4D_row_13(ii,jj)) sum_4D_row_13_correct=false;
        }
    }
    EXPECT_TRUE(sum_4D_row_13_correct);

    // double sum, row to col
    Array<double>::col_major sum_4D_row_to_col_23 = sum(sum(row_4D,3),2);
    EXPECT_TRUE( sum_4D_row_to_col_23.dims() == 2 );
    EXPECT_TRUE( sum_4D_row_to_col_23.shape(0) == 6 );
    EXPECT_TRUE( sum_4D_row_to_col_23.shape(1) == 7 );
    bool sum_4D_row_to_col_23_correct=true;
    for( std::size_t ii=0; ii<6; ++ii){
        for( std::size_t jj=0; jj<7; ++jj){
            total=0;
            for( std::size_t kk=0; kk<8; ++kk){
                for( std::size_t ll=0; ll<9; ++ll){
                    total += row_4D(ii,jj,kk,ll);
                }
            }
            if(total!=sum_4D_row_to_col_23(ii,jj)) sum_4D_row_to_col_23_correct=false;
        }
    }
    EXPECT_TRUE(sum_4D_row_to_col_23_correct);

    Array<double>::col_major sum_4D_row_to_col_00 = sum(sum(row_4D,0),0);
    EXPECT_TRUE( sum_4D_row_to_col_00.dims() == 2 );
    EXPECT_TRUE( sum_4D_row_to_col_00.shape(0) == 8 );
    EXPECT_TRUE( sum_4D_row_to_col_00.shape(1) == 9 );
    bool sum_4D_row_to_col_00_correct=true;
    count = 0;
    for( std::size_t ii=0; ii<8; ++ii){
        for( std::size_t jj=0; jj<9; ++jj){
            total=0;
            for( std::size_t kk=0; kk<6; ++kk){
                for( std::size_t ll=0; ll<7; ++ll){
                    total += row_4D(kk,ll,ii,jj);
                }
            }
            if(total!=sum_4D_row_to_col_00(ii,jj)) sum_4D_row_to_col_00_correct=false;
            ++count;
        }
    }
    EXPECT_TRUE(sum_4D_row_to_col_00_correct);

    Array<double>::col_major sum_4D_row_to_col_10 = sum(sum(row_4D,0),1);
    EXPECT_TRUE( sum_4D_row_to_col_10.dims() == 2 );
    EXPECT_TRUE( sum_4D_row_to_col_10.shape(0) == 7 );
    EXPECT_TRUE( sum_4D_row_to_col_10.shape(1) == 9 );
    bool sum_4D_row_to_col_10_correct=true;
    for( std::size_t jj=0; jj<9; ++jj){
        for( std::size_t ii=0; ii<7; ++ii){
            total=0;
            for( std::size_t kk=0; kk<6; ++kk){
                for( std::size_t ll=0; ll<8; ++ll){
                    total += row_4D(kk,ii,ll,jj);
                }
            }
            if(total!=sum_4D_row_to_col_10(ii,jj)) sum_4D_row_to_col_10_correct=false;
        }
    }
    EXPECT_TRUE(sum_4D_row_to_col_10_correct);

    Array<double>::col_major sum_4D_row_to_col_01 = sum(sum(row_4D,1),0);
    EXPECT_TRUE( sum_4D_row_to_col_01.dims() == 2 );
    EXPECT_TRUE( sum_4D_row_to_col_01.shape(0) == 8 );
    EXPECT_TRUE( sum_4D_row_to_col_01.shape(1) == 9 );
    bool sum_4D_row_to_col_01_correct=true;
    for( std::size_t ii=0; ii<8; ++ii){
        for( std::size_t jj=0; jj<9; ++jj){
            total=0;
            for( std::size_t kk=0; kk<6; ++kk){
                for( std::size_t ll=0; ll<7; ++ll){
                    total += row_4D(kk,ll,ii,jj);
                }
            }
            if(total!=sum_4D_row_to_col_01(ii,jj)) sum_4D_row_to_col_01_correct=false;
        }
    }
    EXPECT_TRUE(sum_4D_row_to_col_01_correct);

    Array<double>::col_major sum_4D_row_to_col_02 = sum(sum(row_4D,2),0);
    EXPECT_TRUE( sum_4D_row_to_col_02.dims() == 2 );
    EXPECT_TRUE( sum_4D_row_to_col_02.shape(0) == 7 );
    EXPECT_TRUE( sum_4D_row_to_col_02.shape(1) == 9 );
    bool sum_4D_row_to_col_02_correct=true;
    for( std::size_t ii=0; ii<7; ++ii){
        for( std::size_t jj=0; jj<9; ++jj){
            total=0;
            for( std::size_t kk=0; kk<6; ++kk){
                for( std::size_t ll=0; ll<8; ++ll){
                    total += row_4D(kk,ii,ll,jj);
                }
            }
            if(total!=sum_4D_row_to_col_02(ii,jj)) sum_4D_row_to_col_02_correct=false;
        }
    }
    EXPECT_TRUE(sum_4D_row_to_col_02_correct);

    Array<double>::col_major sum_4D_row_to_col_12 = sum(sum(row_4D,2),1);
    EXPECT_TRUE( sum_4D_row_to_col_12.dims() == 2 );
    EXPECT_TRUE( sum_4D_row_to_col_12.shape(0) == 6 );
    EXPECT_TRUE( sum_4D_row_to_col_12.shape(1) == 9 );
    bool sum_4D_row_to_col_12_correct=true;
    for( std::size_t jj=0; jj<9; ++jj){
        for( std::size_t ii=0; ii<6; ++ii){
            total=0;
            for( std::size_t kk=0; kk<7; ++kk){
                for( std::size_t ll=0; ll<8; ++ll){
                    total += row_4D(ii,kk,ll,jj);
                }
            }
            if(total!=sum_4D_row_to_col_12(ii,jj)) sum_4D_row_to_col_12_correct=false;
        }
    }
    EXPECT_TRUE(sum_4D_row_to_col_12_correct);

    Array<double>::col_major sum_4D_row_to_col_11 = sum(sum(row_4D,1),1);
    EXPECT_TRUE( sum_4D_row_to_col_11.dims() == 2 );
    EXPECT_TRUE( sum_4D_row_to_col_11.shape(0) == 6 );
    EXPECT_TRUE( sum_4D_row_to_col_11.shape(1) == 9 );
    bool sum_4D_row_to_col_11_correct=true;
    for( std::size_t jj=0; jj<9; ++jj){
        for( std::size_t ii=0; ii<6; ++ii){
            total=0;
            for( std::size_t kk=0; kk<7; ++kk){
                for( std::size_t ll=0; ll<8; ++ll){
                    total += row_4D(ii,kk,ll,jj);
                }
            }
            if(total!=sum_4D_row_to_col_11(ii,jj)) sum_4D_row_to_col_11_correct=false;
        }
    }
    EXPECT_TRUE(sum_4D_row_to_col_11_correct);

    Array<double>::col_major sum_4D_row_to_col_13 = sum(sum(row_4D,3),1);
    EXPECT_TRUE( sum_4D_row_to_col_13.dims() == 2 );
    EXPECT_TRUE( sum_4D_row_to_col_13.shape(0) == 6 );
    EXPECT_TRUE( sum_4D_row_to_col_13.shape(1) == 8 );
    bool sum_4D_row_to_col_13_correct=true;
    for( std::size_t jj=0; jj<8; ++jj){
        for( std::size_t ii=0; ii<6; ++ii){
            total=0;
            for( std::size_t kk=0; kk<7; ++kk){
                for( std::size_t ll=0; ll<9; ++ll){
                    total += row_4D(ii,kk,jj,ll);
                }
            }
            if(total!=sum_4D_row_to_col_13(ii,jj)) sum_4D_row_to_col_13_correct=false;
        }
    }
    EXPECT_TRUE(sum_4D_row_to_col_13_correct);

    // double sum, col to row
    Array<double>::row_major sum_4D_col_to_row_23 = sum(sum(col_4D,3),2);
    EXPECT_TRUE( sum_4D_col_to_row_23.dims() == 2 );
    EXPECT_TRUE( sum_4D_col_to_row_23.shape(0) == 6 );
    EXPECT_TRUE( sum_4D_col_to_row_23.shape(1) == 7 );
    bool sum_4D_col_to_row_23_correct=true;
    for( std::size_t ii=0; ii<6; ++ii){
        for( std::size_t jj=0; jj<7; ++jj){
            total=0;
            for( std::size_t kk=0; kk<8; ++kk){
                for( std::size_t ll=0; ll<9; ++ll){
                    total += col_4D(ii,jj,kk,ll);
                }
            }
            if(total!=sum_4D_col_to_row_23(ii,jj)) sum_4D_col_to_row_23_correct=false;
        }
    }
    EXPECT_TRUE(sum_4D_col_to_row_23_correct);

    Array<double>::row_major sum_4D_col_to_row_00 = sum(sum(col_4D,0),0);
    EXPECT_TRUE( sum_4D_col_to_row_00.dims() == 2 );
    EXPECT_TRUE( sum_4D_col_to_row_00.shape(0) == 8 );
    EXPECT_TRUE( sum_4D_col_to_row_00.shape(1) == 9 );
    bool sum_4D_col_to_row_00_correct=true;
    count = 0;
    for( std::size_t ii=0; ii<8; ++ii){
        for( std::size_t jj=0; jj<9; ++jj){
            total=0;
            for( std::size_t kk=0; kk<6; ++kk){
                for( std::size_t ll=0; ll<7; ++ll){
                    total += col_4D(kk,ll,ii,jj);
                }
            }
            if(total!=sum_4D_col_to_row_00(ii,jj)) sum_4D_col_to_row_00_correct=false;
            ++count;
        }
    }
    EXPECT_TRUE(sum_4D_col_to_row_00_correct);

    Array<double>::row_major sum_4D_col_to_row_10 = sum(sum(col_4D,0),1);
    EXPECT_TRUE( sum_4D_col_to_row_10.dims() == 2 );
    EXPECT_TRUE( sum_4D_col_to_row_10.shape(0) == 7 );
    EXPECT_TRUE( sum_4D_col_to_row_10.shape(1) == 9 );
    bool sum_4D_col_to_row_10_correct=true;
    for( std::size_t jj=0; jj<9; ++jj){
        for( std::size_t ii=0; ii<7; ++ii){
            total=0;
            for( std::size_t kk=0; kk<6; ++kk){
                for( std::size_t ll=0; ll<8; ++ll){
                    total += col_4D(kk,ii,ll,jj);
                }
            }
            if(total!=sum_4D_col_to_row_10(ii,jj)) sum_4D_col_to_row_10_correct=false;
        }
    }
    EXPECT_TRUE(sum_4D_col_to_row_10_correct);

    Array<double>::row_major sum_4D_col_to_row_01 = sum(sum(col_4D,1),0);
    EXPECT_TRUE( sum_4D_col_to_row_01.dims() == 2 );
    EXPECT_TRUE( sum_4D_col_to_row_01.shape(0) == 8 );
    EXPECT_TRUE( sum_4D_col_to_row_01.shape(1) == 9 );
    bool sum_4D_col_to_row_01_correct=true;
    for( std::size_t ii=0; ii<8; ++ii){
        for( std::size_t jj=0; jj<9; ++jj){
            total=0;
            for( std::size_t kk=0; kk<6; ++kk){
                for( std::size_t ll=0; ll<7; ++ll){
                    total += col_4D(kk,ll,ii,jj);
                }
            }
            if(total!=sum_4D_col_to_row_01(ii,jj)) sum_4D_col_to_row_01_correct=false;
        }
    }
    EXPECT_TRUE(sum_4D_col_to_row_01_correct);

    Array<double>::row_major sum_4D_col_to_row_02 = sum(sum(col_4D,2),0);
    EXPECT_TRUE( sum_4D_col_to_row_02.dims() == 2 );
    EXPECT_TRUE( sum_4D_col_to_row_02.shape(0) == 7 );
    EXPECT_TRUE( sum_4D_col_to_row_02.shape(1) == 9 );
    bool sum_4D_col_to_row_02_correct=true;
    for( std::size_t ii=0; ii<7; ++ii){
        for( std::size_t jj=0; jj<9; ++jj){
            total=0;
            for( std::size_t kk=0; kk<6; ++kk){
                for( std::size_t ll=0; ll<8; ++ll){
                    total += col_4D(kk,ii,ll,jj);
                }
            }
            if(total!=sum_4D_col_to_row_02(ii,jj)) sum_4D_col_to_row_02_correct=false;
        }
    }
    EXPECT_TRUE(sum_4D_col_to_row_02_correct);

    Array<double>::row_major sum_4D_col_to_row_12 = sum(sum(col_4D,2),1);
    EXPECT_TRUE( sum_4D_col_to_row_12.dims() == 2 );
    EXPECT_TRUE( sum_4D_col_to_row_12.shape(0) == 6 );
    EXPECT_TRUE( sum_4D_col_to_row_12.shape(1) == 9 );
    bool sum_4D_col_to_row_12_correct=true;
    for( std::size_t jj=0; jj<9; ++jj){
        for( std::size_t ii=0; ii<6; ++ii){
            total=0;
            for( std::size_t kk=0; kk<7; ++kk){
                for( std::size_t ll=0; ll<8; ++ll){
                    total += col_4D(ii,kk,ll,jj);
                }
            }
            if(total!=sum_4D_col_to_row_12(ii,jj)) sum_4D_col_to_row_12_correct=false;
        }
    }
    EXPECT_TRUE(sum_4D_col_to_row_12_correct);

    Array<double>::row_major sum_4D_col_to_row_11 = sum(sum(col_4D,1),1);
    EXPECT_TRUE( sum_4D_col_to_row_11.dims() == 2 );
    EXPECT_TRUE( sum_4D_col_to_row_11.shape(0) == 6 );
    EXPECT_TRUE( sum_4D_col_to_row_11.shape(1) == 9 );
    bool sum_4D_col_to_row_11_correct=true;
    for( std::size_t jj=0; jj<9; ++jj){
        for( std::size_t ii=0; ii<6; ++ii){
            total=0;
            for( std::size_t kk=0; kk<7; ++kk){
                for( std::size_t ll=0; ll<8; ++ll){
                    total += col_4D(ii,kk,ll,jj);
                }
            }
            if(total!=sum_4D_col_to_row_11(ii,jj)) sum_4D_col_to_row_11_correct=false;
        }
    }
    EXPECT_TRUE(sum_4D_col_to_row_11_correct);

    Array<double>::row_major sum_4D_col_to_row_13 = sum(sum(col_4D,3),1);
    EXPECT_TRUE( sum_4D_col_to_row_13.dims() == 2 );
    EXPECT_TRUE( sum_4D_col_to_row_13.shape(0) == 6 );
    EXPECT_TRUE( sum_4D_col_to_row_13.shape(1) == 8 );
    bool sum_4D_col_to_row_13_correct=true;
    for( std::size_t jj=0; jj<8; ++jj){
        for( std::size_t ii=0; ii<6; ++ii){
            total=0;
            for( std::size_t kk=0; kk<7; ++kk){
                for( std::size_t ll=0; ll<9; ++ll){
                    total += col_4D(ii,kk,jj,ll);
                }
            }
            if(total!=sum_4D_col_to_row_13(ii,jj)) sum_4D_col_to_row_13_correct=false;
        }
    }
    EXPECT_TRUE(sum_4D_col_to_row_13_correct);

    // triple sum
    Array<double>::col_major sum_4D_col_012 = sum(sum(sum(col_4D,2),1),0);
    EXPECT_TRUE( sum_4D_col_012.dims() == 1 );
    EXPECT_TRUE( sum_4D_col_012.shape(0) == 9 );
    bool sum_4D_col_012_correct=true;
    for( std::size_t ii=0; ii<9; ++ii){
        total=0;
        for( std::size_t jj=0; jj<6; ++jj){
            for( std::size_t kk=0; kk<7; ++kk){
                for( std::size_t ll=0; ll<8; ++ll){
                    total += col_4D(jj,kk,ll,ii);
                }
            }
        }
        if(total!=sum_4D_col_012(ii)) sum_4D_col_012_correct=false;
    }
    EXPECT_TRUE(sum_4D_col_012_correct);

    Array<double>::col_major sum_4D_col_120 = sum(sum(sum(col_4D,0),2),1);
    EXPECT_TRUE( sum_4D_col_120.dims() == 1 );
    EXPECT_TRUE( sum_4D_col_120.shape(0) == 7 );
    bool sum_4D_col_120_correct=true;
    for( std::size_t ii=0; ii<7; ++ii){
        total=0;
        for( std::size_t jj=0; jj<6; ++jj){
            for( std::size_t kk=0; kk<8; ++kk){
                for( std::size_t ll=0; ll<9; ++ll){
                    total += col_4D(jj,ii,kk,ll);
                }
            }
        }
        if(total!=sum_4D_col_120(ii)) sum_4D_col_120_correct=false;
    }
    EXPECT_TRUE(sum_4D_col_120_correct);
}
