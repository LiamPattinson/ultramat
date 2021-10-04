#include "ultramat/include/Dense/DenseStriper.hpp"
#include <gtest/gtest.h>

using namespace ultra;

TEST(DenseStriperTest,DenseStriper){
    Shape shape{5,3,1,7};
    DenseStriper col_striper_begin( 1, DenseOrder::col_major, shape);
    DenseStriper col_striper_end( 1, DenseOrder::col_major, shape, 1);
    DenseStriper row_striper_begin( 1, DenseOrder::row_major, shape);
    DenseStriper row_striper_end( 1, DenseOrder::row_major, shape, 1);

    EXPECT_TRUE( col_striper_begin < col_striper_end );
    EXPECT_TRUE( col_striper_begin <= col_striper_end );
    EXPECT_TRUE( row_striper_begin < row_striper_end );
    EXPECT_TRUE( row_striper_begin <= row_striper_end );

    DenseStriper col_striper( col_striper_begin);
    DenseStriper row_striper( row_striper_begin);

    for( std::size_t ii=0; ii<4; ++ii){
        EXPECT_EQ( col_striper.shape(ii), shape[ii] );
        EXPECT_EQ( col_striper_end.shape(ii), shape[ii] );
        EXPECT_EQ( row_striper.shape(ii), shape[ii] );
        EXPECT_EQ( row_striper_end.shape(ii), shape[ii] );
    }
    for( std::size_t ii=0; ii<5; ++ii){
        EXPECT_EQ( col_striper.index(ii), 0 );
        EXPECT_EQ( row_striper.index(ii), 0 );
        EXPECT_EQ( col_striper_end.index(ii), ii==4 );
        EXPECT_EQ( row_striper_end.index(ii), ii==0 );
    }

    // increment and see what happens
    
    EXPECT_TRUE( col_striper == col_striper_begin );
    EXPECT_TRUE( row_striper == row_striper_begin );

    for( std::size_t step=1; step<=35; ++step){
        EXPECT_TRUE(col_striper != col_striper_end);
        EXPECT_TRUE(row_striper != row_striper_end);
        EXPECT_TRUE(col_striper < col_striper_end);
        EXPECT_TRUE(row_striper < row_striper_end);
        EXPECT_TRUE(col_striper_end > col_striper);
        EXPECT_TRUE(row_striper_end > row_striper);
        ++col_striper;
        ++row_striper;
        EXPECT_TRUE(col_striper <= col_striper_end);
        EXPECT_TRUE(row_striper <= row_striper_end);
        EXPECT_TRUE(col_striper_end >= col_striper);
        EXPECT_TRUE(row_striper_end >= row_striper);

        EXPECT_EQ(col_striper.index(0), (step%5));
        EXPECT_EQ(row_striper.index(0), (step == 35));
        EXPECT_EQ(col_striper.index(1), 0); // stripe dim, should stay at 0
        EXPECT_EQ(row_striper.index(1), ((step/7)%5));
        EXPECT_EQ(col_striper.index(2), 0); // shape size of 1 should stay at 0
        EXPECT_EQ(row_striper.index(2), 0); // stripe dim, should stay at 0
        EXPECT_EQ(col_striper.index(3), ((step/5)%7));
        EXPECT_EQ(row_striper.index(3), 0); //
        EXPECT_EQ(col_striper.index(4), (step == 35));
        EXPECT_EQ(row_striper.index(4), (step%7));
    }

    EXPECT_TRUE(col_striper == col_striper_end);
    EXPECT_TRUE(row_striper == row_striper_end);

    // Do the same in reverse
    for( std::ptrdiff_t step=34; step>=0; --step){
        EXPECT_TRUE(col_striper_end >= col_striper);
        EXPECT_TRUE(row_striper_end >= row_striper);
        EXPECT_TRUE(col_striper <= col_striper_end);
        EXPECT_TRUE(row_striper <= row_striper_end);
        --col_striper;
        --row_striper;
        EXPECT_TRUE(col_striper != col_striper_end);
        EXPECT_TRUE(row_striper != row_striper_end);
        EXPECT_TRUE(col_striper < col_striper_end);
        EXPECT_TRUE(row_striper < row_striper_end);
        EXPECT_TRUE(col_striper_end > col_striper);
        EXPECT_TRUE(row_striper_end > row_striper);

        EXPECT_EQ(col_striper.index(0), (step%5));
        EXPECT_EQ(row_striper.index(0), (step == 35));
        EXPECT_EQ(col_striper.index(1), 0); // stripe dim, should stay at 0
        EXPECT_EQ(row_striper.index(1), ((step/7)%5));
        EXPECT_EQ(col_striper.index(2), 0); // shape size of 1 should stay at 0
        EXPECT_EQ(row_striper.index(2), 0); // stripe dim, should stay at 0
        EXPECT_EQ(col_striper.index(3), ((step/5)%7));
        EXPECT_EQ(row_striper.index(3), 0); //
        EXPECT_EQ(col_striper.index(4), (step == 35));
        EXPECT_EQ(row_striper.index(4), (step%7));
    }

    EXPECT_TRUE( col_striper == col_striper_begin );
    EXPECT_TRUE( row_striper == row_striper_begin );

    // Increment by amounts greater than 1.
    
    col_striper += 19;
    row_striper += 19;

    EXPECT_EQ(col_striper.index(0), (19%5));
    EXPECT_EQ(row_striper.index(0), (19 == 35));
    EXPECT_EQ(col_striper.index(1), 0); // stripe dim, should stay at 0
    EXPECT_EQ(row_striper.index(1), ((19/7)%5));
    EXPECT_EQ(col_striper.index(2), 0); // shape size of 1 should stay at 0
    EXPECT_EQ(row_striper.index(2), 0); // stripe dim, should stay at 0
    EXPECT_EQ(col_striper.index(3), ((19/5)%7));
    EXPECT_EQ(row_striper.index(3), 0); //
    EXPECT_EQ(col_striper.index(4), (19 == 35));
    EXPECT_EQ(row_striper.index(4), (19%7));

    col_striper -= 13;
    row_striper -= 13;
    
    EXPECT_EQ(col_striper.index(0), (6%5));
    EXPECT_EQ(row_striper.index(0), (6 == 35));
    EXPECT_EQ(col_striper.index(1), 0); // stripe dim, should stay at 0
    EXPECT_EQ(row_striper.index(1), ((6/7)%5));
    EXPECT_EQ(col_striper.index(2), 0); // shape size of 1 should stay at 0
    EXPECT_EQ(row_striper.index(2), 0); // stripe dim, should stay at 0
    EXPECT_EQ(col_striper.index(3), ((6/5)%7));
    EXPECT_EQ(row_striper.index(3), 0); //
    EXPECT_EQ(col_striper.index(4), (6 == 35));
    EXPECT_EQ(row_striper.index(4), (6%7));

    col_striper += 21;
    row_striper += 21;

    EXPECT_EQ(col_striper.index(0), (27%5));
    EXPECT_EQ(row_striper.index(0), (27 == 35));
    EXPECT_EQ(col_striper.index(1), 0); // stripe dim, should stay at 0
    EXPECT_EQ(row_striper.index(1), ((27/7)%5));
    EXPECT_EQ(col_striper.index(2), 0); // shape size of 1 should stay at 0
    EXPECT_EQ(row_striper.index(2), 0); // stripe dim, should stay at 0
    EXPECT_EQ(col_striper.index(3), ((27/5)%7));
    EXPECT_EQ(row_striper.index(3), 0); //
    EXPECT_EQ(col_striper.index(4), (27 == 35));
    EXPECT_EQ(row_striper.index(4), (27%7));

}
