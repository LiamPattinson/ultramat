// ArrayIteration.cpp
//
// Performance test for speed of iteration over multidimensional arrays.

#include<iostream>
#include<iomanip>
#include<vector>
#include<map>
#include<functional>
#include<string>
#include<algorithm>
#include<utility>
#include<chrono>
#include<cstdlib>

#include <ultramat.hpp>
#include <armadillo>
#include <Eigen/Dense>

const std::size_t ROWS = 1000;
const std::size_t COLS = 10000;
const int NUM_TESTS = 5;

template<class T>
using TestFunction = std::function<void(T&, std::uniform_real_distribution<double>&, std::mt19937_64&)>;

template<class T,class F>
double speed_test(T& t, F test_function){
    std::uniform_real_distribution<double> dist(0,1);
    std::mt19937_64 rng;
    double total_time = 0;
    for( int test=0; test<NUM_TESTS; ++test){
        auto start = std::chrono::steady_clock::now();
        test_function(t,dist,rng);
        auto end = std::chrono::steady_clock::now();
        total_time += std::chrono::duration<double,std::milli>(end-start).count();
    }
    return total_time/NUM_TESTS;
}

template<class T>
void round_bracket_col_major( T& t, std::uniform_real_distribution<double>& dist, std::mt19937_64& rng){
    for( std::size_t jj=0; jj<COLS; ++jj){
        for( std::size_t ii=0; ii<ROWS; ++ii){
            t(ii,jj) += dist(rng);
        }
    }
}

template<class T>
void round_bracket_row_major( T& t, std::uniform_real_distribution<double>& dist, std::mt19937_64& rng){
    for( std::size_t ii=0; ii<ROWS; ++ii){
        for( std::size_t jj=0; jj<COLS; ++jj){
            t(ii,jj) += dist(rng);
        }
    }
}

template<class T>
void iteration( T& t, std::uniform_real_distribution<double>& dist, std::mt19937_64& rng){
    for( auto it = t.begin(); it<t.end(); ++it) *it += dist(rng);
}

void fast_iteration( ultra::Array<double>& t, std::uniform_real_distribution<double>& dist, std::mt19937_64& rng){
    for( auto it = t.begin_fast(); it<t.end_fast(); ++it) *it += dist(rng);
}

void stripe_iteration( ultra::Array<double>& t, std::uniform_real_distribution<double>& dist, std::mt19937_64& rng){
    for( std::size_t stripe = 0; stripe < t.num_stripes(); ++stripe){
        for( auto it = t.begin_stripe(stripe); it<t.end_stripe(stripe); ++it) *it += dist(rng);
    }
}

int main(void){

    std::uniform_real_distribution<double> dist(0,10);
    std::mt19937_64 rng;

    arma::mat arma_mat(ROWS,COLS);
    Eigen::MatrixXd eigen_mat(ROWS,COLS);
    ultra::Array<double> ultra_mat_row_major(std::vector<std::size_t>{ROWS,COLS},ultra::Array<double>::row_major);
    ultra::Array<double> ultra_mat_col_major(std::vector<std::size_t>{ROWS,COLS},ultra::Array<double>::col_major);

    for( auto&& x : arma_mat) x = dist(rng);
    for( auto&& x : ultra_mat_row_major) x = dist(rng);
    for( auto&& x : ultra_mat_col_major) x = dist(rng);
    // Eigen versions before 3.4 are boring and don't allow ranged iteration, boo
    for( std::size_t jj=0; jj<COLS; ++jj){
        for( std::size_t ii=0; ii<ROWS; ++ii){
            eigen_mat(ii,jj) = dist(rng);
        }
    }

    // Get results of each test
    std::map<std::string,double> results{
        {"Armadillo round brackets", speed_test(arma_mat,round_bracket_col_major<arma::mat>)},
        {"Armadillo iterator", speed_test(arma_mat,iteration<arma::mat>)},
        {"Eigen round brackets", speed_test(eigen_mat,round_bracket_col_major<Eigen::MatrixXd>)},
        {"Ultramat row major round brackets", speed_test(ultra_mat_row_major,round_bracket_row_major<ultra::Array<double>>)},
        {"Ultramat col major round brackets", speed_test(ultra_mat_col_major,round_bracket_col_major<ultra::Array<double>>)},
        {"Ultramat row major iterator", speed_test(ultra_mat_row_major,iteration<ultra::Array<double>>)},
        {"Ultramat col major iterator", speed_test(ultra_mat_col_major,iteration<ultra::Array<double>>)},
        {"Ultramat row major stripes", speed_test(ultra_mat_row_major,stripe_iteration)},
        {"Ultramat col major stripes", speed_test(ultra_mat_col_major,stripe_iteration)},
        {"Ultramat row major fast", speed_test(ultra_mat_row_major,fast_iteration)},
        {"Ultramat col major fast", speed_test(ultra_mat_col_major,fast_iteration)}
    };

    // Print report
    std::cout << std::setw(48) << "===========================\n"
              << std::setw(48) <<" Ultramat Performance Test \n"
              << std::setw(48) << "===========================\n";
    std::cout << std::setw(36) << "Test" << " | " << std::setw(12) << "Av. Time /ms" << '\n';
    std::cout << std::fixed << std::setprecision(5);
    for( auto&& result : results){
        std::cout << std::setw(36) << result.first  << std::setw(12) << result.second << '\n';
    }
    
    return EXIT_SUCCESS;
}
