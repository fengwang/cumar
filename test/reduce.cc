#include "../include/cumar.hpp"
#include "../include/cumar_misc.hpp"
#include <string>
#include <vector>
#include <numeric>
#include <iostream>
#include <memory>
#include <cmath>
#include <algorithm>
#include <iomanip>

int main()
{
    using namespace cumar;
    set_device(0);

	typedef double working_type;

    if ( 1 )
    {
        unsigned long n = 1;
        std::vector<working_type> a;
        a.resize(n);
        std::fill( a.begin(), a.end(), 1 );
        working_type* a_ = host_to_device( a.data(), a.data()+n );
        working_type red;
        {
            timer t{"Testing ..."};
            red = reduce()()( "[]( double a, double b ){ return a+b; }" )( a_, a_+n );
        }
        std::cout.precision( 15 );
        std::cout << "Reduce test 1: " << red << " -- " << n << " expected.\n";
        deallocate( a_ );
    }

    if ( 1 )
    {
        unsigned long n = 11;
        std::vector<working_type> a;
        a.resize(n);
        std::fill( a.begin(), a.end(), 1 );
        working_type* a_ = host_to_device( a.data(), a.data()+n );
        working_type red;
        {
            timer t{"Testing ..."};
            red = reduce()()( "[]( double a, double b ){ return a+b; }" )( a_, a_+n );
        }
        std::cout.precision( 15 );
        std::cout << "Reduce test 2: " << red << " -- " << n << " expected.\n";
        deallocate( a_ );
    }

    if ( 1 )
    {
        unsigned long n = 111;
        std::vector<working_type> a;
        a.resize(n);
        std::fill( a.begin(), a.end(), 1 );
        working_type* a_ = host_to_device( a.data(), a.data()+n );
        working_type red;
        {
            timer t{"Testing ..."};
            red = reduce()()( "[]( double a, double b ){ return a+b; }" )( a_, a_+n );
        }
        std::cout.precision( 15 );
        std::cout << "Reduce test 3: " << red << " -- " << n << " expected.\n";
        deallocate( a_ );
    }

    if ( 1 )
    {
        unsigned long n = 1111;
        std::vector<working_type> a;
        a.resize(n);
        std::fill( a.begin(), a.end(), 1 );
        working_type* a_ = host_to_device( a.data(), a.data()+n );
        working_type red;
        {
            timer t{"Testing ..."};
            red = reduce()()( "[]( double a, double b ){ return a+b; }" )( a_, a_+n );
        }
        std::cout.precision( 15 );
        std::cout << "Reduce test 4: " << red << " -- " << n << " expected.\n";
        deallocate( a_ );
    }

    if ( 1 )
    {
        unsigned long n = 11111;
        std::vector<working_type> a;
        a.resize(n);
        std::fill( a.begin(), a.end(), 1 );
        working_type* a_ = host_to_device( a.data(), a.data()+n );
        working_type red;
        {
            timer t{"Testing ..."};
            red = reduce()()( "[]( double a, double b ){ return a+b; }" )( a_, a_+n );
        }
        std::cout.precision( 15 );
        std::cout << "Reduce test 5: " << red << " -- " << n << " expected.\n";
        deallocate( a_ );
    }

    if ( 1 )
    {
        unsigned long n = 111111;
        std::vector<working_type> a;
        a.resize(n);
        std::fill( a.begin(), a.end(), 1 );
        working_type* a_ = host_to_device( a.data(), a.data()+n );
        working_type red;
        {
            timer t{"Testing ..."};
            red = reduce()()( "[]( double a, double b ){ return a+b; }" )( a_, a_+n );
        }
        std::cout.precision( 15 );
        std::cout << "Reduce test 6: " << red << " -- " << n << " expected.\n";
        deallocate( a_ );
    }

    if ( 1 )
    {
        unsigned long n = 1111111;
        std::vector<working_type> a;
        a.resize(n);
        std::fill( a.begin(), a.end(), 1 );
        working_type* a_ = host_to_device( a.data(), a.data()+n );
        working_type red;
        {
            timer t{"Testing ..."};
            red = reduce()()( "[]( double a, double b ){ return a+b; }" )( a_, a_+n );
        }
        std::cout.precision( 15 );
        std::cout << "Reduce test 7: " << red << " -- " << n << " expected.\n";
        deallocate( a_ );
    }

    if ( 0 ) // memory not fit with GF750M
    {
        unsigned long n = 11111111;
        std::vector<working_type> a;
        a.resize(n);
        std::fill( a.begin(), a.end(), 1 );
        working_type* a_ = host_to_device( a.data(), a.data()+n );
        working_type red;
        {
            timer t{"Testing ..."};
            red = reduce()()( "[]( double a, double b ){ return a+b; }" )( a_, a_+n );
        }
        std::cout.precision( 15 );
        std::cout << "Reduce test 8: " << red << " -- " << n << " expected.\n";
        deallocate( a_ );
    }

    if ( 0 ) // memory not fit any more with K20
    {
        unsigned long n = 111111111;
        std::vector<working_type> a;
        a.resize(n);
        std::fill( a.begin(), a.end(), 1 );
        working_type* a_ = host_to_device( a.data(), a.data()+n );
        working_type red;
        {
            timer t{"Testing max reduce"};
            red = reduce()()( "[]( float a, float b ){ return a+b; }" )( a_, a_+n );
        }
        std::cout.precision( 15 );
        std::cout << "Reduce test 9: " << red << " -- " << n << " expected.\n";
        deallocate( a_ );
    }

    if ( 1 )
    {
        unsigned long n = 1111111;
        std::vector<working_type> a;
        a.resize(n);
        auto const& generator = [](){ double x = 0.0; return [=]() mutable { x += 0.1; return x; }; };
        std::generate( a.begin(), a.end(), generator() );
        working_type* a_ = host_to_device( a.data(), a.data()+n );
        working_type red;
        {
            timer t{"Testing something....."};
            red = reduce()()( "[]( double a, double b ){ return a>b?a:b; }" )( a_, a_+n );
        }
        std::cout.precision( 15 );
        std::cout << "Reduce test 10: " << red << " -- " << 0.1*n << " expected.\n";
        deallocate( a_ );
    }

    if ( 1 )
    {
        unsigned long n = 1111111;
        std::vector<working_type> a;
        a.resize(n);
        std::fill( a.begin(), a.end(), 0.0 );
        working_type* a_ = host_to_device( a.data(), a.data()+n );
        working_type red;
        {
            timer t{"Testing Opertions."};
            red = reduce()()( "[]( double a, double b ){ return a+b+1.0; }" )( a_, a_+n );
        }
        std::cout.precision( 15 );
        std::cout << "Reduce test 11: " << red << " operations executed.\n";
        deallocate( a_ );
    }

    reset_device();

    return 0;
}

