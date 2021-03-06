#include "../include/cumar.hpp"
#include "../include/cumar_misc.hpp"
#include <string>
#include <vector>
#include <numeric>
#include <iostream>
#include <memory>
#include <cmath>
#include <algorithm>

int main()
{
    using namespace cumar;

	typedef float working_type;

	unsigned long n = 1111111;
	//unsigned long n = 1111111;

    std::vector<working_type> a;
    std::vector<working_type> b;

    a.resize(n);
	std::fill( a.begin(), a.end(), 1 );
    b.resize(n);
	std::fill( b.begin(), b.end(), -1 );

    set_device(0);


    working_type* a_ = host_to_device( a.data(), a.data()+n );
    working_type* b_ = host_to_device( b.data(), b.data()+n );
    working_type* c_ = allocate<working_type>(n);

    {
        {
            timer( "Map 1" );
            map()()( "[](float a, float b, float& c){ c = a + b + 1.0; } " )( a_, a_+n, b_, c_ );
        }
        device_to_host( c_, c_+n, a.data() );
        std::cout << "Test Case 1: " << std::accumulate( a.begin(), a.end(), 0.0 ) << " -- " << n << " expected.\n";
    }

    {
        {
            timer( "Map 1.5" );
            map()()( "[](float a, float b, float& c){ c = a + b + 1.0; } " )( a_, a_+n, b_, c_ );
        }
        device_to_host( c_, c_+n, a.data() );
        std::cout << "Test Case 1.5: " << std::accumulate( a.begin(), a.end(), 0.0 ) << " -- " << n << " expected.\n";
    }

    {
        float x = 1.1;
        {
            timer( "Map 2" );
            map()( "x", x )( "[](float a, float b, float& c){ c = a + b + 1.0 - x; } " )( a_, a_+n, b_, c_ );
        }
        device_to_host( c_, c_+n, a.data() );
        std::cout << "Test Case 2: " << std::accumulate( a.begin(), a.end(), 0.0 ) << " -- " << -0.1 *  n  << " expected.\n";
    }

    {
        float x = 1.1;
        {
            timer( "Map 2.5" );
            map()( "x", x )( "[](float a, float b, float& c){ c = a + b + 1.0 - x; } " )( a_, a_+n, b_, c_ );
        }
        device_to_host( c_, c_+n, a.data() );
        std::cout << "Test Case 2.5: " << std::accumulate( a.begin(), a.end(), 0.0 ) << " -- " << -0.1 *  n  << " expected.\n";
    }

    {
        float x = 1.1;
        float yy = 1.2;
        {
            timer( "Map 3" );
            map()( "x", x, "yy", yy )( "[](float a, float b, float& c){ c = a + b + 1.0 - x + sin(yy); } " )( a_, a_+n, b_, c_ );
        }
        device_to_host( c_, c_+n, a.data() );
        std::cout << "Test Case 3: " << std::accumulate( a.begin(), a.end(), 0.0 ) << " -- " << (-0.1+std::sin(yy)) *  n  << " expected.\n";
    }

    {
        float x = 1.1;
        float yy = 1.2;
        {
            timer( "Map 3.5" );
            map()( "x", x, "yy", yy )( "[](float a, float b, float& c){ c = a + b + 1.0 - x + sin(yy); } " )( a_, a_+n, b_, c_ );
        }
        device_to_host( c_, c_+n, a.data() );
        std::cout << "Test Case 3.5: " << std::accumulate( a.begin(), a.end(), 0.0 ) << " -- " << (-0.1+std::sin(yy)) *  n  << " expected.\n";
    }
    {
        float index = 0;
        auto const& generator = [&index, n](){ index += 1.0; return static_cast<float>(index / n); };
        std::generate( a.begin(), a.end(), generator );
        host_to_device( a.data(), a.data()+n, a_ );
        std::generate( b.begin(), b.end(), generator );
        host_to_device( b.data(), b.data()+n, b_ );
        {
            timer( "Map 4" );
            map()()( "[]( float a, float b, float& c ){ c = b - a; } " )( a_, a_+n, b_, c_ );
        }
        device_to_host( c_, c_+n, a.data() );
        std::cout << "Test Case 4: " << std::accumulate( a.begin(), a.end(), 0.0 ) << " -- " <<  n  << " expected.\n";
    }



    deallocate( a_ );
    deallocate( b_ );
    deallocate( c_ );


    return 0;
}

