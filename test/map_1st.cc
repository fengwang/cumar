#include "../include/cumar.hpp"
#include "../include/cumar_misc.hpp"
#include <vector>
#include <numeric>
#include <iostream>
#include <algorithm>

int main()
{
    using namespace cumar;

    unsigned long n = 1111111;

    std::vector<double> a(n, 1.0);
    std::vector<double> b(n, -1.0);

    double* a_ = host_to_device( a.data(), a.data()+n );  // device ptr
    double* b_ = host_to_device( b.data(), b.data()+n );  // device ptr
    double* c_ = allocate<double>(n);                           // device ptr

    map()()( "[](double a, double b, double& c){ c = a + b + 1.0; }" )( a_, a_+n, b_, c_ );

    device_to_host( c_, c_+n, a.data() );
    std::cout << "Map Test : " << std::accumulate( a.begin(), a.end(), 0.0 ) << " -- " << n << " expected.\n";

    deallocate( a_ );
    deallocate( b_ );
    deallocate( c_ );

    return 0;
}

