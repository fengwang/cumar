#include "../include/cumar.hpp"
#include <vector>
#include <iostream>
#include <algorithm>

int main()
{
    using namespace cumar;

    unsigned long n = 1111111;

    std::vector<double> a(n, 0.0);
    std::generate( a.begin(), a.end(), [](){ double x = 0.0; return [=]() mutable { x += 1.0; return x; }; }() );

    double* a_ = host_to_device_clone( a.data(), a.data()+n );
    double red = reduce()()( "[]( double a, double b ){ return a>b?a:b; }" )( a_, a_+n );

    std::cout << "Reduce test 10: " << red << " -- " << n << " expected.\n";

    deallocate( a_ );

    return 0;
}

