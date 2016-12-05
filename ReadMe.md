CUMAR (CUda MApReduce) is an easy to use library helps to develop [MapReduce](https://www.wikiwand.com/en/MapReduce) algorithm in pure C++.
With this library, the power of [CUDA](https://www.wikiwand.com/en/CUDA) is utilized and the coder is rescued from undescriptable __nvcc__ features/bugs, hopefully.

### Examples

#### __Map__

The `map` is basically for __[map](http://www.wikiwand.com/en/Map_(higher-order_function)) n lists__.

A very common `c = a + b + 1.0` example.

```c++
#include "../include/cumar.hpp"
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

    double* a_ = host_to_device_clone( a.data(), a.data()+n );  // returns a device ptr
    double* b_ = host_to_device_clone( b.data(), b.data()+n );  // returns a device ptr
    double* c_ = allocate<double>(n);                           // returns a device ptr

    map()()( "[](double a, double b, double& c){ c = a + b + 1.0; }" )( a_, a_+n, b_, c_ );

    device_to_host_copy( c_, c_+n, a.data() );
    std::cout << "Map Test: " << std::accumulate( a.begin(), a.end(), 0.0 ) << " -- " << n << " expected.\n";

    deallocate( a_ );
    deallocate( b_ );
    deallocate( c_ );

    return 0;
}
```

In the example above, the important steps are

- copy contents in host vector `a` to device with a simple clone operation `double* a_ = host_to_device_clone(a.data(), a.data()+n);`, and the returned pointer, `a_`, holds device memory;
- copy contents of host vector `b` to device;
- create memory for device pointer `c_` with a device allocation `double* c_ = allocate<double>(n);`
- execute `map` operation, each element triplet applies a string lambda `[](double a, double b, double& c){ c = a + b + 1.0; }` . In a plain C++ view, is is equivalent to 

```c++
for ( unsigned long i = 0; i != n; ++i )
    [](double a, double b, double& c){ c = a + b + 1.0; }( a[i], b[i], c[i] );
```

- copy computation result from device (pinter `c_`) to host (vector `a`) with `device_to_host_copy( c_, c_+n, a.data() );`


The `map` funcion works with arbitrary numbers of arguments (but at least one)

+ for `a = b + c + d + e + f + g + h`, the `map` step will look like

		map()()( "[](double& a, double b, double c, double d, double e, double f, double g  double h ){ a = b + c + d + e + f + g + h; " )( a, b, c, d, e, f, g );

To take care of additional captured variables, the arguments inside the second bracket is supposed to be employed

+ for `a = b * x + c * y + z`, where `x`, `y` and `z` are constant, the corresponding code will look like

```C++
	double x = 1.0;
	double y = 2.0;
	double z = 3.0;
	map()("x", x, "y", y, "z", z)( "[](double& a, double b, double c){ a = b*x + c*y + z; }" )( a, b, c ); 
```


#### __Reduce__

A very simple example

```C++
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

    std::cout << "Reduce test: " << red << " -- " << n << " expected.\n";

    deallocate( a_ );

    return 0;
}
```

The important steps are

- generate a vector containing `1, 2, 3, ..., n` with `std::generator` and a lambda object;
- copy contents of this vector to device, returns a pointer, `a_`, holding the device memory;
- execute fold operation `max` from the string lambda, `"[]( double a, double b ){ return a>b?a:b; }"`, which is equivalent to

		double red = max( a[0], max( a[1], max( a[2], ... ) ) );


### Implementation Details

TODO


### Tested Platforms


+ Mac OS X, clang++ 3.8.1, CUDA version 8.0.28
+ Arch Linux, clang++ 3.9.0, CUDA version 8.0.44
+ Ubuntu Linux, clang++ 3.6.0, CUDA version 7.0.27

 

