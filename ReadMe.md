CUMAR (CUda MApReduce) is an easy to use library helps to develop [MapReduce](https://www.wikiwand.com/en/MapReduce) algorithms in pure C++.

With this library, the super powers of [CUDA](https://www.wikiwand.com/en/CUDA) are utilized and the coders are rescued from undescriptable __nvcc__ features/bugs, hopefully.

### Examples

#### __Map__ (designed for __[map](http://www.wikiwand.com/en/Map_(higher-order_function)) n lists__).

A very primitive `c = a + b + 1.0` [example](https://github.com/fengwang/cumar/blob/master/test/map_1st.cc).

```c++
int main()
{
    using namespace cumar;
    unsigned long n = 1111111;

    std::vector<double> a(n, 1.0);
    std::vector<double> b(n, -1.0);

    double* a_ = host_to_device( a.data(), a.data()+n );  // returns a device ptr
    double* b_ = host_to_device( b.data(), b.data()+n );  // returns a device ptr
    double* c_ = allocate<double>(n);                     // returns a device ptr

    map()()( "[](double a, double b, double& c){ c = a + b + 1.0; }" )( a_, a_+n, b_, c_ );

    device_to_host( c_, c_+n, a.data() );
    std::cout << "Map Test: " << std::accumulate( a.begin(), a.end(), 0.0 ) << " -- " << n << " expected.\n";

    deallocate( a_ );
    deallocate( b_ );
    deallocate( c_ );
}
```

The important steps are

- copy contents from host vector `a` to device with a simple clone operation `double* a_ = host_to_device(a.data(), a.data()+n);`, returning a device pointer holding device memory;
- copy contents of host vector `b` to device;
- create memory for device pointer `c_`, with a device allocation `double* c_ = allocate<double>(n);`
- execute `map` operation, each element triplet applies a string lambda `[](double a, double b, double& c){ c = a + b + 1.0; }` . In a plain C++ view, this is equivalent to

```c++
for ( unsigned long i = 0; i != n; ++i )
    [](double a, double b, double& c){ c = a + b + 1.0; }( a[i], b[i], c[i] );
```

- copy computation result from device (pinter `c_`) to host (vector `a`) with `device_to_host( c_, c_+n, a.data() );`


The `map` funcion works with at least 2 arguments

+ for `a = b + c + d + e + f + g + h`, the `map` step will look like

        map()()( "[](double& a, double b, double c, double d, double e, double f, double g  double h ){ a = b + c + d + e + f + g + h; " )( a, a+n, b, c, d, e, f, g );

To take care of additional variables attained from context, place their symobls and values in the second bracket

+ for `a = b * x + c * y + z`, where `x`, `y` and `z` are constant variables, the corresponding code example takes a form of

```C++
    double x = 1.0;
    double y = 2.0;
    double z = 3.0;
    map()("x", x, "y", y, "z", z)( "[](double& a, double b, double c){ a = b*x + c*y + z; }" )( a, b, c );
```


#### __Reduce__ (without initial value)

A very simple [example](https://github.com/fengwang/cumar/blob/master/test/reduce_1st.cc) demonstrating max reduce

```C++
using namespace cumar;

unsigned long n = 1111111;

std::vector<double> a(n, 0.0);
std::generate( a.begin(), a.end(), [](){ double x = 0.0; return [=]() mutable { x += 1.0; return x; }; }() );

double* a_ = host_to_device( a.data(), a.data()+n );
double red = reduce()()( "[]( double a, double b ){ return a>b?a:b; }" )( a_, a_+n );

std::cout << "Reduce test: " << red << " -- " << n << " expected.\n";

deallocate( a_ );
```

The important steps are

- generate a vector holding `1, 2, 3, ..., n`, by employing  `std::generator` and a lambda object;
- copy contents of this vector to device memory, returns a device pointer;
- execute fold operation `max` from a string lambda, `"[]( double a, double b ){ return a>b?a:b; }"`, which is equivalent to

        double red = max( a[0], max( a[1], max( a[2], ... ) ) );

### Implementation Details

TODO


### Tested Platforms


+ Mac OS X, clang++ 5.0.0; CUDA version 9.0.197



