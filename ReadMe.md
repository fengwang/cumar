# CUMAR

CUMAR (CUDA MAp Reduce) is a pure C++ library accelerating [MapReduce](https://www.wikiwand.com/en/MapReduce) development on GPU, without the pain of CUDA.

### Example Usage of [__map__](http://www.wikiwand.com/en/Map_(higher-order_function))

Let there be two vectors, A and B, of same length. To calculate their elementwise product C with a specified operation

> C[i] = A[i] - B[i] + A[i] * B[i]

We need to write some typical trivial host code like this:

```C++
//main.cc
void impl_operation( double* A, double* B, double* C, int N );

double* A = ...;
double* B = ...;
double* C = ...;
int N = ...;

impl_operation( A, B, C, N );
```

and device side CUDA code:

```CUDA
// impl_operation.cu
__global__ __impl_operation( double* A, double* B, double* C, int N )
{
    int const index = blockDim.x * blockIdx.x + threadIdx.x;

    if ( index < N )
        C[index] = A[index] - B[index] + A[index] * B[index];

}

void impl_operation( double* A, double* B, double* C, int N )
{
    int blocks = 128;
    int grids = (N + blocks - 1) / blocks;

    __impl_operation<<<grids, blocks>>>( A, B, C, N );
}
```

With the help cumar library, we can implement this operation in pure C++ within a few lines:

```C++
double* A = ...;
double* B = ...;
double* C = ...;
int N = ...;
cumar::map()("[](double a, double b, double& c){ c = a - b + a*b; }")( A, A+N, B, C );
```

Depending on the specifications of the working GPU, cumar library will automately generate optimized CUDA code and ptx code in memory, then launch it.
For a typical GTX 1080 GPU and an array length of 11111, the generated files are dumped s

+ [fxaudahpqbfwqg.cu](ptx/fxaudahpqbfwqg.cu)
+ [fxaudahpqbfwqg.cu.ptx](ptx/fxaudahpqbfwqg.cu.ptx)

### Example Usage of [__reduce__](https://en.wikipedia.org/wiki/Fold_(higher-order_function))

To reduce the maximum value of an array

> mx = MAX(A)

we need to write trivial host code like:

```C++
double impl_max_reduce( double* A, int N );

int N = ...;
double* A = ...;

double mx = impl_max_reduce( A, N );
```

and complex device code as is demonstrated in [this ppt](http://developer.download.nvidia.com/compute/cuda/1.1-Beta/x86_website/projects/reduction/doc/reduction.pdf) at page 35 (please note this code is only valid before CUDA7, after CUDA7 `__synchthread()` must be called even threadid is less than 32 ).

With the help cumar library, we can implement this operation in pure C++ within a few lines:
```c++
double* A = ...;
int N = ...;
double red = cumar::reduce()()( "[]( double a, double b ){ return a>b?a:b; }" )( A, A+N );
```

For a typical GPU GTX 1080 and a very large array length `N=1111111`,  cuda will automately generate two CUDA kernels:
+ [vmwesflblkspp.cu](ptx/vmwesflblkspp.cu)
+ [luusddxrbspjv.cu](ptx/luusddxrbspjv.cu)

then compile and launch them sequentially.



