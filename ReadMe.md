CUMAR (__CU__da __MA__p__R__educe) is a C++ library accelerating [MapReduce](https://www.wikiwand.com/en/MapReduce) development on GPU.

With this library, the super powers of [Nvidia GPU](https://www.wikiwand.com/en/CUDA) are utilized without coding in CUDA.

## Example Usage

### exmaple for a single array [map](http://www.wikiwand.com/en/Map_(higher-order_function))


Let there be two vectors, A and B. To calculate their elementwise product C with a specified operation

```
C[i] = A[i] - B[i] + A[i] * B[i]

```

We need to write some typical trivial host code like this:

```C++
//main.cc
void impl_operation( float* A, float* B, float* C, int N );

float* A = ...;
float* B = ...;
float* C = ...;
int N = ...;

impl_operation( A, B, C, N );
```

with device side CUDA code:

```CUDA
// impl_operation.cu
__global__ __impl_operation( float* A, float* B, float* C, int N )
{
    int const index = blockDim.x * blockIdx.x + threadIdx.x;

    if ( index < N )
        C[index] = A[index] - B[index] + A[index] * B[index];

}

void impl_operation( float* A, float* B, float* C, int N )
{
    int blocks = 128;
    int grids = (N + blocks - 1) / blocks;

    __impl_operation<<<grids, blocks>>>( A, B, C, N );
}
```

With the help cumar library, we can implement this operation in pure C++ within a few lines:

```C++
float* A = ...;
float* B = ...;
float* C = ...;
int N = ...;
cumar::map()("[](double a, double b, double& c){ c = a - b + a*b; }")( A, A+N, B, C );
```

Depending the working GPU, cumar library will automately generate optimized CUDA code and ptx code, then launch it.
For a typical GTX 1080 GPU, the generated files ard dumped under the `ptx` folder

+ [fxaudahpqbfwqg.cu](ptx/fxaudahpqbfwqg.cu)
+ [fxaudahpqbfwqg.cu.ptx](ptx/fxaudahpqbfwqg.cu.ptx)






### __Reduce__ (without initial value)

To reduce the maximum value of an array

> mx = MAX(A)

where A is of size 1111111, we can code it this way

```c++
unsigned long const n = 1111111;
double* A = ...;
double red = cumar::reduce()()( "[]( double a, double b ){ return a>b?a:b; }" )( A, A+n );
```
with a chipset model of `NVIDIA GeForce GT 750M`, it will, for the first run with shared memory 8192 bytes,  a dimension setup of Grids (46, 1, 1) and Blocks ( 1024, 1, 1 ), generate code

```
__device__ __forceinline__ double dr_xhvpqsohikrkyd( double a, double b ){ return a>b?a:b; }


extern "C" __global__  __launch_bounds__ (1024) void gr_xhvpqsohikrkyd(const double * __restrict__ input, double * __restrict__ output)
{
    extern __shared__ double shared_cache[];

    unsigned long const thread_index_in_current_block = threadIdx.x;
    unsigned long const thread_index = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned long const start_index = thread_index;

    double current_thread_reduction = input[start_index];//thread and block configuration guarantees boundary condition here

    current_thread_reduction = dr_xhvpqsohikrkyd( current_thread_reduction, input[start_index+47104] );
    current_thread_reduction = dr_xhvpqsohikrkyd( current_thread_reduction, input[start_index+94208] );
    current_thread_reduction = dr_xhvpqsohikrkyd( current_thread_reduction, input[start_index+141312] );
    current_thread_reduction = dr_xhvpqsohikrkyd( current_thread_reduction, input[start_index+188416] );
    current_thread_reduction = dr_xhvpqsohikrkyd( current_thread_reduction, input[start_index+235520] );
    current_thread_reduction = dr_xhvpqsohikrkyd( current_thread_reduction, input[start_index+282624] );
    current_thread_reduction = dr_xhvpqsohikrkyd( current_thread_reduction, input[start_index+329728] );
    current_thread_reduction = dr_xhvpqsohikrkyd( current_thread_reduction, input[start_index+376832] );
    current_thread_reduction = dr_xhvpqsohikrkyd( current_thread_reduction, input[start_index+423936] );
    current_thread_reduction = dr_xhvpqsohikrkyd( current_thread_reduction, input[start_index+471040] );
    current_thread_reduction = dr_xhvpqsohikrkyd( current_thread_reduction, input[start_index+518144] );
    current_thread_reduction = dr_xhvpqsohikrkyd( current_thread_reduction, input[start_index+565248] );
    current_thread_reduction = dr_xhvpqsohikrkyd( current_thread_reduction, input[start_index+612352] );
    current_thread_reduction = dr_xhvpqsohikrkyd( current_thread_reduction, input[start_index+659456] );
    current_thread_reduction = dr_xhvpqsohikrkyd( current_thread_reduction, input[start_index+706560] );
    current_thread_reduction = dr_xhvpqsohikrkyd( current_thread_reduction, input[start_index+753664] );
    current_thread_reduction = dr_xhvpqsohikrkyd( current_thread_reduction, input[start_index+800768] );
    current_thread_reduction = dr_xhvpqsohikrkyd( current_thread_reduction, input[start_index+847872] );
    current_thread_reduction = dr_xhvpqsohikrkyd( current_thread_reduction, input[start_index+894976] );
    current_thread_reduction = dr_xhvpqsohikrkyd( current_thread_reduction, input[start_index+942080] );
    current_thread_reduction = dr_xhvpqsohikrkyd( current_thread_reduction, input[start_index+989184] );
    current_thread_reduction = dr_xhvpqsohikrkyd( current_thread_reduction, input[start_index+1036288] );
    if ( start_index < 1111111 - 1083392 )
        current_thread_reduction = dr_xhvpqsohikrkyd( current_thread_reduction, input[start_index+1083392] );


    shared_cache[thread_index_in_current_block] = current_thread_reduction;

    __syncthreads();

    if (1024 > 1024)
    {
        if ( (thread_index_in_current_block < 1024) && (thread_index_in_current_block+1024 < 1024) )
            shared_cache[thread_index_in_current_block] = dr_xhvpqsohikrkyd( shared_cache[thread_index_in_current_block], shared_cache[thread_index_in_current_block+1024] );
        __syncthreads();
    }

    if (1024 > 512)
    {
        if ( (thread_index_in_current_block < 512) && (thread_index_in_current_block+512 < 1024) )
            shared_cache[thread_index_in_current_block] = dr_xhvpqsohikrkyd( shared_cache[thread_index_in_current_block], shared_cache[thread_index_in_current_block+512] );
        __syncthreads();
    }

    if (1024 > 256)
    {
        if ( (thread_index_in_current_block < 256) && (thread_index_in_current_block+256 < 1024) )
            shared_cache[thread_index_in_current_block] = dr_xhvpqsohikrkyd( shared_cache[thread_index_in_current_block], shared_cache[thread_index_in_current_block+256] );
        __syncthreads();
    }

    if (1024 > 128)
    {
        if ( (thread_index_in_current_block < 128) && (thread_index_in_current_block+128 < 1024) )
            shared_cache[thread_index_in_current_block] = dr_xhvpqsohikrkyd( shared_cache[thread_index_in_current_block], shared_cache[thread_index_in_current_block+128] );
        __syncthreads();
    }

    if (1024 > 64)
    {
        if ( (thread_index_in_current_block < 64) && (thread_index_in_current_block+64 < 1024) )
            shared_cache[thread_index_in_current_block] = dr_xhvpqsohikrkyd( shared_cache[thread_index_in_current_block], shared_cache[thread_index_in_current_block+64] );
        __syncthreads();
    }

    if (1024 > 32)
    {
        if ( (thread_index_in_current_block < 32) && (thread_index_in_current_block+32 < 1024) )
            shared_cache[thread_index_in_current_block] = dr_xhvpqsohikrkyd( shared_cache[thread_index_in_current_block], shared_cache[thread_index_in_current_block+32] );
        __syncthreads();
    }

    if ( (1024 > 16) && (thread_index_in_current_block < 16) && (thread_index_in_current_block+16 < 1024) )
            shared_cache[thread_index_in_current_block] = dr_xhvpqsohikrkyd( shared_cache[thread_index_in_current_block], shared_cache[thread_index_in_current_block+16] );
    __syncthreads();

    if ( (1024 > 8) && (thread_index_in_current_block < 8) && (thread_index_in_current_block+8 < 1024) )
            shared_cache[thread_index_in_current_block] = dr_xhvpqsohikrkyd( shared_cache[thread_index_in_current_block], shared_cache[thread_index_in_current_block+8] );
    __syncthreads();

    if ( (1024 > 4) && (thread_index_in_current_block < 4) && (thread_index_in_current_block+4 < 1024) )
            shared_cache[thread_index_in_current_block] = dr_xhvpqsohikrkyd( shared_cache[thread_index_in_current_block], shared_cache[thread_index_in_current_block+4] );
    __syncthreads();

    if ( (1024 > 2) && (thread_index_in_current_block < 2) && (thread_index_in_current_block+2 < 1024) )
            shared_cache[thread_index_in_current_block] = dr_xhvpqsohikrkyd( shared_cache[thread_index_in_current_block], shared_cache[thread_index_in_current_block+2] );
    __syncthreads();

    if ( (1024 > 1) && (thread_index_in_current_block < 1) && (thread_index_in_current_block+1 < 1024) )
            shared_cache[thread_index_in_current_block] = dr_xhvpqsohikrkyd( shared_cache[thread_index_in_current_block], shared_cache[thread_index_in_current_block+1] );
    __syncthreads();

    if (thread_index_in_current_block == 0) output[blockIdx.x] = shared_cache[0];
}
```

and for the second run with shared memory 256 bytes, a dimension setup of Grids (1, 1, 1) and Blocks (32, 1 1), generates code

```
__device__ __forceinline__ double dr_xhvpqsohikrkyd( double a, double b ){ return a>b?a:b; }


extern "C" __global__  __launch_bounds__ (32) void gr_xhvpqsohikrkyd(const double * __restrict__ input, double * __restrict__ output)
{
    extern __shared__ double shared_cache[];

    unsigned long const thread_index_in_current_block = threadIdx.x;
    unsigned long const thread_index = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned long const start_index = thread_index;

    double current_thread_reduction = input[start_index];//thread and block configuration guarantees boundary condition here

    if ( start_index < 46 - 32 )
        current_thread_reduction = dr_xhvpqsohikrkyd( current_thread_reduction, input[start_index+32] );


    shared_cache[thread_index_in_current_block] = current_thread_reduction;

    __syncthreads();

    if (32 > 1024)
    {
        if ( (thread_index_in_current_block < 1024) && (thread_index_in_current_block+1024 < 32) )
            shared_cache[thread_index_in_current_block] = dr_xhvpqsohikrkyd( shared_cache[thread_index_in_current_block], shared_cache[thread_index_in_current_block+1024] );
        __syncthreads();
    }

    if (32 > 512)
    {
        if ( (thread_index_in_current_block < 512) && (thread_index_in_current_block+512 < 32) )
            shared_cache[thread_index_in_current_block] = dr_xhvpqsohikrkyd( shared_cache[thread_index_in_current_block], shared_cache[thread_index_in_current_block+512] );
        __syncthreads();
    }

    if (32 > 256)
    {
        if ( (thread_index_in_current_block < 256) && (thread_index_in_current_block+256 < 32) )
            shared_cache[thread_index_in_current_block] = dr_xhvpqsohikrkyd( shared_cache[thread_index_in_current_block], shared_cache[thread_index_in_current_block+256] );
        __syncthreads();
    }

    if (32 > 128)
    {
        if ( (thread_index_in_current_block < 128) && (thread_index_in_current_block+128 < 32) )
            shared_cache[thread_index_in_current_block] = dr_xhvpqsohikrkyd( shared_cache[thread_index_in_current_block], shared_cache[thread_index_in_current_block+128] );
        __syncthreads();
    }

    if (32 > 64)
    {
        if ( (thread_index_in_current_block < 64) && (thread_index_in_current_block+64 < 32) )
            shared_cache[thread_index_in_current_block] = dr_xhvpqsohikrkyd( shared_cache[thread_index_in_current_block], shared_cache[thread_index_in_current_block+64] );
        __syncthreads();
    }

    if (32 > 32)
    {
        if ( (thread_index_in_current_block < 32) && (thread_index_in_current_block+32 < 32) )
            shared_cache[thread_index_in_current_block] = dr_xhvpqsohikrkyd( shared_cache[thread_index_in_current_block], shared_cache[thread_index_in_current_block+32] );
        __syncthreads();
    }

    if ( (32 > 16) && (thread_index_in_current_block < 16) && (thread_index_in_current_block+16 < 32) )
            shared_cache[thread_index_in_current_block] = dr_xhvpqsohikrkyd( shared_cache[thread_index_in_current_block], shared_cache[thread_index_in_current_block+16] );
    __syncthreads();

    if ( (32 > 8) && (thread_index_in_current_block < 8) && (thread_index_in_current_block+8 < 32) )
            shared_cache[thread_index_in_current_block] = dr_xhvpqsohikrkyd( shared_cache[thread_index_in_current_block], shared_cache[thread_index_in_current_block+8] );
    __syncthreads();

    if ( (32 > 4) && (thread_index_in_current_block < 4) && (thread_index_in_current_block+4 < 32) )
            shared_cache[thread_index_in_current_block] = dr_xhvpqsohikrkyd( shared_cache[thread_index_in_current_block], shared_cache[thread_index_in_current_block+4] );
    __syncthreads();

    if ( (32 > 2) && (thread_index_in_current_block < 2) && (thread_index_in_current_block+2 < 32) )
            shared_cache[thread_index_in_current_block] = dr_xhvpqsohikrkyd( shared_cache[thread_index_in_current_block], shared_cache[thread_index_in_current_block+2] );
    __syncthreads();

    if ( (32 > 1) && (thread_index_in_current_block < 1) && (thread_index_in_current_block+1 < 32) )
            shared_cache[thread_index_in_current_block] = dr_xhvpqsohikrkyd( shared_cache[thread_index_in_current_block], shared_cache[thread_index_in_current_block+1] );
    __syncthreads();

    if (thread_index_in_current_block == 0) output[blockIdx.x] = shared_cache[0];
}
```

It is also possible to pass argument(s) when calling `cumar::reduce()`. Similiar as `cumar::map()`, just put them in the second brackets. Here is an example summing up all the elements in the array with an extra random element `alpha`

```
double alpha = rand();
cumar::map()("alpha", alpha)( "[](double a, double b){return a+b+lambda;}" )( A, A+n );
```

### More examples

Please refere to the files located under folder `./test`.









