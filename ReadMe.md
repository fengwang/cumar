CUMAR (__CU__da __MA__p__R__educe) is a C++ library accelerating [MapReduce](https://www.wikiwand.com/en/MapReduce) development on GPU. 

With this library, the super powers of [Nvidia GPU](https://www.wikiwand.com/en/CUDA) are utilized without coding in CUDA.

## Example Usage

### exmaple for a single array [map](http://www.wikiwand.com/en/Map_(higher-order_function))


To implement 
> A = 1.0

where A is an array of size 10024, we can code it concisely 

```c++
//map_example.cc
unsigned long n = 10024;
float * A = ...;//allocated on device
cumar::map()()("[](float &a){ a = 1.0; }")(A, A+n);
```

in which the lambda object passed the desired GPU operation in a string form

```
[](float &a){ a = 1.0; }
``` 


Then run a typical compilation and link command (Mac OS X for example)

```
clang++ -c -std=c++17 -stdlib=libc++ -O2 -I/Developer/NVIDIA/CUDA-8.0/include -o ./cumar.o src/cumar.cc ./map_example.cc

clang++ -lc++ -lc++abi -O3 -lcudart -lnvrtc -L/Developer/NVIDIA/CUDA-8.0/lib -framework CUDA -o ./map_example ./cumar.o
```

`./map_example` will carefully generate optimized CUDA code and configuration in memory matching the technical specifications of current working GPU, then execute it. The generated code will be cached in case of reuse.

For instance, with a chipset model of `NVIDIA GeForce GT 750M`, it will generate CUDA code below

```
__device__ __forceinline__  void
df_ktdaezcfwmoxyc(float& a){  a = 1.0f; }


extern "C"
__global__ void  __launch_bounds__ ( 192 )
gf_ktdaezcfwmoxyc(float* __restrict__   a)
{
    unsigned long const const index = (blockDim.x * blockIdx.x + threadIdx.x);
    df_ktdaezcfwmoxyc( a[index]) ;
    df_ktdaezcfwmoxyc( a[index+384]) ;
    df_ktdaezcfwmoxyc( a[index+768]) ;
    df_ktdaezcfwmoxyc( a[index+1152]) ;
    df_ktdaezcfwmoxyc( a[index+1536]) ;
    df_ktdaezcfwmoxyc( a[index+1920]) ;
    df_ktdaezcfwmoxyc( a[index+2304]) ;
    df_ktdaezcfwmoxyc( a[index+2688]) ;
    df_ktdaezcfwmoxyc( a[index+3072]) ;
    df_ktdaezcfwmoxyc( a[index+3456]) ;
    df_ktdaezcfwmoxyc( a[index+3840]) ;
    df_ktdaezcfwmoxyc( a[index+4224]) ;
    df_ktdaezcfwmoxyc( a[index+4608]) ;
    df_ktdaezcfwmoxyc( a[index+4992]) ;
    df_ktdaezcfwmoxyc( a[index+5376]) ;
    df_ktdaezcfwmoxyc( a[index+5760]) ;
    df_ktdaezcfwmoxyc( a[index+6144]) ;
    df_ktdaezcfwmoxyc( a[index+6528]) ;
    df_ktdaezcfwmoxyc( a[index+6912]) ;
    df_ktdaezcfwmoxyc( a[index+7296]) ;
    df_ktdaezcfwmoxyc( a[index+7680]) ;
    df_ktdaezcfwmoxyc( a[index+8064]) ;
    df_ktdaezcfwmoxyc( a[index+8448]) ;
    df_ktdaezcfwmoxyc( a[index+8832]) ;
    df_ktdaezcfwmoxyc( a[index+9216]) ;
    df_ktdaezcfwmoxyc( a[index+9600]) ;
    if ( index < 40 )
        df_ktdaezcfwmoxyc( a[index+9984]) ;
}
```

and corresponding ptx code

```
.version 5.0
.target sm_30
.address_size 64
.visible .entry gf_ktdaezcfwmoxyc(
	.param .u64 gf_ktdaezcfwmoxyc_param_0
)
.maxntid 192, 1, 1
{
	.reg .pred 	%p<2>;
	.reg .b32 	%r<7>;
	.reg .b64 	%rd<5>;


	ld.param.u64 	%rd2, [gf_ktdaezcfwmoxyc_param_0];
	cvta.to.global.u64 	%rd3, %rd2;
	mov.u32 	%r1, %ctaid.x;
	mov.u32 	%r2, %ntid.x;
	mov.u32 	%r3, %tid.x;
	mad.lo.s32 	%r4, %r1, %r2, %r3;
	mul.wide.u32 	%rd4, %r4, 4;
	add.s64 	%rd1, %rd3, %rd4;
	mov.u32 	%r5, 1065353216;
	st.global.u32 	[%rd1], %r5;
	st.global.u32 	[%rd1+1536], %r5;
	st.global.u32 	[%rd1+3072], %r5;
	st.global.u32 	[%rd1+4608], %r5;
	st.global.u32 	[%rd1+6144], %r5;
	st.global.u32 	[%rd1+7680], %r5;
	st.global.u32 	[%rd1+9216], %r5;
	st.global.u32 	[%rd1+10752], %r5;
	st.global.u32 	[%rd1+12288], %r5;
	st.global.u32 	[%rd1+13824], %r5;
	st.global.u32 	[%rd1+15360], %r5;
	st.global.u32 	[%rd1+16896], %r5;
	st.global.u32 	[%rd1+18432], %r5;
	st.global.u32 	[%rd1+19968], %r5;
	st.global.u32 	[%rd1+21504], %r5;
	st.global.u32 	[%rd1+23040], %r5;
	st.global.u32 	[%rd1+24576], %r5;
	st.global.u32 	[%rd1+26112], %r5;
	st.global.u32 	[%rd1+27648], %r5;
	st.global.u32 	[%rd1+29184], %r5;
	st.global.u32 	[%rd1+30720], %r5;
	st.global.u32 	[%rd1+32256], %r5;
	st.global.u32 	[%rd1+33792], %r5;
	st.global.u32 	[%rd1+35328], %r5;
	st.global.u32 	[%rd1+36864], %r5;
	st.global.u32 	[%rd1+38400], %r5;
	setp.gt.u32	%p1, %r4, 39;
	@%p1 bra 	BB0_2;

	st.global.u32 	[%rd1+39936], %r5;

BB0_2:
	ret;
}
```
and run the code with dimension setup of Grids ( 2, 1, 1 ) and Blocks ( 192, 1, 1 ).



### exmaple for argument(s) passing mapping

To implement 
> A = x+y;

where A is an array of size 10024 and x and y are two runtime arguments, we can hijack `x` and `y` as two MACROs, and feed them to the map function

```c++
unsigned long n = 10024;
float x = rand();
float y = rand();
float * A = ...;
cumar::map()("x", x, "y", y)("[](float &a){ a = x+y; }")(A, A+n);
```

### example for arbitary number array mapping

To implement

> A = B + C - D + E + 1.0;

where A, B, C, D and E are array of size 10024, we can code it this way

```
cumar::map()()("[](float& a, float b, float c, float d, fload e){ a = b + c - d + e + 1.0f; } ")(A, A+n, B, C, D, E);
```



### __Reduce__ (without initial value)

To reduce the maximum value of an array 

> mx = MAX(A)

where A is of size 1111111, we can code it this way

```c++
unsigned long const n = 1111111;
double* A = ...;
double red = cumar::reduce()()( "[]( double a, double b ){ return a>b?a:b; }" )( A, A+n );
```
with a chipset model of `NVIDIA GeForce GT 750M`, it will generate CUDA code


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


It is also possible to pass argument(s) when calling `cumar::reduce()`, similiar as `cumar::map()`, just put them in the second brackets. Below is an example summing up all the elements in the array with and extra element `alpha`

```
double alpha = rand()
cumar::map()("alpha", alpha)( "[](double a, double b){return a+b+lambda;}" )( A, A+n );
``` 

### More examples

Please the test files located under folder `./test`









