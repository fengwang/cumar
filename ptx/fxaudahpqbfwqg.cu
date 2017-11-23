
__device__ __forceinline__  void 
df_jsigstfmqkjxv(double a, double b, double& c){ c = a + a*b - b; }


extern "C"
__global__ void  __launch_bounds__ ( 128 )
gf_jsigstfmqkjxv(double* __restrict__  a, double* __restrict__  b, double* __restrict__   c) 
{
    unsigned long const const index = (blockDim.x * blockIdx.x + threadIdx.x);
    df_jsigstfmqkjxv( a[index],  b[index],  c[index]) ;
    if ( index < 3431 )
        df_jsigstfmqkjxv( a[index+7680],  b[index+7680],  c[index+7680]) ;
}

