
__device__ __forceinline__ double dr_xhvpqsohikrkyd( double a, double b ){ return a>b?a:b; }


extern "C" __global__  __launch_bounds__ (1024) void gr_xhvpqsohikrkyd(const double * __restrict__ input, double * __restrict__ output)
{
    extern __shared__ double shared_cache[];

    unsigned long const thread_index_in_current_block = threadIdx.x;
    unsigned long const thread_index = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned long const start_index = thread_index;

    double current_thread_reduction = input[start_index];//thread and block configuration guarantees boundary condition here

    current_thread_reduction = dr_xhvpqsohikrkyd( current_thread_reduction, input[start_index+143360] );
    current_thread_reduction = dr_xhvpqsohikrkyd( current_thread_reduction, input[start_index+286720] );
    current_thread_reduction = dr_xhvpqsohikrkyd( current_thread_reduction, input[start_index+430080] );
    current_thread_reduction = dr_xhvpqsohikrkyd( current_thread_reduction, input[start_index+573440] );
    current_thread_reduction = dr_xhvpqsohikrkyd( current_thread_reduction, input[start_index+716800] );
    current_thread_reduction = dr_xhvpqsohikrkyd( current_thread_reduction, input[start_index+860160] );
    if ( start_index < 1111111 - 1003520 )
        current_thread_reduction = dr_xhvpqsohikrkyd( current_thread_reduction, input[start_index+1003520] );


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
