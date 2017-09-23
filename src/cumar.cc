#include "./warnings.hpp"

SUPPRESS_WARNINGS


#include "../include/cumar_misc.hpp"

#include <cuda.h>
#include <nvrtc.h>
#include <cuda_runtime.h>

#include <map>
#include <cassert>
#include <functional>
#include <fstream>
#include <streambuf>
#include <iostream>
#include <utility>
#include <vector>
#include <sstream>
#include <iterator>
#include <tuple>
#include <cstdio>
#include <string>

struct cumar_result_assert
{
    void operator()( const cudaError_t& result, const char* const file, const unsigned long line ) const
    {
        if ( 0 == result ) { return; }
        fprintf( stderr, "%s:%lu: cudaError occured:\n[[ERROR]]: %s\n", file, line, cudaGetErrorString(result) );
        abort();
    }

    void operator()( const CUresult& result, const char* const file, const unsigned long line ) const
    {
        if ( 0 == result ) { return; }

        const char* msg;
        cuGetErrorString( result, &msg );
        const char* name;
        cuGetErrorName( result, &name );

        fprintf( stderr, "%s:%lu: CUresult error occured:\n[[ERROR]]: %s --- %s\n", file, line, name, msg );
        abort();
    }

    void operator()( const nvrtcResult& result, const char* const file, const unsigned long line ) const
    {
        if ( 0 == result ) { return; }

        fprintf( stderr, "%s:%lu: nvrtcResult error occured:\n[[ERROR]]: %s\n", file, line, nvrtcGetErrorString(result) );
        abort();
    }
};

#ifdef cumar_assert
#undef cumar_assert
#endif

#define cumar_assert(result) cumar_result_assert{}(result, __FILE__, __LINE__)

int cumar_get_cores_per_processor()
{
    auto&& converter = []( int major, int minor )
    {
        typedef struct
        {
            int SM; // 0xMm (hexidecimal notation), M = SM Major version, and m = SM minor version
            int Cores;
        } sSMtoCores;
        sSMtoCores nGpuArchCoresPerSM[] =
        {
            { 0x20, 32 }, // Fermi Generation (SM 2.0) GF100 class
            { 0x21, 48 }, // Fermi Generation (SM 2.1) GF10x class
            { 0x30, 192}, // Kepler Generation (SM 3.0) GK10x class
            { 0x32, 192}, // Kepler Generation (SM 3.2) GK10x class
            { 0x35, 192}, // Kepler Generation (SM 3.5) GK11x class
            { 0x37, 192}, // Kepler Generation (SM 3.7) GK21x class
            { 0x50, 128}, // Maxwell Generation (SM 5.0) GM10x class
            { 0x52, 128}, // Maxwell Generation (SM 5.2) GM20x class
            { 0x53, 128}, // Maxwell Generation (SM 5.3) GM20x class
            { 0x60, 64 }, // Pascal Generation (SM 6.0) GP100 class
            { 0x61, 128}, // Pascal Generation (SM 6.1) GP10x class
            { 0x62, 128}, // Pascal Generation (SM 6.2) GP10x class
            {   -1, -1 }
        };
        int index = 0;

        while ( nGpuArchCoresPerSM[index].SM != -1 )
        {
            if ( nGpuArchCoresPerSM[index].SM == ( ( major << 4 ) + minor ) )
                return nGpuArchCoresPerSM[index].Cores;

            index++;
        }

        assert( !"Failed to get arch cores for current GPU!");
        return 0;
    };

    int current_id;
    cumar_assert( cudaGetDevice( &current_id ) );
    cudaDeviceProp prop;
    cumar_assert(cudaGetDeviceProperties( &prop, current_id ));
    return converter( prop.major, prop.minor );
}

void cumar_reset_device()
{
    cumar_assert( cudaDeviceReset() );
}

int cumar_get_processors()
{
    int current_id;
    cumar_assert( cudaGetDevice( &current_id ) );
    cudaDeviceProp prop;
    cumar_assert(cudaGetDeviceProperties( &prop, current_id ));
    return prop.multiProcessorCount;
}

unsigned long long cumar_get_memory_in_bytes()
{
    int current_id;
    cumar_assert( cudaGetDevice( &current_id ) );
    cudaDeviceProp prop;
    cumar_assert(cudaGetDeviceProperties( &prop, current_id ));
    return static_cast<unsigned long long>(prop.totalGlobalMem);
}

int cumar_get_major_capability()
{
    int current_id;
    cumar_assert( cudaGetDevice( &current_id ) );
    cudaDeviceProp prop;
    cumar_assert(cudaGetDeviceProperties( &prop, current_id ));
    return prop.major;
}

int cumar_get_minor_capability()
{
    int current_id;
    cumar_assert( cudaGetDevice( &current_id ) );
    cudaDeviceProp prop;
    cumar_assert(cudaGetDeviceProperties( &prop, current_id ));
    return prop.minor;
}

void cumar_set_device( int id )
{
    int current_id;
    cumar_assert( cudaGetDevice( &current_id ) );

    if ( current_id != id )
        cumar_assert( cudaSetDevice( id ) );
}

int cumar_get_device()
{
    int current_id;
    cumar_assert( cudaGetDevice( &current_id ) );
    return current_id;
}

int cumar_device_count()
{
    int count;
    cumar_assert( cuDeviceGetCount( &count ) );
    return count;
}

int cumar_get_max_thread_x()
{
    int current_id;
    cumar_assert( cudaGetDevice( &current_id ) );
    cudaDeviceProp prop;
    cumar_assert(cudaGetDeviceProperties( &prop, current_id ));
    return prop.maxThreadsDim[0];
}

void cumar_allocate( void** p, unsigned long n )
{
    cumar_assert( cudaMalloc( p, n ) );
    cumar_assert( cudaMemset( *p, 0, n ) );
}

void cumar_managed_allocate( void** p, unsigned long n )
{
    cumar_assert( cudaMallocManaged( p, n ) );
    cumar_assert( cudaMemset( *p, 0, n ) );
}

void cumar_deallocate( void* p )
{
    cumar_assert( cudaFree( p ) );
}

void cumar_memcopy_host_to_device( const void* src, unsigned long n, void* dst )
{
    cumar_assert( cudaMemcpy( dst, src, n, cudaMemcpyHostToDevice  ) );
}

void cumar_memcopy_device_to_host( const void* src, unsigned long n, void* dst )
{
    cumar_assert( cudaMemcpy( dst, src, n, cudaMemcpyDeviceToHost  ) );
}

void cumar_device_synchronize()
{
    cumar_assert( cudaDeviceSynchronize() );
}

static std::string const make_file_name( std::string const& source )
{
    auto id = std::hash<std::string>{}(source);
    auto const& generator = []( unsigned long id ) -> char { return static_cast<char>(id+'a'); };
    std::string ans;
    for ( auto id = std::hash<std::string>{}(source); id != 0; id /= 26 )
        ans.push_back( generator(id % 26) );

    return ans;
}

void nvrtc_make_ptx( std::string const& source, std::string& ptx )
{
    std::string option_1{ "--gpu-architecture=compute_XX" };
    *(option_1.rbegin()) = '0' + cumar_get_minor_capability();
    *(option_1.rbegin()+1) = '0' + cumar_get_major_capability();
    //std::string option_1{ "--gpu-architecture=compute_30" };
    std::string option_2{ "--use_fast_math" };
    std::string option_3{ "--std=c++11" };
    std::string option_4{ "--debug" };
    std::string option_5{ "--device-debug" };
    std::string const& file_name = std::string{"./ptx/"} + make_file_name( source+option_1 ) + std::string{".cu"};
    std::string const& ptx_name = file_name + std::string{".ptx"};

    static std::map< std::string, std::string > ptx_cache; // <- look up from cache
    if ( auto it = ptx_cache.find( ptx_name ); it != ptx_cache.end() )
    {
        //std::cout << "Loading Generated ptx Code\n";
        ptx = (*it).second;
        return;
    }

#ifdef DEBUG
    // Typical compilation time is around 1
    {//check if this file has been compiled
        std::ifstream ifs( ptx_name.c_str() );
        if ( ifs.good() )
        {
            std::cout << "Loading compiled ptx file from " << ptx_name << "\n";
            std::string str{ std::istreambuf_iterator<char>{ifs}, std::istreambuf_iterator<char>{} };
            ptx.swap(str);
            return;
        }
    }
    {
        std::cout << "Saving generated cuda file to " << file_name << "\n";
        //std::cout << source << "\n\n";
        std::ofstream ofs( file_name.c_str() );

        if ( ofs.good() )
        {
            ofs << source;
            std::cout << "Done.\n";
        }
        else
            std::cout << "Failed to save.\n";
    }
#endif

    nvrtcProgram prog;
    cumar_assert( nvrtcCreateProgram( &prog, source.c_str(), file_name.c_str(), 0, 0, 0 ) );
    char const* options[] = { option_1.c_str(), option_2.c_str(), option_3.c_str() };
    cumar_assert( nvrtcCompileProgram( prog, sizeof(options)/sizeof(options[0]), options ) );
    size_t sz;
    cumar_assert( nvrtcGetPTXSize( prog, &sz ) );
    ptx.resize( sz );
    cumar_assert( nvrtcGetPTX( prog, &ptx[0] ) );
    cumar_assert( nvrtcDestroyProgram( &prog ) );
    ptx_cache[ptx_name] = ptx; // saving to cache

#ifdef DEBUG
    {
        std::cout << "Saving compiled ptx file to " << ptx_name << "\n";
        std::ofstream ofs( ptx_name.c_str() );

        if ( ofs.good() )
        {
            ofs << ptx;
            std::cout << "Done.\n";
        }
        else
            std::cout << "Failed to save.\n";
    }
#endif
}

void make_nvrtc_launcher( char const* const ptx, char const* const func, int gx, int gy, int gz, int tx, int ty, int tz, void** args, int shared_memory_in_byte )
{
    assert( ptx && "Empty ptx code!" );
    assert( func && "Empty kernel function!" );
    assert( gx > 0 && "Grid dimension x not positive!" );
    assert( gy > 0 && "Grid dimension y not positive!" );
    assert( gz > 0 && "Grid dimension z not positive!" );
    assert( tx > 0 && "Thread dimension x not positive!" );
    assert( ty > 0 && "Thread dimension y not positive!" );
    assert( tz > 0 && "Thread dimension z not positive!" );
    assert( args && "Null arguments!" );
    assert( shared_memory_in_byte >= 0 && "Negative shared memory size!" );

    CUdevice cuDevice;
    CUcontext context;
    CUmodule module;
    CUfunction kernel;
    cumar_assert( cuInit( 0 ) );
    cumar_assert( cuDeviceGet( &cuDevice, 0 ) );
    cumar_assert( cuCtxCreate( &context, 0, cuDevice ) );
    cumar_assert( cuModuleLoadDataEx( &module, ptx, 0, 0, 0 ) );
    cumar_assert( cuModuleGetFunction( &kernel, module, func ) );
    cumar_assert( cuLaunchKernel( kernel, gx, gy, gz, tx, ty, tz, shared_memory_in_byte, 0, args, nullptr ) );
    cumar_assert( cuCtxSynchronize() );
    cumar_assert( cuModuleUnload( module ) );
    cumar_assert( cuCtxDestroy( context ) );
}

static std::string generate_device_function( std::string const& operation_code_, std::string const& device_function_name_ )
{
    std::string const heading_device_code{ "__device__ __forceinline__  void \n" };

    unsigned long const capture_position = operation_code_.find( "(" );
    assert( capture_position != std::string::npos ); // <- must find bracket
    std::string const rest_device_code{ operation_code_, capture_position };
    assert( rest_device_code.size() );

    return heading_device_code + device_function_name_ + rest_device_code + std::string{"\n\n"};
}

static void split_string( std::string const& s_, char delim_, std::vector<std::string>& elems_ )
{
    elems_.clear();
    std::stringstream ss;
    ss.str( s_ );
    std::string tmp;
    while ( std::getline( ss, tmp, delim_ ) )
    {
        //collect trimed  args
        auto const& l_itor = std::find_if_not( tmp.begin(), tmp.end(), [](int c){return std::isspace(c);} );
        elems_.push_back( std::string( l_itor, std::find_if_not( tmp.rbegin(), std::string::reverse_iterator(l_itor), [](int c){return std::isspace(c);} ).base() ) );
    }
}

// "double xx" ----> "double* __restrict__ xx"
static void decorate_argument( std::string& elem_ )
{
    unsigned long const pos = elem_.find( " " );
    assert( pos != std::string::npos && "No blank found in argument pair" );
    elem_ = std::string{ elem_.begin(),  elem_.begin()+pos } + std::string{"* __restrict__ "} + std::string{ elem_.begin()+pos, elem_.end() };
}

std::tuple<std::string, std::string, std::string> make_map_code( std::string const& operation_code_, unsigned long length_, unsigned long grids_, unsigned long blocks_, unsigned long operations_ )
{
    assert( length_ && "length of the array is zero" );
    assert( grids_ && "grid is zero" );
    assert( blocks_ && "block is zero" );
    assert( operations_ && "operation is zero" );

    bool const branch_pred = (grids_ * blocks_ * operations_) != length_ ? true : false; // <- branch prediction required or not

    std::string const& operation_code_hash = make_file_name( operation_code_ );
    std::string const& device_function_name = std::string{"df_"} + operation_code_hash;
    std::string const& device_function_code = generate_device_function( operation_code_, device_function_name );

    //global function decorations
    std::string const& global_function_name = std::string{"gf_"} + operation_code_hash;

    std::string global_code{ "extern \"C\"\n__global__ void "};

    global_code += std::string{" __launch_bounds__ ( "} + std::to_string(blocks_) + std::string{" )\n"};

    global_code += global_function_name;
    global_code += std::string{"("};

    //take out arguments
    unsigned long const start_pos = operation_code_.find( "(" );
    assert( start_pos != std::string::npos);
    unsigned long const end_pos = operation_code_.find( ")" );
    assert( end_pos != std::string::npos );
    std::string argument_list{ operation_code_.begin()+start_pos+1, operation_code_.begin()+end_pos }; // <- "double& x_0, double x_1, double x_2, double x_3"
    assert( std::count( argument_list.begin(), argument_list.end(), '&' ) && "No operations" );
    std::for_each( argument_list.begin(), argument_list.end(), []( char& ch ){ if ( ch == '&' ) ch = ' '; } ); // remove '&'

    std::vector<std::string> elems;
    split_string( argument_list, ',', elems ); // <- elems is "double x_0", "double x_1", "double x_2", "double x_3"

    for ( auto& elem : elems ) // <- "double* __restrict__ x_0", "double* __ restrict__ x_1", ...
    {
        decorate_argument( elem );
        global_code += elem;
        global_code += std::string{", "};
    }

    *(global_code.rbegin()+1) = ')';
    global_code += std::string{"\n{\n"};

    std::string const operations = std::to_string( operations_ );
    std::string const index = std::string{"    unsigned long const const index = (blockDim.x * blockIdx.x + threadIdx.x);\n"};
    global_code += index;

    //  <- "x_0", "x_1", ...
    for ( auto& elem : elems )
    {
            unsigned long pos = elem.find_last_of( " " );
            assert( pos != std::string::npos && "Failed to extract black" );
            elem = std::string{ elem.begin() + pos, elem.end() };
    }

    for ( unsigned long current_op = 0; current_op != operations_; ++current_op )
    {
        std::string const offset_str = std::to_string( current_op * grids_ * blocks_ );

        if ( branch_pred && (current_op == (operations_-1)) )
            global_code += std::string{ "    if ( index" } + std::string{" < "} + std::to_string(length_- current_op*grids_*blocks_) + std::string{ " )\n    " };
            //global_code += std::string{ "    if ( index + " } + offset_str + std::string{" < "} + std::to_string(length_) + std::string{ " )\n    " };


        std::string ptr_offset = std::string{"[index+"} + offset_str + std::string{"]"};
        if ( current_op == 0 )
            ptr_offset = std::string{"[index"} + std::string{"]"};

        global_code += std::string{"    "} + device_function_name + std::string{"("};
        for ( auto const& elem : elems )
            global_code += elem + ptr_offset + std::string{", "};
        *(global_code.rbegin()+1) = ')'; // <- replace the last ',' with ')'
        global_code += std::string{";\n"};
    }
    global_code += std::string{"}\n\n"};

    return std::make_tuple( device_function_code, global_code, global_function_name);

}// make_map_code

static std::string isolate_type( std::string const& operation_code_ )
{
    assert( operation_code_.size() && "Empty function code!" );

    unsigned long const capture_position = operation_code_.find( "(" );
    assert( capture_position != std::string::npos && "No function arguments starter bracket found" ); // <- must find bracket

    unsigned long arg_position = capture_position + 1;
    while ( operation_code_[arg_position] == ' ' )
        ++arg_position;

    unsigned long comma_position = operation_code_.find( "," ) - 1;
    assert( comma_position != std::string::npos && "No comma found in device function, two arguments required!" );
    while( operation_code_[comma_position] == ' ' )
        --comma_position;
    assert( comma_position > arg_position );

    unsigned long blank_postion = arg_position;
    while( operation_code_[blank_postion] != ' ' )
        ++blank_postion;
    assert( blank_postion < comma_position );

    return std::string{ operation_code_.begin()+arg_position, operation_code_.begin()+blank_postion };
}

static std::string generate_reduce_device_function( std::string const& operation_code_, std::string const& device_function_name_ )
{
    assert( operation_code_.size() && "Empty device function!" );
    assert( device_function_name_.size() && "Empty device function name!" );
    std::string const heading_device_code{ "__device__ __forceinline__ " };

    unsigned long const capture_position = operation_code_.find( "(" );
    assert( capture_position != std::string::npos ); // <- must find bracket

    std::string const return_string = isolate_type( operation_code_ );

    std::string const rest_device_code{ operation_code_, capture_position };
    assert( rest_device_code.size() );

    return heading_device_code + return_string + std::string{" "} + device_function_name_ + rest_device_code + std::string{"\n\n"};
}

static void replace_all( std::string& str_, std::string const& from_, std::string const& to_ )
{
    if ( from_.empty() )
        return;

    unsigned long start_pos = 0;

    while ( ( start_pos = str_.find( from_, start_pos ) ) != std::string::npos )
    {
        str_.replace( start_pos, from_.length(), to_ );
        start_pos += to_.length(); // In case 'to' contains 'from', like replacing 'x' with 'yx'
    }
}

std::tuple<std::string, std::string, std::string> make_reduce_code( std::string const& operation_code_, unsigned long length_, unsigned long grids_, unsigned long blocks_, unsigned long operations_ )
{
    assert( operation_code_.size() && "Empty device function!" );
    assert( length_ && "length of the array is zero!" );
    assert( grids_ && "grid is zero!" );
    assert( blocks_ && "block is zero!" );

    std::string const& operation_code_hash = make_file_name( operation_code_ );
    std::string const& device_function = std::string{"dr_"} + operation_code_hash;
    std::string const& device_function_code = generate_reduce_device_function( operation_code_, device_function );

    //global function decorations
    std::string const& host_function = std::string{"gr_"} + operation_code_hash;
    std::string const& length = std::to_string( length_ );
    std::string const& blocks = std::to_string( blocks_ );
    std::string const& grids = std::to_string( grids_ );
    std::string const& stride = std::to_string( blocks_*grids_ );
    std::string const& working_type = isolate_type( operation_code_ );

    std::string const safe_single_pass_stride_code_template
    {
        "    current_thread_reduction = DEVICE_FUNCTION( current_thread_reduction, input[start_index+CURRENT_STRIDE] );\n"
    };

    std::string single_pass_stride_code_template
    {
        "    if ( start_index < LENGTH - CURRENT_STRIDE )\n"
        "        current_thread_reduction = DEVICE_FUNCTION( current_thread_reduction, input[start_index+CURRENT_STRIDE] );\n"
    };

    std::string stride_code;
    for ( unsigned long i = 1; i != operations_-1; ++i )
    {
        std::string single_pass_stride_code = safe_single_pass_stride_code_template;
        std::string const& current_stride = std::to_string( i * blocks_ * grids_ );
        replace_all( single_pass_stride_code, std::string{"CURRENT_STRIDE"}, current_stride );
        stride_code += single_pass_stride_code;
    }
    if ( operations_ >= 2 ) // always true?
    {
        std::string const& current_stride = std::to_string( (operations_-1) * blocks_ * grids_ );
        replace_all( single_pass_stride_code_template, std::string{"CURRENT_STRIDE"}, current_stride );
        stride_code += single_pass_stride_code_template;
    }

    std::string global_code
    {
        "extern \"C\" __global__  __launch_bounds__ (BLOCKS) void HOST_FUNCTION(const WORKING_TYPE * __restrict__ input, WORKING_TYPE * __restrict__ output)\n"
        "{\n"
        "    extern __shared__ WORKING_TYPE shared_cache[];\n"
        "\n"
        "    unsigned long const thread_index_in_current_block = threadIdx.x;\n"
        "    unsigned long const thread_index = blockIdx.x * blockDim.x + threadIdx.x;\n"
        "    unsigned long const start_index = thread_index;\n"
        "\n"
        "    WORKING_TYPE current_thread_reduction = input[start_index];//thread and block configuration guarantees boundary condition here\n"
        "\n"
        "STRIDE_CODE\n"
        "\n"
        "    shared_cache[thread_index_in_current_block] = current_thread_reduction;\n"
        "\n"
        "    __syncthreads();\n"
        "\n"
        "    if (BLOCKS > 1024)\n"
        "    {\n"
        "        if ( (thread_index_in_current_block < 1024) && (thread_index_in_current_block+1024 < BLOCKS) )\n"
        "            shared_cache[thread_index_in_current_block] = DEVICE_FUNCTION( shared_cache[thread_index_in_current_block], shared_cache[thread_index_in_current_block+1024] );\n"
        "        __syncthreads();\n"
        "    }\n"
        "\n"
        "    if (BLOCKS > 512)\n"
        "    {\n"
        "        if ( (thread_index_in_current_block < 512) && (thread_index_in_current_block+512 < BLOCKS) )\n"
        "            shared_cache[thread_index_in_current_block] = DEVICE_FUNCTION( shared_cache[thread_index_in_current_block], shared_cache[thread_index_in_current_block+512] );\n"
        "        __syncthreads();\n"
        "    }\n"
        "\n"
        "    if (BLOCKS > 256)\n"
        "    {\n"
        "        if ( (thread_index_in_current_block < 256) && (thread_index_in_current_block+256 < BLOCKS) )\n"
        "            shared_cache[thread_index_in_current_block] = DEVICE_FUNCTION( shared_cache[thread_index_in_current_block], shared_cache[thread_index_in_current_block+256] );\n"
        "        __syncthreads();\n"
        "    }\n"
        "\n"
        "    if (BLOCKS > 128)\n"
        "    {\n"
        "        if ( (thread_index_in_current_block < 128) && (thread_index_in_current_block+128 < BLOCKS) )\n"
        "            shared_cache[thread_index_in_current_block] = DEVICE_FUNCTION( shared_cache[thread_index_in_current_block], shared_cache[thread_index_in_current_block+128] );\n"
        "        __syncthreads();\n"
        "    }\n"
        "\n"
        "    if (BLOCKS > 64)\n"
        "    {\n"
        "        if ( (thread_index_in_current_block < 64) && (thread_index_in_current_block+64 < BLOCKS) )\n"
        "            shared_cache[thread_index_in_current_block] = DEVICE_FUNCTION( shared_cache[thread_index_in_current_block], shared_cache[thread_index_in_current_block+64] );\n"
        "        __syncthreads();\n"
        "    }\n"
        "\n"
        "    if (BLOCKS > 32)\n"
        "    {\n"
        "        if ( (thread_index_in_current_block < 32) && (thread_index_in_current_block+32 < BLOCKS) )\n"
        "            shared_cache[thread_index_in_current_block] = DEVICE_FUNCTION( shared_cache[thread_index_in_current_block], shared_cache[thread_index_in_current_block+32] );\n"
        "        __syncthreads();\n"
        "    }\n"
        "\n"
        "    if ( (BLOCKS > 16) && (thread_index_in_current_block < 16) && (thread_index_in_current_block+16 < BLOCKS) )\n"
        "            shared_cache[thread_index_in_current_block] = DEVICE_FUNCTION( shared_cache[thread_index_in_current_block], shared_cache[thread_index_in_current_block+16] );\n"
        "\n"
        "    if ( (BLOCKS > 8) && (thread_index_in_current_block < 8) && (thread_index_in_current_block+8 < BLOCKS) )\n"
        "            shared_cache[thread_index_in_current_block] = DEVICE_FUNCTION( shared_cache[thread_index_in_current_block], shared_cache[thread_index_in_current_block+8] );\n"
        "\n"
        "    if ( (BLOCKS > 4) && (thread_index_in_current_block < 4) && (thread_index_in_current_block+4 < BLOCKS) )\n"
        "            shared_cache[thread_index_in_current_block] = DEVICE_FUNCTION( shared_cache[thread_index_in_current_block], shared_cache[thread_index_in_current_block+4] );\n"
        "\n"
        "    if ( (BLOCKS > 2) && (thread_index_in_current_block < 2) && (thread_index_in_current_block+2 < BLOCKS) )\n"
        "            shared_cache[thread_index_in_current_block] = DEVICE_FUNCTION( shared_cache[thread_index_in_current_block], shared_cache[thread_index_in_current_block+2] );\n"
        "\n"
        "    if ( (BLOCKS > 1) && (thread_index_in_current_block < 1) && (thread_index_in_current_block+1 < BLOCKS) )\n"
        "            shared_cache[thread_index_in_current_block] = DEVICE_FUNCTION( shared_cache[thread_index_in_current_block], shared_cache[thread_index_in_current_block+1] );\n"
        "\n"
        "    if (thread_index_in_current_block == 0) output[blockIdx.x] = shared_cache[0];\n"
        "}\n"
    };

    replace_all( global_code, std::string{ "STRIDE_CODE" }, stride_code );
    replace_all( global_code, std::string{ "HOST_FUNCTION" }, host_function );
    replace_all( global_code, std::string{ "WORKING_TYPE" },  working_type );
    replace_all( global_code, std::string{ "BLOCKS" },  blocks );
    replace_all( global_code, std::string{ "LENGTH" },  length );
    replace_all( global_code, std::string{ "DEVICE_FUNCTION" },  device_function );

    return std::make_tuple( device_function_code, global_code, host_function );
}// make_map_code

namespace cumar
{

    struct timer::events
    {
        cudaEvent_t start;
        cudaEvent_t stop;
    };

    timer::timer( std::string const& start_info_ ) : ev( std::make_unique<events>() )
    {
        std::cout << "CUDA timer started ";
        if ( start_info_.size() )
            std::cout << " with info: " << start_info_;
        std::cout << "\n";
        cumar_assert( cudaEventCreate( &((*ev).start) ) );
        cumar_assert( cudaEventCreate( &((*ev).stop) ) );
        cumar_assert( cudaEventRecord( (*ev).start, 0 ) );
    }

    timer::~timer()
    {
        cumar_assert( cudaEventRecord( (*ev).stop, 0 ) );
        cumar_assert( cudaThreadSynchronize() );
        float elapsed_time;
        cumar_assert( cudaEventElapsedTime( &elapsed_time, (*ev).start, (*ev).stop ) );
        cumar_assert( cudaEventDestroy( (*ev).start ) );
        cumar_assert( cudaEventDestroy( (*ev).stop ) );
        std::cout << "Elapsed Time " << elapsed_time << " ms.\n";
    }

}

#undef cumar_assert


RESTORE_WARNINGS

