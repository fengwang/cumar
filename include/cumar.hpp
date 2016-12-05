#if 0
Copyright (c) 2016, Feng Wang (wang_feng@live.com)
All rights reserved.

Redistribution and use in source and binary forms, with or without modification, are permitted provided that the following conditions are met:

1. Redistributions of source code must retain the above copyright notice, this list of conditions and the following disclaimer.

2. Redistributions in binary form must reproduce the above copyright notice, this list of conditions and the following disclaimer in the documentation and/or other materials provided with the distribution.

3. Neither the name of the copyright holder nor the names of its contributors may be used to endorse or promote products derived from this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#endif
#ifndef DWJUWOBYHOTAGIAAHOTWEGPBLECMWYVTUVGJIDGRUCQRVQEVWSGTVWIICNONGWSPSRIOMJUKQ
#define DWJUWOBYHOTAGIAAHOTWEGPBLECMWYVTUVGJIDGRUCQRVQEVWSGTVWIICNONGWSPSRIOMJUKQ

#include <algorithm>
#include <array>
#include <cassert>
#include <cmath>
#include <functional>
#include <memory>
#include <sstream>
#include <string>
#include <tuple>
#include <type_traits>
#include <utility>

namespace cumar
{

    namespace cumar_private
    {
        template <typename Arg, typename... Args>
        struct overloader : overloader<Arg>, overloader<Args...>
        {
            overloader( Arg a_, Args... b_ ) noexcept : overloader<Arg>( a_ ), overloader<Args...>( b_... ) {}
        };

        template <typename Arg>
        struct overloader<Arg> : Arg
        {
            overloader( Arg a_ ) noexcept : Arg( a_ ) {}
        };

        template <typename ... Overloaders>
        auto make_overloader( Overloaders ... overloader_ ) noexcept
        {
            return overloader<Overloaders...>( overloader_... );
        }

        template <typename T, typename U>
        T const lexical_cast( U const& from )
        {
            T var;

            std::stringstream ss;
            ss << from;
            ss >> var;

            return var;
        }
    };

    inline int get_cores_per_processor()
    {
        extern int cumar_get_cores_per_processor();
        return cumar_get_cores_per_processor();
    }

    inline int get_processors()
    {
        extern int cumar_get_processors();
        return cumar_get_processors();
    }

    inline int get_cores()
    {
        return get_cores_per_processor() * get_processors();
    }

    inline int get_max_thread_x()
    {
        extern int cumar_get_max_thread_x();
        return cumar_get_max_thread_x();
    }

    //returns [grid size] [block size] [operations per thread]
    inline std::tuple<int,int,int> const make_map_configuration( int N_ )
    {
        assert( N_ > 0 );
        int const operation_threshold = 128;
        int const blocks = get_cores_per_processor();
        int const grids = get_processors();
        int const cores = get_cores();

        if ( cores > N_ )
        {
            int const operations = 1;
            int const grids = (N_+blocks-1) / blocks;
            return std::make_tuple( grids, blocks, operations );
        }

        auto const& gen_operations = []( int N, int grids, int blocks ) noexcept
        {
            long long const gb = grids * blocks;
            long long const op = ( gb + N - 1 ) / gb;
            return static_cast<int>( op );
        };

        int actual_grids = grids;
        int actual_blocks = blocks;
        int actual_operations = gen_operations( N_, actual_grids, actual_blocks );

        while ( actual_operations > operation_threshold )
        {
            actual_grids += grids;
            actual_operations = gen_operations( N_, actual_grids, actual_blocks );
        }

        return std::make_tuple( actual_grids, actual_blocks, actual_operations );
    }

    template< typename T >
    inline T* allocate( unsigned long n_ )
    {
        void cumar_allocate( void**, unsigned long );
        T* ans;
        cumar_allocate( reinterpret_cast<void**>(&ans), n_*sizeof(T) );
        return ans;
    }

    template< typename T >
    inline T* managed_allocate( unsigned long n_ )
    {
        void cumar_managed_allocate( void**, unsigned long );
        T* ans;
        cumar_managed_allocate( reinterpret_cast<void**>(&ans), n_*sizeof(T) );
        return ans;
    }

    template< typename T >
    inline void deallocate( T* ptr_ )
    {
        void cumar_deallocate( void* );
        cumar_deallocate( reinterpret_cast<void*>(ptr_) );
    }

    inline void synchronize()
    {
        void cumar_device_synchronize();
        cumar_device_synchronize();
    }

    template< typename T >
    inline void host_to_device_copy( T* host_begin_, T* host_end_, T* device_begin_ )
    {
        void cumar_memcopy_host_to_device( const void*, unsigned long, void* );
        unsigned long const n = sizeof(T)*(host_end_-host_begin_);
        cumar_memcopy_host_to_device( reinterpret_cast<const void*>(host_begin_), n, reinterpret_cast<void*>(device_begin_) );
    }

    template< typename T >
    inline void device_to_host_copy( T* device_begin_, T* device_end_, T* host_begin_ )
    {
        void cumar_memcopy_device_to_host( const void*, unsigned long, void* );
        unsigned long const n = sizeof(T)*(device_end_-device_begin_);
        cumar_memcopy_device_to_host( reinterpret_cast<const void*>(device_begin_), n, reinterpret_cast<void*>(host_begin_) );
    }

    template< typename T >
    inline T* host_to_device_clone( T* host_begin_, T* host_end_ )
    {
        unsigned long const n = sizeof(T)*(host_end_-host_begin_);
        T* ans = allocate<T>(n);
        host_to_device_copy( host_begin_, host_end_, ans );
        return ans;
    }

    template< typename T >
    inline T* clone( T* host_begin_, T* host_end_ )
    {
        unsigned long const n = sizeof(T)*(host_end_-host_begin_);
        T* ans = managed_allocate<T>(n);
        device_to_device_copy( host_begin_, host_end_, ans );
        synchronize();
        return ans;
    }

    inline std::string const make_ptx( std::string const& source_code_ )
    {
        void nvrtc_make_ptx( std::string const&, std::string& );
        std::string ans;
        nvrtc_make_ptx( source_code_, ans );
        return ans;
    }

    inline auto make_launcher( std::string const& ptx, std::string const& kernel_name, int const shared_memory = 0 )
    {
        assert( shared_memory >= 0 && "Shared memory size cannot be negative!" );
        void make_nvrtc_launcher( char const* const ptx_, char const* const kernel_, int gx, int gy, int gz, int tx, int ty, int tz, void** args, int shared_memory_in_bytes );
        return [&]( auto&& ... dims ) noexcept
        {
            return [&]( auto&& ... args ) noexcept
            {
                auto&& ag = []( auto&& ... args_ ) noexcept { return std::array<void*,sizeof...(args_)>{ {reinterpret_cast<void*>(std::addressof(std::forward<decltype(args_)>(args_)))... }}; }( args... );
                make_nvrtc_launcher( ptx.c_str(), kernel_name.c_str(), dims..., &ag[0], shared_memory );
            };
        };
    }

    inline auto map( std::string const& predefinition_ = std::string{""} ) noexcept
    {
        return [=]( auto ... custom_defines_ ) noexcept // <- ( "id_1", value_1, "id_2", value_2, ... )  -> "custom_define id_1 value_1 \ncustom_define id_2 value_2\n"
        {
            return [=]( std::string const& lambda_code_ ) noexcept // <- operations
            {
                static_assert( (sizeof...(custom_defines_) & 1) == 0, "precaptured variables' types and values not match" );
                return [=]( auto&& first_, auto&& last_, auto&& ... rests_ ) noexcept // <- all iterators
                {
                    unsigned long length = last_ - first_;
                    std::string generated_macro;
                    auto const& composed_generator = cumar_private::make_overloader
                    (
                        [&generated_macro]( auto&& generator_, std::string const& id_, auto&& value_, auto&& ... rests_ ) noexcept
                        {
                            generated_macro += std::string{"#define "} + id_ + std::string{" "} + cumar_private::lexical_cast<std::string>(value_) + std::string{"\n"};
                            generator_( generator_, rests_... );
                        },
                        [&generated_macro]( auto ) noexcept { generated_macro += std::string{"\n"}; }
                    );
                    composed_generator( composed_generator, custom_defines_... ); // <- all macros generated
                    std::string generated_demacro;
                    auto const& composed_degenerator = cumar_private::make_overloader
                    (
                        [&generated_demacro]( auto&& generator_, std::string const& id_, auto&& value_, auto&& ... rests_ ) noexcept
                        {
                            generated_demacro += std::string{"#undef "} + id_  + std::string{"\n"};
                            generator_( generator_, rests_... );
                        },
                        [&generated_demacro]( auto ) noexcept { generated_demacro += std::string{"\n"}; }
                    );
                    composed_degenerator( composed_degenerator, custom_defines_... ); // <- all macros degenerated
                    auto const& config = make_map_configuration( length );
                    int const grids = std::get<0>( config );
                    int const blocks = std::get<1>( config );
                    int const operations = std::get<2>( config ); //TODO: limit operations in range[1,128]
                    std::tuple<std::string,std::string, std::string> make_map_code( std::string const& lambda_code_, unsigned long length_, unsigned long grids_, unsigned long blocks_, unsigned long operations_ );
                    auto const& device_global_kernel =  make_map_code( lambda_code_, length, grids, blocks, operations );
                    std::string const& code = predefinition_ + generated_macro + std::get<0>(device_global_kernel) + generated_demacro + std::get<1>(device_global_kernel);
                    std::string const& kernel = std::get<2>(device_global_kernel);
                    auto&& ptx = make_ptx( code );
                    auto&& launcher = make_launcher( ptx, kernel );
                    launcher( grids, 1, 1, blocks, 1, 1 )( first_, rests_... );
                    return ptx;
                }; // first_, last_, rests_...
            };// lambda_code_
        };// custom_defines_...
    }// predefinition_

    inline auto reduce( std::string const& predefinition_ = std::string{""} ) noexcept
    {
        return [=]( auto ... custom_defines_ ) noexcept // <- ( "id_1", value_1, "id_2", value_2, ... )  -> "custom_define id_1 value_1 \ncustom_define id_2 value_2\n"
        {
            return [=]( std::string const& lambda_code_ ) noexcept // <- operations
            {
                static_assert( (sizeof...(custom_defines_) & 1) == 0, "precaptured variables' types and values not match" );
                return [=]( auto&& first_, auto&& last_ ) noexcept // <- all iterators
                {
                    std::string generated_macro;
                    auto const& composed_generator = cumar_private::make_overloader
                    (
                        [&generated_macro]( auto&& generator_, std::string const& id_, auto&& value_, auto&& ... rests_ ) noexcept
                        {
                            generated_macro += std::string{"#define "} + id_ + std::string{" "} + cumar_private::lexical_cast<std::string>(value_) + std::string{"\n"};
                            generator_( generator_, rests_... );
                        },
                        [&generated_macro]( auto ) noexcept { generated_macro += std::string{"\n"}; }
                    );
                    composed_generator( composed_generator, custom_defines_... ); // <- all macros generated
                    std::string generated_demacro;
                    auto const& composed_degenerator = cumar_private::make_overloader
                    (
                        [&generated_demacro]( auto&& generator_, std::string const& id_, auto&& value_, auto&& ... rests_ ) noexcept
                        {
                            generated_demacro += std::string{"#undef "} + id_  + std::string{"\n"};
                            generator_( generator_, rests_... );
                        },
                        [&generated_demacro]( auto ) noexcept { generated_demacro += std::string{"\n"}; }
                    );
                    composed_degenerator( composed_degenerator, custom_defines_... ); // <- all macros degenerated
                    auto const& prev_power2 = []( unsigned long x ) noexcept // isolate the left-most 1-bit in x
                    {
                        x |= x >> 32;
                        x |= x >> 16;
                        x |= x >> 8;
                        x |= x >> 4;
                        x |= x >> 2;
                        x |= x >> 1;
                        x ^= x >> 1;
                        return x;
                    };
                    auto const& approx_sqrt = []( unsigned long x ) noexcept
                    {
                        unsigned long ans = std::max(x >> 1, 1UL);
                        for ( unsigned long i = 0; i != 10; ++i )
                            ans = ( ans + (x+ans-1) / ans ) >> 1;
                        return ans;
                    };
                    unsigned long const length = last_ - first_;
                    unsigned long const max_blocks = get_max_thread_x(); // TODO: need consider shared memory size here
                    unsigned long const operation_threshold = get_cores_per_processor();
                    unsigned long processors = get_processors();
                    unsigned long blocks = 0;
                    unsigned long grids = 0;
                    unsigned long operations = 0;
                    if ( length <= max_blocks )
                    {
                        blocks = prev_power2( length );
                        grids = 1;
                        operations = 2;
                    }
                    //else if ( length > max_blocks && length <= max_blocks * operation_threshold * processors )
                    else if ( length > max_blocks && length <= max_blocks * operation_threshold )
                    {
                        blocks = max_blocks;
                        grids = 1;
                        operations = ( length + blocks - 1 ) / blocks;
                    }
                    else
                    {
                        blocks = max_blocks;
                        unsigned long const factor = processors * blocks;
                        unsigned long const root = approx_sqrt( (length+factor-1) / factor );
                        grids = processors * root;
                        unsigned long const gactor = blocks * grids;
                        operations = ( length + gactor - 1 ) / gactor;
                    }
                    std::tuple<std::string,std::string, std::string> make_reduce_code( std::string const& lambda_code_, unsigned long length_, unsigned long grids_, unsigned long blocks_, unsigned long operations_ );
                    auto const& device_global_kernel =  make_reduce_code( lambda_code_, length, grids, blocks, operations );
                    std::string const& code = predefinition_ + generated_macro + std::get<0>(device_global_kernel) + generated_demacro + std::get<1>(device_global_kernel);
                    std::string const& kernel = std::get<2>(device_global_kernel);
                    auto&& ptx = make_ptx( code );
                    using result_type = std::remove_reference_t<std::remove_cv_t<decltype(*first_)>>;
                    result_type* device_dst = allocate<result_type>( grids );
                    int shared_memory_in_bytes = blocks * sizeof(result_type);
                    auto&& launcher = make_launcher( ptx, kernel, shared_memory_in_bytes );
                    launcher( grids, 1, 1, blocks, 1, 1 )( first_, device_dst );
                    if ( grids > 1 ) // second time reduce
                    {
                        unsigned long const new_length = grids;
                        unsigned long new_grids = 1;
                        unsigned long new_blocks = 0;
                        unsigned long new_operations = 0;
                        if ( new_length <= max_blocks )
                        {
                            new_blocks = prev_power2( new_length );
                            new_operations = 2;
                        }
                        else
                        {
                            new_blocks = max_blocks;
                            new_operations = ( new_length + new_blocks - 1 ) / new_blocks;
                        }
                        auto const& new_device_global_kernel =  make_reduce_code( lambda_code_, new_length, new_grids, new_blocks, new_operations );
                        std::string const& new_code = predefinition_ + generated_macro + std::get<0>(new_device_global_kernel) + generated_demacro + std::get<1>(new_device_global_kernel);
                        std::string const& new_kernel = std::get<2>(new_device_global_kernel);
                        auto&& new_ptx = make_ptx( new_code );
                        int new_shared_memory_in_bytes = new_blocks * sizeof(result_type);
                        auto&& new_launcher = make_launcher( new_ptx, new_kernel, new_shared_memory_in_bytes );
                        new_launcher( new_grids, 1, 1, new_blocks, 1, 1 )( device_dst, device_dst );
                    }
                    result_type ans;
                    device_to_host_copy( device_dst, device_dst+1, &ans );
                    deallocate( device_dst );
                    return ans;
                }; // first_, last_, rests_...
            };// lambda_code_
        };// custom_defines_...
    }// predefinition_

#if 0
    // TODO:
    //
    inline auto direct_map() // <- map from host memory
    {
        return [](){};
    }

    inline auto multi_map() // <- map enabling multi-devices
    {
    }

    inline auto direct_multi_map() // <- map from host memory enabling multi-devices
    {
        return [](){};
    }

    inline auto direct_reduce() // <- reduce from host memory
    {
        return [](){};
    }

    inline auto multi_reduce() // <- reduce enabling multi-device
    {
    }

    inline auto direct_multi_reduce() // <- reduce from host memory enabling multi-devices
    {
        return [](){};
    }
#endif

    struct timer
    {
        struct events;
        std::unique_ptr<events> ev;
        timer( std::string const& start_info_ = std::string{""} );
        ~timer();
    };

    inline void reset_device()
    {
        void cumar_reset_device();
        cumar_reset_device();
    }

    inline void set_device( int id_ )
    {
        void cumar_set_device( int id_ );
        cumar_set_device( id_ );
    }

    inline int get_device()
    {
        extern int cumar_get_device();
        return cumar_get_device();
    }

    inline unsigned long long get_memory_in_bytes()
    {
        extern unsigned long long cumar_get_memory_in_bytes();
        return cumar_get_memory_in_bytes();
    }

    inline int get_major_capability()
    {
        extern int cumar_get_major_capability();
        return cumar_get_major_capability();
    }

    inline int get_minor_capability()
    {
        extern int cumar_get_minor_capability();
        return cumar_get_minor_capability();
    }

    inline int device_count()
    {
        extern int cumar_device_count();
        return cumar_device_count();
    }


}//namespace cumar

#endif//DWJUWOBYHOTAGIAAHOTWEGPBLECMWYVTUVGJIDGRUCQRVQEVWSGTVWIICNONGWSPSRIOMJUKQ

