#ifndef VNSDFWVBSKPKJJSEKINGGGXDQQMPQJPLFNHIKBERONSQJOEUNQYQAUUKIFUVMTWAVBRSYDYOT
#define VNSDFWVBSKPKJJSEKINGGGXDQQMPQJPLFNHIKBERONSQJOEUNQYQAUUKIFUVMTWAVBRSYDYOT

#include <algorithm>
#include <array>
#include <cassert>
#include <cmath>
#include <functional>
#include <sstream>
#include <string>
#include <tuple>
#include <type_traits>
#include <utility>

int cumar_get_cores_per_processor();
int cumar_get_max_thread_x();
int cumar_get_processors();
void cumar_allocate( void**, unsigned long );
void cumar_deallocate( void* );
void cumar_memcopy_device_to_host( const void*, unsigned long, void* );
void nvrtc_make_ptx( std::string const&, std::string& );
void make_nvrtc_launcher( char const* const ptx_, char const* const kernel_, int gx, int gy, int gz, int tx, int ty, int tz, void** args, int shared_memory_in_bytes );

namespace cumar
{

    namespace cumar_private
    {

        template< bool... > struct bool_pack;
        template< bool... bs > using all_true = std::is_same< bool_pack<bs..., true>, bool_pack<true, bs...>>;
        template< typename ... Args > using all_pointer = all_true< std::is_pointer_v<Args>... >;

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

        inline auto make_macro()
        {
            return [=]( auto ... custom_defines_ ) noexcept // <- ( "id_1", value_1, "id_2", value_2, ... )  -> "custom_define id_1 value_1 \ncustom_define id_2 value_2\n"
            {
                static_assert( (sizeof...(custom_defines_) & 1) == 0, "precaptured variables' types and values not match" );

                std::string generated_macro;
                std::string generated_demacro;

                auto const& composed_generator = make_overloader
                (
                    [&generated_macro]( auto&& generator_, std::string const& id_, auto&& value_, auto&& ... rests_ ) noexcept
                    {
                        generated_macro += std::string{"#define "} + id_ + std::string{" "} + lexical_cast<std::string>(value_) + std::string{"\n"};
                        generator_( generator_, rests_... );
                    },
                    [&generated_macro]( auto ) noexcept { generated_macro += std::string{"\n"}; }
                );

                auto const& composed_degenerator = make_overloader
                (
                    [&generated_demacro]( auto&& generator_, std::string const& id_, auto&& value_, auto&& ... rests_ ) noexcept
                    {
                        generated_demacro += std::string{"#undef "} + id_  + std::string{"\n"};
                        generator_( generator_, rests_... );
                    },
                    [&generated_demacro]( auto ) noexcept { generated_demacro += std::string{"\n"}; }
                );

                composed_generator( composed_generator, custom_defines_... ); // <- all macros generated
                composed_degenerator( composed_degenerator, custom_defines_... ); // <- all macros degenerated

                return std::make_tuple( generated_macro, generated_demacro );
            };
        }

        template< typename T >
        inline T* allocate( unsigned long n_ )
        {
            T* ans;
            cumar_allocate( reinterpret_cast<void**>(&ans), n_*sizeof(T) );
            return ans;
        }

        template< typename T >
        inline void deallocate( T* ptr_ )
        {
            cumar_deallocate( reinterpret_cast<void*>(ptr_) );
        }

        template< typename T >
        inline void device_to_host( T* device_begin_, T* device_end_, T* host_begin_ )
        {
            unsigned long const n = sizeof(T)*(device_end_-device_begin_);
            cumar_memcopy_device_to_host( reinterpret_cast<const void*>(device_begin_), n, reinterpret_cast<void*>(host_begin_) );
        }

        inline std::string const make_ptx( std::string const& source_code_ )
        {
            std::string ans;
            nvrtc_make_ptx( source_code_, ans );
            return ans;
        }

        inline auto make_launcher( std::string const& ptx, std::string const& kernel_name, int const shared_memory = 0 )
        {
            assert( shared_memory >= 0 && "Shared memory size cannot be negative!" );
            return [&]( auto&& ... dims ) noexcept
            {
                return [&]( auto&& ... args ) noexcept
                {
                    auto&& ag = []( auto&& ... args_ ) noexcept { return std::array<void*,sizeof...(args_)>{ {reinterpret_cast<void*>(std::addressof(std::forward<decltype(args_)>(args_)))... }}; }( args... );
                    make_nvrtc_launcher( ptx.c_str(), kernel_name.c_str(), dims..., &ag[0], shared_memory );
                };
            };
        }

        //returns [grid size] [block size] [operations per thread]
        inline std::tuple<int,int,int> const make_map_configuration( int N_ )
        {
            assert( N_ > 0 );


            int const operation_threshold = 128;
            int const blocks = cumar_get_cores_per_processor();
            int const grids = cumar_get_processors();
            int const cores = blocks * grids;

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

        // [grids, blocks, operations]
        inline std::tuple<unsigned long, unsigned long, unsigned long> make_reduce_configuration( unsigned long const N_ )
        {
            assert( N_ > 0 && "Negative array length!" );


            unsigned long const length = N_;
            unsigned long const max_blocks = cumar_get_max_thread_x();

            if ( max_blocks >= length )
            {
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
                return std::make_tuple( 1, prev_power2( length ), 2 );
            }

            unsigned long const threshold = 128;
            if ( length <= max_blocks * threshold )
                return std::make_tuple( 1, max_blocks, ( length + max_blocks - 1 ) / max_blocks );

            auto const& approx_sqrt = []( unsigned long x ) noexcept
            {
                unsigned long ans = std::max(x >> 1, 1UL);
                for ( unsigned long i = 0; i != 10; ++i )
                    ans = ( ans + (x+ans-1) / ans ) >> 1;
                return ans;
            };

            unsigned long const processors = cumar_get_processors();
            unsigned long const factor = processors * max_blocks;
            unsigned long const root = approx_sqrt( ( length + factor - 1 ) / factor );
            unsigned long const gactor = max_blocks * processors * root;
            return std::make_tuple( processors * root, max_blocks, ( length + gactor - 1 ) / gactor );
        }

    }// namespace cumar_private

}//namespace cumar

#endif//VNSDFWVBSKPKJJSEKINGGGXDQQMPQJPLFNHIKBERONSQJOEUNQYQAUUKIFUVMTWAVBRSYDYOT

