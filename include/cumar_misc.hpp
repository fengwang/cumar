#ifndef YSYUCREPNEITFCWOWKBAFXWBJAPYHCVYXRGNJUALGAPRBMVDVANKERLAUGOOIJOBXHDMAKSTO
#define YSYUCREPNEITFCWOWKBAFXWBJAPYHCVYXRGNJUALGAPRBMVDVANKERLAUGOOIJOBXHDMAKSTO

#include <cassert>
#include <memory>
#include <string>

extern void cumar_set_device( int id_ );
extern void cumar_reset_device();
extern void cumar_allocate( void**, unsigned long );
extern void cumar_deallocate( void* );
extern void cumar_memcopy_host_to_device( const void*, unsigned long, void* );
extern void cumar_memcopy_device_to_host( const void*, unsigned long, void* );

namespace cumar
{

    inline void set_device( int id_ )
    {
        cumar_set_device( id_ );
    }

    inline void reset_device()
    {
        cumar_reset_device();
    }

    template< typename T >
    inline T* allocate( unsigned long n_ )
    {
        assert ( n_  && "allocation size should not be zero!" );

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
    inline void host_to_device( T* host_begin_, T* host_end_, T* device_begin_ )
    {
        unsigned long const n = sizeof(T)*(host_end_-host_begin_);
        cumar_memcopy_host_to_device( reinterpret_cast<const void*>(host_begin_), n, reinterpret_cast<void*>(device_begin_) );
    }

    template< typename T >
    inline T* host_to_device( T* host_begin_, T* host_end_ )
    {
        unsigned long const n = sizeof(T)*(host_end_-host_begin_);
        T* ans = allocate<T>(n);
        host_to_device( host_begin_, host_end_, ans );
        return ans;
    }

    template< typename T >
    inline void device_to_host( T* device_begin_, T* device_end_, T* host_begin_ )
    {
        unsigned long const n = sizeof(T)*(device_end_-device_begin_);
        cumar_memcopy_device_to_host( reinterpret_cast<const void*>(device_begin_), n, reinterpret_cast<void*>(host_begin_) );
    }

    template< typename T >
    inline T* device_to_host( T* device_begin_, T* device_end_ )
    {
        unsigned long const n = sizeof(T) * (device_end_ - device_begin_);
        T* ans = new T[n];
        device_to_host( device_begin_, device_begin_, ans );
        return ans;
    }

    struct timer
    {
        struct events;
        std::unique_ptr<events> ev;
        timer( std::string const& start_info_ = std::string{""} );
        ~timer();
    };

}//namespace cumar

#endif//YSYUCREPNEITFCWOWKBAFXWBJAPYHCVYXRGNJUALGAPRBMVDVANKERLAUGOOIJOBXHDMAKSTO

