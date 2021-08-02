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

#include "cumar_private.hpp"

std::tuple<std::string,std::string, std::string> make_map_code( std::string const& lambda_code_, unsigned long length_, unsigned long grids_, unsigned long blocks_, unsigned long operations_ );
std::tuple<std::string,std::string, std::string> make_reduce_code( std::string const& lambda_code_, unsigned long length_, unsigned long grids_, unsigned long blocks_, unsigned long operations_ );

namespace cumar
{

    inline auto map( std::string const& predefinition_ = std::string{""} ) noexcept
    {
        return [=]( auto ... custom_defines_ ) noexcept
        {
            return [=]( std::string const& lambda_code_ ) noexcept
            {
                return [=]( auto first_, auto last_, auto ... rests_ ) noexcept // <- all iterators
                {
                    static_assert( std::is_same_v< decltype(first_), decltype(last_) >, "first two argument type not match!" );
                    static_assert( cumar_private::all_pointer<decltype(first_), decltype(last_), decltype(rests_)...>::value, "arguments contains non-pointer entry!" );

                    auto const& [generated_macro, generated_demacro] = cumar_private::make_macro()( custom_defines_... );

                    unsigned long length = last_ - first_;
                    auto const [grids, blocks, operations] = cumar_private::make_map_configuration( length );

                    auto const& [device_code, global_code, kernel_name ] = make_map_code( lambda_code_, length, grids, blocks, operations );
                    auto const& ptx = cumar_private::make_ptx( predefinition_ + generated_macro + device_code + generated_demacro + global_code );
                    auto&& launcher = cumar_private::make_launcher( ptx, kernel_name );
                    launcher( grids, 1, 1, blocks, 1, 1 )( first_, rests_... );
                    return ptx;
                }; // first_, last_, rests_...
            };// lambda_code_
        };// custom_defines_...
    }// predefinition_

    inline auto reduce( std::string const& predefinition_ = std::string{""} ) noexcept
    {
        return [=]( auto ... custom_defines_ ) noexcept
        {
            return [=]( std::string const& lambda_code_ ) noexcept
            {
                return [=]( auto first_, auto last_ ) noexcept // <- all iterators
                {
                    static_assert( std::is_same_v< decltype(first_), decltype(last_) >, "first two argument type not match!" );
                    static_assert( cumar_private::all_pointer< decltype(first_), decltype(last_) >::value, "arguments contains non-pointer entry!" );

                    auto const& [generated_macro, generated_demacro] = cumar_private::make_macro()( custom_defines_... );

                    unsigned long length = last_ - first_;
                    std::tuple<unsigned long, unsigned long, unsigned long> config = cumar_private::make_reduce_configuration( length );

                    using result_type = std::remove_reference_t<std::remove_cv_t<decltype(*first_)>>;
                    result_type* device_out = cumar_private::allocate<result_type>( std::get<0>( config ) );
                    result_type* device_in = first_;

                    for (;;)
                    {
                        auto const& [grids, blocks, operations] = config;

                        auto const& [device_code, global_code, kernel_name ] = make_reduce_code( lambda_code_, length, grids, blocks, operations );
                        auto&& ptx = cumar_private::make_ptx( predefinition_ + generated_macro + device_code + generated_demacro + global_code );
                        int shared_memory_in_bytes = blocks * sizeof(result_type);
                        auto&& launcher = cumar_private::make_launcher( ptx, kernel_name, shared_memory_in_bytes );
                        launcher( grids, 1, 1, blocks, 1, 1 )( device_in, device_out );

                        if ( grids == 1 ) break;

                        length = grids;
                        config = cumar_private::make_reduce_configuration( length );
                        device_in = device_out;
                    }

                    result_type ans;
                    cumar_private::device_to_host( device_out, device_out+1, &ans );
                    cumar_private::deallocate( device_out );
                    return ans;
                }; // first_, last_, rests_...
            };// lambda_code_
        };// custom_defines_...
    }// predefinition_


}//namespace cumar

#endif//DWJUWOBYHOTAGIAAHOTWEGPBLECMWYVTUVGJIDGRUCQRVQEVWSGTVWIICNONGWSPSRIOMJUKQ

