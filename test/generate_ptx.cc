#include "../include/cumar.hpp"
#include "../include/cumar_misc.hpp"

#include <string>
#include <fstream>
#include <sstream>
#include <iostream>
#include <cassert>

int main( int argc, char** argv )
{
    if ( 1 == argc )
    {
        std::cout << "Usage:\ngenerate_ptx example.cu\n";
        return 0;
    }

    assert( argc == 2 );

    std::ifstream ifs( argv[1] );
    std::stringstream iss;
    std::copy( std::istreambuf_iterator<char>( ifs ), std::istreambuf_iterator<char>(), std::ostreambuf_iterator<char>( iss ) );
    std::string code = iss.str();

    std::cout << cumar::cumar_private::make_ptx( code ) << "\n";

    return 0;
}
