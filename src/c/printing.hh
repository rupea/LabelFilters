#ifndef PRINTING_HH
#define PRINTING_HH

#include "printing.h"
#include <iostream>
#include <stdexcept>

namespace detail {
    template<typename T>
        inline std::ostream&
        io_txt( std::ostream& os, T const& x, char const* ws="\n" )
        { return os << x << ws; }
    template<typename T>
        inline std::istream&
        io_txt( std::istream& is, T& x )
        { return is >> x; }
    template<typename T>
        inline std::ostream&
        io_bin( std::ostream& os, T const& x )
        { return os.write(reinterpret_cast<char const*>(&x),sizeof(T)); }
    template<typename T>
        inline std::istream&
        io_bin( std::istream& is, T& x )
        { return is.read (reinterpret_cast<char*>(&x),sizeof(T)); }

    // specializations
    //   strings as length + (no intervening space) + blob
    template<> inline std::ostream& io_txt( std::ostream& os, std::string const& x, char const* ws/*="\n"*/ ){
        uint32_t len=(uint32_t)(x.size() * sizeof(std::string::traits_type::char_type));
        io_txt(os,len,"");      // no intervening whitespace
        if(os.fail()) throw std::overflow_error("failed string-len-->std::ostream");
        os<<x<<ws;
        if(os.fail()) throw std::overflow_error("failed string-data-->std::ostream");
        return os;
    }
    template<> inline std::istream& io_txt( std::istream& is, std::string& x ){
        uint32_t len;
        io_txt(is,len);
        if(is.fail()) throw std::underflow_error("failed std::istream-->string-len");
        x.resize(len,'\0');     // reserve string memory
        is.read(&x[0], len);    // read full string content
        if(is.fail()) throw std::underflow_error("failed std::istream-->string-data");
        return is;
    }
    template<> inline std::ostream& io_bin( std::ostream& os, std::string const& x ){
        uint32_t len=(uint32_t)(x.size() * sizeof(std::string::traits_type::char_type));
        io_bin(os,len);
        if(os.fail()) throw std::overflow_error("failed string-len-->std::ostream");
        os.write(x.data(),len);
        if(os.fail()) throw std::overflow_error("failed string-data-->std::ostream");
        return os;
    }
    template<> inline std::istream& io_bin( std::istream& is, std::string& x ){
        uint32_t len;
        io_bin(is,len);
        if(is.fail()) throw std::underflow_error("failed std::istream-->string-len");
        x.resize(len,'\0');     // reserve string memory
        is.read(&x[0], len);    // read full string content
        if(is.fail()) throw std::underflow_error("failed std::istream-->string-data");
        return is;
    }
}

#endif // PRINTING_HH
