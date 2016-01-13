#ifndef PRINTING_HH
#define PRINTING_HH

#include "printing.h"
#include <iostream>
#include <stdexcept>
#include <boost/function_output_iterator.hpp>
#include <boost/iterator/function_input_iterator.hpp>

namespace detail {
    template<typename T> inline std::ostream&
        io_txt( std::ostream& os, T const& x, char const* ws="\n" )
        { return os << x << ws; }
    template<typename T> inline std::istream&
        io_txt( std::istream& is, T& x )
        { return is >> x; }
    template<typename T> inline std::ostream&
        io_bin( std::ostream& os, T const& x )
        { return os.write(reinterpret_cast<char const*>(&x),sizeof(T)); }
    template<typename T> inline std::istream&
        io_bin( std::istream& is, T& x )
        { return is.read (reinterpret_cast<char*>(&x),sizeof(T)); }

    inline std::ostream& io_bin( std::ostream& os, void const* p, size_t bytes ){
        return os.write((char*)p,bytes);
    }
    inline std::istream& io_bin( std::istream& is, void      * p, size_t bytes ){
        return is.read((char*)p,bytes);
    }

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
    // -------- helpers for boost iterator adapters --------
    /** unary function that outputs binary 'T' to an ostream (no err check) */
    template<typename T> struct OutBinary {
        OutBinary( std::ostream& os ) : os(os) {}
        void operator()( T const& t ) { os.write( reinterpret_cast<char const*>(&t), sizeof(T) ); }
        std::ostream& os;
    };
    /** nullary function returning next binary 'T' from an istream (no err check) */
    template<typename T> struct InBinary {
        typedef T result_type;
        InBinary( std::istream& is ) : is(is) {}
        result_type operator()() { T t; is.read( reinterpret_cast<char*>(&t), sizeof(T) ); return t; }
        std::istream& is;
    };
    // -------- boost::dynamic_bitset I/O --------------
#define TBITSET template<typename Block, typename Alloc> inline
#define BITSET  boost::dynamic_bitset<Block,Alloc>
#if 0 // Oh, dynamic_bitset << and >> are provided, so default impl is OK
    TBITSET std::ostream& io_txt( std::ostream& os, BITSET const& x, char const* ws/*="\n"*/ ){
        using namespace std;
        std::cout<<" io_txt(os,BS,ws) "; std::cout.flush();
        os<<x<<ws;    // ---------> reverse of expected output.  x[0] is output LAST !
        //for(size_t i=0U; i<x.size(); ++i) os<<x[i];     // XXX slow?
        //os<<ws;
        if(os.fail()) throw std::overflow_error("failed txt dynamic_bitset-->std::ostream");
        return os;
    }
    TBITSET std::istream& io_txt( std::istream& is, BITSET      & x ){
        using namespace std;
        is>>x;        // big-endian output.  Little-endian seems more natural.
        if(is.fail()) throw std::underflow_error("failed txt std::istream-->dynamic_bitset");
        return is;
    }
#endif
    TBITSET std::ostream& io_bin( std::ostream& os, BITSET const& bs ){
        using namespace boost;
        io_bin(os, (uint64_t)(bs.size()));
        io_bin(os, (uint64_t)(bs.num_blocks()));
        to_block_range(bs, make_function_output_iterator(OutBinary< Block >(os)));
        if(os.fail()) throw std::overflow_error("failed bitset-data-->std::ostream");
        return os;
    }
    TBITSET std::istream& io_bin( std::istream& is, BITSET      & bs ){
        using namespace boost;
        uint64_t nbits, nblocks;
        io_bin(is, nbits);
        io_bin(is, nblocks);
        bs.resize( nbits );
        InBinary<Block> in(is);
        from_block_range( make_function_input_iterator(in,uint64_t{0}),
                          make_function_input_iterator(in,nblocks),
                          bs );
        if(is.fail()) throw std::underflow_error("failed std::istream-->bitset-data");
        return is;
    }
#undef BITSET
#undef TBITSET
#define TARRAY template<class T, std::size_t N> inline
    TARRAY std::ostream& io_txt( std::ostream& os, std::array<T,N> const& x, char const* ws/*="\n"*/ ){
        for(auto const& t: x ) io_txt(os,t,ws);
        return os;
    }
    TARRAY std::istream& io_txt( std::istream& is, std::array<T,N>      & x ){
        for(auto & t: x) io_txt(is,t);
        return is;
    }
    TARRAY std::ostream& io_bin( std::ostream& os, std::array<T,N> const& x ){
        //std::cout<<" io_bin(os,array<T,"<<N<<">"; std::cout.flush();
        for(auto const& t: x) io_bin(os,t);
        return os;
    }
    TARRAY std::istream& io_bin( std::istream& is, std::array<T,N>      & x ){
        for(auto & t: x) io_bin(is,t);
        return is;
    }
    // specialization array<char,N> does not need space between every char
    // Note: specialization must respecify the default for ws
    template< std::size_t N > inline
        std::ostream& io_txt( std::ostream& os, std::array<char,N> const& x, char const* ws="\n" ){
            //std::cout<<" io_txt(os,array<char,"<<N<<">,ws)"; std::cout.flush();
            io_bin( os, x );
            os<<ws;
            return os;
        }
    template< std::size_t N > inline
        std::istream& io_txt( std::istream& is, std::array<char,N>      & x ){
            is>>std::skipws;
            io_bin( is, x );
            return is;
        }

#undef TARRAY

    template<> inline std::ostream& io_txt( std::ostream& os, boolmatrix const& x, char const* ws/*="\n"*/ ){
        io_txt(os, x.cbase(), ws);
        return os;
    }
    template<> inline std::istream& io_txt( std::istream& is, boolmatrix& x ){
        io_txt(is, x.base());
        return is;
    }
    template<> inline std::ostream& io_bin( std::ostream& os, boolmatrix const& x ){
        io_bin(os, x.cbase());
        return os;
    }
    template<> inline std::istream& io_bin( std::istream& is, boolmatrix& x ){
        io_bin(is, x.base());
        return is;
    }

#define TMATRIX template<typename Derived>
#define MATRIX  Eigen::PlainObjectBase< Derived >
#if 1
    // Ohoh, but compiler needs help to resolve template types...
    TMATRIX std::ostream& eigen_io_txt( std::ostream& os, MATRIX const& x, char const *ws/*="\n"*/ ){
        using namespace std;
        uint64_t rows = x.rows();
        uint64_t cols = x.cols();
        //cout<<" eigen_io_txt( "<<rows<<" x "<<cols<<" ): ";
        os<<x.rows()<<' '<<x.cols();
        size_t const sz = size_t(rows)*size_t(cols);
        typename MATRIX::Scalar const* data = x.data();
        for(size_t i=0U; i<sz; ++i)
            os << ' ' << *data++;
        os<<ws;
        return os;
    }
    TMATRIX std::istream& eigen_io_txt( std::istream& is, MATRIX      & x ){
        using namespace std;
        // is.operator>>( MATRIX ) is NOT AVAILABLE in Eigen
        //cout<<" eigen_io_txt MATRIX-input ";
        uint64_t rows,cols;
        io_txt(is,rows);
        io_txt(is,cols);
        //cout<<" \trows "<<rows<<" cols "<<cols<<endl;
        x.resize(rows,cols);
        size_t const sz = size_t(rows)*size_t(cols);
        typename MATRIX::Scalar * data = x.data();
        for(size_t i=0U; i<sz; ++i)
            is >> *data++;
        return is;
    }
#endif
    TMATRIX std::ostream& eigen_io_bin( std::ostream& os, MATRIX const& x ){
        using namespace std;
        //cout<<" MATRIX-output rows "<<x.rows()<<" cols "<<x.cols()<<endl;
        // well, actually rows,cols are of typename MATRIX::INDEX
        uint64_t rows = x.rows();
        uint64_t cols = x.cols();
        io_bin(os,rows);
        io_bin(os,cols);
        io_bin(os,(void const*)x.data(),size_t(rows*cols*sizeof(typename MATRIX::Scalar)));
        return os;
    }
    TMATRIX std::istream& eigen_io_bin( std::istream& is, MATRIX      & x ){
        using namespace std;
        //cout<<" MATRIX-input"<<endl;
        uint64_t rows,cols;
        io_bin(is,rows);
        io_bin(is,cols);
        //cout<<" \trows "<<rows<<" cols "<<cols<<endl;
        x.resize(rows,cols);
        io_bin(is,(void*)x.data(),size_t(rows*cols*sizeof(typename MATRIX::Scalar)));
        return is;
    }
#undef MATRIX
#undef TMATRIX
}

template <typename Block, typename Alloc> inline
  void save_bitvector(std::ostream& out, const boost::dynamic_bitset<Block, Alloc>& bs)
{
  using namespace std;
  size_t num_bits = bs.size();
  size_t num_blocks = bs.num_blocks();
  std::vector<Block> blocks(num_blocks);
  to_block_range(bs, blocks.begin());  
  out.write((char*)&num_bits, sizeof(size_t));
  if (out.fail())
    {
      cerr << "Error writing file" << endl;
    }
  out.write((char*)&num_blocks, sizeof(size_t));
  if (out.fail())
    {
      cerr << "Error writing file" << endl;
    }
  out.write((char*)(&(blocks[0])), num_blocks*sizeof(Block));  
  if (out.fail())
    {
      cerr << "Error writing file" << endl;
    }
}

template <typename Block, typename Alloc> inline
  int load_bitvector(std::istream& in, boost::dynamic_bitset<Block, Alloc>& bs)
{
  using namespace std;
  size_t num_bits,num_blocks;
  in.read((char*)&num_bits, sizeof(size_t));
  if (in.fail())
    {
      cerr << "Error reading file" << endl;
      return -1;
    }
  in.read((char*)&num_blocks, sizeof(size_t));
  if (in.fail())
    {
      cerr << "Error reading file" << endl;
      return -1;
    }
  std::vector<Block> blocks(num_blocks);
  in.read((char*)(&(blocks[0])), num_blocks*sizeof(Block));
  if (in.fail())
    {
      cerr << "Error reading file" << endl;
      return -1;
    }
  bs.resize(num_bits);
  from_block_range(blocks.begin(), blocks.end(), bs);
  bs.resize(num_bits);
  return 0;
}

    template<typename EigenType> inline
void print_mat_size(const EigenType& mat)
{
    using namespace std;
    cout << "(" << mat.rows() << ", " << mat.cols() << ")";
}
    template<> inline
void print_mat_size(const Eigen::VectorXd& mat)
{
    using namespace std;
    cout << "(" << mat.size() << ")";
}


#endif // PRINTING_HH
