#ifndef PRINTING_HH
#define PRINTING_HH

#include "printing.h"
#include <iostream>
#include <iomanip>
#include <sstream>
#include <stdexcept>
#include "roaring.hh"
#include <limits>

namespace detail {
  template<typename T> inline std::ostream&
  io_txt( std::ostream& os, T const& x, char const* ws /*="\n"*/ )
  { return os << x << ws; }
  template<typename T> inline std::istream&
  io_txt( std::istream& is, T& x )
  { return is >> x; }
  template<typename T> inline std::ostream&
  io_bin( std::ostream& os, T const& x )
  { return os.write(reinterpret_cast<char const*>(&x),sizeof(T)); }
  template<typename T> inline std::istream&
  io_bin( std::istream& is, T& x )
  {
    if( ! is.good() )
      throw std::runtime_error("io_bin: read failure");
    is.read (reinterpret_cast<char*>(&x),sizeof(T));
    return is;
  }
  
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
        if(len){
            os.write(x.data(),len);
            if(os.fail()) throw std::overflow_error("failed string-data-->std::ostream");
        }
        return os;
    }
    template<> inline std::istream& io_bin( std::istream& is, std::string& x ){
        uint32_t len;
        io_bin(is,len);
        if(is.fail()) throw std::underflow_error("failed std::istream-->string-len");
        x.resize(len,'\0');     // reserve string memory
        if(len){
            is.read(&x[0], len);    // read full string content
            if(is.fail()) throw std::underflow_error("failed std::istream-->string-data");
        }
        assert( is.good() );
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
    // -------- Roaring I/O --------------
  inline std::ostream& io_txt( std::ostream& os, Roaring const& x, char const* ws/*="\n"*/ ){
    os<<x.toString()<<ws;  
    if(os.fail()) throw std::overflow_error("failed txt Roaring-->std::ostream");
    return os;
  }
  
  inline std::istream& io_txt( std::istream& is, Roaring  & x ){
    std::string buf;
    is.ignore(std::numeric_limits<std::streamsize>::max(),'{');
    std::getline(is,buf,'}');
    if(is.fail()) throw std::underflow_error("failed txt std::istream-->Roaring");
    std::stringstream ss(buf);
    std::vector<uint32_t> array;    
    uint32_t val;    
    while (ss>>val)
      {
	array.push_back(val);
	ss.ignore(std::numeric_limits<std::streamsize>::max(), ',');
      }
    x = Roaring(array.size(), array.data());
    return is;
  }

  inline std::ostream& io_txt( std::ostream& os, std::vector<Roaring> const& x, char const* ws/*="\n"*/ ){
    os << x.size() << "\n";
    for(auto & xi: x) io_txt(os,xi,ws);
    if(os.fail()) throw std::overflow_error("failed txt dynamic_bitset-->std::ostream");
    return os;
  }
  
  inline std::istream& io_txt( std::istream& is, std::vector<Roaring>& x ){
    size_t n;
    is >> n;
    x.resize(n);    
    for(auto & xi: x) io_txt(is,xi);
    if(is.fail()) throw std::underflow_error("failed bitset-data-->std::ostream");
    return is;
  }    
  
  inline std::ostream& io_bin( std::ostream& os, Roaring const& r ){
    uint32_t bytes = (uint32_t)r.getSizeInBytes();
    io_bin(os, bytes);
    char* buf = new char[bytes];
    r.write(buf);
    os.write(buf,bytes);
    delete[] buf;
    if(os.fail()) throw std::overflow_error("failed Roaring-data-->std::ostream");    
    return os;
  }

  inline std::istream& io_bin( std::istream& is, Roaring& r ){
    uint32_t bytes;
    io_bin(is, bytes);
    char* buf = new char [bytes];
    is.read(buf,bytes);
    if(is.fail()) 
      {
	delete[] buf;
	throw std::underflow_error("failed std::istream-->Roaring-data");
      }
    r = Roaring::readSafe(buf,bytes);
    delete[] buf;
    return is;
  }

  inline std::ostream& io_bin( std::ostream& os, std::vector<Roaring> const& x )
  {
    uint32_t const rows = static_cast<uint32_t>(x.size());    
    io_bin(os,rows);
    for(auto const& xi: x) io_bin(os,xi);
    if(os.fail()) throw std::overflow_error("failed bitset-data-->std::ostream");
    return os;
  }

  inline std::istream& io_bin( std::istream& is, std::vector<Roaring>& x )
  {
    uint32_t rows;
    io_bin(is,rows);
    x.resize(rows);
    for(auto & xi: x) io_bin(is,xi);
    if(is.fail()) throw std::underflow_error("failed bitset-data-->std::ostream");
    return is;
  }

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

#define TMATRIX template<typename Derived> inline
#define MATRIX  Eigen::PlainObjectBase< Derived >
    // Ohoh, but compiler needs help to resolve template types...
  
  // if x is row-major then write x. If x is col major then write transpose(x)
  TMATRIX std::ostream& eigen_io_txt( std::ostream& os, MATRIX const& x, char const *ws/*="\n"*/ ){
    using namespace std;
    uint64_t outer_size = x.outerSize();
    uint64_t inner_size = x.innerSize();
    os<<outer_size<<' '<<inner_size << ws;
    typename MATRIX::Scalar const* data = x.data();
    for(size_t i=0U; i<outer_size; ++i)
      {
	for (size_t j = 0U; j < inner_size; ++j)
	  {
	    os << *data++ << ' ';
	  }	    
	os<<ws;
      }	
    return os;
  }
  
  // if x is row-major then read. If x is col major then read and transpose. 
  TMATRIX std::istream& eigen_io_txt( std::istream& is, MATRIX      & x ){
    using namespace std;
    uint64_t outer_size,inner_size;
    io_txt(is,outer_size);
    io_txt(is,inner_size);
    x.IsRowMajor?x.resize(outer_size, inner_size):x.resize(inner_size,outer_size);
    size_t const sz = size_t(outer_size)*size_t(inner_size);
    typename MATRIX::Scalar * data = x.data();
    for(size_t i=0U;i<sz; ++i)
      is >> *data++;
    return is;
  }
  
  // dense output
  // if x is row-major then write x. If x is col major then write transpose(x)
  TMATRIX std::ostream& eigen_io_bin_plain( std::ostream& os, MATRIX const& x ){
    using namespace std;
    // well, actually rows,cols are of typename MATRIX::INDEX
    uint64_t outer_size = x.outerSize();
    uint64_t inner_size = x.innerSize();
    io_bin(os,outer_size);
    io_bin(os,inner_size);
    io_bin(os,(void const*)x.data(),size_t(outer_size*inner_size*sizeof(typename MATRIX::Scalar)));
    return os;
  }
  
  // if x is row-major then write x. If x is col major then write transpose(x)
  TMATRIX std::ostream& eigen_io_bin_float( std::ostream& os, MATRIX const& x ){
    uint64_t outer_size = x.outerSize();
    uint64_t inner_size = x.innerSize();
    using namespace std;
    io_bin(os,outer_size);
    io_bin(os,inner_size);
    for(size_t i=0U; i<outer_size*inner_size; ++i){
      float val = static_cast<float>( x.data()[i] );
      io_bin(os,val);
    }
    return os;
  }
  
  // if x is row-major then write x. If x is col major then write transpose(x)
  TMATRIX std::ostream& eigen_io_bin( std::ostream& os, MATRIX const& x ){
    if( std::is_same<typename MATRIX::Scalar, double>::value ){
      eigen_io_bin_float( os, x );
    }else{
      eigen_io_bin_plain( os, x );
    }
    return os;
  }

  // dense input
  // if x is row-major then read. If x is col major then read and transpose. 
  TMATRIX std::istream& eigen_io_bin_plain( std::istream& is, MATRIX      & x ){
    using namespace std;
    uint64_t outer_size,inner_size;
    io_bin(is,(uint64_t&)outer_size);
    io_bin(is,(uint64_t&)inner_size);
    x.IsRowMajor?x.resize(outer_size, inner_size):x.resize(inner_size,outer_size);
    io_bin(is,(void*)x.data(),size_t(outer_size*inner_size*sizeof(typename MATRIX::Scalar)));
    return is;
  }

  // if x is row-major then read. If x is col major then read and transpose. 
  TMATRIX std::istream& eigen_io_bin_float( std::istream& is, MATRIX      & x ){
    using namespace std;
    uint64_t outer_size,inner_size;
    io_bin(is,(uint64_t&)outer_size);
    io_bin(is,(uint64_t&)inner_size);
    x.IsRowMajor?x.resize(outer_size, inner_size):x.resize(inner_size,outer_size);
    typename MATRIX::Scalar *xdata = x.data();
    uint64_t nData = outer_size*inner_size;
    for(float tmp[1024] ; nData > 1024U; nData -= 1024U ){
      io_bin(is,(void*)tmp,size_t(1024U*sizeof(float)));
      for(uint32_t j=0U; j<1024U; ++j){
	*xdata++ = static_cast<typename MATRIX::Scalar>( tmp[j] );
      }
    }
    for( ; nData; --nData ){
      float tmp;
      io_bin(is,tmp);
      *xdata++ = static_cast<typename MATRIX::Scalar>( tmp );
    }
    return is;
  }
  
  // if x is row-major then read. If x is col major then read and transpose.
  // doubles were saved as floats, so reconvert floats to double. 
  TMATRIX std::istream& eigen_io_bin( std::istream& is, MATRIX      & x ){
    if( std::is_same<typename MATRIX::Scalar, double>::value ){
      eigen_io_bin_float( is, x );
    }else{
      eigen_io_bin_plain( is, x );
    }
    return is;
  }
#undef MATRIX
#undef TMATRIX

    // ----------------------- SPARSE --------------------------

#define TMATRIX template<typename Scalar, int Options, typename Index> inline
#define MATRIX  Eigen::SparseMatrix< Scalar, Options, Index >
  // if x is row-major then write x. If x is col major then write transpose(x)
  /* format (columns are 0-based):
   * rows cols
   * col:val col:val ... col:val 
   * col:val col:val ... col:val*/
  
  TMATRIX std::ostream& eigen_io_txt( std::ostream& os, MATRIX const& x, char const *ws/*="\n"*/ ){
    using namespace std;
    os<<x.outerSize()<<' '<<x.innerSize() << ws;
    for(int i=0U; i<x.outerSize(); ++i)
      {
	for(typename MATRIX::InnerIterator it(x,i); it; ++it)
	  {
	    os << it.index() << ":" << it.value() << " ";
	  }
	os << ws;
      }
    return os;
  }

  
  // if x is row-major then read. If x is col major then read and transpose. 
  /* format (columns are 0-based):
   * rows cols
   * col:val col:val ... col:val 
   * col:val col:val ... col:val*/
  TMATRIX std::istream& eigen_io_txt( std::istream& is, MATRIX      & x ){
    using namespace std;
    size_t outer_size,inner_size;
    io_txt(is,outer_size);
    io_txt(is,inner_size);
    is >> ws;
    size_t idx;
    Scalar val;
    char sep;
    typedef Eigen::Triplet<Scalar> T;
    std::vector<T> Triplets;
    std::string line;
    for(size_t i = 0; i<outer_size; i++){
      getline(is,line);
      istringstream iss(line);
      iss>>ws;
      while(iss.good())
	{
	  if( !(iss>>idx>>sep>>val))
	    {
	      throw std::runtime_error("sparse text format parse error");
	    }
	  if( sep != ':' )
	    {
	      throw std::runtime_error("sparse text format parse error");
	    }	  
	  x.IsRowMajor?Triplets.push_back(T(i,idx,val)):Triplets.push_back(T(idx,i,val));
	  iss>>ws;
	}
      if (!iss.eof()) throw std::runtime_error("sparse text format parse error");
    }         
    x.IsRowMajor?x.resize(outer_size, inner_size):x.resize(inner_size,outer_size);
    x.setFromTriplets(Triplets.begin(),Triplets.end());
    return is;
  }								   
  
  // generic sparse binary output
  // values are saved as floats. bool values are not saved. 
  // format:
  // size_in_bytes of value type (0 for binary, 4 for float and 8 for double)
  // size_in_bytes of index (1,2,4 or 8)
  // rows, cols, #nonzeros, outerIndex array, innerIndex array, valueIndex array (not saved for bool matrices).
  
  TMATRIX std::ostream& eigen_io_bin( std::ostream& os, MATRIX const& x){
    using namespace std;
    typedef float Real;     // we will convert to 'real' for binary i/o (maybe save space)

    // copy x so that it is not modified. Elimnate all zero values (important for bool matrices)
    // and make sure the matrix is in compressed format. 
    MATRIX xc = x.pruned();
    xc.makeCompressed();
    
    char val_size = std::is_same<Scalar, bool>::value?0U:sizeof(Real); //convert values to "real" 
    char idx_size = 0;
    
    //lambda functions to avoid code duplication
    auto idx_io = [&os,&idx_size](Index idx)      
      {
	switch (idx_size)
	  {
	  case 1:io_bin(os,static_cast<uint8_t>(idx));break;
	  case 2:io_bin(os,static_cast<uint16_t>(idx));break;
	  case 4:io_bin(os,static_cast<uint32_t>(idx));break;
	  case 8:io_bin(os,static_cast<uint64_t>(idx));break;
	  default: io_bin(os,idx);
	  }
      };
    
    auto val_io = [&os,&val_size](Scalar val)      
      {
	switch (val_size)
	  {
	  case 0: break;
	  case 4: io_bin(os,static_cast<float>(val)); break;
	  case 8: io_bin(os,static_cast<double>(val)); break;
	  default: io_bin(os, val);
	  }
      };
    
  
    size_t nData = std::max({xc.rows(),xc.cols(),xc.nonZeros()})+1;

    if(nData < numeric_limits<uint8_t>::max()){
      idx_size = 1;
    }
    else if(nData < numeric_limits<uint16_t>::max()){
      idx_size = 2;
    }
    else if(nData < numeric_limits<uint32_t>::max()){
      idx_size = 4;
    }
    else if(nData < numeric_limits<uint64_t>::max()){
      idx_size = 8;
    }
    else idx_size = sizeof(Index);

    io_bin(os, val_size);
    io_bin(os, idx_size);
    idx_io(xc.outerSize());
    idx_io(xc.innerSize());
    idx_io(xc.nonZeros());
    for (size_t i = 0; i <= xc.outerSize(); ++i)
      {
	idx_io(xc.outerIndexPtr()[i]);
      }
    for (size_t i = 0; i < xc.outerIndexPtr()[xc.outerSize()]; ++i)
      {
	idx_io(xc.innerIndexPtr()[i]);
      }
    if (val_size > 0){ // for bool do not write values since we know it is always 1. 
      for (size_t i = 0; i < xc.outerIndexPtr()[xc.outerSize()]; ++i)
	{
	  val_io(xc.valuePtr()[i]);
	}
    }
    return os;
  }

  
  
  TMATRIX std::istream& eigen_io_bin( std::istream& is, MATRIX      & x){
    using namespace std;

    char val_size,idx_size;
    Index outer_size = 0U;
    Index inner_size = 0U;
    Index nData = 0U;
    
    //lambda functions to avoid code duplication
    auto idx_io = [&is,&idx_size](Index& idx)      
      {
	switch (idx_size)
	  {
	  case 1:uint8_t  uint8 ; io_bin(is,uint8) ; idx=static_cast<Index>(uint8) ;break;
	  case 2:uint16_t uint16; io_bin(is,uint16); idx=static_cast<Index>(uint16);break;
	  case 4:uint32_t uint32; io_bin(is,uint32); idx=static_cast<Index>(uint32);break;
	  case 8:uint64_t uint64; io_bin(is,uint64); idx=static_cast<Index>(uint64);break;
	  default: io_bin(is,idx);
	  }
      };
        
    io_bin(is,val_size);
    io_bin(is,idx_size);
    
    idx_io(outer_size);
    idx_io(inner_size);
    idx_io(nData);
    
    x.IsRowMajor?x.resize(outer_size, inner_size):x.resize(inner_size,outer_size);
    x.resizeNonZeros( nData );
    
    size_t buff_size = std::max({outer_size+1*idx_size, nData*idx_size, nData*val_size});
    
    char* buff = new char[buff_size];
    
    //lambda functions to avoid code duplication
    auto idx_cast = [&buff, &idx_size](Index* idx, size_t size)      
      {
	switch (idx_size)
	  {
	  case 1:
	  {
	    uint8_t*  u8ptr  = reinterpret_cast<uint8_t*>(buff);
	    for(size_t i=0U; i<size; ++i) *idx++ = static_cast<Index>(*u8ptr++);
	    break;
	  }
	  case 2:
	  {
	    uint16_t*  u16ptr  = reinterpret_cast<uint16_t*>(buff);
	    for(size_t i=0U; i<size; ++i) *idx++ = static_cast<Index>(*u16ptr++);
	    break;
	  }	  
	  case 4: 
	  {
	    uint32_t*  u32ptr  = reinterpret_cast<uint32_t*>(buff);
	    for(size_t i=0U; i<size; ++i) *idx++ = static_cast<Index>(*u32ptr++);
	    break;
	  }
	  case 8:
	  {
	    uint64_t*  u64ptr  = reinterpret_cast<uint64_t*>(buff);
	    for(size_t i=0U; i<size; ++i) *idx++ = static_cast<Index>(*u64ptr++);
	    break;
	  }
	  default: throw runtime_error("Index size can only be 1,2,4 or 8.");
	  }
      };
    
    auto val_cast = [&buff, &val_size](Scalar* val, size_t size)      
      {
	switch (val_size)
	  {
	  case 0: for(size_t i=0U; i<size; ++i) *val++ = 1; break; //boolean. All values are 1.
	  case 4:
	  {
	    float*  fptr = reinterpret_cast<float*>(buff);
	    for(size_t i=0U; i<size; ++i) *val++ = static_cast<Scalar>(*fptr++);
	    break;
	  }
	  case 8:
	  {
	    double* dptr = reinterpret_cast<double*>(buff);
	    for(size_t i=0U; i<size; ++i) *val++ = static_cast<Scalar>(*dptr++);
	    break;
	  }
	  default: throw runtime_error("Value size can only be 0,4,8.");
	  }
      };
    
    //read the outerIndex array 
    io_bin(is, buff, (outer_size+1)*idx_size);
    idx_cast(x.outerIndexPtr(), outer_size+1);
    //read the innerIndex array
    io_bin(is, buff, nData*idx_size);
    idx_cast(x.innerIndexPtr(), nData);  
    //read the value array. 
    io_bin(is, buff, nData*val_size);// if val_size is 0 then nothing is read
    val_cast(x.valuePtr(), nData);
    return is;
  }
#undef MATRIX
#undef TMATRIX
  
}//detail::

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

    template< typename EigenType > inline
std::string print_report(EigenType const& x)
{
    return std::string();
}

    template< typename DERIVED > inline
std::string print_report(const Eigen::SparseMatrixBase<DERIVED>& x)
{
    std::ostringstream oss;
    int nnz = x.nonZeros();
    oss << "x:non-zeros: " << nnz << ", avg. nnz/row: " << nnz / x.rows();
    return oss.str();
}

template<typename EigenType> inline
PrettyDimensions prettyDims( EigenType const& x )
{
    return PrettyDimensions{{static_cast<std::size_t>(x.rows()),static_cast<std::size_t>(x.cols()),0U}, 2U};
}

#endif // PRINTING_HH
