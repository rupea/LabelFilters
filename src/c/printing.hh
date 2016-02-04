#ifndef PRINTING_HH
#define PRINTING_HH

#include "printing.h"
#include <iostream>
#include <sstream>
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

#define TMATRIX template<typename Derived> inline
//#define MATRIX  Eigen::PlainObjectBase< Derived >
#define MATRIX  Eigen::PlainObjectBase< Derived >
    // Ohoh, but compiler needs help to resolve template types...
    TMATRIX std::ostream& eigen_io_txt( std::ostream& os, MATRIX const& x, char const *ws/*="\n"*/ ){
        using namespace std;
        uint64_t rows = x.outerSize();
        uint64_t cols = x.innerSize();
        //cout<<" eigen_io_txt( "<<rows<<" x "<<cols<<" ): ";
        os<<x.outerSize()<<' '<<x.innerSize();
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
    // dense output
    TMATRIX std::ostream& eigen_io_bin_plain( std::ostream& os, MATRIX const& x ){
        using namespace std;
        //cout<<" MATRIX-output rows "<<x.rows()<<" cols "<<x.cols()<<endl;
        // well, actually rows,cols are of typename MATRIX::INDEX
        uint64_t rows = x.outerSize();
        uint64_t cols = x.innerSize();
        io_bin(os,rows);
        io_bin(os,cols);
        io_bin(os,(void const*)x.data(),size_t(rows*cols*sizeof(typename MATRIX::Scalar)));
        return os;
    }
    TMATRIX std::ostream& eigen_io_bin_float( std::ostream& os, MATRIX const& x ){
        //cout<<" MATRIX-output double->float rows "<<x.rows()<<" cols "<<x.cols()<<endl;
        uint64_t rows = x.outerSize();
        uint64_t cols = x.innerSize();
        io_bin(os,rows);
        io_bin(os,cols);
        //assert( static_cast<Derived>(x).data().size() == rows*cols );
        for(size_t i=0U; i<rows*cols; ++i){
            float val = static_cast<float>( x.data()[i] );
            io_bin(os,val);
        }
        return os;
    }
    TMATRIX std::ostream& eigen_io_bin( std::ostream& os, MATRIX const& x ){
        if( std::is_same<typename MATRIX::Scalar, double>::value ){
            std::cout<<" (double-->float write demotion)"; std::cout.flush();
            eigen_io_bin_float( os, x );
        }else{
            eigen_io_bin_plain( os, x );
        }
        return os;
    }
    // dense input
    TMATRIX std::istream& eigen_io_bin_plain( std::istream& is, MATRIX      & x ){
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
    TMATRIX std::istream& eigen_io_bin_float( std::istream& is, MATRIX      & x ){
        uint64_t rows,cols;
        io_bin(is,rows);
        io_bin(is,cols);
        x.resize(rows,cols);
        //std::cout<<" MATRIX-input float->double  rows "<<rows<<" cols "<<cols<<std::endl;
        uint64_t nData = rows*cols;
        typename MATRIX::Scalar *xdata = x.data();
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
    TMATRIX std::istream& eigen_io_bin( std::istream& is, MATRIX      & x ){
        if( std::is_same<typename MATRIX::Scalar, double>::value ){
            std::cout<<" (float-->double read demotion)"; std::cout.flush();
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
    TMATRIX std::ostream& eigen_io_txt( std::ostream& os, MATRIX const& x, char const *ws/*="\n"*/ ){
        using namespace std;
        //cout<<" eigen_io_txt-SPARSE-"<<(x.isCompressed()? "compressed ":"uncompressed "); cout.flush();
        if( x.isCompressed() ){
            os<<x.outerSize()<<' '<<x.innerSize(); os.flush();
            int const nData = x.outerIndexPtr()[ x.outerSize() ];
            os<<' '<<nData; os.flush();       // makes input 'reserve' efficient

            os<<"\n\t"; //os<<"outerIndexPtr[] ";
            for(int i=0U; i<x.outerSize()   + 1   ; ++i) os<<" "<<x.outerIndexPtr()[i];

            os<<"\n\t"; //os<<"innerIndexPtr[] ";
            for(int i=0U; i< nData; ++i) os<<" "<<x.innerIndexPtr()[i];

            os<<"\n\t"; //os<<"valuePtr[] ";
            for(int i=0U; i< nData; ++i) os<<" "<<x.valuePtr()[i];
        }else{
            // after MATRIX::compress(), innerNonZerPtr() returns NULL, so cannot use the following
            os<<x.outerSize()<<' '<<x.innerSize(); os.flush();
            int const nData = x.nonZeros();           // not required
            os<<' '<<nData; os.flush();
            os<<"\n\t"; //os<<"outerIndexPtr[] ";
            typename MATRIX::Index inzSum = 0U;
            for(int i=0U; i<x.outerSize()   + 1   ; ++i){
                //os<<" <"<<x.outerIndexPtr()[i]<<"> "; // outerIndex includes unused memory slots
                os<<" "<<inzSum; os.flush();
                inzSum += x.innerNonZeroPtr()[i];
            }
            os<<"\n\t"; //os<<"innerIndexPtr[] ";
            for(int i=0U; i<x.outerSize(); ++i)
                for(typename MATRIX::InnerIterator it(x,i); it; ++it)
                    os<<" "<<it.col(); os.flush();
            os<<"\n\t"; //os<<"valuePtr[] ";
            for(int i=0U; i<x.outerSize(); ++i)
                for(typename MATRIX::InnerIterator it(x,i); it; ++it)
                    os<<" "<<it.value(); os.flush();
        }
        os<<ws;
        return os;
    }
    TMATRIX std::istream& eigen_io_txt( std::istream& is, MATRIX      & x ){
        using namespace std;
        // is.operator>>( MATRIX ) is NOT AVAILABLE in Eigen
        //cout<<" eigen_io_txt SparseM-input "<<endl;
        size_t rows,cols,nData;
        io_txt(is,rows);
        io_txt(is,cols);
        io_txt(is,nData);
        //cout<<"\trows "<<rows<<" cols "<<cols<<endl;
        x.resize(rows,cols);
        x.setZero();
        x.makeCompressed();
        x.reserve( nData );
        for(size_t i=0U; i<rows+1U; ++i){
            io_txt( is, x.outerIndexPtr()[i] );
            //cout<<" oip["<<i<<"]="<<x.outerIndexPtr()[i]<<endl;
        }
        size_t osz = x.outerIndexPtr()[rows];
        for(size_t i=0U; i<osz; ++i){
            io_txt( is, x.innerIndexPtr()[i] );
            //cout<<" iip["<<i<<"]="<<x.innerIndexPtr()[i]<<endl;
        }
        for(size_t i=0U; i<osz; ++i){
            io_txt( is, x.valuePtr()[i] );
            //cout<<" val["<<i<<"]="<<x.valuePtr()[i]<<endl;
        }

        return is;
    }
    // generic sparse binary output (extra 'isBool' flag)
    TMATRIX std::ostream& eigen_io_bin_impl( std::ostream& os, MATRIX const& x, bool const isBool ){
        using namespace std;
        cout<<" SPARSE-"<<(isBool?"bool":"float")<<"-output rows "<<x.rows()<<" cols "
            <<x.cols()<<" isCompressed()="<<x.isCompressed()<<endl;
        typedef float Real;     // we will convert to 'real' for binary i/o (maybe save space)
        typedef uint64_t Idx;
#define IDX_IO(IDX,TYPE) do{ TYPE idx=static_cast<TYPE>(IDX); io_bin(os,idx); \
    /*cout<<" idx "<<idx<<endl;*/ \
}while(0)
#define REAL_IO(REAL) do{ Real r=static_cast<Real>(REAL); io_bin(os,r); \
    /*cout<<" oval "<<r<<endl;*/ \
}while(0)
        if( x.isCompressed() ){
            int const verbose = 0;
            if(verbose){cout<<" TEST COMPRESSED SPARSE BINARY OUTPUT"<<endl; cout.flush();}
            //os<<x.outerSize()<<' '<<x.innerSize(); os.flush();
            Idx const rows = x.outerSize();
            Idx const cols = x.innerSize();
            io_bin(os,rows);
            io_bin(os,cols);
            //os<<' '<<nData; os.flush();       // makes input 'reserve' efficient
            Idx const nData = x.outerIndexPtr()[ x.outerSize() ]; // # of possibly non-zero items
            io_bin(os,nData);
            if(nData < numeric_limits<uint_least8_t>::max()){
                for(Idx i=0U; i<rows  + 1   ; ++i){
                    assert( static_cast<Idx>(x.outerIndexPtr()[i]) <= nData );
                    IDX_IO(x.outerIndexPtr()[i], uint_least8_t);
                }
            }else if(nData < numeric_limits<uint_least16_t>::max()){
                for(Idx i=0U; i<rows  + 1   ; ++i){
                    assert( static_cast<Idx>(x.outerIndexPtr()[i]) <= nData );
                    IDX_IO(x.outerIndexPtr()[i], uint_least16_t);
                }
            }else if(nData < numeric_limits<uint_least32_t>::max()){
                for(Idx i=0U; i<rows  + 1   ; ++i){
                    assert( static_cast<Idx>(x.outerIndexPtr()[i]) <= nData );
                    IDX_IO(x.outerIndexPtr()[i], uint_least32_t);
                }
            }else{ // original, VERY wide Idx
                //os<<"\n\t"; //os<<"outerIndexPtr[] ";
                //for(int i=0U; i<x.outerSize()   + 1   ; ++i) os<<" "<<x.outerIndexPtr()[i];
                for(Idx i=0U; i<rows  + 1   ; ++i){
                    assert( static_cast<Idx>(x.outerIndexPtr()[i]) <= nData );
                    IDX_IO(x.outerIndexPtr()[i], Idx);
                }
            }
            if(cols < numeric_limits<uint_least8_t>::max()){
                for(Idx i=0U; i< nData; ++i){
                    assert( static_cast<Idx>(x.innerIndexPtr()[i]) < cols );
                    IDX_IO(x.innerIndexPtr()[i], uint_least8_t);
                }
            }else if(cols < numeric_limits<uint_least16_t>::max()){
                for(Idx i=0U; i< nData; ++i){
                    assert( static_cast<Idx>(x.innerIndexPtr()[i]) < cols );
                    IDX_IO(x.innerIndexPtr()[i], uint_least16_t);
                }
            }else if(cols < numeric_limits<uint_least32_t>::max()){
                for(Idx i=0U; i< nData; ++i){
                    assert( static_cast<Idx>(x.innerIndexPtr()[i]) < cols );
                    IDX_IO(x.innerIndexPtr()[i], uint_least32_t);
                }
            }else{ // original, VERY wide Idx
                //os<<"\n\t"; //os<<"innerIndexPtr[] ";
                //for(int i=0U; i< nData; ++i) os<<" "<<x.innerIndexPtr()[i];
                // XXX can be a single i/o if no type conversion.
                for(Idx i=0U; i< nData; ++i){
                    assert( static_cast<Idx>(x.innerIndexPtr()[i]) < cols );
                    IDX_IO(x.innerIndexPtr()[i], Idx);
                }
            }

            //os<<"\n\t"; //os<<"valuePtr[] ";
            //for(int i=0U; i< nData; ++i) os<<" "<<x.valuePtr()[i];
            if( ! isBool ){
                // XXX can be a single i/o if no type conversion.
                for(Idx i=0U; i< nData; ++i) REAL_IO(x.valuePtr()[i]);
            }else{
                boost::dynamic_bitset<uint64_t> dbs(nData,false);
                for(Idx i=0U; i< nData; ++i)
                    dbs.set(i,x.valuePtr()[i]);         // bool valuePtr[i] --> bitset[i]
                // save bools as dynamic_bitset (save storage)
                io_bin(os,dbs);
                // NOTE: slight inefficiency because nData is now stored twice (again as dbs header item)
            }
        }else{
            //cout<<" TEST UNCOMPRESSED SPARSE BINARY OUTPUT"<<endl; cout.flush();
            // after MATRIX::compress(), innerNonZerPtr() returns NULL, so cannot use the following
            //os<<x.outerSize()<<' '<<x.innerSize(); os.flush();
            Idx const rows = x.outerSize();
            Idx const cols = x.innerSize();
            //cout<<" o x c "<<rows<<" x "<<cols; cout.flush();
            io_bin(os,rows);
            io_bin(os,cols);
            //int const nData = x.nonZeros();           // not required
            //os<<' '<<nData; os.flush();       // makes input 'reserve' efficient
            Idx const nData = x.nonZeros();             // ISSUE XXX differs from valuePtr() treatment of compressed!
            //cout<<" nData="<<nData; cout.flush();
            io_bin(os,nData);
            //typename MATRIX::Index inzSum;
            Idx inzSum=0U;
            //os<<"\n\t"; //os<<"outerIndexPtr[] ";
            if(nData < numeric_limits<uint_least8_t>::max()){
                for(Idx i=0U; i<rows  + 1   ; ++i){
                    IDX_IO(x.outerIndexPtr()[i], uint_least8_t);
                    inzSum += static_cast<Idx>(x.innerNonZeroPtr()[i]);
                }
            }else if(nData < numeric_limits<uint_least16_t>::max()){
                for(Idx i=0U; i<rows  + 1   ; ++i){
                    IDX_IO(x.outerIndexPtr()[i], uint_least16_t);
                }
            }else if(nData < numeric_limits<uint_least32_t>::max()){
                for(Idx i=0U; i<rows  + 1   ; ++i){
                    IDX_IO(x.outerIndexPtr()[i], uint_least32_t);
                }
            }else{ // original, VERY wide Idx
                for(int i=0U; i<x.outerSize()   + 1   ; ++i){
                    //os<<" "<<inzSum; os.flush();
                    //io_bin(os,inzSum);
                    IDX_IO(inzSum, Idx);
                    inzSum += static_cast<Idx>(x.innerNonZeroPtr()[i]);     // Index-->Idx (unsigned, known size)
                }
            }
            //os<<"\n\t"; //os<<"innerIndexPtr[] ";
            if(cols < numeric_limits<uint_least8_t>::max()){
                for(int i=0U; i<x.outerSize(); ++i)
                    for(typename MATRIX::InnerIterator it(x,i); it; ++it)
                        IDX_IO(it.col(), uint_least8_t);
            }else if(cols < numeric_limits<uint_least16_t>::max()){
                for(int i=0U; i<x.outerSize(); ++i)
                    for(typename MATRIX::InnerIterator it(x,i); it; ++it)
                        IDX_IO(it.col(), uint_least16_t);
            }else if(cols < numeric_limits<uint_least32_t>::max()){
                for(int i=0U; i<x.outerSize(); ++i)
                    for(typename MATRIX::InnerIterator it(x,i); it; ++it)
                        IDX_IO(it.col(), uint_least32_t);
            }else{
                for(int i=0U; i<x.outerSize(); ++i)
                    for(typename MATRIX::InnerIterator it(x,i); it; ++it){
                        //os<<" "<<it.col(); os.flush();
                        IDX_IO(it.col(), Idx);
                    }
            }
            //os<<"\n\t"; //os<<"valuePtr[] ";
            size_t n=0U;
            if( ! isBool ){
                for(int i=0U; i<x.outerSize(); ++i){
                    for(typename MATRIX::InnerIterator it(x,i); it; ++it){
                        //os<<" "<<it.value(); os.flush();
                        REAL_IO(it.value());
                        ++n;
                    }
                }
            }else{
                boost::dynamic_bitset<uint64_t> dbs(nData,false);
                for(int i=0U; i<x.outerSize(); ++i){
                    for(typename MATRIX::InnerIterator it(x,i); it; ++it){
                        dbs.set(n,it.value());       // sparse bool item COULD be false, still
                        // NB: could optimize for common case of ALL values true XXX
                        ++n;
                    }
                }
                // save bools as dynamic_bitset (save storage)
                io_bin(os,dbs);
            }
            assert( n == nData );
        }
#undef REAL_IO
#undef IDX_IO
        return os;
    }
    // default handles arbitrary values as 'float'
    TMATRIX std::ostream& eigen_io_bin( std::ostream& os, MATRIX const& x ){
        return eigen_io_bin_impl( os, x, /*isBool=*/false );
    }
    // override for bool
    template<int Options, typename Index> inline
    std::ostream& eigen_io_bin( std::ostream& os, Eigen::SparseMatrix<bool,Options,Index> const& x ){
        return eigen_io_bin_impl( os, x, /*isBool=*/true );
    }

    TMATRIX std::istream& eigen_io_bin_impl( std::istream& is, MATRIX      & x, bool const isBool ){
        using namespace std;
        int const verbose = 0;
        if(verbose){cout<<" TEST COMPRESSED SPARSE BINARY INPUT"<<endl; cout.flush();}
        //io_bin(is,(void*)x.data(),size_t(rows*cols*sizeof(typename MATRIX::Scalar)));
        typedef float Real;     // we will convert to 'real' for binary i/o (maybe save space)
        typedef uint64_t Idx;
        Real val;
#define NEXT_IDX(VAR) (io_bin(is,VAR), static_cast<typename MATRIX::Index>(VAR))
#define NEXT_VAL (io_bin(is,val), val)
        Idx rows,cols,nData;
        io_bin(is,rows);
        io_bin(is,cols);
        io_bin(is,nData);
        //cout<<"\trows "<<rows<<" cols "<<cols<<endl;
        x.resize(rows,cols);
        //x.setZero();
        //x.makeCompressed();
        x.resizeNonZeros( nData );      assert( x.data().size() == nData );
        if(verbose){
            cout<<" sparse binary input o x i = "<<rows<<" x "<<cols<<" nData="<<nData<<endl;
            cout<<" outer/inner/data Size() = "<<x.outerSize()<<" "<<x.innerSize()<<" "<<x.data().size()
                <<" rows,cols = "<<x.rows()<<","<<x.cols()<<" x.size()="<<x.size()<<endl;
        }
        assert( x.outerSize() == rows );
        assert( x.innerSize() == cols );
        assert( x.data().size() == nData );
        // XXX look at 'low-level insertBack ???'

        {
            auto idxp = x.outerIndexPtr();
            if(nData < numeric_limits<uint_least8_t>::max()){
                uint_least8_t tmp; for(size_t i=0U; i<rows + 1U ; ++i) *idxp++ = NEXT_IDX(tmp);
            }else if(nData < numeric_limits<uint_least16_t>::max()){
                uint_least16_t tmp; for(size_t i=0U; i<rows + 1U ; ++i) *idxp++ = NEXT_IDX(tmp);
            }else if(nData < numeric_limits<uint_least32_t>::max()){
                uint_least32_t tmp; for(size_t i=0U; i<rows + 1U ; ++i) *idxp++ = NEXT_IDX(tmp);
            }else{ // original, VERY wide Idx
                Idx tmp;
                for(size_t i=0U; i<rows + 1U ; ++i){
                    *idxp++ = NEXT_IDX(tmp);
                    //cout<<" oip["<<i<<"]="<<x.outerIndexPtr()[i]<<endl;
                }
            }
        }
        if(verbose){cout<<"x.outerIndexPtr()[rows] = "<<x.outerIndexPtr()[rows]<<endl;cout.flush();}
        assert( x.outerIndexPtr()[rows] == nData );

        {
            auto idxp = x.innerIndexPtr();
            if(nData < numeric_limits<uint_least8_t>::max()){
                uint_least8_t tmp; for(size_t i=0U; i<nData; ++i) *idxp++ = NEXT_IDX(tmp);
            }else if(nData < numeric_limits<uint_least16_t>::max()){
                uint_least16_t tmp; for(size_t i=0U; i<nData; ++i) *idxp++ = NEXT_IDX(tmp);
            }else if(nData < numeric_limits<uint_least32_t>::max()){
                uint_least32_t tmp; for(size_t i=0U; i<nData; ++i) *idxp++ = NEXT_IDX(tmp);
            }else{
                Idx tmp;
                for(size_t i=0U; i<nData; ++i){
                    *idxp++ = NEXT_IDX(tmp);
                    //cout<<" iip["<<i<<"]="<<x.innerIndexPtr()[i]<<endl;
                }
            }
        }
        if( ! isBool ){
            auto valp = x.valuePtr();
            for(size_t i=0U; i<nData; ++i){
                *valp++ = NEXT_VAL;
                //cout<<" val["<<i<<"]="<<x.valuePtr()[i]<<endl;
            }
        }else{
            boost::dynamic_bitset<uint64_t> dbs;
            io_bin(is,dbs);
            auto valp = x.valuePtr();   // now this is bool*
            for(size_t i=0U; i<nData; ++i){ // copy dbs[i] ---> valuePtr()[i]
                *valp++ = dbs[i];
            }
        }
#undef NEXT_VAL
#undef NEXT_IDX
        return is;
    }
    // default handles arbitrary values as 'float'
    TMATRIX std::istream& eigen_io_bin( std::istream& is, MATRIX      & x ){
        return eigen_io_bin_impl( is, x, /*isBool=*/false );
    }
    // override for bool
    template<int Options, typename Index> inline
    std::istream& eigen_io_bin( std::istream& is, Eigen::SparseMatrix<bool,Options,Index>      & x ){
        return eigen_io_bin_impl( is, x, /*isBool=*/true );
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
template< typename DERIVED > inline
std::string print_report(const Eigen::SparseMatrixBase<DERIVED>& x)
{
    std::ostringstream oss;
    int nnz = x.nonZeros();
    oss << "x:non-zeros: " << nnz << ", avg. nnz/row: " << nnz / x.rows();
    return oss.str();
}



#endif // PRINTING_HH
