#include <iostream>
#include <stdio.h>
#include "Eigen/Dense"
#include "Eigen/Sparse"
#include "typedefs.h"
#include "printing.hh"
#include <sstream>

using Eigen::VectorXd;
using namespace std;

#define MAGIC_EQU( A, B ) (A[0]==B[0] && A[1]==B[1] && A[2]==B[2] && A[3]==B[3])
namespace detail {
    std::array<char,4> magicSparseMbBin = {'S', 'M', 'b', 'b' };
    std::array<char,4> magicSparseMbTxt = {'S', 'M', 'b', 't' };
}//detail::

ostream& operator<<(ostream& os, PrettyDimensions const& pd )
{
    os<<'[';
    decltype(pd.dim) d=0U;
    if(pd.dim) do{
        os<< pd.dims[d];
        if(++d >= pd.dim) break;
        os<<'x';
    }while(1);
    os<<']';
    assert( pd.dim <= PrettyDimensions::maxDim );
    return os;
}

// *******************************
// Prints the progress bar
void print_progress(string s, int t, int max_t)
{
  double p = ((double) ((double) t * 100.0)) / (double) max_t;
  int percent = (int) p;

  string str = "\r" + s + "=";
  for (int i = 0; i < (int) percent / 10; i++)
    {
      str = str + "==";
    }
  
  int c = 1000;
  char buff[c];
  sprintf(buff,
	  " > (%d%%) @%d                     ",
	  percent, t);
  str = str + buff;
  
  cout << str;
  
  if (percent == 100)
    {
      cout << std::endl;
    }
}

template< typename EigenType > std::string print_report(EigenType const& x);

void print_report(const int projection_dim, const int batch_size,
                  const int noClasses, const double C1, const double C2, const double lambda, const int w_size,
                  std::string x_report) //const EigenType& x)
{
    using namespace std;
    cout << "projection_dim: " << projection_dim << ", batch_size: "
        << batch_size << ", noClasses: " << noClasses << ", C1: " << C1
        << ", C2: " << C2 << ", lambda: " << lambda << ", size w: " << w_size;
    if(x_report.size()) cout<< ", "<<x_report; // print_report(x);
    cout << "\n-----------------------------\n";

}

namespace detail {
    std::ostream& eigen_io_binbool( std::ostream& os, SparseMb const& x )
    {
        if( ! x.isCompressed() )
            throw std::runtime_error(" eigen_io_binbool output requires a COMPRESSED matrix"
                                     " pruned to contain only true values");
        // following Map is not resolved by gcc-4.9
        //auto const trueCount = ( Eigen::Map<Eigen::VectorXi>(x.valuePtr(), x.nonZeros()).array()!=false ).count();
        size_t trueCount = 0U;
        for(int i=0; i<x.nonZeros(); ++i) trueCount += (x.valuePtr()[i] == true);
        if( trueCount != x.nonZeros() )
        throw std::runtime_error(" eigen_io_binbool output requires a compressed matrix"
                                 " pruned to contain ONLY TRUE values");
        typedef uint64_t Idx;
#define IDX_IO(IDX,TYPE) do{ TYPE idx=static_cast<TYPE>(IDX); io_bin(os,idx); }while(0)
        io_bin( os, magicSparseMbBin );
        Idx const rows = x.rows();
        Idx const cols = x.cols();
        io_bin(os,rows);
        io_bin(os,cols);
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
            for(Idx i=0U; i<rows  + 1   ; ++i){
                assert( static_cast<Idx>(x.outerIndexPtr()[i]) <= nData );
                IDX_IO(x.outerIndexPtr()[i], Idx);
            }
        }
        //
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
            for(Idx i=0U; i< nData; ++i){
                assert( static_cast<Idx>(x.innerIndexPtr()[i]) < cols );
                IDX_IO(x.innerIndexPtr()[i], Idx);
            }
        }
        // all 'values' are 'true' --- they are not saved in the ostream
        return os;
    }
#undef IDX_IO

    std::istream& eigen_io_binbool( std::istream& is, SparseMb      & x )
    {
        std::array<char,4> magic;
        io_bin( is, magic );
        if( ! MAGIC_EQU(magic,magicSparseMbBin) )
            throw runtime_error(" Expected magicSparseMbBin header not found");
        typedef uint64_t Idx;
#define NEXT_IDX(VAR) (io_bin(is,VAR), static_cast<SparseMb::Index>(VAR))
        Idx rows,cols,nData;
        io_bin(is,rows);
        io_bin(is,cols);
        io_bin(is,nData);
        x.resize(rows,cols);
        x.resizeNonZeros( nData );      assert( x.data().size() == nData );
        assert( x.outerSize() == rows );
        assert( x.innerSize() == cols );
        assert( x.data().size() == nData );
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
        { // set all 'values' to 'true' --- they are not present in the istream
            auto valp = x.valuePtr();
            for(size_t i=0U; i<nData; ++i){
                *valp++ = true;
            }
        }
#undef NEXT_IDX
        return is;
    }
    std::ostream& eigen_io_txtbool( std::ostream& os, SparseMb const& x ){
        typedef uint64_t Idx;
        Idx const rows = x.rows();
        Idx const cols = x.cols();
        io_txt(os,rows);
        io_txt(os,cols);
        int line=0;                     // one output line per row (example)
        bool firstitem = true;
        for(int i=0; i<x.outerSize(); ++i){
            for(SparseMb::InnerIterator it(x,i); it; ++it){
                bool const val = it.value();
                cout<<" line="<<line<<" it.col,row = "<<it.col()<<","<<it.row()
                    <<" val="<<val<<endl;
                if( val != false ){
                    if( it.row() == line && !firstitem ) os<<" ";
                    else{
                        while( line != it.row() ) { os<<"\n"; ++line; }
                        firstitem = true;
                    }
                    os<<it.col();       // containing list of 'true' labels
                    firstitem = false;
                }
            }
        }
        return os;
    }
    std::istream& eigen_io_txtbool( std::istream& is, SparseMb      & x ){
        cout<<"WARNING: eigen_io_txtbool needs to be tested"<<endl;
        typedef uint64_t Idx;
        Idx rows;
        Idx cols;
        io_txt(is,rows);
        io_txt(is,cols);
        is>>ws;         // OK now do linewise parse, 1 line per example
        std::string line;
        typedef Eigen::Triplet<bool> T;
        std::vector<T> tripletList;
        tripletList.reserve( rows );    // at least!!!
        for(int row=0; getline(is,line); ++row ){    // one row per line
            istringstream iss(line);
            Idx idx;
            while(iss>>idx){
                tripletList.push_back( T(row,idx,true) );
            }
        }
        x.setFromTriplets( tripletList.begin(), tripletList.end() );
        return is;
    }
}//detail::
