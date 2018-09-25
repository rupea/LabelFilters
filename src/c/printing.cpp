/*  Copyright (C) 2017 NEC Laboratories America, Inc. ("NECLA"). All rights reserved.
 *
 * This source code is licensed under the license found in the LICENSE file in
 * the root directory of this source tree. An additional grant of patent rights
 * can be found in the PATENTS file in the same directory.
 */
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
        int const verbose=1;
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
        if(verbose) cout<<" r,c,nData = "<<rows<<", "<<cols<<", "<<nData<<endl;
        {
            auto idxp = x.outerIndexPtr();
            if(nData < numeric_limits<uint_least8_t>::max()){
                if(verbose){cout<<" u8 oindex..."; cout.flush();}
                uint_least8_t tmp; for(size_t i=0U; i<rows + 1U ; ++i) *idxp++ = NEXT_IDX(tmp);
            }else if(nData < numeric_limits<uint_least16_t>::max()){
                if(verbose){cout<<" u16 oindex..."; cout.flush();}
                uint_least16_t tmp; for(size_t i=0U; i<rows + 1U ; ++i) *idxp++ = NEXT_IDX(tmp);
            }else if(nData < numeric_limits<uint_least32_t>::max()){
                if(verbose){cout<<" u32 oindex..."; cout.flush();}
                uint_least32_t tmp; for(size_t i=0U; i<rows + 1U ; ++i) *idxp++ = NEXT_IDX(tmp);
            }else{ // original, VERY wide Idx
                if(verbose){cout<<" Idx oindex..."; cout.flush();}
                Idx tmp;
                for(size_t i=0U; i<rows + 1U ; ++i){
                    *idxp++ = NEXT_IDX(tmp);
                }
            }
        }
        if( x.outerIndexPtr()[rows] != nData )
            throw std::runtime_error(" wrong number of oindex values!");
        {
            auto idxp = x.innerIndexPtr();
            if(cols < numeric_limits<uint_least8_t>::max()){
                if(verbose){cout<<" u8 iindex..."; cout.flush();}
                uint_least8_t tmp; for(size_t i=0U; i<nData; ++i) *idxp++ = NEXT_IDX(tmp);
            }else if(cols < numeric_limits<uint_least16_t>::max()){
                if(verbose){cout<<" u16 iindex..."; cout.flush();}
                uint_least16_t tmp; for(size_t i=0U; i<nData; ++i) *idxp++ = NEXT_IDX(tmp);
            }else if(cols < numeric_limits<uint_least32_t>::max()){
                if(verbose){cout<<" u32 iindex..."; cout.flush();}
                uint_least32_t tmp; for(size_t i=0U; i<nData; ++i) *idxp++ = NEXT_IDX(tmp);
            }else{
                if(verbose){cout<<" Idx iindex..."; cout.flush();}
                Idx tmp;
                for(size_t i=0U; i<nData; ++i){
                    *idxp++ = NEXT_IDX(tmp);
                }
            }
        }
        if(verbose){cout<<" setting 'true'"; cout.flush();}
        { // set all 'values' to 'true' --- they are not present in the istream
            auto valp = x.valuePtr();
            for(size_t i=0U; i<nData; ++i){
                *valp++ = true;
            }
        }
        if(verbose)cout<<" eigen_io_binbool input OK"<<endl;
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
    /** eigen_io_txtbool \em should return with \c is.eof(), and may normally
     * return with \c is.fail(). */
    std::istream& eigen_io_txtbool( std::istream& is, SparseMb      & x ){
        int const verbose=0;
        typedef uint64_t Idx;
        Idx rows;
        Idx cols;
        io_txt(is,rows);
        io_txt(is,cols);
        if( rows<=0 || cols <= 0 )
            throw std::runtime_error("bad eigen_io_txtbool rows/cols input");
        is>>ws;         // OK now do linewise parse, 1 line per example
        if(verbose)cout<<" eigen_io_txtbool: rows="<<rows<<" cols="<<cols<<endl;
        std::string line;
        typedef Eigen::Triplet<bool> T;
        std::vector<T> tripletList;
        tripletList.reserve( rows );    // at least!!!
        for(int row=0; getline(is,line); ++row ){    // one row per line
            if(verbose>1){cout<<"row="<<row<<" line="<<line<<" parse: "; cout.flush();}
            istringstream iss(line);
            Idx idx;
            while(iss>>idx){
                if(verbose>=2){cout<<" "<<idx; cout.flush();}
                tripletList.push_back( T(row,idx,true) );
            }
            if(verbose>=2)cout<<endl;
        }
        x.resize(rows,cols);
        x.setFromTriplets( tripletList.begin(), tripletList.end() );
        if(verbose)cout<<" eigen_io_txtbool read SUCCESS"<<endl;
        return is;
    }

  template< typename X_REAL >
  std::istream& eigen_read_libsvm( std::istream& is,
				   typename Eigen::SparseMatrix<X_REAL,Eigen::RowMajor> &xSparse,
				   Eigen::SparseMatrix<bool,Eigen::RowMajor> &y ){
    xSparse.resize(0,0);
    y.resize(0,0);
    
    int const verbose=1; //
    typedef size_t Idx;
    typedef Eigen::Triplet<bool> B;
    typedef typename Eigen::Triplet<X_REAL> D;
    std::vector<B> yTriplets;
    std::vector<D> xTriplets;
    std::string line;
    std::vector<Idx> yIdx;
    std::vector<Idx> xIdx;
    std::vector<double> xVal;
    size_t row=0U;
    Idx maxClass=0U;
    Idx minClass=std::numeric_limits<Idx>::max();
    Idx maxXidx=0U;
    Idx minXidx=std::numeric_limits<Idx>::max();
    Idx nEx=0U;
    Idx nFeats=0U;
    Idx nClass=0U;
    bool checkHeader = true;

    for(;getline(is,line);){
      istringstream iss(line);
      iss>>ws;
      char const c=iss.peek();
      if(iss.eof() || c == '#') 
	{
	  // empty line or comment line. Skipped.
	  continue; 
	}
      if (checkHeader)
	{
	  // test if XML format. Check if header is present.
	  // only do it once.
	  checkHeader = false;
	  istringstream header(line);	  
	  if (!(header>>ws>>nEx)) 
	    {
	      //this was not a header so no XML format. Reset values and reparse the line. 
	      nEx = 0U;
	    }
	  else if ( !(header>>ws>>nFeats))
	    {
	      //this was not a header so no XML format. Reset values and reparse the line. 
	      nEx = 0U;	      
	      nFeats = 0U;
	    }
	  else if (!(header>>ws>>nClass))
	    {
	      //this was not a header so no XML format. Reset values and reparse the line. 
	      nEx = 0U;
	      nFeats = 0U;
	      nClass = 0U;
	    }
	  else 
	    { 
	      // all read ok. now test that there is nothing else on the line except potential comments
	      header>>ws;	  
	      if (header.eof() || header.peek() == '#') 
		{
		  //correct header has been read, do not parse this line again.
		  if (verbose) 
		    {
		      cout << "Header detected. "<< nEx << " examples, " 
			   << nFeats << " features, " 
			   << nClass << " classes." << endl;
		    }
		  continue;
		}	    
	      else 
		{
		  // there is more on the line. Must not be a header. 
		  nEx = 0U;
		  nFeats = 0U;
		  nClass = 0U;
		}
	    }
	}	
      yIdx.clear();
      xIdx.clear();
      xVal.clear();
      try{
	char sep='x';
	Idx idx;
	while(iss>>idx){
	  yIdx.push_back(idx);
	  iss>>ws;
	  if((sep=iss.peek()) == ':') break;
	  if (sep == ',') iss>>sep>>ws;
	}
	if(sep==':'){
	  iss>>sep;
	  xIdx.push_back( yIdx.back() );
	  yIdx.pop_back();
	  double val;
	  if( !(iss>>val) )
	    throw std::runtime_error(" bad double input?");
	  xVal.push_back(val);
	  for(;iss.good();){
	    iss>>ws;
	    if (iss.eof())
	      break;
	    if(iss.peek() == '#' ) //ignore trailing comments 
	      break;
	    if( !(iss>>idx>>sep>>val))
	      {
		throw std::runtime_error(" libsvm-fmt parse error");
	      }
	    if( sep != ':' ) throw std::runtime_error(" bad sep");
	    xIdx.push_back(idx);
	    xVal.push_back(val);
	  }		  
	  if (iss.fail()) 
	    {
	      throw std::runtime_error(" libsvm parse error"); 
	    }
	}else{
	  throw std::runtime_error(" illegal input line");
	}
      }catch(std::exception const& e){
	cerr<<" Error: " <<e.what() << endl
	    << "Offending line: " << line <<endl;
	throw;
      }
      // move class and data items onto respective TripletLists
      for(size_t i=0U; i<yIdx.size(); ++i){
	if( yIdx[i] > maxClass ) maxClass = yIdx[i];
	if( yIdx[i] < minClass ) minClass = yIdx[i];
	yTriplets.push_back( B(row,yIdx[i],true) );
      }
      for(size_t i=0U; i<xIdx.size(); ++i){
	if( xIdx[i] > maxXidx ) maxXidx = xIdx[i];
	if( xIdx[i] < minXidx ) minXidx = xIdx[i];
	xTriplets.push_back( D(row,xIdx[i],xVal[i]) );
      }
      ++row;
    }
    if (is.fail()&&!is.eof())
      {
	throw runtime_error("Error reading line.");
      }
    
    if(nEx > 0 && row != nEx)
      throw runtime_error("Found more/less examples than declared in the header");

    // solver complains if have any class {0,1,2,...,nClasses-1} with
    // no assigned examples.
    if( minClass > 0 )
      {
	cerr<< "WARNING: No labels with indices below " << minClass << endl 
	    << "The code assumes label indices start at 0!" <<endl;
      }    
    if (nClass > 0)
      {
	if (maxClass > nClass - 1)
	  throw runtime_error("Found more classes than decleared in the header");
	if (maxClass < nClass - 1)
	  {
	    cerr << "Warning: Found fewer classes than decleared in the header";
	    maxClass = nClass-1;
	  }
      }    
    {
      y.resize( row, maxClass+1U );
      y.setFromTriplets( yTriplets.begin(), yTriplets.end() );
      std::vector<B> empty;
      yTriplets.swap(empty); // de-allocate some memory
    }
    
    // libsvm sparse vector has first dimension as '1', we want '0' (ideally)
    // if minXidx > 0, then subtract 1 from all x indices
    // (libsvm format begins sparse format at index "1")
    if( minXidx > 0 )
      {
	cerr << "Warning; No features with index 0. Assuming 1-based feature indices." << endl; 
	--minXidx;
	--maxXidx;
	// Triplet is not modifiable, so construct and replace
	for(size_t i=0U; i<xTriplets.size(); ++i){
	  auto & xi = xTriplets[i];
	  D dnew( xi.row(), xi.col()-1U, xi.value() );
	  xi = dnew;
	}
      }
    if ( nFeats > 0 )
      {
	if (maxXidx > nFeats - 1)
	  throw runtime_error("Found more features than declared in the header");
	maxXidx = nFeats - 1;
      }
    {
      xSparse.resize(row,maxXidx+1U);
      xSparse.setFromTriplets(xTriplets.begin(), xTriplets.end());
      std::vector<D> empty;
      xTriplets.swap(empty); // de-allocate xTriplets memory
    }
    if(verbose>=1){cout<<" GOOD libsvm-like text input, minClass="<<minClass
		       <<" maxClass="<<maxClass
		       <<" minXidx="<<minXidx<<" maxXidx="<<maxXidx<<" rows="<<row<<endl;}
    
    return is;
  }  
  
  template
  std::istream& eigen_read_libsvm( std::istream& is,
				   SparseMf &x,
				   SparseMb &y );
  template
  std::istream& eigen_read_libsvm( std::istream& is,
				   SparseM &x,
				   SparseMb &y );
}//detail::



