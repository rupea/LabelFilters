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
#include "printing.hh"
#include <sstream>

using Eigen::VectorXd;
using namespace std;


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
  
  std::istream& eigen_read_libsvm( std::istream& is,
				   SparseM &xSparse,
				   SparseMb &y,
				   int const verbose /*=0*/){
    typedef size_t Idx;
    typedef Eigen::Triplet<bool> B;
    typedef typename Eigen::Triplet<double> D;
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
	  if (!(header>>nEx>>nFeats>>nClass)) 
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
      // not empty, comment or header. Parse the line.
      try{
	parse_labels(iss, yIdx);
	parse_features(iss, xIdx, xVal);
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
    
}//detail::



