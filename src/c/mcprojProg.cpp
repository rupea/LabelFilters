
#include "mcprojProg.hpp"
#include "mcxydata.h"
#include "printing.hh"
#include "normalize.h"
#include "mcpredict.hh"
#include "predict.hh"           // low-level tests [optional]
#include "utils.h"              // OUTWIDE

//#include <omp.h>

#include <stdexcept>
#include <iostream>
#include <fstream>
#include <iomanip>
#include <memory> //shared_ptr

namespace opt {
  using namespace std;
  /**
   * - What are the default parameters?
   *   - 1. if given explicitly as \c defparms
   *   - 2. if --solnFile, adopt parms from a previous run
   *   - 3. else use library defaults
   */
  MCprojProgram::MCprojProgram( int argc, char**argv
				, int const verbose/*=0*/ )
    : ::opt::MCprojArgs( argc, argv )
    , ::MCfilter()
    , xy( new ::MCxyData() )
    , feasible()
  {
    // defaults: if given explicitly as \c defparms
    if (A::solnFile.size()==0) 
      {
	throw runtime_error("MCprojProg: must specify label filter, or file to read it from via --solnFile");
      }
   
    if(verbose>=1){
      //++A::verbose;
      cout<<" +MCprojProgram --solnFile="<<A::solnFile;
      if(A::xFile.size()) cout<<" --xFile="<<A::xFile;
      if(A::outFile.size()) cout<<" --output="<<A::outFile;
      if(A::maxProj) cout<<" --proj="<<A::maxProj;
      if(A::outBinary) cout<<" -B";
      if(A::outText)   cout<<" -T";
      if(A::outSparse) cout<<" -S";
      if(A::outDense)  cout<<" -D";
      //            if(A::yFile.size()) cout<<(A::yPerProj?" --Yfile=":" --yfile")<<A::yFile;
      //if(A::threads)   cout<<" --threads="<<A::threads;
      if(A::xnorm)     cout<<" --xnorm";
      if(A::xunit)     cout<<" --xunit";
      if(A::xscale!=1.0) cout<<" --xscale="<<A::xscale;
      cout<<endl;
    }
  }

  MCprojProgram::MCprojProgram( ::MCsoln const& soln, shared_ptr<::MCxyData> data /*=nullptr*/,  std::string mod/*=std::string()*/ )
    : ::opt::MCprojArgs( mod )
    , ::MCfilter(soln)
    , xy( data )
    , feasible()
  {}
  MCprojProgram::~MCprojProgram(){}
  
  void MCprojProgram::tryRead( int const verb/*=0*/ )
  {
    int const verbose = A::verbose + verb;  // verb modifies the initial value from MCprojArgs --verbose
    if(verbose>=1){cout<<" MCprojProgram::tryRead()"; cout.flush();}
    {
      if (A::solnFile.size()){
	try{
	  this->read( A::solnFile );
	}catch(std::exception const& e){
	  cerr<<"Error reading solnfile "<<A::solnFile<<endl<<e.what()<<endl;
	  throw;
	}		  
	if(verbose>=2) F::pretty(cout);
      }
    }
    
    if (A::xFile.size())
      {
	readData(A::xFile);
      }
  }

  void MCprojProgram::readData(std::string const& xfile)
  {
    // obtain this->xy->MCxyData::{xDense, denseOk, xSparse,sparseOk}
    xy = std::make_shared<::MCxyData>();
    if(verbose>=1) cout<<" try reading DenseM from xfile="<<xfile<<endl;
    xy->xread(xfile);    // tries binary dense, sparse, (others?)            
    if( A::xnorm ){
      // SHOULD USE THE MEAN/SDEV CALCULATED FOR THE TRAINIGN SET, NOT THE ONES FOR THE PROJECTION SET!!!
      ::Eigen::VectorXd xmean;
      ::Eigen::VectorXd xstdev;
      if(verbose>=1){
	cout<<" xnorm!"<<endl;
      }
      //for sparse data do not center (should be an option)
      cerr << "Warning: column normalization is done using statistics on the prediction set.\n"
	   << "         Training set statistics should probably be used" << endl
	   << endl; 
      if (xy->sparseOk)
	cerr << "Warning: --xnorm is used with sparse data. Features are not centered to maintain sparsity" << endl;	      
      xy->xstdnormal(xmean, xstdev, true, !xy->sparseOk, false);
      if(verbose>=1){
	cout<<"xmeans"<<prettyDims(xmean)<<":\n"<<xmean.transpose()<<endl;
	cout<<"xstdev"<<prettyDims(xstdev)<<":\n"<<xstdev.transpose()<<endl;
      }
    }
    
    if( A::xunit ){
      xy->xunitnormal();
    }
    
    if( A::xscale != 1.0 ){
      xy-> xscale( A::xscale );
    }
    
    if(verbose>=1){
      if( xy->denseOk ){
	cout<<"--------- xy->xDense"<<prettyDims(xy->xDense)<<":\n";
	if(xy->xDense.size()<1000&&verbose>=3) cout<<xy->xDense<<endl;
      }else{ //xy->sparseOk
	cout<<"--------- xy->xSparse"<<prettyDims(xy->xSparse)<<":\n";
	if(xy->xSparse.size()<1000&&verbose>=3) cout<<xy->xSparse<<endl;
      }
    }
  }

  void MCprojProgram::setData(shared_ptr<::MCxyData> data)
  {
    xy = data;
    if (A::xnorm || A::xunit || A::xscale!=1.0)
      {
	cerr << "Warning: setData is used. Data will not be normalized or scaled.";
      } 
  }
  

  void MCprojProgram::tryProj( int const verb/*=0*/ )
  {
    int const verbose = A::verbose + verb;  // verb modifies the initial value from MCprojArgs --verbose
    if(verbose>=1) cout<<" MCprojProgram::tryProj() "<<(xy->denseOk?"dense":xy->sparseOk?"sparse":"HUH?")<<endl;
    
    // if the data or the solution has not been read, do it now. 
    if((!xy && A::xFile.size()) || (F::isempty() && A::solnFile.size())) 
      {
	tryRead();
      }	
    if (!xy)
      {	
	throw runtime_error("tryProj called without data");
      }
    
    if( xy->denseOk ){
      F::filter(feasible, xy->xDense, A::maxProj );
    }else if( xy->sparseOk ){
      F::filter(feasible, xy->xSparse, A::maxProj );
    }else{
      throw std::runtime_error("neither sparse nor dense training x was available");
    }      
  }
  

  // void MCprojProgram::dumpFeasible(std::ostream& os
  // 				   , std::vector<boost::dynamic_bitset<>> const& vbs
  // 				   , bool denseFmt/*=false*/)
  // {
  //   for(uint32_t i=0U; i<vbs.size(); ++i){
  //     auto const& fi = vbs[i];
  //     if( denseFmt ){
  // 	for(uint32_t c=0U; c<fi.size(); ++c) os<<fi[c];
  //     }else{
  // 	for(uint32_t c=0U; c<fi.size(); ++c) if( fi[c] ) os<<c<<" ";
  //     }
  //     os<<endl;
  //   }
  // }


  void MCprojProgram::trySave( int const verb/*=0*/ )
  {
    int const verbose = A::verbose + verb;  // verb modifies the initial value from MCprojArgs --verbose
    if(verbose>=1) cout<<" MCprojProgram::trySave()"
		       <<"\tSaving feasible "<<(outBinary?"Binary":"Text")<<" "
		       <<(outSparse?"Sparse":"Dense")<<" classes --> "
		       <<(outFile.size()? outFile: string("cout"))
		       <<endl;
    
    ofstream ofs;
    if (outFile.size()==0){
      cerr<<"Warning: Binary flag ignored for cout"<<endl;
      outBinary = false;
    }else{
      ofs.open(outFile);
    }
    ostream& out = outFile.size()?ofs:cout;
    if( ! out.good() ) throw std::runtime_error("trouble opening outFile");
    
    if(outBinary){	
      if(outSparse){
	cerr<<"Warning: Sparse flag ignored fro binary dump -- see printing.hh to implement it 'magically'"<<endl;
      }
      detail::io_bin( out, feasible );
      if(verbose>=2) cout<<"Note:\tIn C++, use printing.hh\n\t\tio_bin(ifstream(\""<<outFile
			 <<"\"),vector<boost::dynamic_bitset<>>&)\n\tcan read the projections binary file";
    }else{ // outText
      out<<"## "<<feasible.size()<<" instances" << endl;
      out <<"## "<<F::lower_bounds.rows()<<" classes" <<endl;
      dumpFeasible( out, feasible, outDense );      
      if( ! out.good() ) throw std::runtime_error("trouble writing outFile");
    }
    if (A::outFile.size()) ofs.close();
    
  }

}//opt::
