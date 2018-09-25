/*  Copyright (C) 2017 NEC Laboratories America, Inc. ("NECLA"). All rights reserved.
 *
 * This source code is licensed under the license found in the LICENSE file in
 * the root directory of this source tree. An additional grant of patent rights
 * can be found in the PATENTS file in the same directory.
 */
#include "linearModel.h"
#include "linearModel.hh"
#include "linearModel_detail.hh"
#include "printing.hh"
#include "mcxydata.h"
#include <fstream>


// ... linearModel magic headers (simplify I/O)
std::array<char,4> linearModel::magic_Sparse = {0,'W','s','4'}; // values saved as floats
std::array<char,4> linearModel::magic_Dense  = {0,'W','d','4'}; // values saved as floats
// ...

#define MAGIC_EQU( A, B ) (A[0]==B[0] && A[1]==B[1] && A[2]==B[2] && A[3]==B[3])


linearModel::linearModel()
  : WDense(),
    denseOk(false),
    WSparse(),
    sparseOk(false)
{}

linearModel::linearModel(ovaDenseColM const& W, Eigen::RowVectorXd const& I)
  :WDense(W),
   denseOk(true),
   WSparse(),
   sparseOk(false),
   intercept(I)
{}

linearModel::linearModel(ovaSparseColM const& W, Eigen::RowVectorXd const& I)
  :WDense(),
   denseOk(false),
   WSparse(W),
   sparseOk(true),
   intercept(I)
{}


void linearModel::write(std::string fname, bool binary /*=true*/) const{ // write binary or text (either sparse/dense)
  ofstream ofs;
  try
    {
      ofs.open(fname);
      if( ! ofs.good() ) throw std::runtime_error("trouble opening the output file");
      if( !sparseOk && !denseOk )
	throw std::runtime_error("no model to be saved");
      if( sparseOk )
	{
	  if (binary)
	    {
	      detail::io_bin(ofs,linearModel::magic_Sparse);
	      detail::eigen_io_bin(ofs, WSparse);
	      if (intercept.size() > 0) detail::eigen_io_bin(ofs, intercept);
	    }
	  else
	    {
	      detail::eigen_io_txt(ofs,WSparse);
	      if (intercept.size() > 0) detail::eigen_io_txt(ofs, intercept);
	    }
	}
      else
	{
	  assert(denseOk);
	  if (binary)
	    {
	      detail::io_bin(ofs,linearModel::magic_Dense);
	      detail::eigen_io_bin(ofs, WDense);
	      if (intercept.size() > 0) detail::eigen_io_bin(ofs, intercept);
	    }
	  else
	    {
	      throw std::runtime_error("Dense text format not implemented for linear models");
	      //detail::eigen_io_txt(ofs, WDense);
	      //if (intercept.size() > 0) detail::eigen_io_txt(ofs, intercept);
	    }
	}
      if( ! ofs.good() ) throw std::runtime_error("trouble writing to the output file");
      ofs.close();
    }
  catch(std::exception const& e)
    {
      cerr<<" trouble writing "<<fname<<" : unknown exception"<<endl;
      ofs.close();
      throw;
    }
  }
  

  // read with magic header
void linearModel::read( std::string fname ){
  ifstream ifs;
  std::array<char,4> magicHdr;
  // TODO XXX try Dense-Text, Sparse-Text too?
  try{
    ifs.open(fname);
    if( ! ifs.good() ) throw std::runtime_error("trouble opening fname");
    detail::io_bin(ifs,magicHdr);
    if( MAGIC_EQU(magicHdr,linearModel::magic_Dense)){
      read(ifs, false, true);
    }else if( MAGIC_EQU(magicHdr,linearModel::magic_Sparse)){
      read(ifs, true, true);
    }else{      
      // not binary. Try sparse text (libsvm-like format)
      ifs.seekg(ios::beg);	
      read(ifs, true, false);
    }
    // Not here [yet]: dense text format? milde repo?
    ifs.close();
  }catch(std::exception const& e){
    cerr<<" oops reading the linear model data from "<<fname<<" ... "<<e.what()<<endl;
    ifs.close();
    throw;
  }
}

// read without magic header. 
void linearModel::read( ifstream& ifs, bool sparse, bool binary){   
  try{
    if( binary && !sparse){
      detail::eigen_io_bin(ifs, WDense);
      assert( WDense.cols() > 0U );            
      if( ifs.fail() ) throw std::underflow_error("problem reading linear model  weight matrix from file with eigen_io_bin");
      ifs.get(); // test if an intercept vector is present
      if ( !ifs.eof() )
	{
	  // intercept is present 
	  ifs.unget();
	  detail::eigen_io_bin(ifs, intercept);
	  if (ifs.fail()) throw std::runtime_error("problem reading linear model intercepts form file with eigen_io_bin");
	  if (intercept.size() != WDense.cols()) throw std::runtime_error("Dimmensions of intercept and weight matrix do not match");
	  ifs.get();  // should trigger eof if BINARY dense file exactly the write length
	  if( ! ifs.eof() ) throw std::overflow_error("linear model read did not use full file");
	}
      denseOk=true;
      // if there was data in sparse format invalidate it and free the memeory.
      if (sparseOk)
	{
	  sparseOk = false;
	  WSparse = ovaSparseColM();	
	}
    }else if( binary && sparse ){
      detail::eigen_io_bin( ifs, WSparse );
      if( ifs.fail() ) throw std::underflow_error("problem reading SparseM from xfile with eigen_io_bin");
      assert( WSparse.cols() > 0U );
      ifs.get(); // test if an intercept vector is present
      if ( !ifs.eof() )
	{
	  // intercept is present 
	  ifs.unget();
	  detail::eigen_io_bin(ifs, intercept);
	  if (ifs.fail()) throw std::runtime_error("problem reading linear model intercepts form file with eigen_io_bin");
	  if (intercept.size() != WSparse.cols()) throw std::runtime_error("Dimmensions of intercept and weight matrix do not match");
	  ifs.get();  // should trigger eof if BINARY dense file exactly the write length
	  if( ! ifs.eof() ) throw std::overflow_error("linear model read did not use full file");
	}
      sparseOk=true;
      // if there was data in sparse format invalidate it and free the memeory.
      if (denseOk)
	{
	  denseOk = false;
	  WDense = ovaDenseColM();
	}
    }else if (!binary && sparse){      
      // sparse text (libsvm-like format)
      detail::eigen_io_txt(ifs, WSparse);
      if (ifs.fail()) throw std::runtime_error("Error reading file");
      ifs >> ws;
      if (!ifs.eof())
	{ // read the intercepts
	  detail::eigen_io_txt(ifs,intercept);
	  if (ifs.fail()) throw std::runtime_error("problem reading linear model intercepts form file with eigen_io_bin");
	  if (intercept.size() != WSparse.cols()) throw std::runtime_error("Dimmensions of intercept and weight matrix do not match");
	  ifs >> ws;
	  if( ! ifs.eof() ) throw std::overflow_error("linear model read did not use full file");
	}      
      sparseOk = true;
      if (denseOk)
	{
	  denseOk= false;
	  WDense = ovaDenseColM();
	}
    }else{
      throw std::runtime_error("Format not implemented");
    }   
    // Not here [yet]: dense text format? milde repo?
  }catch(std::exception const& e){
    ifs.close();
    throw;
  }
}

// Explicitly instantiate predict into the library

template
size_t linearModel::predict (PredictionSet& predictions, DenseM const& x, ActiveSet const* feasible, bool verbose, predtype keep_thresh, size_t keep_size) const;
template
size_t linearModel::predict (PredictionSet& predictions, SparseM const& x, ActiveSet const* feasible, bool verbose, predtype keep_thresh, size_t keep_size) const;
//template
//size_t linearModel::predict (PredictionSet& predictions, ExtConstSparseM const& x, ActiveSet const* feasible, bool verbose, predtype keep_thresh, size_t keep_size) const;
  
