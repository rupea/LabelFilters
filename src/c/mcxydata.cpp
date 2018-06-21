#include "mcxydata.h"
#include "mcxydata_detail.h"
#include "mcxydata_detail.hh"
//#include "normalize.hh"
#include "printing.h"
#include "printing.hh"

//#include <stdexcept>
#include <iostream>
#include <fstream>
//#include <sstream>
//#include <iomanip>

// ... MCxyData magic headers (simplify I/O)
std::array<char,4> MCxyData::magic_xSparse = {0,'X','s','4'}; // value saved as floats
std::array<char,4> MCxyData::magic_xDense  = {0,'X','d','4'}; // values saved as floats 
// x text mode not supported so far.
std::array<char,4> MCxyData::magic_yBin    = {0,'Y','s','b'};
// y text mode readable but has no magic.
// ...


using namespace std;
using namespace Eigen;
using namespace detail;

#define MAGIC_EQU( A, B ) (A[0]==B[0] && A[1]==B[1] && A[2]==B[2] && A[3]==B[3])

MCxyData::MCxyData() : xDense(), denseOk(false), xSparse(), sparseOk(false)
                       , y(), qscal(0.0), xscal(0.0) {}

MCxyData::MCxyData(DenseM const& x): 
  xDense(x), denseOk(true), xSparse(), sparseOk(false)
  , y(), qscal(0.0), xscal(1.0)
{}

MCxyData::MCxyData(DenseM const& x, SparseMb const& y): 
  xDense(x), denseOk(true), xSparse(), sparseOk(false)
  , y(y), qscal(0.0), xscal(1.0)
{}

MCxyData::MCxyData(SparseM const& x): 
  xDense(),denseOk(false),xSparse(x), sparseOk(true)
  , y(), qscal(0.0), xscal(1.0)
{}

MCxyData::MCxyData(SparseM const& x, SparseMb const& y): 
  xDense(), denseOk(false), xSparse(x), sparseOk(true)
  , y(y), qscal(0.0), xscal(1.0)
{}
  

void MCxyData::xscale( double const mul ){
    if(mul != 1.0){
        if( sparseOk ) xSparse *= mul;
        if( denseOk )  xDense  *= mul;
        if(xscal==0.0) xscal=1.0;
        xscal *= mul;
        qscal *= mul;
    }
}

void MCxyData::xunitnormal(){
  // have normalization indicators? 
  if (denseOk) mcxydata_detail::row_unit_normalize(xDense);
  if (sparseOk) mcxydata_detail::row_unit_normalize(xSparse);
}

void MCxyData::xstdnormal(bool colNorm, bool center){
  VectorXd foo, bar;
  this->xstdnormal(foo,bar,colNorm,center,false);
}

void MCxyData::xstdnormal(VectorXd& mean, VectorXd& stdev, bool colNorm, bool center, bool useMeanStdev){
  // we may not want implicit conversion to dense. 
    if( !denseOk ) throw std::runtime_error("no dense data for xrnormal. If sparse data please convert to dense");
    if( sparseOk ){
      //invalidate the sparse data as the dense one will change. Free the memory.
      //we may not want implicit conversion to sparse since it mith be made dense by the mean removal.
      // this should not be called since data is supposed to be in either sparse or dense format. But just in case.
      sparseOk = false;
      xSparse = SparseM();
    }
    if (colNorm) mcxydata_detail::col_mean_std_normalize( xDense, mean, stdev, center, useMeanStdev); 
    else         mcxydata_detail::row_mean_std_normalize( xDense, mean, stdev, center, useMeanStdev);     
}

void MCxyData::removeRareFeatures(const int minex /*=1*/){
  std::vector<std::size_t> foo, bar;
  removeRareFeatures(foo, bar, minex, false);
}

void MCxyData::removeRareFeatures(std::vector<std::size_t>& feature_map, std::vector<std::size_t>& reverse_feature_map, const int minex /*=1*/, const bool useFeatureMap /*=false*/ ){
  if (denseOk)  mcxydata_detail::remove_rare_features(xDense, feature_map, reverse_feature_map, minex, useFeatureMap);
  if (sparseOk) mcxydata_detail::remove_rare_features(xSparse, feature_map, reverse_feature_map, minex, useFeatureMap);
}

void MCxyData::removeRareLabels(const int minex /*=1*/){
  std::vector<std::size_t> foo, bar;
  removeRareLabels(foo, bar, minex, false);
}

void MCxyData::removeRareLabels(std::vector<std::size_t>& label_map, std::vector<std::size_t>& reverse_label_map, const int minex /*=1*/, const bool useLabelMap /*=false*/ ){
  if (y.cols() > 0)  mcxydata_detail::remove_rare_features(y, label_map, reverse_label_map, minex, useLabelMap);
}

void MCxyData::read( std::string xFile, std::string yFile /*=""*/ ){
  xread(xFile);
  if (yFile.size())
    {
      yread(yFile);
    }
}

void MCxyData::xread( std::string xFile ){
  ifstream xfs;
  std::array<char,4> magicHdr;
  // TODO XXX try Dense-Text, Sparse-Text too?
  try{
    xfs.open(xFile);
    if( ! xfs.good() ) throw std::runtime_error("trouble opening xFile " + xFile);
    detail::io_bin(xfs,magicHdr);
    if( MAGIC_EQU(magicHdr,MCxyData::magic_xDense)){
      detail::eigen_io_bin(xfs, xDense);
      if( xfs.fail() ) throw std::underflow_error("problem reading DenseM from xfile with eigen_io_bin");
      char c;
      xfs >> c;   // should trigger eof if BINARY dense file exactly the write length
      if( ! xfs.eof() ) throw std::overflow_error("xDense read did not use full file");
      xfs.close();
      assert( xDense.cols() > 0U );
      denseOk=true;
      // if there was data in sparse format invalidate it and free the memeory.
      if (sparseOk)
	{
	  sparseOk = false;
	  xSparse = SparseM();
	}
    }else if( MAGIC_EQU(magicHdr,MCxyData::magic_xSparse)){
      detail::eigen_io_bin( xfs, xSparse );
      if( xfs.fail() ) throw std::underflow_error("problem reading SparseM from xfile with eigen_io_bin");
      xfs.close();
      assert( xSparse.cols() > 0U );
      sparseOk=true;
      // if there was data in sparse format invalidate it and free the memeory.
      if (denseOk)
	{
	  denseOk = false;
	  xDense = DenseM();
	}
    }else{
      // not binary. Try libSVM/XML format
      xfs.seekg(ios::beg);
      detail::eigen_read_libsvm(xfs, xSparse, y);
      sparseOk = true;
      if (denseOk)
	{
	  denseOk= false;
	  xDense = DenseM();
	}
      // Not here [yet]: text formats? milde repo?
    }
    xfs.close();
  }catch(std::exception const& e){
    cerr<<" oops reading x data from "<<xFile<<" ... "<<e.what()<<endl;
    throw;
  }
}
void MCxyData::xwrite( std::string fname ) const { // write binary (either sparse/dense)
  ofstream ofs;
  try{
    if( !sparseOk && !denseOk )
      throw std::runtime_error("savex no Eigen x data yet");
    ofs.open(fname);
    if( ! ofs.good() ) throw std::runtime_error("savex trouble opening fname");
    if( sparseOk ){
      detail::io_bin(ofs,MCxyData::magic_xSparse);
      detail::eigen_io_bin(ofs, xSparse);
    }else{ assert(denseOk);
      detail::io_bin(ofs,MCxyData::magic_xDense);
      detail::eigen_io_bin(ofs, xDense);
    }
    if( ! ofs.good() ) throw std::runtime_error("savex trouble writing fname");
    ofs.close();
  }catch(std::exception const& e){
    cerr<<" trouble writing "<<fname<<" : unknown exception"<<endl;
    ofs.close();
    throw;
  }
}
void MCxyData::yread( std::string yFile ){
  std::array<char,4> magicHdr;
  ifstream yfs;
  bool yOk = false;
  try{
    yfs.open(yFile);
    if( ! yfs.good() ) throw std::runtime_error("ERROR: opening SparseMb yfile");
    detail::io_bin(yfs,magicHdr);
    if( MAGIC_EQU(magicHdr,magic_yBin) ){
      detail::eigen_io_binbool( yfs, y );
      assert( y.cols() > 0U );
      if( yfs.fail() ) throw std::underflow_error("problem reading yfile with eigen_io_binbool");
      yOk = true;
    }
  }catch(std::runtime_error const& e){
    cerr<<e.what()<<endl;
    //throw; // continue execution -- try text format
  }catch(std::exception const& e){
    cerr<<"ERROR: during read of classes from "<<yFile<<" -- "<<e.what()<<endl;
    throw;
  }
  if( !yOk ){
    cerr<<"Retrying --yfile as text mode list-of-classes format (eigen_io_txtbool)"<<endl;
    try{
      yfs.close();
      yfs.open(yFile);
      if( ! yfs.good() ) throw std::runtime_error("ERROR: opening SparseMb yfile");
      detail::eigen_io_txtbool( yfs, y );
      assert( y.cols() > 0U );
      // yfs.fail() is expected
      if( ! yfs.eof() ) throw std::underflow_error("problem reading yfile with eigen_io_txtbool");
      yOk=true;
    }
    catch(std::exception const& e){
      cerr<<" --file could not be read in text mode from "<<yFile<<" -- "<<e.what()<<endl;
      throw;
    }
  }
}
void MCxyData::ywrite( std::string yFile ) const { // write binary y data (not very compact!)
    ofstream ofs;
    try{
        ofs.open(yFile);
        if( ! ofs.good() ) throw std::runtime_error("ywrite trouble opening file");
	detail::io_bin(ofs,MCxyData::magic_yBin);
        detail::eigen_io_binbool(ofs, y);
        if( ! ofs.good() ) throw std::runtime_error("ywrite trouble writing file");
        ofs.close();
    }catch(std::exception const& e){
        cerr<<" ywrite trouble writing "<<yFile<<" : unknown exception"<<e.what()<<endl;
        ofs.close();
        throw;
    }
}
std::string MCxyData::shortMsg() const {
    ostringstream oss;
    if( denseOk ) oss<<" dense x "<<prettyDims(xDense);
    if( sparseOk ) oss<<" sparse x "<<prettyDims(xSparse);
    oss<<" SparseMb y "<<prettyDims(y);
    return oss.str();
}
/** convert x to new matrix adding quadratic dimensions to each InnerIterator (row).
 * Optionally scale the quadratic elements to keep them to reasonable range.
 * For example if original dimension is 0..N, perhaps scale the quadratic
 * terms by 1.0/N. */
static void addQuadratic( SparseM & x, double const qscal=1.0 ){
    x.makeCompressed();
    VectorXi xsz(x.outerSize());
#pragma omp parallel for schedule(static)
    for(int i=0U; i<x.outerSize(); ++i){
        xsz[i] = x.outerIndexPtr()[i+1]-x.outerIndexPtr()[i];
    }
    // final inner dim increases by square of inner dim
    SparseM q(x.outerSize(),x.innerSize()+x.innerSize()*x.innerSize());
    // calc exact nnz elements for each row of q
    VectorXi qsz(x.outerSize());
#pragma omp parallel for schedule(static)
    for(int i=0; i<x.outerSize(); ++i){
        qsz[i] = xsz[i] + xsz[i]*xsz[i];
    }
    q.reserve( qsz );           // reserve exact per-row space needed
    // fill q
#pragma omp parallel for schedule(dynamic,128)
    for(int r=0; r<x.outerSize(); ++r){
        for(SparseM::InnerIterator i(x,r); i; ++i){
            q.insert(r,i.col()) = i.value();    // copy the original dimension
            for(SparseM::InnerIterator j(x,r); j; ++j){
                int col = x.innerSize() + i.col()*x.innerSize() + j.col(); // x.innerSize() is 4
                q.insert(r,col) = i.value()*j.value()*qscal;  // fill in quad dims
            }
        }
    }
    q.makeCompressed();
    x.swap(q);
}
static void addQuadratic( DenseM & x, double const qscal=1.0 ){
    DenseM q(x.outerSize(),x.innerSize()+x.innerSize()*x.innerSize());
#pragma omp parallel for schedule(static,128)
    for(int r=0; r<x.outerSize(); ++r){
        for(int i=0; i<x.innerSize(); ++i){
            q.coeffRef(r,i) = x.coeff(r,i);
            for(int j=0; j<x.innerSize(); ++j){
                int col = x.innerSize() + i*x.innerSize() + j;
                q.coeffRef(r,col) = x.coeff(r,i) * x.coeff(r,j) * qscal;
            }
        }
    }
    x.swap(q);
}
/** This can be very slow and should always be
 * done with full parallelism.
 * \throw if no data. */
void MCxyData::quadx(double qscal /*=0.0*/){
    if( !sparseOk && !denseOk ) throw std::runtime_error(" quadx needs x data");
    if( this->qscal != 0.0 ) throw std::runtime_error("quadx called twice!?");
    // divide xi * xj terms by max|x| ?
    // look at quad terms globally and make them mean 0 stdev 1 ?
    // scale so mean is 0 and all values lie within [-2,2] ?
    if( qscal==0.0 ){
        // calculate some "max"-ish value, and invert it.
        // I'll use l_{\inf} norm, or max absolute value,
        // so as not to ruin sparsity (if any)
        if( sparseOk ){
            if(!xSparse.isCompressed()) xSparse.makeCompressed();
#if 0 // not available for sparse..   l_{\inf} norm == max absolute value
            qscal = xSparse.lpNorm<Eigen::Infinity>();
#else
            SparseM::Index const nData = xSparse.outerIndexPtr()[ xSparse.outerSize() ];
            auto       b = xSparse.valuePtr();
            auto const e = b + nData;
            SparseM::Scalar maxabs=0.0;
            for( ; b<e; ++b) maxabs = std::max( maxabs, std::abs(*b) );
            qscal = maxabs;
#endif
        }else{
            // not supported: qscal = xDense.cwiseAbs().max();
            qscal = xDense.lpNorm<Eigen::Infinity>();
        }
        qscal = 1.0/qscal;
        // (note all compares with NaN are false, except !=)
        if( !(qscal > 1.e-12) ) qscal=1.0; // ignore strange/NaN scaling
    }
    if( sparseOk ) addQuadratic( xSparse, qscal );
    if( denseOk )           addQuadratic( xDense,  qscal );
    this->qscal = qscal;
}
