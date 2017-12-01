#ifndef __NORMALIZE_HH
#define __NORMALIZE_HH


#include "normalize.h"
#include <stdexcept>

//#include "typedefs.h"
//#include <iostream>     // debug


/** Generic, dense, row-wise normalization, returning the removed mean and stdev [ejk] */
template< typename DERIVED > inline
void row_mean_std_normalize( Eigen::DenseBase<DERIVED> & x, Eigen::VectorXd & mean, Eigen::VectorXd & stdev, bool center, bool useMeanStdev){
    if (useMeanStdev && stdev.size() != x.rows()) throw std::runtime_error("Dimensions do not match for normalization"); 
    if (useMeanStdev && center && mean.size() != x.rows()) throw std::runtime_error("Dimensions do not match for normalization"); 
    Eigen::VectorXd delta( x.rows() );
    if (!useMeanStdev)
      {
	mean.resize( x.rows() );
	if (center)
	  {
	    mean = x.rowwise().mean();
	  }
	else
	  {
	    mean.setZero();
	  }
      }
    if ( center ){
      x = x.colwise() - mean;         // first center the data
    }
    if (!useMeanStdev)
      {
	stdev.resize( x.rows() );
	if( x.cols() < 2 ) stdev.setOnes();      
	else stdev = (x.rowwise().squaredNorm() / (x.cols()-1)).cwiseSqrt();
      }
    
    // remove mean & stdev!=1.0 from x
    for(int i=0U; i<stdev.size(); ++i ){    // safe-ish inverse stdev
      double g = stdev.coeff(i);
      g = (g>1.e-10? 1.0/g: 1.0);
      x.row(i) *= g;
    }
}

/** Generic, dense, col-wise normalization, returning the removed mean and stdev [ejk] */
template< typename DERIVED > inline
void col_mean_std_normalize( Eigen::DenseBase<DERIVED> & x, Eigen::VectorXd & mean, Eigen::VectorXd & stdev, bool center /*=true*/, bool useMeanStdev /*=false*/){
    if (useMeanStdev && stdev.size() != x.cols()) throw std::runtime_error("Dimensions do not match for normalization"); 
    if (useMeanStdev && center && mean.size() != x.cols()) throw std::runtime_error("Dimensions do not match for normalization"); 
    Eigen::VectorXd delta( x.cols() );
    if (!useMeanStdev)
      {
	mean.resize( x.cols() );
	if (center)
	  {
	    mean = x.colwise().mean();
	  }
	else
	  {
	    mean.setZero();
	  }
      }
    if ( center ){
      x = x.rowwise() - mean.transpose();         // first center the data
    }
    if (!useMeanStdev)
      {
	stdev.resize( x.cols() );
	if( x.rows() < 2 ) stdev.setOnes();      
	else stdev = (x.colwise().squaredNorm() / (x.rows()-1)).cwiseSqrt();
      }
    
    // remove mean & stdev!=1.0 from x
    for(int i=0U; i<stdev.size(); ++i ){    // safe-ish inverse stdev
      double g = stdev.coeff(i);
      g = (g>1.e-10? 1.0/g: 1.0);
      x.col(i) *= g;
    }
}


/* Normalize data by dividing each row by its norm. x must be row-major to work with sparse matrices*/ 
template< typename EigenType > inline 
void row_unit_normalize(EigenType& x){
  if( !x.IsRowMajor ) throw std::runtime_error("row unit normalization only implemented for row major matrices");
  for(size_t r=0U; r<x.rows(); ++r){
    double const f = 1.0 / x.row(r).norm();
    if( !(f < 1.e-10) ){
      x.row(r) *= f;
    }
  }
}

#endif


