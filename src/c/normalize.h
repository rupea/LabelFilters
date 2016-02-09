#ifndef __NORMALIZE_H
#define __NORMALIZE_H

#include "typedefs.h"

// *************************
// Normalize data : centers and makes sure the variance is one
void normalize_col(SparseM& mat);

void normalize_col(DenseM& mat);

void normalize_row(SparseM &mat);

/** x MUST be Dense for assignment to x.row(r) to succeed.
 * Numerically stable alg.  \sa normalize_col(DenseM&). */
template< typename DERIVED >
void column_normalize( Eigen::DenseBase<DERIVED> & x, VectorXd & mean, VectorXd & stdev );

/** disallow column-normalization for SparseM . \throw runtime_error always. */
template< typename DERIVED > inline 
void column_normalize( Eigen::SparseMatrixBase<DERIVED>& x, VectorXd & mean, VectorXd & stdev ){
    //static_assert(false,"Error: Sparse normalization not supported. x-->DenseM for column_normalize(x)");
    throw std::runtime_error("Error: Sparse normalization not supported. x-->DenseM for column_normalize(x)");
}


template< typename DERIVED > inline
void column_normalize( Eigen::DenseBase<DERIVED> & x, VectorXd & mean, VectorXd & stdev ){
    VectorXd delta( x.cols() );
    { // column_mean_stdev ...
        mean.resize( x.cols() );
        mean.setZero();
        stdev.resize( x.cols() );               // we'll develop sum for variance
        stdev.setZero();
        size_t n=0U;
        for(uint32_t r=0U; r<x.rows(); ++r){    // numerically stable (see Wikipedia)
            ++n;
            delta = x.row(r).transpose() - mean;
            mean += delta * (1.0/n);
            stdev += delta.cwiseProduct( x.row(r).transpose() - mean );
        }
        if( n < 2 ) stdev.setOnes();                    // not enough data to calculate stdev
        else stdev = (stdev * (1.0/(n-1))).cwiseSqrt(); // var --> correct scale --> stdev
    }
    { // remove mean & stdev!=1.0 from x
        for(uint32_t r=0U; r<x.rows(); ++r){
            delta = x.row(r).transpose();
            delta -= mean;
            x.row(r) = delta.cwiseQuotient(stdev).transpose();
        }
    }
}
#endif


