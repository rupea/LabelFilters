#ifndef __NORMALIZE_H
#define __NORMALIZE_H
/** \file
 * XXX TODO FIXME names and what the functions do are in sorry state.
 *                whole thing should be reviewed and made consistent.
 *
 * Issues began in normalize.cpp where normalize_col seems to do different
 * things for Dense and Sparse matrices ???
 * Then I [ejk] added to the confusion by following the bizarro convention
 * of normalize_col(SparseM) (which I think is wrong) for DenseM, so that
 * now:
 * 
 * - both SparseM and DenseM have one "col" version that goes row-wise
 * and one version that goes columnwise !!!
 */

#include "typedefs.h"

#include <iostream>     // debug


void normalize_col(DenseM& mat);

/** JUST divides each row by its norm (no mean removal) */
void normalize_row(SparseM &mat);

/** Normalize data : centers and makes sure the variance is one.
 * Fixed, but you might instead just want to \em return the mean
 * and variance, because otherwise you may remove sparsity. */
void normalize_row_remove_mean(SparseM& mat);

/** There was no correct implementation of this, and it would be slow,
 * so it just throws, to remind you to either write it or switch to DenseM */
void normalize_col(SparseM &mat);


/** x MUST be Dense for assignment to x.row(r) to succeed.
 * Numerically stable and generic version of normalize_col(DenseM&). */
template< typename DERIVED >
void col_normalize( Eigen::DenseBase<DERIVED> & x, Eigen::VectorXd & mean, Eigen::VectorXd & stdev );

/** remove mean and stdev, generic form of normalize_row(DenseM&) */
template< typename DERIVED >
void row_normalize( Eigen::DenseBase<DERIVED> & x, Eigen::VectorXd & mean, Eigen::VectorXd & stdev );

/** disallow column-normalization for SparseM . \throw runtime_error always. */
template< typename DERIVED > inline 
void col_normalize( Eigen::SparseMatrixBase<DERIVED>& x, Eigen::VectorXd & mean, Eigen::VectorXd & stdev ){
    throw std::runtime_error("Error: Sparse normalization not supported. x-->DenseM for column_normalize(x)");
}

/** Generic, dense, column-wise normalization, returning the removed mean and stdev [ejk] */
template< typename DERIVED > inline
void col_normalize( Eigen::DenseBase<DERIVED> & x, Eigen::VectorXd & mean, Eigen::VectorXd & stdev ){
    int const verbose=0;
    if(verbose) std::cout<<" col_normalize(x,m,s) ... x"<<x<<std::endl;
    Eigen::VectorXd delta( x.cols() );
    { // compute mean and stdev of each row in one pass ...
        mean.resize( x.cols() );
        mean.setZero();
        if(verbose) std::cout<<" mean init "<<mean.transpose()<<std::endl;
        stdev.resize( x.cols() );              // we'll develop sum for variance
        stdev.setZero();
        for(int r=0U; r<x.rows(); ++r){    // numerically stable (see Wikipedia)
            delta = x.row(r).transpose() - mean;
            mean += delta * (1.0/(r+1U));
            stdev += delta.cwiseProduct( x.row(r).transpose() - mean );
        }
        if(verbose) std::cout<<" mean  "<<mean.transpose()<<std::endl;
        if( x.rows() < 2 ) stdev.setOnes();             // not enough data to calculate stdev
        else stdev = (stdev / (x.rows()-1)).cwiseSqrt(); // var --> correct scale --> stdev
        if(verbose)std::cout<<" stdev final "<<stdev.transpose()<<std::endl;
    }
    { // remove mean & so stdev!=1.0 from x
        for(int i=0U; i<stdev.size(); ++i ){    // safe-ish inverse stdev
            double g = stdev.coeff(i);
            delta.coeffRef(i) = (g>1.e-10? 1.0/g: 1.0);
        }
        if(verbose)std::cout<<" stdev inverse, delta = "<<delta.transpose()<<std::endl;
        for(uint32_t r=0U; r<x.rows(); ++r){
            x.row(r) = (x.row(r).transpose() - mean).cwiseProduct(delta);
        }
    }
}
/** Generic, dense, row-wise normalization, returning the removed mean and stdev [ejk] */
template< typename DERIVED > inline
void row_normalize( Eigen::DenseBase<DERIVED> & x, Eigen::VectorXd & mean, Eigen::VectorXd & stdev ){
    static int const verbose=0;
    Eigen::VectorXd delta( x.rows() );
    // column_mean_stdev ... method of normalize_col ...
    mean.resize( x.rows() );
    mean = x.rowwise().mean();
    if(verbose) std::cout<<"\nrn mean = "<<mean.transpose()<<std::endl;
    x = x.colwise() - mean;         // first center the data
    if(verbose) std::cout<<" x["<<x.rows()<<"x"<<x.cols()<<"], mean removed\n"<<x<<std::endl;
    stdev.resize( x.rows() );
    if( x.cols() < 2 ) stdev.setOnes();
    else stdev = (x.rowwise().squaredNorm() / (x.cols()-1)).cwiseSqrt();
    if(verbose)std::cout<<" stdev size "<<stdev.rows()<<"x"<<stdev.cols()<<" final "<<stdev.transpose()<<std::endl;
    { // remove mean & stdev!=1.0 from x
        for(int i=0U; i<stdev.size(); ++i ){    // safe-ish inverse stdev
            double g = stdev.coeff(i);
            delta.coeffRef(i) = (g>1.e-10? 1.0/g: 1.0);
        }
        if(verbose)std::cout<<" stdev inverse, delta = "<<delta.transpose()<<std::endl;
        for(uint32_t c=0U; c<x.cols(); ++c){
            x.col(c) = x.col(c).cwiseProduct(delta);
            if(verbose) std::cout<<" x.col(c="<<c<<") = "<<x.col(c).transpose()<<std::endl;
        }
    }
}

#if 0 // perhaps harden this code a bit more (for var nans)
template< typename DERIVED >
void column_mean_stdev( Eigen::DenseBase<DERIVED> const& x, Eigen::VectorXd & mean, Eigen::VectorXd &var ){
    mean.resize( x.cols() );
    mean.setZero();
    var.resize( x.cols() );
    var.setZero();
    size_t n=0U;
    Eigen::VectorXd delta( x.cols() );
    for(uint32_t r=0U; r<x.rows(); ++r){
        ++n;
        delta = x.row(r).transpose() - mean;
        mean += delta * (1.0/n);
        var += delta.cwiseProduct( x.row(r).transpose() - mean );
    }
    if( n < 2 ) var.setOnes();                  // not enough data to calculate var
    else var = (var * (1.0/(n-1))).cwiseSqrt(); // var --> correct scale --> stdev
}
#endif

#endif


