#ifndef __NORMALIZE_H
#define __NORMALIZE_H

#include "Eigen/Dense"
#include "Eigen/Sparse"

/** Normalize data colwise by removing mean (if center = true) and dividing by stdev. 
 ** If useMeanSdev is true, use the values in mean and stdev to perform the normalization */
template< typename DERIVED >
void col_mean_std_normalize( Eigen::DenseBase<DERIVED> & x, Eigen::VectorXd & mean, Eigen::VectorXd & stdev, bool center = true, bool useMeanStdev = false);

/** Normalize data rowwise by removing mean (if center = true) and dividing by stdev. 
 ** If useMeanSdev is true, use the values in mean and stdev to perform the normalization */
template< typename DERIVED >
void row_mean_std_normalize( Eigen::DenseBase<DERIVED> & x, Eigen::VectorXd & mean, Eigen::VectorXd & stdev,  bool center = true, bool useMeanStdev = false );

/* Normalize data by dividing each row by its norm. x must be row-major to work with sparse matrices*/ 
template <typename EigenType>
void row_unit_normalize(EigenType& x);

/** disallow column-normalization for SparseM . \throw runtime_error always. */
template< typename DERIVED > inline 
void col_mean_std_normalize( Eigen::SparseMatrixBase<DERIVED>& x, Eigen::VectorXd & mean, Eigen::VectorXd & stdev, bool center = true, bool useMeanStdev = false  ){
    throw std::runtime_error("Error: Sparse coulumn normalization not supported. x-->DenseM for col_mean_std_normalize");
}
/** disallow row-normalization for SparseM . \throw runtime_error always. */
template< typename DERIVED > inline 
void row_mean_std_normalize( Eigen::SparseMatrixBase<DERIVED>& x, Eigen::VectorXd & mean, Eigen::VectorXd & stdev, bool center = true, bool useMeanStdev = false ){
    throw std::runtime_error("Error: Sparse row normalization not supported. x-->DenseM for row_mean_std_normalize ");
}

#endif

