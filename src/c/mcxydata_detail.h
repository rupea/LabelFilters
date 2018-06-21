#ifndef __MCXYDATA_DETAIL_H
#define __MCXYDATA_DETAIL_H

#include "typedefs.h"
#include "Eigen/Dense"
#include "Eigen/Sparse"

namespace mcxydata_detail{


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
  
  /** Removes columns that have fewer than minex non-zero entries. **/
  template< typename DERIVED >
    void remove_rare_features (Eigen::PlainObjectBase<DERIVED> &x, std::vector<std::size_t> feature_map, std::vector<std::size_t> reverse_feature_map, const int minex = 1, bool useFeatureMap = false);

  template< typename Scalar, int Options, typename Index >    
    void remove_rare_features (Eigen::SparseMatrix<Scalar, Options, Index> &x, std::vector<std::size_t>& feature_map, std::vector<std::size_t>& reverse_feature_map, const int minex = 1, const bool useFeatureMap = false);

  //  template <typename EigenType>
  //  void remove_rare_labels(EigenType x, SparseMb y, const int minex = 1, const bool remove_ex = false); // new label map??

}

#endif //__MCXYDATA_DETAIL_H


