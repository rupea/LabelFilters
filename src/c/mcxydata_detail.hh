#ifndef __MCXYDATA_DETAIL_HH
#define __MCXYDATA_DETAIL_HH

#include "mcxydata_detail.h"

#include <iostream>

namespace mcxydata_detail{
  using namespace std;
  
  template <typename EigenType>
  void get_feature_map(vector<size_t>& feature_map, vector<size_t>& reverse_feature_map, EigenType const& x, const int minex)
  {  
    // calculate the number of ex where feature is non-zero
    vector<size_t> nex(x.cols(),0);
    for (size_t i = 0; i < x.rows(); i++)
      {
	for (typename EigenType::InnerIterator it(x.derived(),i); it; ++it)
	  {
	    if (it.value())
	      {
		nex[it.col()] += 1;
	      }
	  }
      }
    
    // get the new feature map
    feature_map.clear();
    reverse_feature_map.clear();
    for (size_t i = 0; i < x.cols(); i++)
      {
	if (nex[i] >= minex)
	  {
	    feature_map.push_back(i);
	    reverse_feature_map.push_back(feature_map.size()-1);
	  }
	else
	  {
	    reverse_feature_map.push_back(x.cols());
	  }
      }
  }

  template< typename Scalar, int Options, typename Index > inline   
  void remove_rare_features (Eigen::SparseMatrix<Scalar, Options, Index>& x, vector<size_t>& feature_map, vector<size_t>& reverse_feature_map, const int minex /*= 1*/, const bool useFeatureMap /*=false*/)
  {

    typedef Eigen::SparseMatrix<Scalar, Options, Index> SM;

    if (useFeatureMap)
      {
	if (reverse_feature_map.size() != x.cols())
	  throw runtime_error("Feature map does not match data");
      }
    else
      {
	get_feature_map(feature_map, reverse_feature_map, x, minex);
      }

    typedef Eigen::Triplet<Scalar> T;
    std::vector<T> Triplets;
    
    for (size_t i = 0; i < x.rows(); i++)
      {
	for (typename SM::InnerIterator it(x,i); it; ++it)
	  {
	    if (reverse_feature_map[it.col()] < x.cols())
	      {
		Triplets.push_back(T(i,reverse_feature_map[it.col()],it.value()));
	      }
	  }
      }
    
    x.setZero();
    x.resize(x.rows(), feature_map.size());
    x.setFromTriplets(Triplets.begin(), Triplets.end());
    x.makeCompressed();
  }
    
  template< typename DERIVED > inline
  void remove_rare_features (Eigen::PlainObjectBase<DERIVED> &x, std::vector<std::size_t> feature_map, std::vector<std::size_t> reverse_feature_map, const int minex /*= 1*/, bool useFeatureMap /*=false*/)
  {
        
    if (useFeatureMap)
      {
	if (reverse_feature_map.size() != x.cols())
	  throw runtime_error("Feature map does not match data");
      }
    else
      {
	get_feature_map(feature_map, reverse_feature_map, x, minex);
      }
    
    for (size_t i = 0; i < feature_map.size(); i++)
      {
	x.col(i) = x.col(feature_map[i]);
      }
    x.conservativeResize(x.rows(), feature_map.size());
  }


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

  
}

#endif //__MCXYDATA_DETAIL_H
