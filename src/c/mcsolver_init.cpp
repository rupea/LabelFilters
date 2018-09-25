/*  Copyright (C) 2017 NEC Laboratories America, Inc. ("NECLA"). All rights reserved.
 *
 * This source code is licensed under the license found in the LICENSE file in
 * the root directory of this source tree. An additional grant of patent rights
 * can be found in the PATENTS file in the same directory.
 */
#include "mcsolver_init.hh"
#include "boolmatrix.h"
#include <iostream>
#include <sstream>

namespace mcsolver_detail{

  using namespace std;
  
  // *************************************
  // need to reimplement this funciton to work with (inside) the WeightVector class
  // this is a costly operation, especially since weights are usually not already orthogonal
  //  orthogonalization might not be needed in high dimensions. 
  // we'll implement this when we get there

  /** Projection to a new vector that is orthogonal to the rest.
   * - sets w to be ortho to weights[0..projection_dim-1]
   * - It is basically Gram-Schmidt Orthogonalization.
   * - sequentially remove projections of \c w onto each of the
   *   \c weights[i] directions.
   */
  void project_orthogonal( VectorXd& w, const DenseM& weights,
				     const int projection_dim, const bool orthogonal_weights/*=false*/)
  {
    if (projection_dim == 0)
      return;
    if (orthogonal_weights)
      {
	for (int i = 0; i < projection_dim; ++i) {
	  double const norm = weights.col(i).norm();
	  if( norm > 1.e-6 ){
	    w.array() -= (weights.col(i) * ((w.transpose() * weights.col(i)) / (norm*norm))).array();
	  }
	}
      }
    else
      {
	// we orthogonalize weights, and then apply to w
	// expensive. 
	DenseM o = weights.topLeftCorner( weights.rows(), projection_dim );  // orthonormalized weights
	assert( o.cols() == projection_dim );
	VectorXd onorm(projection_dim);
	for(int i = 0; i < projection_dim; ++i){
	  onorm[i] = o.col(i).norm();                     // valgrind issues. why?
	}
	for(int i = 0; i < projection_dim; ++i){
	  for(int j = 0; j < i; ++j) {
	    double const nj = o.col(j).norm(); //double const nj = onorm[j];
	    if( nj > 1.e-6 ){
	      o.col(i).array() -= (o.col(j) * ((o.col(i).transpose() * o.col(j)) / (nj*nj))).array();
	    }
	  }
	  onorm[i] = o.col(i).norm();
	}
	for(int i = 0; i < projection_dim; ++i){
	  double const norm = onorm[i]; //o.col(i).norm();        // XXX valgrind
	  if( norm > 1.e-6 ){
	    w.array() -= (o.col(i) * ((w.transpose() * o.col(i)) / (norm*norm))).array();
	  }      
	}
      }
#ifndef NDEBUG
    // check orthogonality
    std::ostringstream err;
    for( int i=0; i<projection_dim; ++i ){
      double z = w.transpose() * weights.col(i);
      if( fabs(z) > 1.e-8 ){
	err<<" ERROR: w wrt "<<weights.cols()<<" vectors,"
	  " orthogonality violated for w vs weights.col("<<i<<")"
	  "\n\t|w'| = "<<w.norm() // <<" |proj_sum| = "<<proj_sum.norm()
	   <<"\n\tdot product = "<<z<<" too large";
      }
    }
    if(err.str().size()){
      throw std::runtime_error(err.str()); // perhaps it is not that serious?
    }
#endif //NDEBUG
  }


  // ************************************
  // Get the number of classes for each examples, 
  // and the weight of each example for the inside interval constraints and 
  // for the outside interval constraints.
  // These quantities do not change if constraints are filtered. 
  void init_nclasses(VectorXi& nclasses, VectorXd& inside_weight, VectorXd& outside_weight, const SparseMb& y, const param_struct& params)
  {
    int n = y.rows();
    nclasses.setZero(n);
    inside_weight.setZero(n);
    outside_weight.setZero(n);
    for (int i=0;i<n;i++) {
      for (SparseMb::InnerIterator it(y,i);it;++it) {
	if (it.value()) {
	  ++nclasses.coeffRef(i);
	}
      }
    }
    int nclassesZero = 0;
    for (int i=0;i<n;i++)
      {
	if( nclasses[i]==0 )
	  {
	    ++nclassesZero;
	    inside_weight.coeffRef(i) = 0.0;
	    outside_weight.coeffRef(i) = 0.0;
	  }
	else
	  {
	    inside_weight.coeffRef(i) = params.ml_wt_class_by_nclasses?1.0/nclasses.coeff(i):1.0;
	    outside_weight.coeffRef(i) = params.ml_wt_by_nclasses?1.0/nclasses.coeff(i):1.0;
	  }
      }
    if(nclassesZero) std::cerr<<"WARNING: it seems "<<nclassesZero<<" examples have been assigned to no class at all. They will be ignored"<<endl;
  }
 


  // ************************************
  // Get the number of examples in each class and the sum of the weight of all examples in each class
  // if remove_class_constraints is on then examples for which the class has been filtered are not counted
  void init_nc(VectorXi& nc, VectorXd& wc, const VectorXd& inside_weight, const SparseMb& y, const param_struct& params, boolmatrix const& filtered)
  {
    int noClasses = y.cols();
    nc.setZero(noClasses);
    wc.setZero(noClasses);
    size_t n = y.rows();
    if (inside_weight.size() != n) {
      throw runtime_error("init_wc has been called with vector nclasses of wrong size");
    }
    for (size_t i=0;i<n;i++) {
      for (SparseMb::InnerIterator it(y,i);it;++it) {
	if (it.value()) {
	  if (!params.remove_class_constraints || !(filtered.get(i,it.col())))
	    {
	      ++nc.coeffRef(it.col());
	      wc.coeffRef(it.col()) += inside_weight.coeff(i);
	    }
	}
      }
    }
  }

  
}
