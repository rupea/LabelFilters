/*  Copyright (C) 2017 NEC Laboratories America, Inc. ("NECLA"). All rights reserved.
 *
 * This source code is licensed under the license found in the LICENSE file in
 * the root directory of this source tree. An additional grant of patent rights
 * can be found in the PATENTS file in the same directory.
 */
#ifndef MCSOLVER_INIT_HH
#define MCSOLVER_INIT_HH

#include "typedefs.h"
#include "parameter.h"
#include "WeightVector.h"
#include "boolmatrix.h"
#include <vector>

namespace mcsolver_detail{

  using namespace Eigen;

  // ************************************
  // Get the number of classes for each examples, 
  // and the weight of each example for the inside interval constraints and 
  // for the outside interval constraints.
  // These quantities do not change if constraints are filtered. 

  void init_nclasses(VectorXi& nclasses, VectorXd& inside_weight, VectorXd& outside_weight, const SparseMb& y, const param_struct& params);
  
  // ************************************
  // Get the number of examples in each class and the sum of the weight of all examples in each class
  // if remove_class_constraints is on then examples for which the class has been filtered are not counted  
  void init_nc(VectorXi& nc, VectorXd& wc, const VectorXd& inside_weight, const SparseMb& y, 
	       const param_struct& params, boolmatrix const& filtered);

  
  /** Projection to a new vector that is orthogonal to the rest.
   * - sets w to be ortho to weights[0..projection_dim-1]
   * - It is basically Gram-Schmidt Orthogonalization.
   * - sequentially remove projections of \c w onto each of the
   *   \c weights[i] directions.
   */
  // not thoroughly tested!!!!
  void project_orthogonal( VectorXd& w, const DenseM& weights,
			   const int projection_dim, const bool orthogonal_weights = false);
  
  template<typename EigenType>
  void difference_means(VectorXd& difference, const EigenType& x, const SparseMb& y, const VectorXi& nc, const int c1, const int c2, const param_struct& params, boolmatrix const& filtered) 
  {
    assert(nc(c1) != 0);
    assert(nc(c2) != 0);
    size_t const d = x.cols();
    size_t const n = x.rows();
    difference.resize(d);
    difference.setZero();
    double wt1 = 1.0 / nc(c1);
    double wt2 = 1.0 / nc(c2);
    for (size_t row=0;row<n; ++row) {
      if (y.coeff(row,c1)) {
	if (!params.remove_class_constraints || !(filtered.get(row,c1)))
	  {	    
	    typename EigenType::InnerIterator it(x,row);
	    for (; it; ++it)
	      difference.coeffRef(it.col())+=it.value()*wt1;
	  }
      }
      if (y.coeff(row,c2)) {
	if (!params.remove_class_constraints || !(filtered.get(row,c2)))
	  {	    	
	    typename EigenType::InnerIterator it(x,row);
	    for (; it; ++it)
	      difference.coeffRef(it.col())-=it.value()*wt2;
	  }
      }
    }
  }
  
  /** initialize projection vector \c w prior to calculating \c weights[*,projection_dim].
   * \p w         WeightedVector for SGD iterations
   * \p x         nExamples x d training data (each row is one input vector)
   * \p y         bool matrix of class info
   * \p nc        number of examples in each class
   * \p weights   d x projections matrix of col-wise projection vectors
   * \p projection_dim the column of weights for which we are initializing \c w
   */
  template<typename EigenType>
  /** \b OHOH: for projection_dim > 0, should probably concentrate on classes that
   * are \b not yet well-separated. */
  void init_w( WeightVector& w, DenseM const& weights,
	       EigenType const& x, SparseMb const& y, VectorXi const& nc,
	       int const projection_dim, param_struct const& params, boolmatrix const& filtered)            
  {
    using namespace std;
    
    int const d = x.cols();
    int const noClasses = y.cols();
    int c1,c2,tries;
    VectorXd init(d);
    bool normalize = true;
    if (params.verbose >= 1)
      {
	cout<<" init_w : weights.cols()="<<weights.cols()<<" projection_dim="<<projection_dim; cout.flush();
      }
    switch (params.init_type){
    case INIT_ZERO:
      if (params.verbose >= 1)
	{
	  cout << " zero"; cout.flush();
	}
      init.setZero(); normalize=false;
      break;      
    case INIT_PREV:
      if(projection_dim <= weights.cols()) 
	{
	  if (params.verbose >= 1)
	    {
	      cout<<" initializing with weights.col("<<projection_dim<<")"<<endl;
	    }
	  init = weights.col(projection_dim);
	  normalize = false; // if initializing with previous solution, do not renormlaize w
	  break;
	}
      else
	{
	  cerr << "Warning: init_type=INIT_PREV but only "<< weights.cols() <<" filters are available for initialization. Initializing using init_type=INIT_DIFF." << endl;
	}
    case INIT_DIFF:
      if (params.verbose >= 1)
	{
	  cout<<" vector-between-2-classes"; cout.flush();
	}
      // initialize w as vector between the means of two random classes.
      // should find cleverer initialization schemes
      tries=0;
      // Actually better to select two far classes, probability prop. to dist between classes?
      // BUT also want to avoid previously chosen directions.
      do{
      c1 = ((int) rand()) % noClasses;
      ++tries;
      }while( nc(c1) == 0 && tries < 10 );
      do{
      c2 = ((int) rand()) % noClasses;
      ++tries;
      }while( (nc(c2) == 0 || c2 == c1) && tries < 50 );
      if( nc(c1) > 0 && nc(c2) > 0 && c2 != c1 ){
	difference_means(init,x,y,nc,c1,c2, params, filtered);
	init.normalize();
	break;
      }
    case INIT_RANDOM:      
      if (params.verbose >= 1)
	{
	  cout<<" random"; cout.flush();
	}
      init.setRandom(); init.normalize();
      break;      
    default:
      throw std::runtime_error("ERROR: initilization type not recognized");
    }
    
    if (params.init_orthogonal)
      {
	try{
	  // orthogonalize to current projection dirns w[*, col<projection_dim]
	  project_orthogonal( init, weights, projection_dim );
	}catch(std::runtime_error const& e){
	  cerr<<e.what();
	  cerr<<" Continuing anyway (just initializing a fresh \"random\" projection vector)"<<endl;
	}
	double inorm = init.norm();
	if( init.norm() < 1.e-6 ) 
	  {
	    if (params.verbose >= 1)
	      {
		cout<<" randomized"; 
	      }
	    init.setRandom(); 
	    init.normalize();
	  }
	else
	  {
	    if (params.verbose >= 1)
	      {
		cout<<" orthogonalized";
	      }
	    init *= (1.0 / inorm);
	  }
      }
    if (params.verbose >= 1)
      {
	cout<<endl;
      }
    assert( init.size() == d );
    if (params.init_norm > 0 && normalize){
      init *= params.init_norm;
    }
    w.init(init);	
  }
  
}
#endif //MCSOLVER_INIT_HH
