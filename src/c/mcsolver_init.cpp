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
  // Get the number of examples in each class

  void init_nc(VectorXi& nc, VectorXi& nclasses, const SparseMb& y)
  {
    int noClasses = y.cols();
    int n = y.rows();
    nc.setZero(noClasses);
    nclasses.setZero(n);
    for (int i=0;i<n;i++) {
      for (SparseMb::InnerIterator it(y,i);it;++it) {
	if (it.value()) {
	  ++nc.coeffRef(it.col());
	  ++nclasses.coeffRef(i);
	}
      }
    }
    ostringstream err;
    // Until supported, check that all data is needed
    // E.g. many places may divide by nc[i] or ...
    int nclassesZero = 0;
    for (int i=0;i<n;i++) if( nclasses[i]==0 ) ++nclassesZero;
    if(nclassesZero) err<<"\nWARNING: it seems "<<nclassesZero<<" examples have been assigned to no class at all";

    int ncZero = 0;
    int i0=0;
    for (int i=0;i<noClasses;i++) if( nc[i]==0 ) {++ncZero; if(i0==0) i0=i;}
    if(ncZero) err<<"\nWARNING: it seems "<<ncZero<<" classes have been assigned to NO training examples (nc["<<i0<<"]==0)";

    if(err.str().size()){
      err<<"\n\tPlease check whether code should support this, since"
	 <<"\n\tboth nc[class] and nclasses[example] may be used as divisors"<<endl;
      //throw runtime_error(err.str());
    }
  }

  // ************************************
  // Get the sum of the weight of all examples in each class

  void init_wc(VectorXd& wc, const VectorXi& nclasses, const SparseMb& y, const param_struct& params, boolmatrix const& filtered)
  {
    double ml_wt_class = 1.0;
    int noClasses = y.cols();
    wc.setZero(noClasses);
    size_t n = y.rows();
    if (nclasses.size() != n) {
      throw runtime_error("init_wc has been called with vector nclasses of wrong size");
    }
    for (size_t i=0;i<n;i++) {
      if (params.ml_wt_class_by_nclasses) {
	ml_wt_class = 1.0/nclasses.coeff(i);
      }
      for (SparseMb::InnerIterator it(y,i);it;++it) {
	if (it.value()) {
	  if (!params.remove_class_constraints || !(filtered.get(i,it.col())))
	    {
	      wc.coeffRef(it.col()) += ml_wt_class;
	    }
	}
      }
    }
  }



}
