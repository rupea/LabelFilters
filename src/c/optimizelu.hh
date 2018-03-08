#ifndef OPTIMIZELU_HH
#define OPTIMIZELU_HH

#include "typedefs.h"
#include "parameter.h"
#include "boolmatrix.h"
#include "Eigen/Dense"
#include <vector>

#define __restricted /* __restricted seems to be an error */

namespace mcsolver_detail
{ 
  using namespace std;
  using namespace Eigen;
  /** get optimal {l,u} bounds given projection and class order.
   * Computationally expensive, so it should be done sparingly.
   */
  void optimizeLU(VectorXd& l, VectorXd& u,
		  const VectorXd& projection, const SparseMb& y,
		  const vector<int>& class_order, const vector<int>& sorted_class,
		  const VectorXd& wc, const VectorXi& nclasses,
		  const boolmatrix& filtered,
		  const double C1, const double C2,
		  const param_struct& params);
  
  
  //*****************************************
  // function used by optimizeLU
  // grad are stored in order of the ranked classes
  // to minimize cash misses and false sharing
  
  /** set sorted class_order values of labels for training example 'idx' */
  static inline void sortedClasses( std::vector<int> & classes, std::vector<int> const& class_order,
				    SparseMb const& y, size_t const idx )
  {
    classes.resize(0);
    for (SparseMb::InnerIterator it(y,idx); it; ++it){
      classes.push_back(class_order[it.col()]);
    }
    std::sort(classes.begin(),classes.end());
  }
  
  //*****************************************
  // function used by optimizeLU 
  // grad are stored in order of the ranked classes
  // to minimize cash misses and false sharing 
  
  static inline void getBoundGrad (VectorXd& __restricted grad, VectorXd& __restricted bound,
				   const size_t idx, const size_t allproj_idx,
				   const std::vector<int>& __restricted sorted_class,
				   const int sc_start, const int sc_end,
				   const std::vector<int>& __restricted classes,
				   const double start_update, const double other_weight,
				   const VectorXd& __restricted allproj,
				   const bool none_filtered, const boolmatrix& __restricted filtered)
    
  {
    std::vector<int>::const_iterator class_iter = std::lower_bound(classes.begin(), classes.end(), sc_start);  
    double update = start_update + (class_iter - classes.begin())*other_weight;
    for (int sc = sc_start; sc < sc_end; sc++)
      {
	if (class_iter != classes.end() && sc == *class_iter)
	  {
	    // example is of this class
	    update += other_weight;
	    class_iter++;
	    continue;
	  }		  
	const int cp = sorted_class[sc];
	if (grad.coeff(sc) >= 0 && (none_filtered || !(filtered.get(idx,cp))))
	  {			      
	    grad.coeffRef(sc) -= update;
	    if (grad.coeff(sc) < 0)
	      {
		bound.coeffRef(cp) = allproj.coeff(allproj_idx);
	      }
	  }
      }  
  }
}

#endif // OPTIMIZELU_HH
