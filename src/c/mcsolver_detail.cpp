/*  Copyright (C) 2017 NEC Laboratories America, Inc. ("NECLA"). All rights reserved.
 *
 * This source code is licensed under the license found in the LICENSE file in
 * the root directory of this source tree. An additional grant of patent rights
 * can be found in the PATENTS file in the same directory.
 */
#include "mcsolver_detail.hh"
#include "parameter.h"

namespace mcsolver_detail{
  using namespace std;

  // ************************
  // function to set eta for each iteration

  double set_eta(param_struct const& params, size_t const t, double const lambda)
  {
    double eta_t;
    switch (params.eta_type)
      {
      case ETA_CONST:
	eta_t = params.eta;
	break;
      case ETA_SQRT:
	eta_t = params.eta/sqrt(t);
	break;
      case ETA_LIN:
	eta_t = params.eta/(1+params.eta*lambda*t);
	break;
      case ETA_3_4:
	eta_t = params.eta/pow(1+params.eta*lambda*t,3*1.0/4);
	break;
      case DEFAULT:
	throw runtime_error("eta_type parameter has not been set. finalize_default_params should have been called before getting here.");	
      default:
	throw runtime_error("Unknown eta_type option");
      }
    if (eta_t < params.min_eta)
      {
	eta_t = params.min_eta;
      }
    return eta_t;
  }


  // ********************************
  // Compute the means of the classes of the projected data
  void proj_means(VectorXd& means, VectorXi const& nc,
		  VectorXd const& projection, SparseMb const& y, 
		  const param_struct& params, boolmatrix const& filtered)
  {
    size_t noClasses = y.cols();
    size_t n = projection.size();
    size_t c,i,k;
    means.resize(noClasses);
    means.setZero();
    for (i=0;i<n;i++)
      {
	for (SparseMb::InnerIterator it(y,i);it;++it)
	  {
	    if (it.value())
	      {
		c = it.col();
		if (!params.remove_class_constraints || !(filtered.get(i,c)))
		  {	    		    
		    means(c)+=projection.coeff(i);
		  }
	      }
	  }
      }
    for (k = 0; k < noClasses; k++)
      {
	// if there are no examples of a ceratin class then the order does not matter 
	// becaues L and U will be set at high/low value. 
	nc(k)?means(k) /= nc(k):0.0; 
      }
  }


  //*****************************************
  // Update the filtered constraints

  void update_filtered(boolmatrix& filtered, const VectorXd& projection,
		       const VectorXd& l, const VectorXd& u, const SparseMb& y,
		       const bool filter_class)
  {
    int noClasses = y.cols();
    int c;
    for (size_t i = 0; i < projection.size(); i++)
      {
	double proj = projection.coeff(i);
	SparseMb::InnerIterator it(y,i);
	while ( it && !it.value() ) ++it;
	c=it?it.col():noClasses;
	for (int cp = 0; cp < noClasses; cp++)
	  {
	    if ( filter_class || cp != c )
	      {
		bool val = (proj<l.coeff(cp))||(proj>u.coeff(cp))?true:false;
		if (val)
		  {
		    filtered.set(i,cp);
		  }
	      }
	    if ( cp == c )
	      {
		++it;
		while ( it && !it.value() ) ++it;
		c=it?it.col():noClasses;
	      }
	  }
      }
  }
}
