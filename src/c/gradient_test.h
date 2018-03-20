#ifndef GRADIENT_TEST_H
#define GRADIENT_TEST_H

#include "typedefs.h"
#include "parameter.h"
#include "WeightVector.h"
#include <iostream>                             // std::cerr

template<typename EigenType>
void finite_diff_test(const WeightVector& w, const EigenType& x, size_t idx,
                      const SparseMb& y, const VectorXi& nclasses, int maxclasses,
		      const VectorXd& inside_weight, const VectorXd& outside_weight,		    
                      const std::vector<int>& sorted_class, const std::vector<int>& class_order,
                      const VectorXd& sortedLU,
                      const boolmatrix& filtered,
                      double C1, double C2, const param_struct& params)
{
  using namespace std;
  double delta = params.finite_diff_test_delta;
  VectorXd proj(1);
  proj.coeffRef(0) = w.project_row(x,idx);
  bool none_filtered = filtered.count()==0;
  double obj = calculate_ex_objective_hinge(idx, proj.coeff(0), y, nclasses, inside_weight, outside_weight, sorted_class, class_order, sortedLU, filtered, none_filtered, C1, C2, params);
  
  VectorXsz index(1);
  index.coeffRef(0) = idx;
  size_t idx_start = 0;
  size_t idx_end = 1;
  int sc_start = 0;
  int sc_end = y.cols();
  VectorXd multipliers;
  VectorXd sortedLU_gradient;
  
  compute_gradients(multipliers, sortedLU_gradient,
		    idx_start, idx_end, sc_start, sc_end,
		    proj, index, y, nclasses, maxclasses,
		    inside_weight, outside_weight,
		    sorted_class, class_order, sortedLU,
		    filtered, C1, C2, params);
  
  WeightVector w_new(w);
  double xnorm = x.row(idx).norm();
  double multsign;
  if (multipliers.coeff(0) > 0)
    multsign = 1.0;
  if (multipliers.coeff(0) < 0)
    multsign = -1.0;
  
  w_new.gradient_update(x, idx, multsign*delta/xnorm);// divide delta by multipliers.coeff(0)*xnorm . the true gradient is multpliers.coeff(0)*x.
  
  double obj_w_grad = calculate_ex_objective_hinge(idx, w_new.project_row(x,idx), y, nclasses, inside_weight, outside_weight, sorted_class, class_order, sortedLU, filtered, none_filtered, C1, C2, params);
  double w_grad_error = fabs(obj_w_grad - obj + multsign*delta*multipliers.coeff(0)*xnorm);
  
  VectorXd sortedLU_new(sortedLU);
  sortedLU_new += sortedLU_gradient * delta / sortedLU_gradient.norm();  // have some delta that is inversely proportional to the norm of the gradient
  
  double obj_LU_grad = calculate_ex_objective_hinge(idx, proj.coeff(0), y, nclasses, inside_weight, outside_weight, sorted_class, class_order, sortedLU_new, filtered, none_filtered, C1, C2, params);
  double LU_grad_error = fabs(obj_LU_grad - obj + delta*sortedLU_gradient.norm());
  
  cerr << "w_grad_error:  " << w_grad_error << "   " << obj_w_grad - obj << "  " << obj_w_grad << "  " << obj << "  " << multsign*delta*multipliers.coeff(0)*xnorm << "   " << xnorm << "  " << idx << "   " << proj.coeff(0) << "  " << w_new.project_row(x,idx)  << "  ";
  
  for (SparseMb::InnerIterator it(y,idx); it; ++it)
    {
      int order = class_order[it.col()];
      cerr << it.col() << ":" << it.value() << ":" << order << ":" <<sortedLU.coeff(2*order) + 1 << ":" << sortedLU.coeff(2*order+1) - 1  << "  ";
    }
  cerr << endl;
  cerr << "LU_grad_error: " << LU_grad_error << "  " << obj_LU_grad - obj << "  " << "  " << obj_LU_grad << "  " << obj << "  " << delta*sortedLU_gradient.norm() << "  " << "  " << idx << "  " << proj.coeff(0) << "  ";
  for (SparseMb::InnerIterator it(y,idx); it; ++it)
    {
        int order = class_order[it.col()];
        cerr << it.col() << ":" << it.value() << ":" << order << ":" << sortedLU.coeff(2*order) + 1 << ":" << sortedLU.coeff(2*order+1) - 1  << " - " << sortedLU_new.coeff(2*order) + 1 << ":" << sortedLU_new.coeff(2*order+1) - 1  << "  ";
    }
    cerr << endl;
}

#endif //GRADIENT_TEST_JH
