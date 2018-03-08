
#include "objective.h"
#include "typedefs.h"
#include "parameter.h"
#include "boolmatrix.h"
#include "WeightVector.h"

namespace mcsolver_detail{

  // *****************************************
  // Function to calculate the objective for one example
  // this almost duplicates the function compute_objective
  // the two functions should be unified
  // this functions is easier to use with the finite_diff_test because
  // it does not require the entire projection vector
  double calculate_ex_objective_hinge(size_t i, double proj, const SparseMb& y,
				      const VectorXi& nclasses,
				      const std::vector<int>& sorted_class,
				      const std::vector<int>& class_order,
				      const VectorXd& sortedLU,
				      const boolmatrix& filtered,
				      bool none_filtered,
				      double C1, double C2,
				      const param_struct& params)
  {
    double obj_val=0;
    int noClasses = y.cols();
    double class_weight, other_weight;
    std::vector<int> classes;
    std::vector<int>::iterator class_iter;
    classes.reserve(nclasses.coeff(i)+1);
    int left_classes, right_classes;
    double left_weight, right_weight;
    int sc,cp;
    const double* sortedLU_iter;

    class_weight = C1;
    other_weight = C2;
    if (params.ml_wt_by_nclasses)
      {
	other_weight /= nclasses.coeff(i);
      }
    if (params.ml_wt_class_by_nclasses)
      {
	class_weight /= nclasses.coeff(i);
      }

    left_classes = 0; //number of classes to the left of the current one
    left_weight = 0; // left_classes * other_weight
    right_classes = nclasses.coeff(i); //number of classes to the right of the current one
    right_weight = other_weight * right_classes;

    // calling y.coeff is expensive so get the classes here
    classes.resize(0);
    for (SparseMb::InnerIterator it(y,i); it; ++it)
      {
	if (it.value())
	  {
	    classes.push_back(class_order[it.col()]);
	  }
      }
    classes.push_back(noClasses); // this will always be the last
    std::sort(classes.begin(),classes.end());

    sc=0;
    class_iter = classes.begin();
    sortedLU_iter=sortedLU.data();
    while (sc < noClasses)
      {
	while(sc < *class_iter)
	  {
	    // while example is not of class cp
	    cp = sorted_class[sc];
	    if (none_filtered || !(filtered.get(i,cp)))
	      {
		obj_val += (left_classes?(left_weight * hinge_loss(*sortedLU_iter - proj)):0)
		  + (right_classes?(right_weight * hinge_loss(proj - *(sortedLU_iter+1))):0);
	      }
	    sc++;
	    sortedLU_iter+=2;
	  }
	if (sc < noClasses) // test if we are done
	  {
	    // example has class cp
	    cp = sorted_class[sc];
	    // compute the loss incured by the example no being withing the bounds
	    //    of class cp
	    // could also test if params.remove_class_constraints is set. If not, this constarint can not be filtered
	    //    essentially add || !params.remove_class_constraint to the conditions below
	    //    would speed things up by a tiny bit .. not significant
	    if (none_filtered || !(filtered.get(i,cp)))
	      {
		obj_val += (class_weight
			    * (hinge_loss(proj - *sortedLU_iter)
			       + hinge_loss(*(sortedLU_iter+1) - proj)));
	      }
	    left_classes++;
	    right_classes--;
	    left_weight += other_weight;
	    right_weight -= other_weight;
	    ++class_iter;
	    sc++;
	    sortedLU_iter+=2;
	  }
      }
    return obj_val;
  }

  // // ***********************************
  // // calculates the objective value for a subset of instances and classes

  // double compute_objective(const VectorXd& projection, const SparseMb& y,
  // 			 const VectorXi& nclasses, int maxclasses,
  // 			 size_t i_start, size_t i_end,
  // 			 int sc_start, int sc_end,
  // 			 const vector<int>& sorted_class,
  // 			 const vector<int>& class_order,
  // 			 const VectorXd& sortedLU,
  // 			 const boolmatrix& filtered,
  // 			 double C1, double C2,
  // 			 const param_struct& params)
  // {
  //   double tmp;
  //   int sc, cp;
  //   int left_classes, right_classes;
  //   double left_weight, right_weight, other_weight, class_weight;
  //   std::vector<int> classes;
  //   std::vector<int>::iterator class_iter;
  //   size_t no_filtered = filtered.count();
  //   classes.reserve(maxclasses+1);
  //   double obj_val = 0.0;
  //   const double* sortedLU_iter;
  //   for (size_t i = i_start; i < i_end; i++)
  //     {
  //       tmp = projection.coeff(i);

  //       class_weight = C1;
  //       other_weight = C2;
  //       if (params.ml_wt_by_nclasses)
  // 	{
  // 	  other_weight /= nclasses.coeff(i);
  // 	}
  //       if (params.ml_wt_class_by_nclasses)
  // 	{
  // 	  class_weight /= nclasses.coeff(i);
  // 	}

  //       left_classes = 0; //number of classes to the left of the current one
  //       left_weight = 0; // left_classes * other_weight
  //       right_classes = nclasses[i]; //number of classes to the right of the current one
  //       right_weight = other_weight * right_classes;

  //       // calling y.coeff is expensive so get the classes here
  //       classes.resize(0);
  //       for (SparseMb::InnerIterator it(y,i); it; ++it)
  // 	{
  // 	  int c = class_order[it.col()];
  // 	  if ( c < sc_start )
  // 	    {
  // 	      left_classes++;
  // 	      left_weight += other_weight;
  // 	      right_classes--;
  // 	      right_weight -= other_weight;
  // 	    }
  // 	  else if (c < sc_end)
  // 	    {
  // 	      classes.push_back(c);
  // 	    }
  // 	}
  //       classes.push_back(sc_end); // this will always be the last
  //       std::sort(classes.begin(),classes.end());

  //       sc=sc_start;
  //       class_iter = classes.begin();
  //       sortedLU_iter = sortedLU.data() + 2*sc_start;
  //       while (sc < sc_end)
  // 	{
  // 	  while(sc < *class_iter)
  // 	    {
  // 	      // while example is not of class cp
  // 	      cp = sorted_class[sc];
  // 	      if (no_filtered == 0 || !(filtered.get(i,cp)))
  // 		{
  // 		  obj_val += (left_classes?(left_weight * hinge_loss(*sortedLU_iter - tmp)):0)
  // 		    + (right_classes?(right_weight * hinge_loss(tmp - *(sortedLU_iter+1))):0);

  // 		}
  // 	      sc++;
  // 	      sortedLU_iter+=2;
  // 	    }
  // 	  if (sc < sc_end) // test if we are done
  // 	    {
  // 	      // example has class cp
  // 	      cp = sorted_class[sc];
  // 	      if (!params.remove_class_constraints || no_filtered == 0 || !(filtered.get(i,cp)))
  // 		{
  // 		  obj_val += (class_weight
  // 			      * (hinge_loss(tmp - *sortedLU_iter)
  // 				 + hinge_loss(*(sortedLU_iter+1) - tmp)));
  // 		}
  // 	      left_classes++;
  // 	      right_classes--;
  // 	      left_weight += other_weight;
  // 	      right_weight -= other_weight;
  // 	      ++class_iter;
  // 	      sc++;
  // 	      sortedLU_iter+=2;
  // 	    }
  // 	}
  //     }
  //   return obj_val;
  // }

  // *******************************
  // Calculates the objective function
  double calculate_objective_hinge(const VectorXd& projection, const SparseMb& y,
				   const VectorXi& nclasses,
				   const std::vector<int>& sorted_class,
				   const std::vector<int>& class_order,
				   const double norm, const VectorXd& sortedLU,
				   const boolmatrix& filtered,
				   double lambda, double C1, double C2,
				   const param_struct& params)
  {
    double obj_val = 0;
    bool none_filtered = filtered.count()==0;
#if MCTHREADS
#pragma omp parallel for default(shared) reduction(+:obj_val)
#endif
    for (size_t i = 0; i < projection.size(); i++)
      {
	obj_val += calculate_ex_objective_hinge(i, projection.coeff(i),  y,
						nclasses,
						sorted_class,class_order,
						sortedLU, filtered,
						none_filtered,
						C1, C2, params);
      }
    obj_val += .5 * lambda * norm * norm;
    return obj_val;
  }

  double calculate_objective_hinge(const VectorXd& projection, const SparseMb& y,
				   const VectorXi& nclasses,
				   const std::vector<int>& sorted_class,
				   const std::vector<int>& class_order,
				   const double norm, const VectorXd& sortedLU,
				   double lambda, double C1, double C2,
				   const param_struct& params)
  {
    const int noClasses = y.cols();
    const int n = projection.size();
    boolmatrix filtered(n,noClasses);
    return calculate_objective_hinge(projection, y,nclasses, sorted_class, class_order, norm, sortedLU, filtered, lambda, C1, C2, params);
  }

}
