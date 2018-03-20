#ifndef _OBJECTIVE_H
#define _OBJECTIVE_H

//#include "Eigen/Dense"
#include "typedefs.h"
#include "parameter.h"
//#include <vector>

class boolmatrix;

namespace mcsolver_detail{

  /** The hinge loss */
    inline double constexpr hinge_loss(double val)
    {
      return ((val<1.0)?(1.0-val):0.0);
    }

    // *****************************************
    // Function to calculate the objective for one example
    // this almost duplicates the function compute_objective
    // the two functions should be unified
    // this functions is easier to use with the finite_diff_test because
    // it does not require the entire projection vector
    double calculate_ex_objective_hinge(std::size_t i, double proj, const SparseMb& y,
					const Eigen::VectorXi& nclasses,
					const Eigen::VectorXd& inside_weight, const Eigen::VectorXd& outside_weight,
					const std::vector<int>& sorted_class,
					const std::vector<int>& class_order,
					const Eigen::VectorXd& sortedLU,
					  const ::boolmatrix& filtered,
					bool none_filtered,
					double C1, double C2,
					const ::param_struct& params);

    /* // *********************************** */
    /* // calculates the objective value for a subset of instances and classes */

    /* double compute_objective(const VectorXd& projection, const SparseMb& y, */
    /* 			 const VectorXi& nclasses, int maxclasses, */
    /* 			 size_t i_start, size_t i_end, */
    /* 			 int sc_start, int sc_end, */
    /* 			 const std::vector<int>& sorted_class, */
    /* 			 const std::vector<int>& class_order, */
    /* 			 const VectorXd& sortedLU, */
    /* 			 const boolmatrix& filtered, */
    /* 			 double C1, double C2, */
    /* 			 const ::param_struct& params); */

    // *******************************
    // Calculates the objective function

    double calculate_objective_hinge(const Eigen::VectorXd& projection, const SparseMb& y,
				     const Eigen::VectorXi& nclasses,
				     const Eigen::VectorXd& inside_weight, const Eigen::VectorXd& outside_weight,
				     const std::vector<int>& sorted_class,
				     const std::vector<int>& class_order,
				     const double norm, const Eigen::VectorXd& sortedLU,
				       const ::boolmatrix& filtered,
				     double lambda, double C1, double C2,
				     const ::param_struct& params);



    double calculate_objective_hinge(const Eigen::VectorXd& projection, const SparseMb& y,
				     const Eigen::VectorXi& nclasses,
				     const Eigen::VectorXd& inside_weight, const Eigen::VectorXd& outside_weight,
				     const std::vector<int>& sorted_class,
				     const std::vector<int>& class_order,
				     const double norm, const Eigen::VectorXd& sortedLU,
				     double lambda, double C1, double C2,
				     const ::param_struct& params);


  
}
#endif // _OBJECTIVE_H
