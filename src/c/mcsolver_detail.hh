#ifndef MCSOLVER_DETAIL_HH
#define MCSOLVER_DETAIL_HH

//functions used in solve to initialize w and other quantities
#include "mcsolver_init.hh"

//functions that perform the gradient calculation and updating
#include "mcupdate.hh"

// functions to calculate the objective values
#include "objective.h"

//functions for optimization of L and U 
#include "optimizelu.hh" 

namespace mcsolver_detail{

// ************************
// function to set eta for each iteration
double set_eta(param_struct const& params, size_t const t, double const lambda);


// ********************************
// Compute the means of the classes of the projected data
void proj_means(VectorXd& means, VectorXi const& nc,
		VectorXd const& projection, SparseMb const& y,
		const param_struct& params, boolmatrix const& filtered);

//*****************************************
// Update the filtered constraints
void update_filtered(boolmatrix& filtered, const VectorXd& projection,
		     const VectorXd& l, const VectorXd& u, const SparseMb& y,
		     const bool filter_class);

}

#endif //MCSOLVER_DETAIL_HH
