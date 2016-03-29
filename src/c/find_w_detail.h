#ifndef FIND_W_DETAIL_H
#define FIND_W_DETAIL_H

#include "typedefs.h"
#include "parameter.h"
#include "boolmatrix.h"
#include "WeightVector.h"
#include "mutexlock.h"                  // MutexType

// ------------- templated for input data = SparseM or DenseM -----------------

/** Initialize w either from weights_avg or randomly.
 * - If weightsCopy == true, just \c w.init(weights.col(projection_dim)) and return
 * - otherwise:
 *   - choose a dirn between two perhaps far class centers
 *   - add a bit of randomness
 *   - project orthogonal to \c weights columns dim < projection_dim.
 */
    template<typename EigenType>
void init_w( WeightVector& w,
             EigenType const& x, SparseMb const& y, VectorXi const& nc,
             DenseM const& weights, int const projection_dim, bool const weightsCopy = false);

#if 0
/** Initializes the lower and upper bound */
template<typename EigenType>
void init_lu(VectorXd& l, VectorXd& u, VectorXd& means, const VectorXi& nc,
             const WeightVector& w,
             EigenType& x, const SparseMb& y);
#endif

/** function to calculate the difference vector beween the mean vectors of two classes */
template<typename EigenType>
void difference_means(VectorXd& difference, const EigenType& x, const SparseMb& y,
                      const VectorXi& nc, const int c1, const int c2);


/** check the gradient calculation using finite differences */
template<typename EigenType>
void finite_diff_test(const WeightVector& w, const EigenType& x, size_t idx,
                      const SparseMb& y, const VectorXi& nclasses, int maxclasses,
                      const std::vector<int>& sorted_class, const std::vector<int>& class_order,
                      const VectorXd& sortedLU,
                      const boolmatrix& filtered,
                      double C1, double C2, const param_struct& params);

/** compute gradient and update w, L and U using safe updates that do not overshoot.
 * - update w first, making sure we do not overshoot
 * - then update LU using projected gradient
 * - only works for batch sizes of 1
 * - may require x input data to have unit norms (?) */
template<typename EigenType>
void update_safe_SGD (WeightVector& w, VectorXd& sortedLU, VectorXd& sortedLU_avg,
                      const EigenType& x, const SparseMb& y, const VectorXd& sqNormsX,
                      const double C1, const double C2, const double lambda,
                      const unsigned long t, const double eta_t,
                      const size_t n, const VectorXi& nclasses, const int maxclasses,
                      const std::vector<int>& sorted_class, const std::vector<int>& class_order,
                      const boolmatrix& filtered,
                      const int sc_chunks, const int sc_chunk_size, const int sc_remaining,
                      const param_struct& params);


/** function to perform batch SGD update of w, L and U */
template<typename EigenType>
void update_minibatch_SGD(WeightVector& w, VectorXd& sortedLU, VectorXd& sortedLU_avg,
		      const EigenType& x, const SparseMb& y,
		      const double C1, const double C2, const double lambda,
		      const unsigned long t, const double eta_t,
		      const size_t n, const size_t batch_size,
		      const VectorXi& nclasses, const int maxclasses,
		      const std::vector<int>& sorted_class, const std::vector<int>& class_order,
		      const boolmatrix& filtered,
                      const size_t sc_chunks, const size_t sc_chunk_size, const size_t sc_remaining,
                      const int idx_chunks, const int idx_chunk_size, const int idx_remaining,
                      MutexType* idx_locks, MutexType* sc_locks,
		      const param_struct& params);
//
// ******************************
// Projection to a new vector that is orthogonal to the rest
// It is basically Gram-Schmidt Orthogonalization
// *************************************
// need to reimplement this funciton to work with (inside) the WeightVector class
// this might be a costly operation that might be not needed
// we'll implement this when we get there

void project_orthogonal( VectorXd& w, const DenseM& weights,
			 const int& projection_dim);

/** function to set eta for each iteration */
double set_eta(param_struct const& params, size_t const t, double const lambda);

/** Compute the means of the classes of the projected data */
void proj_means(VectorXd& means, const VectorXi& nc,
		const VectorXd& projection, const SparseMb& y);

/** Initialize l, u and means.
 * \p projection is the projection of every example x onto w
 */
void init_lu( VectorXd& l, VectorXd& u, VectorXd& means,
              enum Reorder_Type const reorder_type, VectorXd const& projection,
              SparseMb const& y, VectorXi const& nc );

/** Update the filtered constraints */
void update_filtered(boolmatrix& filtered, const VectorXd& projection,
		     const VectorXd& l, const VectorXd& u, const SparseMb& y,
		     const bool filter_class);

/** calculate the multipliers (for the w gradient update)
 * and the gradients for l and u updates on a subset of classes and instances.
 *
 * - computes
 *   - \c multipliers[]       length \c idx_end-idx_start
 *   - \c sortedLU_gradient[] length \c 2*(sc_end-sc_start)
 */
void compute_gradients (VectorXd& multipliers , VectorXd& sortedLU_gradient,
                        // const inputs...
			const size_t idx_start, const size_t idx_end,
			const int sc_start, const int sc_end,
			const VectorXd& proj, const VectorXsz& index,
			const SparseMb& y, const VectorXi& nclasses,
			const int maxclasses,
			const std::vector<int>& sorted_class,
			const std::vector<int>& class_order,
			const VectorXd& sortedLU,
			const boolmatrix& filtered,
			const double C1, const double C2,
			const param_struct& params );

// function to compute the gradient size for w for a single example
double compute_single_w_gradient_size ( const int sc_start, const int sc_end,
					const double proj, const size_t i,
					const SparseMb& y, const VectorXi& nclasses,
					const int maxclasses,
					const std::vector<int>& sorted_class,
					const std::vector<int>& class_order,
					const VectorXd& sortedLU,
					const boolmatrix& filtered,
					const double C1, const double C2,
					const param_struct& params );


// function to update L and U for a single example, given w.
// performs projected gradient updates
void update_single_sortedLU( VectorXd& sortedLU,
			     int sc_start, int sc_end,
			     const double proj, const size_t i,
			     const SparseMb& y, const VectorXi& nclasses,
			     int maxclasses,
			     const std::vector<int>& sorted_class,
			     const std::vector<int>& class_order,
			     const boolmatrix& filtered,
			     double C1, double C2, const double eta_t,
			     const param_struct& params);


// generates num_samples uniform samples between 0 and max-1 with replacement,
//  and sorts them in ascending order
void get_ordered_sample(std::vector<int>& sample, int max, int num_samples);


// function to calculate the multiplier of the gradient for w for a single example.
// subsampling the negative class constraints
double compute_single_w_gradient_size_sample ( int sc_start, int sc_end,
					       const std::vector<int>& sc_sample,
					       const double proj, const size_t i,
					       const SparseMb& y, const VectorXi& nclasses,
					       int maxclasses,
					       const std::vector<int>& sorted_class,
					       const std::vector<int>& class_order,
					       const VectorXd& sortedLU,
					       const boolmatrix& filtered,
					       double C1, double C2,
					       const param_struct& params );

// function to update L and U for a single example, given w.
// subsampling the negative classes
void update_single_sortedLU_sample ( VectorXd& sortedLU,
				     int sc_start, int sc_end,
				     const std::vector<int>& sc_sample,
				     const double proj, const size_t i,
				     const SparseMb& y, const VectorXi& nclasses,
				     int maxclasses,
				     const std::vector<int>& sorted_class,
				     const std::vector<int>& class_order,
				     const boolmatrix& filtered,
				     double C1, double C2, const double eta_t,
				     const param_struct& params);

// ********************************
// Get unique values in the class vector -> classes
std::vector<int> get_classes(VectorXd y);


// *********************************
// Ranks the classes to build the switches
void rank_classes(std::vector<int>& order, std::vector<int>& cranks, const VectorXd& sortKey);

// **********************************************
// get l and u in the original class order

void get_lu (VectorXd& l, VectorXd& u, const VectorXd& sortedLU, const std::vector<int>& sorted_class);

// **********************************
// sort l and u in the new class order

void get_sortedLU(VectorXd& sortedLU, const VectorXd& l, const VectorXd& u,
		  const std::vector<int>& sorted_class);

// *******************************
// Get the number of exampls in each class

void init_nc(VectorXi& nc, VectorXi& nclasses, const SparseMb& y);

// ************************************
// Get the sum of the weight of all examples in each class

void init_wc(VectorXd& wc, const VectorXi& nclasses, const SparseMb& y, const param_struct& params);


// *****************************************************
// get the optimal values for lower and upper bounds given
// a projection and the class order
// computationally expensive so it should be done sparingly
// Note: optimizeLU does not care about input state of l,u
void optimizeLU(VectorXd&l, VectorXd&u,
		const VectorXd& projection, const SparseMb& y,
		const std::vector<int>& class_order, const std::vector<int>& sorted_class,
		const VectorXd& wc, const VectorXi& nclasses,
		const boolmatrix& filtered,
		const double C1, const double C2,
		const param_struct& params,
		bool print = false);

// ******************************
// Convert to a STD vetor from Eigen Vector
void toVector(std::vector<int>& to, const VectorXd& from);


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
double calculate_ex_objective_hinge(size_t i, double proj, const SparseMb& y,
				    const VectorXi& nclasses,
				    const std::vector<int>& sorted_class,
				    const std::vector<int>& class_order,
				    const VectorXd& sortedLU,
				    const boolmatrix& filtered,
				    bool none_filtered,
				    double C1, double C2,
				    const param_struct& params);

// ***********************************
// calculates the objective value for a subset of instances and classes

double compute_objective(const VectorXd& projection, const SparseMb& y,
			 const VectorXi& nclasses, int maxclasses,
			 size_t i_start, size_t i_end,
			 int sc_start, int sc_end,
			 const std::vector<int>& sorted_class,
			 const std::vector<int>& class_order,
			 const VectorXd& sortedLU,
			 const boolmatrix& filtered,
			 double C1, double C2,
			 const param_struct& params);

// *******************************
// Calculates the objective function

double calculate_objective_hinge(const VectorXd& projection, const SparseMb& y,
				 const VectorXi& nclasses,
                                 const std::vector<int>& sorted_class,
                                 const std::vector<int>& class_order,
				 const double norm, const VectorXd& sortedLU,
				 const boolmatrix& filtered,
				 double lambda, double C1, double C2,
				 const param_struct& params);



double calculate_objective_hinge(const VectorXd& projection, const SparseMb& y,
				 const VectorXi& nclasses,
                                 const std::vector<int>& sorted_class,
                                 const std::vector<int>& class_order,
				 const double norm, const VectorXd& sortedLU,
				 double lambda, double C1, double C2,
				 const param_struct& params);



#endif // FIND_W_DETAIL_H
