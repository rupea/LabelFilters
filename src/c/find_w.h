#ifndef __FIND_W_H
#define __FIND_W_H

#include <omp.h>
#include <boost/numeric/conversion/bounds.hpp>
#include <boost/limits.hpp>
#include "constants.h"
#include "typedefs.h"
#include "WeightVector.h"
#include "printing.h"
#include "parameter.h"
#include "boolmatrix.h"
#include "mutexlock.h" 
#include "utils.h"

#ifdef PROFILE
#include <gperftools/profiler.h>
#endif

using Eigen::VectorXd;
using Eigen::VectorXi;

using namespace std;


// *******************************
// The hinge loss -- implemented here to get rid of compiler warnings
inline double hinge_loss(double val)
{
  return ((val<1)?(1-val):0);
}

// ************************
// function to set eta for each iteration
double set_eta(const param_struct& params, size_t t, double lambda);


// ******************************
// Convert to a STD vetor from Eigen Vector
void toVector(std::vector<int>& to, const VectorXd& from);


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
			 const vector<int>& sorted_class, 
			 const vector<int>& class_order,
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


// ********************************
// Get unique values in the class vector -> classes
std::vector<int> get_classes(VectorXd y);


// *********************************
// Ranks the classes to build the switches
void rank_classes(std::vector<int>& order, std::vector<int>& cranks, const VectorXd& sortKey);

// **********************************************
// get l and u in the original class order

void get_lu (VectorXd& l, VectorXd& u, const VectorXd& sortedLU, const vector<int>& sorted_class);

// **********************************
// sort l and u in the new class order

void get_sortedLU(VectorXd& sortedLU, const VectorXd& l, const VectorXd& u, 
		  const vector<int>& sorted_class);

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
void optimizeLU(VectorXd&l, VectorXd&u, 
		const VectorXd& projection, const SparseMb& y, 
		const vector<int>& class_order, const vector<int>& sorted_class,
		const VectorXd& wc, const VectorXi& nclasses,
		const boolmatrix& filtered, 
		const double C1, const double C2,
		const param_struct& params,
		bool print = false);


// ********************************
// Initializes the lower and upper bound
template<typename EigenType>
void init_lu(VectorXd& l, VectorXd& u, VectorXd& means, const VectorXi& nc,
	     const WeightVector& w,
	     EigenType& x, const SparseMb& y)
{
  int noClasses = y.cols();
  size_t n = x.rows();
  size_t c,i,k;
  double pr;
  means.resize(noClasses);
  means.setZero();
  for (k = 0; k < noClasses; k++)
    {
      l(k)=boost::numeric::bounds<double>::highest();
      u(k)=boost::numeric::bounds<double>::lowest();	      
    }
  VectorXd projection;
  w.project(projection,x);
  for (i=0;i<n;i++)
    {
      for (SparseMb::InnerIterator it(y,i);it;++it)
	{	
	  if (it.value())
	    {
	      c = it.col();
	      pr = projection.coeff(i);
	      means(c)+=pr;

	      l(c)=pr<l(c)?pr:l(c);
	      u(c)=pr>u(c)?pr:u(c);
	    }
	}
    }
  for (k = 0; k < noClasses; k++)
    {
      means(k) /= nc(k);
    }
}

// ********************************
// Compute the means of the classes of the projected data
void proj_means(VectorXd& means, const VectorXi& nc,
		const VectorXd& projection, const SparseMb& y);



//*****************************************
// Update the filtered constraints

void update_filtered(boolmatrix& filtered, const VectorXd& projection,  
		     const VectorXd& l, const VectorXd& u, const SparseMb& y, 
		     const bool filter_class);

// function to calculate the difference vector beween the mean vectors of two classes

template<typename EigenType>
  void difference_means(VectorXd& difference, const EigenType& x, const SparseMb& y, const VectorXi& nc, const int c1, const int c2)
{
  size_t d = x.cols();
  size_t n = x.rows();
  difference.resize(d);
  difference.setZero();
  for (size_t row=0;row<n;row++)
    {
      if (y.coeff(row,c1))
	{
	  typename EigenType::InnerIterator it(x,row);
	  for (; it; ++it)
	    {
	      difference.coeffRef(it.col())+=(it.value()/nc(c1));
	    }
	}
      if (y.coeff(row,c2))
	{
	  typename EigenType::InnerIterator it(x,row);
	  for (; it; ++it)
	    {
	      difference.coeffRef(it.col())-=(it.value()/nc(c2));
	    }
	}
    }
}

// ******************************
// Projection to a new vector that is orthogonal to the rest
// It is basically Gram-Schmidt Orthogonalization
// *************************************
// need to reimplement this funciton to work with (inside) the WeightVector class
// this might be a costly operation that might be not needed
// we'll implement this when we get there

// void project_orthogonal(VectorXd& w, const DenseM& weights,
// 			const size_t& projection_dim);



// ***********************************************
// calculate the multipliers (for the w gradient update)
// and the gradients for l and u updates 
// on a subset of classes and instances

void compute_gradients (VectorXd& multipliers , VectorXd& sortedLU_gradient, 
			const size_t idx_start, const size_t idx_end, 
			const int sc_start, const int sc_end,
			const VectorXd& proj, const VectorXsz& index,
			const SparseMb& y, const VectorXi& nclasses, 
			int maxclasses,
			const vector<int>& sorted_class,
			const vector<int>& class_order,
			const VectorXd& sortedLU,
			const boolmatrix& filtered,
			double C1, double C2,
			const param_struct& params );


// ****************************************************
// check the gradient calculation using finite differences

template<typename EigenType>
void finite_diff_test(const WeightVector& w, const EigenType& x, size_t idx, const SparseMb& y, const VectorXi& nclasses, int maxclasses, const vector<int>& sorted_class, const vector<int>& class_order, const VectorXd& sortedLU, const boolmatrix& filtered, double C1, double C2, const param_struct& params)
{
  double delta = params.finite_diff_test_delta;
  VectorXd proj(1);
  proj.coeffRef(0) = w.project_row(x,idx);
  bool none_filtered = filtered.count()==0;
  double obj = calculate_ex_objective_hinge(idx, proj.coeff(0), y, nclasses, sorted_class, class_order, sortedLU, filtered, none_filtered, C1, C2, params);

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

  double obj_w_grad = calculate_ex_objective_hinge(idx, w_new.project_row(x,idx), y, nclasses, sorted_class, class_order, sortedLU, filtered, none_filtered, C1, C2, params);
  double w_grad_error = fabs(obj_w_grad - obj + multsign*delta*multipliers.coeff(0)*xnorm);
  
  VectorXd sortedLU_new(sortedLU);
  sortedLU_new += sortedLU_gradient * delta / sortedLU_gradient.norm();  // have some delta that is inversely proportional to the norm of the gradient 

  double obj_LU_grad = calculate_ex_objective_hinge(idx, proj.coeff(0), y, nclasses, sorted_class, class_order, sortedLU_new, filtered, none_filtered, C1, C2, params);
  double LU_grad_error = fabs(obj_LU_grad - obj + delta*sortedLU_gradient.norm());
  
  cerr << "w_grad_error:  " << w_grad_error << "   " << obj_w_grad - obj << "  " << obj_w_grad << "  " << obj << "  " << multsign*delta*multipliers.coeff(0)*xnorm << "   " << xnorm << "  " << idx << "   " << proj.coeff(0) << "  " << w_new.project_row(x,idx)  << "  ";
      
  for (SparseMb::InnerIterator it(y,idx); it; ++it)
    {
      int order = class_order[it.col()];
      cerr << it.col() << ":" << it.value() << ":" << order << ":" <<sortedLU.coeff(2*order) + 1 << ":" << sortedLU.coeff(2*order+1) - 1  << "  ";
    }
  cerr << endl;
  /* if (idx == 9022) */
  /*   {						 */
  /*     cerr << sortedLU.transpose() - VectorXd::Ones(sortedLU.size()).transpose() << endl; */
  /*     cerr << sortedLU.transpose() +  VectorXd::Ones(sortedLU.size()).transpose() << endl; */
  /*   } */
  cerr << "LU_grad_error: " << LU_grad_error << "  " << obj_LU_grad - obj << "  " << "  " << obj_LU_grad << "  " << obj << "  " << delta*sortedLU_gradient.norm() << "  " << "  " << idx << "  " << proj.coeff(0) << "  ";
  for (SparseMb::InnerIterator it(y,idx); it; ++it)
    {
      int order = class_order[it.col()];
      cerr << it.col() << ":" << it.value() << ":" << order << ":" << sortedLU.coeff(2*order) + 1 << ":" << sortedLU.coeff(2*order+1) - 1  << " - " << sortedLU_new.coeff(2*order) + 1 << ":" << sortedLU_new.coeff(2*order+1) - 1  << "  ";
    }
  cerr << endl;
}       


// *********************************
// Solve the optimization using the gradient decent on hinge loss

template<typename EigenType>
void solve_optimization(DenseM& weights, DenseM& lower_bounds,
			DenseM& upper_bounds,
			VectorXd& objective_val,
			DenseM& weights_avg, DenseM& lower_bounds_avg,
			DenseM& upper_bounds_avg,
			VectorXd& objective_val_avg,
			const EigenType& x, const SparseMb& y,
			const param_struct& params)

{
  #ifdef PROFILE
  ProfilerStart("find_w.profile");
  #endif
  
  double lambda = 1.0/params.C2;
  double C1 = params.C1/params.C2;
  double C2 = 1.0;
  const	int no_projections = params.no_projections;
  cout << "no_projections: " << no_projections << endl;
  const size_t n = x.rows(); 
  const size_t batch_size = (params.batch_size < 1 || params.batch_size > n) ? (size_t) n : params.batch_size;
  const size_t d = x.cols();
  //std::vector<int> classes = get_classes(y);
  cout << "size x: " << x.rows() << " rows and " << x.cols() << " columns.\n";
  cout << "size y: " << y.rows() << " rows and " << y.cols() << " columns.\n";

  const int noClasses = y.cols();
  WeightVector w;
  VectorXd projection, projection_avg;
  VectorXd l(noClasses),u(noClasses);
  VectorXd sortedLU(2*noClasses); // holds l and u interleaved in the curent class sorting order (i.e. l,u,l,u,l,u)
  VectorXd sortedLU_gradient(2*noClasses); // used to improve cache performance
  VectorXd sortedLU_gradient_chunk;
  VectorXd l_avg(noClasses),u_avg(noClasses); // the lower and upper bounds for the averaged gradient
  VectorXd sortedLU_avg(2*noClasses); // holds l_avg and u_avg interleaved in the curent class sorting order (i.e. l_avg,u_avg,l_avg,u_avg,l_avg,u_avg)
  VectorXd means(noClasses); // used for initialization of the class order vector;
  VectorXi nc; // the number of examples in each class 
  VectorXd wc; // the number of examples in each class 
  VectorXi nclasses; // the number of examples in each class 
  int maxclasses; // the maximum number of classes an example might have
  double eta_t, tmp, sj;
  int cp;// current class and the other classes
  size_t obj_idx = 0, obj_idx_avg = 0;
  //  bool order_changed = 1;
  VectorXd proj(batch_size);
  VectorXsz index(batch_size);
  VectorXd multipliers(batch_size);
  VectorXd multipliers_chunk;
  // in the multilabel case each example will have an impact proportinal
  // to the number of classes it belongs to. ml_wt and ml_wt_class
  // allows weighting that impact when updating params for the other classes
  // respectively its own class. 
  size_t  i=0, k=0,idx=0;
  unsigned long t = 1;
  std::vector<int> sorted_class(noClasses), class_order(noClasses);//, prev_class_order(noClasses);// used as the switch
  char iter_str[30];
  
  // how to split the work for gradient update iterations
#ifdef _OPENMP
  int total_chunks = omp_get_max_threads();
  int sc_chunks = total_chunks;// floor(sqrt(total_chunks));
  int idx_chunks = total_chunks/sc_chunks;
  if (params.num_threads < 1)
    omp_set_num_threads(omp_get_max_threads());
  else
    omp_set_num_threads(params.num_threads);
#else
  int idx_chunks = 1;
  int sc_chunks = 1;
#endif 
  MutexType* sc_locks = new MutexType [sc_chunks];
  MutexType* idx_locks = new MutexType [idx_chunks];
  int sc_chunk_size = noClasses/sc_chunks;
  int sc_remaining = noClasses % sc_chunks;
  int idx_chunk_size = batch_size/idx_chunks;
  int idx_remaining = batch_size % idx_chunks;
  

  if (params.report_epoch > 0)
    {
      objective_val.resize(1000 + (no_projections * params.max_iter / params.report_epoch));
    }

  if (params.report_avg_epoch > 0)
    {
      objective_val_avg.resize(1000 + (no_projections * params.max_iter / params.report_avg_epoch));
    }

  init_nc(nc, nclasses, y);
  if (params.optimizeLU_epoch > 0)
    {
      init_wc(wc, nclasses, y, params);
    }

  maxclasses = nclasses.maxCoeff();
  //keep track of which classes have been elimninated for a particular example
  boolmatrix filtered(n,noClasses);
  VectorXd difference(d);
  unsigned long total_constraints = n*noClasses - (1-params.remove_class_constraints)*nc.sum();
  size_t no_filtered=0;
  int projection_dim = 0;

  if (params.resume && params.remove_constraints)
    {
      for (projection_dim = 0; projection_dim < weights.cols(); projection_dim++)
	{
	  // use weights_avg since they will hold the correct weights regardless if 
	  // averaging was performed on a prior run or not
	  w = WeightVector(weights_avg.col(projection_dim));
	  l = lower_bounds_avg.col(projection_dim);
	  u = upper_bounds_avg.col(projection_dim);
	  
	  // should we do this in parallel? 
	  // the main problem is that the bitset is not thread safe (changes to one bit can affect changes to other bits)
	  // should update to use the filter class 
	  // things will not work correctly with remove_class_constrains on. We need to update wc, nclass 
	  //       and maybe nc
	  // check if nclass and nc are used for anything else than weighting examples belonging
	  //       to multiple classes
	  if (params.remove_constraints && projection_dim < no_projections-1)
	    {

	      w.project(projection,x); // could eliminate this since it has most likely been calculated above, but we keep it here for now for clarity
	      update_filtered(filtered, projection, l, u, y, params.remove_class_constraints);
	    }
	      
	  no_filtered = filtered.count();
	  cout << "Filtered " << no_filtered << " out of " << total_constraints << endl;
	  // work on this. This is just a crude approximation.
	  // now every example - class pair introduces nclass(example) constraints
	  // if weighting is done, the number is different
	  // eliminating one example -class pair removes nclass(exmple) potential
	  // if the class not among the classes of the example
	  long int no_remaining = total_constraints - no_filtered;
	  lambda = no_remaining*1.0/(total_constraints*params.C2);
	}
    }

  weights.conservativeResize(d, no_projections);
  weights_avg.conservativeResize(d, no_projections);
  lower_bounds.conservativeResize(noClasses, no_projections);
  upper_bounds.conservativeResize(noClasses, no_projections);
  lower_bounds_avg.conservativeResize(noClasses, no_projections);
  upper_bounds_avg.conservativeResize(noClasses, no_projections);
  
  for(; projection_dim < no_projections; projection_dim++)
    {
            
      // initialize w as vector between the means of two random classes.
      // should find clevered initialization schemes
      int c1 = ((int) rand()) % noClasses;
      int c2 = ((int) rand()) % noClasses;
      if (c1 == c2)
	{
	  c2=(c1+1)%noClasses;
	}
      difference_means(difference,x,y,nc,c1,c2);
      w = WeightVector(difference*10/difference.norm());  // get a better value than 10 .. somethign that would match the margins
      // w.setRandom(); // initialize to a random value
      
      // initialize the l an u  
      init_lu(l,u,means,nc,w,x,y); // use the projection, remove the template, no need to initialize the means
            
      w.project(projection,x);
      switch (params.reorder_type)
	{
	case REORDER_AVG_PROJ_MEANS:
	  // use the current w since averaging has not started yet 
	case REORDER_PROJ_MEANS:	       
	  proj_means(means, nc, projection, y);
	  break;
	case REORDER_RANGE_MIDPOINTS: 
	  means = l+u; //no need to divide by 2 since it is only used for ordering
	  break;
	default:
	  cerr << "Error, reordering " << params.reorder_type << " not implemented" << endl;
	  exit(-1);	      
	}	  
      rank_classes(sorted_class, class_order, means);
      
      
      if (params.optimizeLU_epoch > 0)
	{
	  optimizeLU(l,u,projection,y,class_order, sorted_class, wc, nclasses, filtered, C1, C2, params);
	}	  

      get_sortedLU(sortedLU, l, u, sorted_class);

      if (params.optimizeLU_epoch <= 0)
	{
	  // we do not need the average sortedLU since we will
	  // optimize the bounds at the end based on the 
	  // average w 
	  sortedLU_avg.setZero();      
	}

      print_report<EigenType>(projection_dim,batch_size, noClasses,C1,C2,lambda,w.size(),x);


      t = 0;		    
      while (t < params.max_iter)
	{
	  t++;
	  // print some progress
	  if (!params.report_epoch && t % 1000 == 0)
	    {
	      snprintf(iter_str,30, "Projection %d > ", projection_dim+1);
	      print_progress(iter_str, t, params.max_iter);
	      fflush(stdout);
	    }
	  
	  // perform finite differences test
	  if ( params.finite_diff_test_epoch && (t % params.finite_diff_test_epoch == 0) ) 
	    {
	      for (size_t fdtest=0; fdtest<params.no_finite_diff_tests; fdtest++)
		{
		  idx = ((size_t) rand()) % n;
		  finite_diff_test(w, x, idx, y, nclasses, maxclasses, sorted_class, class_order, sortedLU, filtered, C1, C2, params);
		}
	    }
	  
	  // compute the gradient and update 
	  
	  // first compute all the projections so that we can update w directly
	  for (idx = 0; idx < batch_size; idx++)// batch_size will be equal to n for complete GD
	    {
	      if(batch_size < n)
		{
		  i = ((size_t) rand()) % n;
		}
	      else
		{
		  i=idx;
		}
	      
	      proj.coeffRef(idx) = w.project_row(x,i);
	      index.coeffRef(idx)=i;
	    }	      
	  // now we can update w directly
	  
	  multipliers.setZero();
	  sortedLU_gradient.setZero(); 
	  
# pragma omp parallel for  default(shared) shared(idx_locks,sc_locks) private(multipliers_chunk,sortedLU_gradient_chunk) collapse(2)
	  for (int idx_chunk = 0; idx_chunk < idx_chunks; idx_chunk++)
	    for (int sc_chunk = 0; sc_chunk < sc_chunks; sc_chunk++)
	      {
		// the first chunks will have an extra iteration 
		int idx_start = idx_chunk*idx_chunk_size + (idx_chunk<idx_remaining?idx_chunk:idx_remaining);
		int idx_incr = idx_chunk_size + (idx_chunk<idx_remaining);
		// the first chunks will have an extra iteration 
		int sc_start = sc_chunk*sc_chunk_size + (sc_chunk<sc_remaining?sc_chunk:sc_remaining);
		int sc_incr = sc_chunk_size + (sc_chunk<sc_remaining);
		compute_gradients(multipliers_chunk, sortedLU_gradient_chunk,
				  idx_start, idx_start+idx_incr, 
				  sc_start, sc_start+sc_incr,
				  proj, index, y, nclasses, maxclasses,
				  sorted_class, class_order,
				  sortedLU, filtered, 
				  C1, C2, params);
		
#pragma omp task default(none) shared(sc_chunk, idx_chunk, sortedLU_gradient, multipliers, sc_start, idx_start, sc_incr, idx_incr, sortedLU_gradient_chunk, multipliers_chunk, sc_locks,  idx_locks)
		{
#pragma omp task default(none) shared(idx_chunk, multipliers, multipliers_chunk, idx_start, idx_incr, idx_locks)
		  {
		    idx_locks[idx_chunk].YieldLock();
		    multipliers.segment(idx_start, idx_incr) += multipliers_chunk;
		    idx_locks[idx_chunk].Unlock();
		  }		    			
		  sc_locks[sc_chunk].YieldLock();
		  // shoulc update directly sortedLU
		  // this would be important if batch size is small
		  sortedLU_gradient.segment(2*sc_start, 2*sc_incr) += sortedLU_gradient_chunk;
		  sc_locks[sc_chunk].Unlock();
#pragma omp taskwait		     
		}
#pragma omp taskwait 
	      }
	  
	  // set eta for this iteration
	  eta_t = set_eta(params, t, lambda);
	  
	  //update w
	  if (params.avg_epoch && t >= params.avg_epoch)
	    {
	      // updates both the curent w and the average w
	      w.batch_gradient_update_avg(x, index, multipliers, lambda, eta_t);
	    }
	  else
	    {
	      // update only the current w
	      w.batch_gradient_update(x, index, multipliers, lambda, eta_t);
	    }
	  
	  // update the lower and upper bounds
	  // divide by batch_size here because the gradients have 
	  // not been averaged
	  // should be done above
	  sortedLU += sortedLU_gradient * (eta_t / batch_size); 
	  // update the average version
	  // should do in parallel (maybe Eigen already does it?)
	  // especially for small batch sizes. 
	  if (params.optimizeLU_epoch <= 0 && params.avg_epoch > 0 && t >= params.avg_epoch)
	    {
	      // if we optimize the LU, we do not need to
	      // keep track of the averaged lower and upper bounds 
	      // We optimize the bounds at the end based on the 
	      // average w 	      

	      // do not divide by t-params.avg_epoch + 1 here 
	      // do it when using sortedLU_avg
	      // it might become too big!, but through division it 
              //might become too small 
	      sortedLU_avg += sortedLU;
	    }

	  if ((params.reorder_epoch > 0 && (t % params.reorder_epoch == 0)
	       && params.reorder_type == REORDER_AVG_PROJ_MEANS) 
	      || (params.report_avg_epoch && (t % params.report_avg_epoch == 0)))
	    {
	      w.project_avg(projection_avg,x);		      
	    }
	  
	  if ((params.reorder_epoch > 0 && (t % params.reorder_epoch == 0)
	       && params.reorder_type == REORDER_PROJ_MEANS)
	      || (params.report_epoch > 0 && (t % params.report_epoch == 0))
	      || (params.optimizeLU_epoch > 0 && ( t % params.optimizeLU_epoch == 0)))
	    {
	      w.project(projection,x);		      
	    }
	  
	  // reorder the classes
	  if (params.reorder_epoch && (t % params.reorder_epoch == 0))
	    {
	      // do this in a function?
	      // get the current l and u in the original class order
	      get_lu(l,u,sortedLU,sorted_class);		  
	      if ( params.optimizeLU_epoch <= 0 && params.avg_epoch &&  t >= params.avg_epoch)
		{
		  get_lu(l_avg,u_avg,sortedLU_avg,sorted_class);  
		}
	      switch (params.reorder_type)
		{
		case REORDER_AVG_PROJ_MEANS:
		  // if averaging has not started yet, this defaults projecting 
		  // using the current w 
		  proj_means(means, nc, projection_avg, y);
		  break;
		case REORDER_PROJ_MEANS:	       
		  proj_means(means, nc, projection, y);
		  break;
		case REORDER_RANGE_MIDPOINTS: 
		  means = l+u; //no need to divide by 2 since it is only used for ordering
		  break;
		default:
		  cerr << "Error, reordering " << params.reorder_type << " not implemented" << endl;
		  exit(-1);	      
		}
	      // calculate the new class order
	      rank_classes(sorted_class, class_order, means);
	      // sort the l and u in the order of the classes
	      get_sortedLU(sortedLU, l, u, sorted_class);
	      if ( params.optimizeLU_epoch <= 0 && params.avg_epoch &&  t >= params.avg_epoch)
		{
		  // if we optimize the LU, we do not need to
		  // keep track of the averaged lower and upper bounds 
		  // We optimize the bounds at the end based on the 
		  // average w 	      
		  get_sortedLU(sortedLU_avg, l_avg, u_avg, sorted_class);
		}		  
	    }	     	      
	  
	  // optimize the lower and upper bounds (done after class ranking)
	  // since it depends on the ranks 
	  // if ranking type is REORDER_RANGE_MIDPOINTS, then class ranking depends on this
	  // but shoul still be done before since it is less expensive
	  // (could also be done after the LU optimization
	  if (params.optimizeLU_epoch > 0 && ( t % params.optimizeLU_epoch == 0) )
	    {
	      optimizeLU(l,u,projection,y,class_order, sorted_class, wc, nclasses, filtered, C1, C2, params);
	      get_sortedLU(sortedLU, l, u, sorted_class);
	    }

	  
	  // calculate the objective functions with respect to the current w and bounds
	  if( params.report_epoch && (t % params.report_epoch == 0) )
	    {
	      // use the current w to calculate objective
	      objective_val[obj_idx++] = 
		calculate_objective_hinge( projection, y, nclasses, 
					   sorted_class, class_order, 
					   w.norm(), sortedLU, filtered, 
					   lambda, C1, C2, params); // save the objective value
	      if(PRINT_O)
		{
		  cout << "objective_val[" << t << "]: " << objective_val[obj_idx-1] << " "<< w.norm() << endl;
		}
	    }
	  
	 	  
	  // calculate the objective for the averaged w 
	  // if optimizing LU then this is expensive since 
	  // it runs the optimizaion
	  if( params.report_avg_epoch && (t % params.report_avg_epoch == 0) )
	    {
	      if ( params.avg_epoch && t >= params.avg_epoch)
		{
		  // use the average to calculate objective
		  VectorXd sortedLU_test;
		  if (params.optimizeLU_epoch > 0)
		    {
		      optimizeLU(l_avg, u_avg, projection_avg, y, class_order, sorted_class, wc, nclasses, filtered, C1, C2, params);
		      get_sortedLU(sortedLU_test, l_avg, u_avg, sorted_class);
		    }
		  else
		    {
		      sortedLU_test = sortedLU_avg/(t - params.avg_epoch + 1);
		    }
		  objective_val_avg[obj_idx_avg++] = 
		    calculate_objective_hinge( projection_avg, y, nclasses, 
					       sorted_class, class_order, 
					       w.norm_avg(), sortedLU_test, 
					       filtered, 
					       lambda, C1, C2, params); // save the objective value
		}
	      else
		{
		  if (params.report_epoch > 0 && (t % params.report_epoch==0))
		    {
		      // the objective has just been computed for the current w, use it. 
		      objective_val_avg[obj_idx_avg++] = objective_val[obj_idx - 1];
		    }
		  else
		    {	
		      // since averaging has not started yet, compute the objective for
		      // the current w.
		      // we can use projection_avg because if averaging has not started
		      // this is just the projection using the current w
		      objective_val_avg[obj_idx_avg++] = 
			calculate_objective_hinge( projection_avg, y, nclasses, 
						   sorted_class, class_order, 
						   w.norm(), sortedLU, filtered, 
						   lambda, C1, C2, params); // save the objective value
		    }
		}		    
	      if(PRINT_O)
		{
		  cout << "objective_val_avg[" << t << "]: " << objective_val_avg[obj_idx_avg-1] << " "<< w.norm_avg() << endl;
		}
	    }	  
       
	  
	} // end while t
      
      // define these here just in case I got some of the conditons wrong
      VectorXd projection, projection_avg;
	
      // get l and u if needed 
      // have to do this here because class order might change 
      if ( params.optimizeLU_epoch <= 0 || params.reorder_type == REORDER_RANGE_MIDPOINTS )
	{
	  get_lu(l,u,sortedLU,sorted_class);		  
	}
      
      // optimize LU and compute objective for averaging if it is turned on
      // if t = params.avg_epoch, everything is exactly the same as 
      // just using the current w
      if ( params.avg_epoch && t > params.avg_epoch ) 
	{
	  // get the current l_avg and u_avg if needed
	  if ( params.optimizeLU_epoch <= 0)
	    {
	      get_lu(l_avg,u_avg,sortedLU_avg/(t - params.avg_epoch + 1),sorted_class);
	    }

	  // project all the data on the average w if needed
	  if (params.report_avg_epoch > 0 || params.optimizeLU_epoch > 0)
	    {
	      w.project_avg(projection_avg,x);
	    }
	  // only need to reorder the classes if optimizing LU
	  // or if we are interested in the last obj value
	  // do the reordering based on the averaged w 
	  if (params.reorder_epoch > 0 && (params.optimizeLU_epoch > 0 || params.report_avg_epoch > 0))	
	    {
	      proj_means(means, nc, projection_avg, y);
	      // calculate the new class order
	      rank_classes(sorted_class, class_order, means);
	    }	     	      
	  
	  // optimize the lower and upper bounds (done after class ranking)
	  // since it depends on the ranks 
	  if (params.optimizeLU_epoch > 0)
	    {	     
	      optimizeLU(l_avg,u_avg,projection_avg,y,class_order, sorted_class, wc, nclasses, filtered, C1, C2, params);
	    }
	  
	  // calculate the objective for the averaged w 
	  if( params.report_avg_epoch>0 )
	    {
	      // get the current sortedLU in case bounds or order changed
	      // could test for changes!
	      get_sortedLU(sortedLU_avg, l_avg, u_avg, sorted_class);
	      objective_val_avg[obj_idx_avg++] = 
		calculate_objective_hinge( projection_avg, y, nclasses, 
					   sorted_class, class_order, 
					   w.norm_avg(), sortedLU_avg, 
					   filtered, 
					   lambda, C1, C2, params); // save the objective value
	      if(PRINT_O)
		{
		  cout << "objective_val_avg[" << t << "]: " << objective_val_avg[obj_idx_avg-1] << " "<< w.norm_avg() << endl;
		}
	    }
	}	  
      
	

      // do everything for the current w .
      // it might be wasteful if we are not interested in the current w      
      if (params.report_epoch > 0 || params.optimizeLU_epoch > 0)
	{
	  w.project(projection,x);
	}
      // only need to reorder the classes if optimizing LU
      // or if we are interested in the last obj value
      // do the reordering based on the averaged w 
      if (params.reorder_epoch > 0 && (params.optimizeLU_epoch > 0 || params.report_epoch > 0))	
	{
	  switch (params.reorder_type)
	    {
	    case REORDER_AVG_PROJ_MEANS:
	      // even if reordering is based on the averaged w
	      // do it here based on the w to get the optimal LU and 
	      // the best objective with respect to w
	    case REORDER_PROJ_MEANS:
	      proj_means(means, nc, projection, y);
	      break;
	    case REORDER_RANGE_MIDPOINTS:
	      means = l+u; //no need to divide by 2 since it is only used for ordering
	      break;
	    default:
	      cerr << "Error, reordering " << params.reorder_type << " not implemented" << endl;
	      exit(-1);	      
	    }	 	  
	  // calculate the new class order
	  rank_classes(sorted_class, class_order, means);
	}		  
      
      // optimize the lower and upper bounds (done after class ranking)
      // since it depends on the ranks 
      // if ranking type is REORDER_RANGE_MIDPOINTS, then class ranking depends on this
      // but shoul still be done before since it is less expensive
      // (could also be done after the LU optimization
      // do this for the average class
      if (params.optimizeLU_epoch > 0)
	{	     
	  optimizeLU(l,u,projection,y,class_order, sorted_class, wc, nclasses, filtered, C1, C2, params);
	}
      
      // calculate the objective for the current w
      if( params.report_epoch>0 )
	{
	  // get the current sortedLU in case bounds or order changed
	  get_sortedLU(sortedLU, l, u, sorted_class);	  
	  objective_val[obj_idx++] = 
	    calculate_objective_hinge( projection, y, nclasses, 
				       sorted_class, class_order, 
				       w.norm(), sortedLU, 
				       filtered, 
				       lambda, C1, C2, params); // save the objective value
	  if(PRINT_O)
	    {
	      cout << "objective_val[" << t << "]: " << objective_val[obj_idx-1] << " "<< w.norm() << endl;
	    }
	}
      
      
      VectorXd vect;
      w.toVectorXd(vect);
      weights.col(projection_dim) = vect;
      lower_bounds.col(projection_dim) = l;
      upper_bounds.col(projection_dim) = u;
      if ( params.avg_epoch && t > params.avg_epoch )
	{
	  w.toVectorXd_avg(vect);
	  weights_avg.col(projection_dim) = vect;
	  lower_bounds_avg.col(projection_dim) = l_avg;
	  upper_bounds_avg.col(projection_dim) = u_avg;
	}
      else 
	{
	  weights_avg.col(projection_dim) = vect;
	  lower_bounds_avg.col(projection_dim) = l;
	  upper_bounds_avg.col(projection_dim) = u;
	}
      
      // should we do this in parallel? 
      // the main problem is that the bitset is not thread safe (changes to one bit can affect changes to other bits)
      // should update to use the filter class 
      // things will not work correctly with remove_class_constrains on. We need to update wc, nclass 
      //       and maybe nc
      // check if nclass and nc are used for anything else than weighting examples belonging
      //       to multiple classes
      if (params.remove_constraints && projection_dim < no_projections-1)
	{
	  if (params.avg_epoch && t > params.avg_epoch )
	    {
	      w.project_avg(projection_avg,x); // could eliminate this since it has most likely been calculated above, but we keep it here for now for clarity
	      update_filtered(filtered, projection_avg, l_avg, u_avg, y, params.remove_class_constraints);
	    }
	  else
	    {
	      w.project(projection,x); // could eliminate this since it has most likely been calculated above, but we keep it here for now for clarity
	      update_filtered(filtered, projection, l, u, y, params.remove_class_constraints);
	    }
	  
	  no_filtered = filtered.count();
	  cout << "Filtered " << no_filtered << " out of " << total_constraints << endl;
	  // work on this. This is just a crude approximation.
	  // now every example - class pair introduces nclass(example) constraints
	  // if weighting is done, the number is different
	  // eliminating one example -class pair removes nclass(exmple) potential
	  // if the class not among the classes of the example
	  long int no_remaining = total_constraints - no_filtered;
	  lambda = no_remaining*1.0/(total_constraints*params.C2);
	}
      
      //      C2*=((n-1)*noClasses)*1.0/no_remaining;
      //C1*=((n-1)*noClasses)*1.0/no_remaining;
      
      
        
    } // end for projection_dim
  
  cout << "\n---------------\n" << endl;
  
  objective_val.conservativeResize(obj_idx);
  objective_val_avg.conservativeResize(obj_idx_avg);
  
  
  delete[] sc_locks;
  delete[] idx_locks;
  
  #ifdef PROFILE
  ProfilerStop();
  #endif
};

#endif
