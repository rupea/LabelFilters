#ifndef __FIND_W_H
#define __FIND_W_H

#include <omp.h>
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
				    bool any_filtered,
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

void get_sortedLU(VectorXd& sortedLU, const VectorXd& l, const VectorXd& u, const vector<int>& sorted_class);

// *******************************
// Get the number of exampls in each class

void init_nc(VectorXi& nc, VectorXi& nclasses, const SparseMb& y);

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
      l(k)=std::numeric_limits<double>::max();
      u(k)=std::numeric_limits<double>::min();	      
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
			DenseM& weights_avg, DenseM& lower_bounds_avg,
			DenseM& upper_bounds_avg,
			VectorXd& objective_val,
			EigenType& x, const SparseMb& y,
			bool resumed, const param_struct& params)

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
  VectorXd projection;
  VectorXd l(noClasses),u(noClasses);
  VectorXd sortedLU(2*noClasses); // holds l and u interleaved in the curent class sorting order (i.e. l,u,l,u,l,u)
  VectorXd sortedLU_gradient(2*noClasses); // used to improve cache performance
  VectorXd sortedLU_gradient_chunk;
  VectorXd l_avg(noClasses),u_avg(noClasses); // the lower and upper bounds for the averaged gradient
  VectorXd sortedLU_avg(2*noClasses); // holds l_avg and u_avg interleaved in the curent class sorting order (i.e. l_avg,u_avg,l_avg,u_avg,l_avg,u_avg)
  VectorXd means(noClasses); // used for initialization of the class order vector;
  VectorXi nc(noClasses); // the number of examples in each class 
  VectorXi nclasses(n); // the number of examples in each class 
  int maxclasses; // the maximum number of classes an example might have
  double eta_t, tmp, sj;
  int cp;// current class and the other classes
  size_t obj_idx = 0;
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
  
  weights.conservativeResize(d, no_projections);
  weights_avg.conservativeResize(d, no_projections);
  lower_bounds.conservativeResize(noClasses, no_projections);
  upper_bounds.conservativeResize(noClasses, no_projections);
  lower_bounds_avg.conservativeResize(noClasses, no_projections);
  upper_bounds_avg.conservativeResize(noClasses, no_projections);

  if (params.report_epoch > 0)
    {
      objective_val.resize(1000 + (no_projections * params.max_iter / params.report_epoch));
    }
  
  init_nc(nc, nclasses, y);
  
  maxclasses = nclasses.maxCoeff();
  //keep track of which classes have been elimninated for a particular example
  boolmatrix filtered(n,noClasses);
  VectorXd difference(d);
  unsigned long total_constraints = n*noClasses - (1-params.remove_class_constraints)*nc.sum();
  size_t no_filtered=0;
  
  for(int projection_dim=0; projection_dim < no_projections; projection_dim++)
    {
      
      if ( projection_dim == -1 )
	{
	  w = WeightVector(weights.col(projection_dim));
	}
      else
	{
	  int c1 = ((int) rand()) % noClasses;
	  int c2 = ((int) rand()) % noClasses;
	  if (c1 == c2)
	    {
	      c2=(c1+1)%noClasses;
	    }
	  difference_means(difference,x,y,nc,c1,c2);
	  w = WeightVector(difference*10/difference.norm());  // get a better value than 10 .. somethign that would match the margins
	}
      
      // w.setRandom(); // initialize to a random value
      if (!resumed)
	{
	  //initialize the class_order vector by sorting the means of the projections of each class. Use l to store the means.
	  init_lu(l,u,means,nc,w,x,y);
	  rank_classes(sorted_class, class_order, means);
	  get_sortedLU(sortedLU, l, u, sorted_class);
	}
      else 
	{	  
	  // should not use this  this. The w is not initialized 
	  l = lower_bounds.col(projection_dim);
	  u = upper_bounds.col(projection_dim);
	  if (params.rank_by_mean)
	    {
	      w.project(projection,x);
	      proj_means(means, nc, projection, y);
	    }
	  else
	    {
	      means = l+u;
	    }
	  rank_classes(sorted_class, class_order, means);
	  get_sortedLU(sortedLU, l, u, sorted_class);
	}
      sortedLU_avg.setZero();

      //      order_changed = 1;

      print_report<EigenType>(projection_dim,batch_size, noClasses,C1,C2,lambda,w.size(),x);

      /* // staring optimization */
      /* for (int iter = 0; iter < params.max_reorder && order_changed == 1; iter++) */
      /* 	{ */

      /* 	  // init the optimization specific parameters */
      /* 	  std::copy(class_order.begin(),class_order.end(), prev_class_order.begin()); */

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
	      w.batch_gradient_update_avg(x, index, multipliers, lambda, eta_t);
	    }
	  else
	    {
	      w.batch_gradient_update(x, index, multipliers, lambda, eta_t);
	    }
	  /* w.scale(1.0-lambda*eta_t); */
	  /* for (idx = 0; idx < batch_size; idx++) */
	  /* 	{ */
	  /* 	  if (multipliers.coeff(idx) != 0) */
	  /* 	    { */
	  /* 	      w.gradient_update(x,index.coeff(idx),multipliers.coeff(idx) * (eta_t / batch_size)); */
	  /* 	    } */
	  /* 	} */
	  
	  // update the lower and upper bounds
	  // divide by batch_size here because the gradients have 
	  // not been averaged
	  // should be done above
	  sortedLU += sortedLU_gradient * (eta_t / batch_size); 
	  // update the average version
	  // shoudl do in parallel (maybe Eigen already does it?)
	  // especially for small batch sizes. 
	  if (params.avg_epoch && t >= params.avg_epoch)
	    {
	      // do not divide by t-params.avg_epoch + 1 here 
	      // do it when using sortedLU_avg
	      // it might become too big!, but through division it might
	      // become too small 
	      sortedLU_avg += sortedLU;
	    }
	 	  
	  // calculate the objective
	  if( params.report_epoch && (t % params.report_epoch == 0) )
	    {
	      VectorXd projection;
	      if ( params.avg_epoch && t >= params.avg_epoch)
		{
		  // use the average to calculate objective
		  w.project_avg(projection,x);		      
		  objective_val[obj_idx++] = 
		    calculate_objective_hinge( projection, y, nclasses, 
					       sorted_class, class_order, 
					       w.norm_avg(), sortedLU_avg/(t-params.avg_epoch+1), 
					       filtered, 
					       lambda, C1, C2, params); // save the objective value
		}
	      else
		{
		  // use the current w to calculate objective
		  w.project(projection,x);		      
		  objective_val[obj_idx++] = 
		    calculate_objective_hinge( projection, y, nclasses, 
					       sorted_class, class_order, 
					       w.norm(), sortedLU, filtered, 
					       lambda, C1, C2, params); // save the objective value
		}
	      if(PRINT_O)
		{
		  cout << "objective_val[" << t << "]: " << objective_val[obj_idx-1] << " "<< w.norm() << endl;
		}
	    }
	  
	  // reorder the classes
	  if (params.reorder_epoch && (t % params.reorder_epoch == 0))
	    {
	      // do this in a function?
	      // get the current l and u in the original class order
	      get_lu(l,u,sortedLU,sorted_class);		  
	      if ( params.avg_epoch &&  t > params.avg_epoch)
		{
		  get_lu(l_avg,u_avg,sortedLU_avg,sorted_class);		  
		  if (params.rank_by_mean)
		    {		      
		      w.project_avg(projection,x);
		      proj_means(means, nc, projection, y);
		    }
		  else 
		    {
		      means = l_avg+u_avg; //no need to divide by 2 since it is only used for ordering
		    }
		}
	      else
		{
		  if (params.rank_by_mean)
		    {		      
		      w.project(projection,x); 
		      proj_means(means, nc, projection, y);
		    }
		  else 
		    {
		      means = l+u; //no need to divide by 2 since it is only used for ordering
		    }
		}
	      // calculate the new class order
	      rank_classes(sorted_class, class_order, means);
	      // sort the l and u in the order of the classes
	      get_sortedLU(sortedLU, l, u, sorted_class);
	      if ( params.avg_epoch &&  t >= params.avg_epoch)
		{
		  get_sortedLU(sortedLU_avg, l_avg, u_avg, sorted_class);
		}		  
	    }	     	      
	  	  
	} // end while t
      
      if (params.report_epoch > 0)
	{
	  // save the objective value of the last iteration
	  VectorXd projection;
	  if ( params.avg_epoch &&  t >= params.avg_epoch)
	    {
	      w.project_avg(projection,x);
	      objective_val[obj_idx++] = 
		calculate_objective_hinge(projection, y, nclasses,
					  sorted_class, class_order,
					  w.norm_avg(), sortedLU_avg/(t-params.avg_epoch+1),
					  filtered,
					  lambda, C1, C2, params);
	    }
	  else
	    {
	      w.project(projection,x);
	      objective_val[obj_idx++] = 
		calculate_objective_hinge(projection, y, nclasses,
					  sorted_class, class_order,
					  w.norm(), sortedLU,
					  filtered,
					  lambda, C1, C2, params);
	    }		  
	}
      // Let's check if s changed
      // check if the orders are the same
      //      order_changed = 0;
      // check if the class_order are still the same
      
      // get the current l and u in the original class order
      get_lu(l,u,sortedLU,sorted_class);		  
      if ( params.avg_epoch &&  t >= params.avg_epoch)
	{
	  //divide sortedLU_avg to get the average
	  get_lu(l_avg, u_avg,sortedLU_avg/(t-params.avg_epoch+1),sorted_class);
	}
      /* 	  // order classes using  the average weights */
      /* 	  if (params.rank_by_mean) */
      /* 	    {		       */
      /* 	      w.project_avg(projection,x); */
      /* 	      proj_means(means, nc, projection, y); */
      /* 	    } */
      /* 	  else  */
      /* 	    { */
      /* 	      means = l_avg+u_avg; //no need to divide by 2 since it is only used for ordering */
      /* 	    } */
      /* 	} */
      /* else */
      /* 	{ */
      /* 	  // rank classes using the current weights */
      /* 	  if (params.rank_by_mean) */
      /* 	    {		       */
      /* 	      w.project(projection,x);  */
      /* 	      proj_means(means, nc, projection, y); */
      /* 	    } */
      /* 	  else  */
      /* 	    { */
      /* 	      means = l+u; //no need to divide by 2 since it is only used for ordering */
      /* 	    } */
      /* 	} */
      /* // calculate the new class order */
      /* rank_classes(sorted_class, class_order, means); */
      /* // sort the l and u in the order of the classes */
      /* get_sortedLU(sortedLU, l, u, sorted_class); */
      /* if ( params.avg_epoch &&  t >= params.avg_epoch) */
      /* 	{ */
      /* 	  get_sortedLU(sortedLU_avg, l_avg, u_avg, sorted_class); */
      /* 	}		   */
      
    /*   // check that the ranks are the same  */
    /*   for(int c = 0; c < noClasses; c++) */
    /* 	{ */
    /* 	  if (class_order[c] != prev_class_order[c]) */
    /* 	    { */
    /* 	      order_changed = 1; */
    /* 	      break; */
    /* 	    } */
    /* 	} */
      
    /*   if(PRINT_T==1) */
    /* 	{ */
    /* 	  double obj = obj_idx >= 1 ? objective_val[obj_idx-1] : 0; */
    /* 	  cout << "\nt: " << t << ", obj:" << obj */
    /* 	       << ", l:" << l.transpose() << ", u:" << u.transpose() */
    /* 	       << ", cur_norm: " << w.norm() << endl; */
    /* 	} // end if print */
      
    /*   cout << "\r>> " << iter+1 << ": Done in " << t */
    /* 	   << " iterations ... with w.norm(): " << w.norm() << endl; */
    
    /* } // end for iter */
      
      VectorXd vect;
      w.toVectorXd(vect);
      weights.col(projection_dim) = vect;
      lower_bounds.col(projection_dim) = l;
      upper_bounds.col(projection_dim) = u;
      if ( params.avg_epoch && t >= params.avg_epoch )
	{
	  w.toVectorXd_avg(vect);
	  weights_avg.col(projection_dim) = vect;
	  lower_bounds_avg.col(projection_dim) = l_avg;
	  upper_bounds_avg.col(projection_dim) = u_avg;
	}
      else 
	{
	  w.toVectorXd(vect);
	  weights_avg.col(projection_dim) = vect;
	  lower_bounds_avg.col(projection_dim) = l;
	  upper_bounds_avg.col(projection_dim) = u;
	}
      
      // should we do this in parallel? 
      // the main problem is that the bitset is not thread safe (changes to one bit can affect changes to other bits)
      if (params.remove_constraints && projection_dim < no_projections-1)
	{
	  if (params.avg_epoch && t >= params.avg_epoch )
	    {
	      w.project_avg(projection,x);
	      update_filtered(filtered, projection, l_avg, u_avg, y, params.remove_class_constraints);
	    }
	  else
	    {
	      w.project(projection,x);
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
  
  
  delete[] sc_locks;
  delete[] idx_locks;
  
  #ifdef PROFILE
  ProfilerStop();
  #endif
}

#endif
