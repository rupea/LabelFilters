#ifndef __MCUPDATE_HH
#define __MCUPDATE_HH

#include "mcsolver.h"
#include "Eigen/Dense"
#include "Eigen/Sparse"
#include "mutexlock.h"                  // MutexType
#include "typedefs.h"
#include "parameter.h"
#include "WeightVector.h"
#include <vector>


class boolmatrix;

namespace mcsolver_detail
{
  using namespace Eigen;
  using namespace std;

  template<typename EigenType> inline
  void calc_sqNorms( EigenType const& x, VectorXd& sqNorms ){
    sqNorms.resize(x.rows());
#if MCTHREADS
#pragma omp parallel for schedule(static,1024)
#endif
    for(size_t i=0U; i<x.rows(); ++i)
      sqNorms.coeffRef(i) = x.row(i).squaredNorm();
  }
  
  /** sample with replacement .. i.e. may contain duplicates */
  inline void get_ordered_sample(vector<int>& sample, int max, int num_samples)
  {
    sample.resize(num_samples);
    for (int i=0;i<num_samples;i++)
      {
	sample[i] = ((int)rand()) % max;
      }
    std::sort(sample.begin(),sample.end());
  }
  

  // ***********************************************
  // calculate the multipliers (for the w gradient update)
  // and the gradients for l and u updates
  // on a subset of classes and instances

  void compute_gradients (VectorXd& multipliers , VectorXd& sortedLU_gradient,
			  const size_t idx_start, const size_t idx_end,
			  const int sc_start, const int sc_end,
			  const VectorXd& proj, const VectorXsz& index,
			  const SparseMb& y, const VectorXi& nclasses,
			  const int maxclasses,
			  const vector<int>& sorted_class,
			  const vector<int>& class_order,
			  const VectorXd& sortedLU,
			  const boolmatrix& filtered,
			  const double C1, const double C2,
			  const param_struct& params );


  // function to calculate the multiplier of the gradient for w for a single example.

  double compute_single_w_gradient_size (const int sc_start, const int sc_end,
					 const double proj, const size_t i,
					 const SparseMb& y, const VectorXi& nclasses,
					 const int maxclasses,
					 const vector<int>& sorted_class,
					 const vector<int>& class_order,
					 const VectorXd& sortedLU,
					 const boolmatrix& filtered,
					 const double C1, const double C2,
					 const param_struct& params );

  // function to update L and U for a single example, given w.

  void update_single_sortedLU( VectorXd& sortedLU,
			       int sc_start, int sc_end,
			       const double proj, const size_t i,
			       const SparseMb& y, const VectorXi& nclasses,
			       int maxclasses,
			       const vector<int>& sorted_class,
			       const vector<int>& class_order,
			       const boolmatrix& filtered,
			       double C1, double C2, const double eta_t,
			       const param_struct& params);


  // function to calculate the multiplier of the gradient for w for a single example.
  // subsampling the negative class constraints
  double compute_single_w_gradient_size_sample ( int sc_start, int sc_end,
						 const vector<int>& sc_sample,
						 const double proj, const size_t i,
						 const SparseMb& y, const VectorXi& nclasses,
						 int maxclasses,
						 const vector<int>& sorted_class,
						 const vector<int>& class_order,
						 const VectorXd& sortedLU,
						 const boolmatrix& filtered,
						 double C1, double C2,
						 const param_struct& params );

  // function to update L and U for a single example, given w.
  // subsampling the negative classes

  void update_single_sortedLU_sample ( VectorXd& sortedLU,
				       int sc_start, int sc_end,
				       const vector<int>& sc_sample,
				       const double proj, const size_t i,
				       const SparseMb& y, const VectorXi& nclasses,
				       int maxclasses,
				       const vector<int>& sorted_class,
				       const vector<int>& class_order,
				       const boolmatrix& filtered,
				       double C1, double C2, const double eta_t,
				       const param_struct& params);



  
  /** compute gradient and update w, L and U using safe updates that do not overshoot.
   * - update w first, making sure we do not overshoot
   * - then update LU using projected gradient
   * - only works for batch sizes of 1*/
  
  template<typename EigenType>
  void update_safe_SGD (WeightVector& w, VectorXd& sortedLU, 
			const EigenType& x, const SparseMb& y, const VectorXd& sqNormsX,
			const double C1, const double C2, const double lambda,
			const unsigned long t, const double eta_t,
			const size_t n, const VectorXi& nclasses, const int maxclasses,
			const std::vector<int>& sorted_class, const std::vector<int>& class_order,
			const boolmatrix& filtered,
			const int sc_chunks, const int sc_chunk_size, const int sc_remaining,
			const param_struct& params)
  {
    using namespace std;
      
    double multiplier = 0;
      
#ifndef NDEBUG
    // batch size should be 1
    assert(params.batch_size == 1);
#endif
      
    size_t i = ((size_t) rand()) % n;
    double const proj = w.project_row(x,i);
    double const iSqNorm = sqNormsX.coeff(i);
      
    vector<int> sample;
    if (params.class_samples)
      {
	get_ordered_sample(sample, y.cols(), params.class_samples);
	sample.push_back(y.cols()); // need the last entry of the sample to be the number of classes
      }
      
      
#if MCTHREADS
    //#pragma omp parallel for default(shared) reduction(+:multiplier) schedule(static,1)
#pragma omp parallel for default(shared) reduction(+:multiplier)
#endif
    for (int sc_chunk = 0; sc_chunk < sc_chunks; sc_chunk++)
      {
	// the first chunks will have an extra iteration
	int const sc_start = sc_chunk*sc_chunk_size + (sc_chunk<sc_remaining?sc_chunk:sc_remaining);
	int const sc_incr = sc_chunk_size + (sc_chunk<sc_remaining);
	multiplier +=
	  ( params.class_samples
	    ? compute_single_w_gradient_size_sample(sc_start, sc_start+sc_incr,
						    sample, proj, i,
						    y, nclasses, maxclasses,
						    sorted_class, class_order,
						    sortedLU, filtered, C1, C2, params)
	    : compute_single_w_gradient_size(sc_start, sc_start+sc_incr,
					     proj, i,
					     y, nclasses, maxclasses,
					     sorted_class, class_order,
					     sortedLU, filtered, C1, C2, params) );
      }

    // make sure we do not overshoot with the update
    // this is expensive, so we might want an option to turn it off
    double new_multiplier, new_proj;
    double eta = eta_t;      
    // original values 2.0, 0.5 could be a bit unstable for some problems (ex. mnist)
    double const eta_bigger = 2.0;
    double const eta_smaller = 1.0 / eta_bigger;
    do
      {
	new_proj = proj - eta*lambda*proj - eta*multiplier*iSqNorm;
	new_multiplier=0;
#if MCTHREADS
	//#pragma omp parallel for  default(shared) reduction(+:new_multiplier) schedule(static,1)
#pragma omp parallel for  default(shared) reduction(+:new_multiplier)
#endif
	for (int sc_chunk = 0; sc_chunk < sc_chunks; sc_chunk++)
	  {
	    // the first chunks will have an extra iteration
	    int const sc_start = sc_chunk*sc_chunk_size + (sc_chunk<sc_remaining?sc_chunk:sc_remaining);
	    int const sc_incr = sc_chunk_size + (sc_chunk<sc_remaining);
	    new_multiplier +=
	      ( params.class_samples
		? compute_single_w_gradient_size_sample(sc_start, sc_start+sc_incr,
							sample,
							new_proj, i,
							y, nclasses, maxclasses,
							sorted_class, class_order,
							sortedLU, filtered, C1, C2, params)
		: compute_single_w_gradient_size(sc_start, sc_start+sc_incr,
						 new_proj, i,
						 y, nclasses, maxclasses,
						 sorted_class, class_order,
						 sortedLU, filtered, C1, C2, params) );
	  }
	eta *= eta_smaller;
	if( eta < 1.e-8 ) break;   
      } while (multiplier*new_multiplier < -1e-5);

    
    if( eta >= 1.e-8){ // if we kept overshooting got too small, do not update w at all. 
      eta *= eta_bigger; // last eta did not overshooot so restore it
      if (params.avg_epoch && t >= params.avg_epoch)      // update current and avg w
	{
	  w.batch_gradient_update_avg(x, i, multiplier, lambda, eta);
	}
      else   // update only current w
	{
	  w.batch_gradient_update    (x, i, multiplier, lambda, eta);
	}
    }else{
      new_proj = proj; // we have not updated w
    }

    // update L and U with w fixed.
    // use new_proj since it is exactly the projection obtained with the new w

#if MCTHREADS
    //#pragma omp parallel for default(shared) schedule(static,1)
#pragma omp parallel for default(shared)
#endif
    for (int sc_chunk = 0; sc_chunk < sc_chunks; sc_chunk++)
      {
	int sc_start = sc_chunk*sc_chunk_size + (sc_chunk<sc_remaining?sc_chunk:sc_remaining);
	int sc_incr = sc_chunk_size + (sc_chunk<sc_remaining);
	if (params.class_samples)
	  {
	    update_single_sortedLU_sample(sortedLU, sc_start, sc_start+sc_incr,
					  sample, new_proj, i,
					  y, nclasses, maxclasses,
					  sorted_class, class_order,
					  filtered, C1, C2, eta_t, params);
	  }
	else
	  {
	    update_single_sortedLU(sortedLU, sc_start, sc_start+sc_incr, new_proj, i,
				   y, nclasses, maxclasses, sorted_class, class_order,
				   filtered, C1, C2, eta_t, params);
	  }
      }
  }
  


  /** function to perform batch SGD update of w, L and U */
  template<typename EigenType>
  void update_minibatch_SGD(WeightVector& w, VectorXd& sortedLU,
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
			    const param_struct& params)
  {
    // use statics to avoid the cost of alocation at each iteration?
    static VectorXd proj(batch_size);
    static VectorXsz index(batch_size);
    static VectorXd multipliers(batch_size);
    VectorXd multipliers_chunk;
    VectorXd sortedLU_gradient_chunk;
    size_t i,idx;
  
    // first compute all the projections so that we can update w directly
    assert( batch_size <= n );
#if MCTHREADS
#pragma omp parallel for schedule(static,32)
#endif
    for (idx = 0; idx < batch_size; idx++)// batch_size will be equal to n for complete GD
      {
	i = (batch_size < n? ((size_t) rand()) % n: idx );
	proj.coeffRef(idx)  = w.project_row(x,i);
	index.coeffRef(idx) = i;
      }
  
    // now we can update w and L,U directly
    multipliers.setZero();
#if MCTHREADS
#pragma omp parallel for  default(shared) shared(idx_locks,sc_locks) private(multipliers_chunk,sortedLU_gradient_chunk) collapse(2)
#endif
    for (int idx_chunk = 0; idx_chunk < idx_chunks; idx_chunk++)
      for (size_t sc_chunk = 0; sc_chunk < sc_chunks; sc_chunk++)
	{
	  // the first chunks will have an extra iteration
	  size_t idx_start = idx_chunk*idx_chunk_size + (idx_chunk<idx_remaining?idx_chunk:idx_remaining);
	  size_t idx_incr = idx_chunk_size + (idx_chunk<idx_remaining);
	  // the first chunks will have an extra iteration
	  size_t sc_start = sc_chunk*sc_chunk_size + (sc_chunk<sc_remaining?sc_chunk:sc_remaining);
	  int sc_incr = sc_chunk_size + (sc_chunk<sc_remaining);
	  compute_gradients(multipliers_chunk, sortedLU_gradient_chunk,
			    idx_start, idx_start+idx_incr,
			    sc_start, sc_start+sc_incr,
			    proj, index, y, nclasses, maxclasses,
			    sorted_class, class_order,
			    sortedLU, filtered,
			    C1, C2, params);
	
#if MCTHREADS
#pragma omp task default(none) shared(sc_chunk, idx_chunk, multipliers, sc_start, idx_start, sc_incr, idx_incr, sortedLU, sortedLU_gradient_chunk, multipliers_chunk, sc_locks,  idx_locks)
#endif
	  {
#if MCTHREADS
#pragma omp task default(none) shared(idx_chunk, multipliers, multipliers_chunk, idx_start, idx_incr, idx_locks)
#endif                
	    {
#if MCTHREADS
	      idx_locks[idx_chunk].YieldLock();
#endif
	      multipliers.segment(idx_start, idx_incr) += multipliers_chunk;
#if MCTHREADS
	      idx_locks[idx_chunk].Unlock();
#endif
	    }
#if MCTHREADS
	    sc_locks[sc_chunk].YieldLock();
#endif
	    // update the lower and upper bounds
	    // divide by batch_size here because the gradients have
	    // not been averaged
	    sortedLU.segment(2*sc_start, 2*sc_incr) += sortedLU_gradient_chunk * (eta_t / batch_size);
#if MCTHREADS
	    sc_locks[sc_chunk].Unlock();
#pragma omp taskwait
#endif
	  }
#if MCTHREADS
#pragma omp taskwait
#endif
	}
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
  }
  
  struct MCupdate{
    //put this in a struct to give access to priate fields in MCPermState    
    template<typename EigenType> 
    static void update( WeightVector& w,
			MCpermState & luPerm,      // sortlu and sortlu_avg are input and output
			const double  eta_t, 
			const EigenType& x, const SparseMb& y, const VectorXd& xSqNorms,
			const double C1, const double C2, const double lambda,
			const unsigned long t, const size_t nTrain,
			const VectorXi& nclasses, const int maxclasses,
			const boolmatrix& filtered,
			MCupdateChunking const& updateSettings,
			param_struct const& params)
    {
      // make sortlu* variables valid (if possible, and not already valid)
      luPerm.mkok_sortlu();
      luPerm.mkok_sortlu_avg();
      assert( luPerm.ok_sortlu );
      //assert( luPerm.ok_sortlu_avg ); // sortlu_avg may be undefined (until t>=epoch_avg)
      if (params.optimizeLU_epoch <= 0 && params.avg_epoch > 0 && t    >    params.avg_epoch){
	assert( luPerm.ok_sortlu_avg );
      }
      VectorXd& sortedLU                          = luPerm.sortlu;
      VectorXd& sortedLU_acc                      = luPerm.sortlu_acc;
      std::vector<int> const& sorted_class        = luPerm.perm;
      std::vector<int> const& class_order         = luPerm.rev;
      
      size_t const& batch_size    = updateSettings.batch_size;
      // the following bunch should disappear soon
      int sc_chunks        = updateSettings.sc_chunks;
      int sc_chunk_size    = updateSettings.sc_chunk_size;
      int sc_remaining     = updateSettings.sc_remaining;
      
      int idx_chunks       = updateSettings.idx_chunks;
      int idx_chunk_size   = updateSettings.idx_chunk_size;
      int idx_remaining    = updateSettings.idx_remaining;
      MutexType* idx_locks       = updateSettings.idx_locks;
      MutexType* sc_locks        = updateSettings.sc_locks;
      
      // After some point 'update' BEGINS TO ACCUMULATE sortedLU into sortedLU
      assert( luPerm.ok_sortlu_avg == true ); // accumulator begins at all zeros, so true
      if (params.update_type == SAFE_SGD)
	{
	  mcsolver_detail::update_safe_SGD(w, sortedLU,
					   x, y, xSqNorms,
					   C1, C2, lambda, t, eta_t, nTrain, // nTrain is just x.rows()
					   nclasses, maxclasses, sorted_class, class_order, filtered,
					   sc_chunks, sc_chunk_size, sc_remaining,
					   params);
	}
      else if (params.update_type == MINIBATCH_SGD)
	{
	  mcsolver_detail::update_minibatch_SGD(w, sortedLU,
						x, y, C1, C2, lambda, t, eta_t, nTrain, batch_size,
						nclasses, maxclasses, sorted_class, class_order, filtered,
						sc_chunks, sc_chunk_size, sc_remaining,
						idx_chunks, idx_chunk_size, idx_remaining,
						idx_locks, sc_locks,
						params);
	}
      else
	{
	  throw runtime_error("Unrecognized update type");
	}
      
      
      // After some point 'update' BEGINS TO ACCUMULATE sortedLU into sortedLU_acc
      // let's move the accumulation here since it would avoid code duplication and simplify the code 
      
      if (params.optimizeLU_epoch <= 0 && params.avg_epoch > 0 && t >= params.avg_epoch)
	{
	  // move this thing inside MCPermState?
	  
	  // if we optimize the LU, we do not need to
	  // keep track of the averaged lower and upper bounds
	  // We optimize the bounds at the end based on the
	  // average w  
	  // probably the parallelization is overkill. 
	  
	  // need to do something special when sampling classes to avoid the O(noClasses) complexity.
	  // for now we leave it like this since we almost always we optimize LU at the end
	  
#if MCTHREADS
	  //#pragma omp parallel for default(shared) schedule(static,1)
#pragma omp parallel for default(shared)
	  for (int sc_chunk = 0; sc_chunk < sc_chunks; sc_chunk++)
	    {
	      int sc_start = sc_chunk*sc_chunk_size + (sc_chunk<sc_remaining?sc_chunk:sc_remaining);
	      int sc_incr = sc_chunk_size + (sc_chunk<sc_remaining);	  
	      sortedLU_acc.segment(2*sc_start, 2*sc_incr) += sortedLU.segment(2*sc_start, 2*sc_incr);
	    }
#else
	  sortedLU_acc += sortedLU;
#endif
	  ++luPerm.nAccSortlu;
	}
      
      luPerm.chg_sortlu();        // sortlu change ==> ok_lu now false      
    }
  };
}

	  
#endif //__MCUPDATE_HH
