#ifndef PREDICT_HH
#define PREDICT_HH

#include "predict.h"
#include "utils.h"
#include "constants.h" // MCTHREADS

#include <boost/numeric/conversion/bounds.hpp>
#include <boost/limits.hpp>
#include <vector>
#include <boost/dynamic_bitset.hpp>

#ifdef PROFILE
#include <gperftools/profiler.h>
#endif

template <typename Eigentype> inline
ActiveDataSet* getactive( VectorXsz& no_active, const Eigentype& x,
                          const DenseColM& wmat, const DenseColM& lmat, const DenseColM& umat,
                          bool verbose = false)
{
  #ifdef PROFILE
  ProfilerStart("projected_getactive.profile");
  #endif
  DenseM const projections = x*wmat;
  ActiveDataSet* active;

  // non-templated from this point forward (no longer need 'x')
  active = projectionsToActiveSet( no_active, projections, lmat, umat, verbose );

  #ifdef PROFILE
  ProfilerStop();
  #endif
  return active;
}


template <typename Eigentype> inline
PredictionSet* predict ( Eigentype const& x, DenseColMf const& w,
                         ActiveDataSet const* active, size_t& nact,
                         bool verbose             /*= false*/,
                         predtype keep_thresh     /*= boost::numeric::bounds<predtype>::lowest()*/,
                         size_t keep_size         /*= boost::numeric::bounds<size_t>::highest()*/,
                         size_t const start_class /*= 0*/)
{
  using namespace std;
  size_t n = x.rows();
  size_t noClasses = w.cols();
  nact = 0;
  PredictionSet* predictions = new PredictionSet(n); // preallocates n elements
  if (verbose)
    {
      cout << "Predicting " << n << "    " << noClasses << endl;
    }
  bool prune = !(keep_thresh == boost::numeric::bounds<predtype>::lowest() && keep_size == boost::numeric::bounds<size_t>::highest());

  size_t i;
  if (active == NULL)
    {
      #ifdef PROFILE
      ProfilerStart("full_predict.profile");
      #endif
      Eigen::RowVectorXd outs(noClasses);
#pragma omp parallel for firstprivate(outs) default(shared)
      for (i = 0; i < n; i++)
	{	  	
	  //outs = x.row(i)*(w.cast<double>());	
	  DotProductInnerVector(outs,w,x,i);
	  // should preallocate to be more efficient
	  PredVec* pv = predictions->NewPredVecAt(i,noClasses);
	  for (size_t c = 0; c < noClasses; c++)
	    {
	      pv->add_pred(static_cast<predtype> (outs.coeff(c)),c+start_class);
	    }
	  if (prune)
	    {
	      pv->prune(keep_size, keep_thresh);
	    }
	}
      nact = n*noClasses;
      #ifdef PROFILE
      ProfilerStop();
      #endif
    }
  else
    {
      #ifdef PROFILE
      ProfilerStart("projected_predict.profile");
      #endif
      assert(active->size() == n);
      if(n>0)
	{
	  assert((*active)[0]->size() == noClasses);
	}
      size_t totalactive = 0;
#pragma omp parallel for default(shared) reduction(+:totalactive)
      for (i = 0; i < n; i++)
	{
          boost::dynamic_bitset<>* act = (*active)[i];
	  size_t nactive = act->count();
	  totalactive += nactive;
	  PredVec* pv = predictions->NewPredVecAt(i,nactive);
	  size_t c = act->find_first();	
	  while (c < noClasses)
	    {
	      // could eliminate the multiplication for only one active class
	      predtype out = static_cast<predtype>(DotProductInnerVector(w.col(c),x,i));
	      pv->add_pred(out,c+start_class);
	      c = act->find_next(c);
	    }
	  if (prune)
	    {
	      pv->prune(keep_size, keep_thresh);
	    }	
	}
      nact = totalactive;

      #ifdef PROFILE
      ProfilerStop();
      #endif
    }
  return predictions;
}


/** Beware: \c w here is a linear xform that gets pre-applied to x. */
template <typename Eigentype> inline
void predict( PredictionSet* predictions,
              Eigentype const& x, DenseColMf const& w,
              ActiveDataSet const* active, size_t& nact,
              bool verbose             /*= false*/,
              predtype keep_thresh     /*= boost::numeric::bounds<predtype>::lowest()*/,
              size_t keep_size         /*= boost::numeric::bounds<size_t>::highest()*/,
              size_t const start_class /*= 0*/)
{
  size_t n = x.rows();
  size_t noClasses = w.cols();
  nact = 0;
  assert(predictions->size() == n);
  if (verbose)
    {
      cout << "Predicting " << n << "    " << noClasses << endl;
    }
  bool prune = !(keep_thresh == boost::numeric::bounds<predtype>::lowest() && keep_size == boost::numeric::bounds<size_t>::highest());

  size_t i;
  if (active == NULL)
    {
      #ifdef PROFILE
      ProfilerStart("full_predict.profile");
      #endif

      Eigen::RowVectorXd outs(noClasses);
#pragma omp parallel for firstprivate(outs) default(shared)
      for (i = 0; i < n; i++)
	{	  	
	  //outs = x.row(i)*(w.cast<double>());	
	  DotProductInnerVector(outs,w,x,i);
	  // should preallocate to be more efficient	
	  PredVec* pv;
	  if (start_class == 0) // assumes chunk 0 is always the first
	    {
	      pv = predictions->NewPredVecAt(i,noClasses);
	    }
	  else
	    {
	      pv = predictions->GetPredVec(i);
	      pv->reserve_extra(noClasses);
	    }
	  for (size_t c = 0; c < noClasses; c++)
	    {
	      pv->add_pred(static_cast<predtype> (outs.coeff(c)),c+start_class);
	    }
	  if (prune)
	    {
	      pv->prune(keep_size, keep_thresh);
	    }
	}
      nact = n*noClasses;
      #ifdef PROFILE
      ProfilerStop();
      #endif
    }
  else
    {
      #ifdef PROFILE
      ProfilerStart("projected_predict.profile");
      #endif
      assert(active->size() == n);
      if(n>0)
	{
	  assert((*active)[0]->size() == noClasses);
	}
      size_t totalactive = 0;
#pragma omp parallel for default(shared) reduction(+:totalactive)
      for (i = 0; i < n; i++)
	{
          boost::dynamic_bitset<>* act = (*active)[i];
	  size_t nactive = act->count();
	  totalactive += nactive;	
	  PredVec* pv;
	  if (start_class == 0) // assumes chunk 0 is always the first
	    {
	      pv = predictions->NewPredVecAt(i,nactive);
	    }
	  else
	    {
	      pv = predictions->GetPredVec(i);
	      pv->reserve_extra(nactive);
	    }
	  size_t c = act->find_first();	
	  while (c < noClasses)
	    {
	      // predtype out = static_cast<predtype>(DotProductInnerVector(w.col(c),x,i));
	      predtype out = static_cast<predtype>((x.row(i)*w.col(c).cast<double>())(0,0));
	      pv->add_pred(out,c+start_class);
	      c = act->find_next(c);
	    }
	  if (prune)
	    {
	      pv->prune(keep_size, keep_thresh);
	    }	
	}
      nact = totalactive;

      #ifdef PROFILE
      ProfilerStop();
      #endif
    }
}

#endif // PREDICT_HH
