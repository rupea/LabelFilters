#ifndef __PREDICT_H
#define __PREDICT_H

#include "Eigen/Dense"
#include "Eigen/Sparse"
#include <vector>
#include <boost/dynamic_bitset.hpp>
#include "filter.h"
#include "typedefs.h"
#include "PredictionSet.h"

#ifdef PROFILE
#include <gperftools/profiler.h>
#endif


using namespace std;
  
template <typename Eigentype>
PredictionSet* predict ( const Eigentype& x, const DenseColMf& w, const ActiveDataSet* active, size_t& nact, predtype keep_thresh = std::numeric_limits<predtype>::min(), size_t keep_size = std::numeric_limits<size_t>::max())
{
  size_t n = x.rows();
  size_t noClasses = w.cols();
  nact = 0;
  PredictionSet* predictions = new PredictionSet(n); // preallocates n elements
  cout << "Predicting " << n << "    " << noClasses << endl;

  bool prune = !(keep_thresh == std::numeric_limits<predtype>::min() && keep_size == std::numeric_limits<size_t>::max());
  
  size_t i;
  if (active == NULL)
    {
      #ifdef PROFILE
      ProfilerStart("full_predict.profile");
      #endif  
#pragma omp parallel for default(shared)
      for (i = 0; i < n; i++)
	{	  
	  DenseM outs;
	  outs = x.row(i)*(w.cast<double>());	  
	  // should preallocate to be more efficient
	  PredVec* pv = predictions->NewPredVecAt(i,noClasses); 
	  for (size_t c = 0; c < noClasses; c++)
	    {
	      pv->add_pred(static_cast<predtype> (outs.coeff(c)),c);
	    }
	  if (prune)
	    {
	      pv->prune(keep_size, keep_thresh);
	    }
	}
      #ifdef PROFILE
      ProfilerStop();
      #endif  
    }
  else
    {
      #ifdef PROFILE
      ProfilerStart("projected_predict.profile");
      #endif  
      size_t totalactive = 0;
#pragma omp parallel for default(shared) reduction(+:totalactive) 
      for (i = 0; i < n; i++)      
	{
	  dynamic_bitset<>* act = (*active)[i];
	  size_t nactive = act->count();
	  totalactive += nactive;
	  PredVec* pv = predictions->NewPredVecAt(i,nactive);
	  size_t c = act->find_first();	  
	  while (c < noClasses)
	    {
	      // could eliminate the multiplication for only one active class
	      predtype out = static_cast<predtype> ((x.row(i)*(w.col(c).cast<double>()))(0,0));
	      pv->add_pred(out,c);
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


template <typename Eigentype>
ActiveDataSet* getactive(const Eigentype& x, const DenseColM& wmat, const DenseColM& lmat, const DenseColM& umat)
{
  #ifdef PROFILE
  ProfilerStart("projected_getactive.profile");
  #endif  
  DenseM projections = x*wmat;
  size_t n = projections.rows();
  size_t noClasses = lmat.rows();
  ActiveDataSet* active = new ActiveDataSet(n);
  for (ActiveDataSet::iterator it = active->begin(); it != active->end(); ++it)
    {
      *it = new dynamic_bitset<>(noClasses);
      (*it) -> set();
    }

  for (int i = 0; i < projections.cols(); i++)
    {
      cout << i << endl;
      VectorXd proj = projections.col(i);
      VectorXd l = lmat.col(i);
      VectorXd u = umat.col(i);
      cout << "init filter" << endl;
      Filter f(l,u);      
      cout << i << "  update filter" << endl; 
      size_t j;
      ActiveDataSet::iterator it;
      size_t count = 0;
#pragma omp parallel for default(shared) reduction(+:count)
      for (j=0; j < n; j++)
	{
	  dynamic_bitset<>* act = (*active)[j];
	  (*act) &= *(f.filter(proj.coeff(j)));
	  count += act -> count();
	}
      cout << i << " active  " << count << " out of " << n*noClasses << " (" << count*1.0/(n*noClasses) << ")" << endl; 
    }
  #ifdef PROFILE
  ProfilerStop();
  #endif  
  return active;
}

#endif
