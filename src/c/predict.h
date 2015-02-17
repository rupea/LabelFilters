#ifndef __PREDICT_H
#define __PREDICT_H

#include "Eigen/Dense"
#include "Eigen/Sparse"
#include <boost/numeric/conversion/bounds.hpp>
#include <boost/limits.hpp>
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
PredictionSet* predict ( const Eigentype& x, const DenseColMf& w, const ActiveDataSet* active, size_t& nact, bool verbose = false, predtype keep_thresh = boost::numeric::bounds<predtype>::lowest(), size_t keep_size = boost::numeric::bounds<size_t>::highest(), const size_t start_class=0)
{
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
#pragma omp parallel for default(shared)
      for (i = 0; i < n; i++)
	{	  
	  DenseM outs;
	  outs = x.row(i)*(w.cast<double>());	  
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
      assert(active->length() == n);
      if(n>0)
	{
	  assert((*active)[0]->size() == noClasses);
	}
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


template <typename Eigentype>
ActiveDataSet* getactive(vector<size_t>& no_active, const Eigentype& x, const DenseColM& wmat, const DenseColM& lmat, const DenseColM& umat, const string projname="", bool verbose = false)
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

  no_active.clear();
  for (int i = 0; i < projections.cols(); i++)
    {
      VectorXd proj = projections.col(i);
      VectorXd l = lmat.col(i);
      VectorXd u = umat.col(i);
      if (verbose) 
	{
	  cout << "Init filter" << endl;
	}
      Filter f(l,u);      
      if (verbose)
	{
	  cout << "Update filter, projection " << i << endl; 
	}
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
      no_active.push_back(count);
    }
  #ifdef PROFILE
  ProfilerStop();
  #endif  
  return active;
}

#endif
