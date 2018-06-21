#ifndef __LINEARMODEL_DETAIL_HH
#define __LINEARMODEL_DETAIL_HH

#include "linearModel_detail.h"
#include "PredictionSet.h"
#include "roaring.hh"
#include "utils.h"
#include "typedefs.h"
#include <cstddef>

#include <ctime>

#ifdef PROFILE
#include <gperftools/profiler.h>
#endif

namespace linearmodel_detail{
  using namespace std;
  
  /** predict x*w.  \c w here is a linear xform that gets pre-applied to x. */
  template <typename Eigentype, typename ovaType>
  size_t predict( PredictionSet& predictions,  // output
		  Eigentype const& x, ovaType const& w, Eigen::RowVectorXd const& intercept,
		  ActiveSet const* feasible,
		  bool verbose         /*= false*/,
		  predtype keep_thresh /*= boost::numeric::bounds<predtype>::lowest()*/,
		  size_t keep_size     /*= boost::numeric::bounds<size_t>::highest()*/)
  {
    
    assert (x.cols()==w.rows());
    size_t n = x.rows();
    size_t noClasses = w.cols();
    size_t npreds = 0;
    
    if (verbose)
      {
	cout << "Predicting " << n << "    " << noClasses << endl;
      }
    
    predictions.clear();
    predictions.resize(n);
    predictions.shrink_to_fit();  // free memory in case there was a larger prediction set allocated. 
    
    predictions.set_prune_params(keep_size, keep_thresh);
    
    size_t i;
    if (feasible == NULL)
      {
#ifdef PROFILE
	ProfilerStart("full_predict.profile");
#endif
	Eigen::RowVectorXd outs(noClasses);
#if MCTHREADS
#pragma omp parallel for firstprivate(outs) default(shared)
#endif
	for (i = 0; i < n; i++)
	  {
	    DotProductInnerVector(outs,x,i,w);
	    if (intercept.size())
	      {
		outs += intercept;
	      }
	    PredVec& pv = predictions[i];
	    for (size_t c = 0; c < noClasses; c++)
	      {
		pv.add_and_prune(static_cast<predtype> (outs.coeff(c)),c);
	      }
	  }	 
	npreds = n*noClasses;
#ifdef PROFILE
	ProfilerStop();
#endif
      }
    else
      {
#ifdef PROFILE
	ProfilerStart("projected_predict.profile");
#endif
	if(feasible->size() != n)
	  {
	    throw std::runtime_error("Dimensions of feasible set and data  do not agree");
	  }
	npreds = 0;
#if MCTHREADS
#pragma omp parallel default(shared) reduction(+:npreds)
#endif
	{
	  // CRoaring is fatster if using toArray than using iterators. 
	  // allocate here to avoid multiple alocations. Will hold the active classes
	  uint32_t* act = new uint32_t[noClasses]; 
#if MCTHREADS
#pragma omp for
#endif
	  for (i = 0; i < n; i++)
	    {
	      feasible->at(i).toUint32Array(act);
	      size_t nactive = feasible->at(i).cardinality();	  
	      npreds += nactive;
	      PredVec& pv = predictions[i];
	      for (uint32_t* cptr = act; cptr != act + nactive; cptr++)
		{
		  predtype out = static_cast<predtype>(DotProductInnerVector(x,i,w,*cptr) + intercept.coeff(*cptr)) ;
		  pv.add_and_prune(out,*cptr);
		}
	    }
	  delete[] act;
	}
#ifdef PROFILE
	ProfilerStop();
#endif
      }
    return npreds;
  }

}//linearmodel_detail::
#endif //__LINEARMODEL_DETAIL_HH
  
