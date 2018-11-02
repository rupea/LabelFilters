/*  Copyright (C) 2017 NEC Laboratories America, Inc. ("NECLA"). All rights reserved.
 *
 * This source code is licensed under the license found in the LICENSE file in
 * the root directory of this source tree. An additional grant of patent rights
 * can be found in the PATENTS file in the same directory.
 */
#ifndef MCFILTER_HH
#define MCFILTER_HH

#include "mcfilter.h"
#include "roaring.hh"
#include "typedefs.h" //DenseM, ActiveSet
#include "mutexlock.h"
#include "profile.h"
#include <iostream>

template< typename EIGENTYPE >
void MCfilter::filter(/*out*/ ActiveSet& active, /*in*/ EIGENTYPE const& x, int np/* = -1*/) const{  

  size_t const nExamples = x.rows();  
  assert(x.cols() == this->d);
  
  np = np>=0?np:nFilters();
  if (np > nFilters())
    {
      std::cerr << "Warning: MCfilter::filter requested " << np 
		<< " projections, but only " << nProj
		<< " are available" << std::endl;
      np = nFilters();
    }

  time_t start, stop;
  time(&start);
  DenseM const projections = (x * weights.leftCols(np));

  active.clear();
  // initialize with all labels active.   
  active.reserve(nExamples);  
  Roaring full; //empty set
  full.flip(0,nClass); // full set
  full.setCopyOnWrite(false);
  for(size_t i=0U; i<nExamples; ++i){
    active.emplace_back(full);
  }
  if (np == 0 ) return;
      
  MCfilter const* fp = this; // to work with omp
  if (_logtime > 0){
  PROFILER_START("log_filtering.profile");
#if MCTHREADS
#pragma omp parallel for shared(np, fp, active) default(none)
#endif
    for(size_t e=0U; e<nExamples; ++e){
      for(size_t p=0U; p<std::min(np,_logtime); ++p){
	Roaring const* dbitset = fp->_filters[p]->filter(projections.coeff(e,p));
	active[e] &= *dbitset;
      }
    }
  }
  PROFILER_STOP;
  
  if (_logtime < np)
    {
      time(&start);
      PROFILER_START("linear_filter.profile");
      // the parallelism here is limited by the nubmer of filters.
      // better results might be achievable with more complex schemes.
      std::vector<MutexType> mutex(nExamples);
#if MCTHREADS
#pragma omp parallel for default(shared)
#endif
      for(size_t p=_logtime; p<np; ++p){
	fp->_filters[p]->filterBatch(projections.col(p), active, mutex);
      }
      PROFILER_STOP;
    }       	 
  time(&stop);
  if (_verbose) std::cout << stop - start << " seconds to apply " << np << " filters" << std::endl;
}

#endif //MCFILTER_HH
