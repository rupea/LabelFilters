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
  DenseM const projections = (x * weights.leftCols(np));



  // initialize with all labels active.   
  active.clear();
  active.reserve(nExamples);  
  Roaring full; //empty set
  full.flip(0,nClass); // full set
  full.setCopyOnWrite(false);
  for(size_t i=0U; i<nExamples; ++i){
    active.emplace_back(full);
  }

  assert( active.size() == nExamples );
  
  // TODO if ! nProj >> nExamples, provide a faster impl ???
  MCfilter const* fp = this; // to work with omp
#if MCTHREADS
#pragma omp parallel for shared(fp, active)
#endif
  for(size_t e=0U; e<nExamples; ++e){
    for(size_t p=0U; p<np; ++p){
      Roaring const* dbitset = fp->_filters[p]->filter(projections.coeff(e,p));
      active[e] &= *dbitset;
    }
  }
}

#endif //MCFILTER_HH
