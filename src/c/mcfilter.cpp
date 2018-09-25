/*  Copyright (C) 2017 NEC Laboratories America, Inc. ("NECLA"). All rights reserved.
 *
 * This source code is licensed under the license found in the LICENSE file in
 * the root directory of this source tree. An additional grant of patent rights
 * can be found in the PATENTS file in the same directory.
 */
#include "mcfilter.hh"


void MCfilter::init_filters()
{
  // delete any existing filters
  delete_filters();
  
  _filters.resize(nProj);
  MCfilter* fp = this; // for openmp
#if MCTHREADS
#pragma omp parallel for shared(fp)
#endif
  for(size_t p=0U; p<nProj; ++p)
    {
      fp->_filters[p] = new Filter(fp->lower_bounds.col(p), fp->upper_bounds.col(p));
    }
}    


MCfilter::MCfilter()
  :MCsoln(), _filters()
{}

MCfilter::MCfilter(MCsoln const& s):
  MCsoln(s), _filters()
{
  init_filters();
}

void MCfilter::delete_filters()
{
  for (size_t i = 0; i < _filters.size(); i++)
    {
      delete _filters[i];
    }    
  _filters.clear();
}  
    
MCfilter::~MCfilter()
{
  delete_filters();
}

// Explicitly instantiate MCfilter into the library

template
void MCfilter::filter(ActiveSet& active, DenseM const& x, int np) const;
template
void MCfilter::filter(ActiveSet& active, SparseM const& x, int np) const;
//template
//void MCfilter::filter(ActiveSet& active, ExtConstSparseM const& x, int np) const;

