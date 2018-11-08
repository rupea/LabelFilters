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
  _logtime = 0;
}    

void MCfilter::init_logtime(int np /*=-1*/)
{
  time_t start, stop;
  // if filters have not been initialized, do it now. This should not happen.  
  if (isempty())
    {
      init_filters();
    }
  if (np < 0 || np > nProj) np = nProj;

  MCfilter* fp = this; // for openmp
  time(&start);
#if MCTHREADS
#pragma omp parallel for shared(fp)
#endif
  for(size_t p=0U; p<np; ++p)
    {
      fp->_filters[p]->init_map();
    }
  time(&stop);
  if (_verbose) std::cout << stop-start << " seconds to initialize log-time filtering" << std::endl;
  _logtime = np;
}    


MCfilter::MCfilter(int const vb /*=defaultVerbose*/)
  :MCsoln(), _filters(), _verbose(vb), _logtime(0)
{}
		   
MCfilter::MCfilter(MCsoln const& s, int const vb /*=defaultVerbose*/ ):
  MCsoln(s), _filters(), _verbose(vb), _logtime(0)
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
  _logtime = false;
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

