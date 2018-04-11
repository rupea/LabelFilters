#include "mcfilter.hh"

void MCfilter::init_filters()
{
  // delete any existing filters
  _filters.clear();
  _filters.shrink_to_fit(); //free memory if any has been alocated.
  
  _filters.reserve(nProj);  
#if MCTHREADS
#pragma omp parallel for shared(_filters, lower_bounds, upper_bounds)
#endif
  for(size_t p=0U; p<nProj; ++p)
    {
      _filters.emplace_back(lower_bounds.col(p), upper_bounds.col(p));
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


// Explicitly instantiate MCfilter into the library

template
void MCfilter::filter(std::vector<Roaring>& active, DenseM const& x, int np);
template
void MCfilter::filter(std::vector<Roaring>& active, SparseM const& x, int np);
template
void MCfilter::filter(std::vector<Roaring>& active, ExtConstSparseM const& x, int np);

