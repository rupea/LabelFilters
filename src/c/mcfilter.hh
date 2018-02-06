#ifndef MCFILTER_HH
#define MCFILTER_HH

#include "mcfilter.h"
#include <boost/dynamic_bitset.hpp>
#include "typedefs.h" //DenseM

template< typename EIGENTYPE >
void MCfilter::filter(/*out*/ std::vector<boost::dynamic_bitset<>>& active, /*in*/ EIGENTYPE const& x, int np/* = 0*/)
{  

  size_t const nExamples = x.rows();  
  assert(x.cols() == this->d);
  
  np = np?np:nProj;
  if (np > nProj)
    {
      cerr << "Warning: MCfilter::filter requested " << np << " projections, but only " << nProj
	   << " are available" << endl;
      np = nProj;
    }
  DenseM const projections = (x * weights.leftCols(np));

  
  //active = new ActiveDataSet(nExamples);
  if( active.size() > nExamples ){
    active.resize(nExamples);
  }
  for(size_t i=0U; i<active.size(); ++i){
    active[i].clear();
    active[i].resize(nClass,true);
  }
  if( active.size() < nExamples ){
    if( active.size() == 0U ){
      active.emplace_back( boost::dynamic_bitset<>() );
      active.back().resize(nClass,true);
    }
    size_t bk = active.size() - 1U;
    while( active.size() < nExamples )
      active.emplace_back( active[bk] );  // can copy-construct all-true as a copy in 1 step
  }
  assert( active.size() == nExamples );
  
  // TODO if ! nProj >> nExamples, provide a faster impl ???
#if MCTHREADS
#pragma omp parallel for shared(_filters, projections, active, lower_bounds, upper_bounds)
#endif
  for(size_t e=0U; e<nExamples; ++e){
    for(size_t p=0U; p<np; ++p){
      boost::dynamic_bitset<> const* dbitset = _filters[p].filter(projections.coeff(e,p));
      active[e] &= *dbitset;
    }
  }
}

#endif //MCFILTER_HH
