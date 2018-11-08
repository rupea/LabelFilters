/*  Copyright (C) 2017 NEC Laboratories America, Inc. ("NECLA"). All rights reserved.
 *
 * This source code is licensed under the license found in the LICENSE file in
 * the root directory of this source tree. An additional grant of patent rights
 * can be found in the PATENTS file in the same directory.
 */
#include "filter.h"
#include "Eigen/Dense"
#include "utils.h"
#include "roaring.hh"
#include "mutexlock.h"
#include <vector>
#include <iostream>

using Eigen::VectorXd;
using namespace std;

int const Filter::verbose = 0;

    Filter::Filter(const VectorXd& l, const VectorXd& u)
      :_l(l), _u(u), _sortedLU( 2*l.size() ), _sortedClasses(2*l.size())
      , _map()
      , _mapok(false)
#ifndef NDEBUG
      , nCalls(0)
#endif
{
    for(size_t i=0; i<l.size(); ++i){
        _sortedLU.coeffRef(2*i) = l.coeff(i);
        if (l.coeff(i) >= u.coeff(i)) {
            if(verbose) cerr<<"Warning, L >= U for class "<<i<<" (L="<<l.coeff(i)<<",U="<<u.coeff(i)
                <<")\n     using interval [L, L+1e-6]" << endl;
            //better to deal with this by eliminating the class.
            //This way there won't be any cases that will match this class
            _sortedLU.coeffRef(2*i+1) = l.coeff(i)+1e-6;	
        } else {
            _sortedLU.coeffRef(2*i+1) = u.coeff(i);
        }
    }
    //    vector<int> ranks(_sortedLU.size());
    //    sort_index(_sortedLU, ranks);
    sort_index(_sortedLU, _sortedClasses);
    std::sort(_sortedLU.data(), _sortedLU.data()+_sortedLU.size());
    for (auto &c :_sortedClasses) c = c/2;
    if(verbose>=2){
        cout<<" Filter    lower : "<<l.transpose()<<endl;
        cout<<" Filter    upper : "<<u.transpose()<<endl;
        cout<<" Filter _sortedLU: "<<_sortedLU.transpose()<<endl;
        cout<<" Filter     ranks: "; for(auto const r: _sortedClasses) cout<<" "<<r; cout<<endl;
    }
}

Filter::~Filter()
{
#ifndef NDEBUG
    if(verbose>=1){
        uint64_t nClass = _sortedLU.size() / 2U;
        if( nCalls < nClass ){
            cerr<<" Inefficiency Warning: Filter object for "<<nClass<<" classes only called "<<nCalls<<" times."<<endl;
        }
    }
#endif
}

void Filter::init_map()
{
  if (_mapok) return; // already initialized 
  size_t const noClasses = _sortedLU.size()/2;
  size_t const nBitmaps = 2U*noClasses + 1U;
  assert( _map.size() == 0U );
  _map.reserve( nBitmaps );
  _map.emplace_back();  // begin with a default [empty] bitset
  _map.back().setCopyOnWrite(true);
  if(verbose>=2) cout<<" Filter map["<<_map.size()-1U<<"]="<<_map.back().toString()<<endl;
  for(size_t i=0U; i<nBitmaps-1U; ++i){
    _map.emplace_back( _map.back() ); // push a copy of the last bitmap
    // this depends on lower bounds being smaller than upper bounds.
    _map.back().flip( _sortedClasses[i], _sortedClasses[i]+1 ); 
  }
  assert( _map.size() == nBitmaps );
  _mapok = true;
}

void Filter::filterBatch (Eigen::VectorXd proj, ActiveSet& active, vector<MutexType>& mutex) const
{
  size_t nEx = proj.size();

  if (active.size() != nEx)
    {
      throw("ERROR: Filter::filterBatch, proj and active not the same size");
    }
  if (mutex.size() != nEx)
    {
      throw("ERROR: Filter::filterBatch, active and mutex not the same size");
    }
  
  vector<int> ranks(nEx);
  sort_index(proj, ranks);
  
  std::sort(proj.data(), proj.data()+proj.size());
  size_t n = 0;
  size_t i = 0;
  Roaring current;
  while (n < nEx)
    {
      while (proj[n] > _sortedLU[i] && i < _sortedLU.size())
	{
	  int c=_sortedClasses[i++];
	  current.flip(c,c+1);
	}
      mutex[ranks[n]].Lock();
      active[ranks[n]] &= current;
      mutex[ranks[n]].Unlock();
      n++;
    }
}
  
