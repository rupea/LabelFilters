#include "filter.h"
#include "Eigen/Dense"
#include "utils.h"
#include "roaring.hh"
#include <vector>
#include <iostream>

using Eigen::VectorXd;
using namespace std;

int const Filter::verbose = 0;

    Filter::Filter(const VectorXd& l, const VectorXd& u)
    : _sortedLU( 2*l.size() )
      , _map()
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
    vector<int> ranks(_sortedLU.size());
    sort_index(_sortedLU, ranks);
    std::sort(_sortedLU.data(), _sortedLU.data()+_sortedLU.size());
    if(verbose>=2){
        cout<<" Filter    lower : "<<l.transpose()<<endl;
        cout<<" Filter    upper : "<<u.transpose()<<endl;
        cout<<" Filter _sortedLU: "<<_sortedLU.transpose()<<endl;
        cout<<" Filter     ranks: "; for(auto const r: ranks) cout<<" "<<r; cout<<endl;
    }
    init_map(ranks);
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

void Filter::init_map(vector<int>& ranks)
{
  size_t const noClasses = ranks.size()/2U;
  size_t const nBitmaps = 2U*noClasses + 1U;
  assert( _map.size() == 0U );
  _map.reserve( nBitmaps );
  _map.emplace_back();  // begin with a default [empty] bitset
  _map.back().setCopyOnWrite(true);
  if(verbose>=2) cout<<" Filter map["<<_map.size()-1U<<"]="<<_map.back().toString()<<endl;
  for(size_t i=0U; i<nBitmaps-1U; ++i){
    _map.emplace_back( _map.back() ); // push a copy of the last bitmap
    // this depends on lower bounds being smaller than upper bounds. 
    ranks[i]&1?_map.back().remove(ranks[i]/2):_map.back().add(ranks[i]/2); // toggle 1 bit as cross a lower or upper bound
    // could try if this is faster
    // _map.back().flip( ranks[i]/2, ranks[i]/2+1 ); 
    if(verbose>=2) cout<<" Filter map["<<_map.size()-1U<<"]="<<_map.back().toString()<<endl;
  }
  assert( _map.size() == nBitmaps );
}

