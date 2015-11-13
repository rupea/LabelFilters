#include <boost/dynamic_bitset.hpp>
#include <vector>
#include "Eigen/Dense"
#include "filter.h"
#include "utils.h"

#include <iostream>

using Eigen::VectorXd;
using namespace boost;
using namespace std;

Filter::Filter(const VectorXd& l, const VectorXd& u)
{
  _sortedLU = new VectorXd(2*l.size());
  for (int i=0;i<l.size();i++)
    {
      _sortedLU->coeffRef(2*i) = l.coeff(i);
      if (l.coeff(i) >= u.coeff(i))
	{
	  // cerr << "Warning, L >= U for class " << i << " (L=" << l.coeff(i) << ",U=" << u.coeff(i) << ")" << endl;
	  // cerr << "     using interval [L, L+1e-6]" << endl;
	  //better to deal with this by eliminating the class. This way there won't be any cases that will match this class
	  _sortedLU->coeffRef(2*i+1) = l.coeff(i)+1e-6;	
	}
      else
	{
	  _sortedLU->coeffRef(2*i+1) = u.coeff(i);
	}
    }
  vector<int> ranks(_sortedLU->size());
  sort_index(*_sortedLU, ranks);
  std::sort(_sortedLU->data(), _sortedLU->data() + _sortedLU->size());
  init_map(ranks);
}

Filter::~Filter()
{
  for (vector<dynamic_bitset<>*>::iterator map_it=_map->begin(); map_it != _map->end();++map_it)
    {
      delete (*map_it);
    }
  delete _map;
  delete _sortedLU;
}

void Filter::init_map(vector<int>& ranks)
{
  size_t noClasses = ranks.size()/2;
  _map = new vector<dynamic_bitset<>*>(2*noClasses+1);
  vector<dynamic_bitset<>*>::iterator map_it = _map->begin();
  *map_it++ = new dynamic_bitset<>(noClasses);
  size_t i = 0;
  while (map_it != _map->end())
    {
      *map_it = new dynamic_bitset<>(**(map_it-1));
      (*map_it)->set(ranks[i]/2,!(ranks[i]&1));
      map_it++;
      i++;
    }
}

