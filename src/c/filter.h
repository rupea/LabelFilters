#ifndef __FILTER_H
#define __FILTER_H

using Eigen::VectorXd;
using namespace boost;
using namespace std;

class Filter
{
 public:
  Filter(const VectorXd& l, const VectorXd& u);
  ~Filter();
  const dynamic_bitset<>* filter (double xproj) const;
  size_t noClasses();
 private:  
  size_t _noClasses;
  VectorXd* _sortedLU;
  vector<dynamic_bitset<>*>* _map;
  void init_map(vector<int>& ranks);
};

inline size_t Filter::noClasses() 
{
  return _noClasses;
}

inline const dynamic_bitset<>* Filter::filter(double xproj) const
{
  return((*_map)[lower_bound(_sortedLU->data(),_sortedLU->data() + _sortedLU->size(), xproj) - _sortedLU->data()]);
}

#endif
