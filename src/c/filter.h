#ifndef __FILTER_H
#define __FILTER_H

#include "Eigen/Dense"
#include <boost/dynamic_bitset.hpp>
#include <vector>
//using Eigen::VectorXd;
//using namespace boost;
//using namespace std;

class Filter
{
 public:
  Filter(const Eigen::VectorXd& l, const Eigen::VectorXd& u);
  ~Filter();
  const boost::dynamic_bitset<>* filter (double xproj) const;
 private:  
  Eigen::VectorXd* _sortedLU;
  std::vector<boost::dynamic_bitset<>*>* _map;
  void init_map(std::vector<int>& ranks);
};

inline const boost::dynamic_bitset<>* Filter::filter(double xproj) const
{
  return((*_map)[std::lower_bound(_sortedLU->data(),_sortedLU->data() + _sortedLU->size(), xproj) - _sortedLU->data()]);
}

#endif
