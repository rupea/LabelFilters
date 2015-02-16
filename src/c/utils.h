#ifndef __UTILS_H
#define __UTILS_H

#include "typedefs.h"

using Eigen::VectorXd;


// *********************************
// functions and structures for sorting and keeping indeces
// Should implement bound checking but it is faster this way.
template<typename IntType>
struct IndexComparator
{
  const VectorXd* v;
  IndexComparator(const VectorXd* m)
  {
    v = m;
  }
  bool operator()(IntType i, IntType j)
  {
    return (v->coeff(i) < v->coeff(j));
  }
};

template<typename IntType>
void sort_index(const VectorXd& m, std::vector<IntType>& cranks)
{
  for (IntType i = 0; i < m.size(); i++)
    {
      cranks[i] = i;
    }
  IndexComparator<IntType> cmp(&m);
  std::sort(cranks.begin(), cranks.end(), cmp);
};


// ************************************
// Convert a label vector to a label matrix
// Assumes that the label vector contains labels from 1 to noClasses

SparseMb labelVec2Mat (const VectorXd& yVec);


#endif
