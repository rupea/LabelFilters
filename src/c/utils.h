#ifndef __UTILS_H
#define __UTILS_H

#include "typedefs.h"

using Eigen::VectorXd;

// *********************************
// functions and structures for sorting and keeping indeces
struct IndexComparator;

void sort_index(const VectorXd& m, std::vector<int>& cranks);


// ************************************
// Convert a label vector to a label matrix
// Assumes that the label vector contains labels from 1 to noClasses

SparseMb labelVec2Mat (const VectorXd& yVec);


#endif
