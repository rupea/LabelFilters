#include <vector>
#include "Eigen/Dense"
#include "Eigen/Sparse"
#include "utils.h"


// ************************************
// Convert a label vector to a label matrix
// Assumes that the label vector contains labels from 1 to noClasses

SparseMb labelVec2Mat (const VectorXd& yVec)
{
  long int n = yVec.size();
  long int noClasses = yVec.maxCoeff();
  std::vector< Eigen::Triplet<bool> > tripletList;
  tripletList.reserve(n);
  long int i;
  for (i = 0; i<n; i++)
    {
      // label list starts from 1
      tripletList.push_back(Eigen::Triplet<bool> (i, yVec.coeff(i)-1, true));
    }
  
  SparseMb y(n,noClasses);
  y.setFromTriplets(tripletList.begin(),tripletList.end());
  return y;
}
