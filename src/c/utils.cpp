#include <vector>
#include "Eigen/Dense"
#include "Eigen/Sparse"
#include "utils.h"
#include <iostream>
#include <fstream>

// ************************************
// Convert a label vector to a label matrix
// Assumes that the label vector contains labels from 1 to noClasses
using namespace std;

SparseMb labelVec2Mat (const VectorXd& yVec)
{
  size_t const n = yVec.size();
  size_t const minclass = yVec.minCoeff();      // originally this was assumed to be '1'
  size_t const noClasses = yVec.maxCoeff() - minclass + 1U;
  std::vector< Eigen::Triplet<bool> > tripletList;
  tripletList.reserve(n);
  for(size_t i = 0; i<n; ++i) {
    // label list starts from 1 (minclass)
    tripletList.push_back(Eigen::Triplet<bool> (i, yVec(i)-minclass, true));
  }
  
  SparseMb y(n,noClasses);
  y.setFromTriplets(tripletList.begin(),tripletList.end());
  return y;
}


// check if a file exists and is ready for reading

bool fexists(const char *filename)
{
  std::ifstream ifile(filename);
  return ifile;
}


// delete the contents of an ActiveDataSet to free up memory

void free_ActiveDataSet(ActiveDataSet*& active)
{
    if (active){   
        for(ActiveDataSet::iterator actit = active->begin(); actit !=active->end();actit++){
            delete (*actit);
        }
        delete active;
        active=nullptr;
    }		
}
