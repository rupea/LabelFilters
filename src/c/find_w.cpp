#include <iostream>
#include <vector>
#include <stdio.h>
#include <typeinfo>
#include <math.h>
#include <stdlib.h>
#include "Eigen/Dense"
#include "Eigen/Sparse"
#include "constants.h"
#include "typedefs.h"
#include "WeightVector.h"
#include "printing.h"
#include "find_w.h"

using Eigen::VectorXd;
using Eigen::VectorXi;

using namespace std;



// ******************************
// Convert to a STD vetor from Eigen Vector
void toVector(std::vector<int>& to, const VectorXd& from)
{
  for (int i = 0; i < from.size(); i++)
    {
      to.push_back((int) from(i));
    }
}


// *******************************
// Calculates the objective function

/********* template functions are implemented in the header
template<typename EigenType>
double calculate_objective_hinge(const VectorXd& w,
				 const EigenType& x, const VectorXd& y,
				 const VectorXd& l, const VectorXd& u, const std::vector<int>& class_order,
				 double C1, double C2);

template<typename EigenType>
double calculate_objective_hinge(const VectorXd& w,
				 const EigenType& x, const VectorXd& y,
				 const VectorXd& l, const VectorXd& u, const VectorXd& class_order,
                                 double C1, double C2);
**********/

// ********************************
// Get unique values in the class vector -> classes

std::vector<int> get_classes(VectorXd& y)
{
  std::vector<int> v;
  for (int i = 0; i < y.rows(); i++)
    {
      if (std::find(v.begin(), v.end(), y[i]) == v.end()) // if the label does not exist
	{
	  v.push_back((int) y[i]);
	}
    }
  return v;
}

// *********************************
// functions and structures for sorting and keeping indeces
struct IndexComparator
{
  VectorXd v;
  IndexComparator(VectorXd& m)
  {
    v = m;
  }
  bool operator()(int i, int j)
  {
    return (v(i) < v(j));
  }
};

void sort_index(VectorXd& m, std::vector<int>& cranks)
{
  for (int i = 0; i < m.size(); i++)
    {
      cranks[i] = i;
    }
  IndexComparator cmp(m);
  std::sort(cranks.begin(), cranks.end(), cmp);

}

// *********************************
// Ranks the classes to build the switches
void rank_classes(std::vector<int>& indices, std::vector<int>& cranks, VectorXd& l, VectorXd& u)
{
  VectorXd m = ((u + l) * .5);
  sort_index(m, indices);
  for (int i = 0; i < m.size(); i++)
    {
      cranks[indices[i]] = i;
    }
}


// *******************************
// Get the number of exampls in each class

void init_nc(VectorXi& nc, VectorXi& nclasses, const SparseMb& y)
{  
  int noClasses = y.cols();
  if (nc.size() != noClasses) 
    {
      cerr << "init_nc has been called with vector nc of wrong size" << endl;
      exit(-1);
    }
  int n = y.rows();  
  if (nclasses.size() != n) 
    {
      cerr << "init_nc has been called with vector nclasses of wrong size" << endl;
      exit(-1);
    }
  for (int k = 0; k < noClasses; k++)
    {
      nc(k)=0;
    }
  for (int i=0;i<n;i++)
    {
      nclasses[i]=0;
      for (SparseMb::InnerIterator it(y,i);it;++it)
	{
	  if (it.value())
	    {
	      nc(it.col())++;
	      nclasses(it.row())++;
	    }
	}
    }
}

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



// ********************************
// Initializes the lower and upper bound

/********* template functions are implemented in the header
template<typename EigenType>
void init_lu(VectorXd& l, VectorXd& u, VectorXd& means, const VectorXi& nc, const VectorXd& w,
	     EigenType& x, const VectorXd& y,
	     const int noClasses);
****************/

// *************************************
// need to reimplement this funciton to work with (inside) the WeightVector class
// this might be a costly operation that might be not needed
// we'll implement this when we get there

// // ******************************
// // Projection to a new vector that is orthogonal to the rest
// // It is basically Gram-Schmidt Orthogonalization
// void project_orthogonal(VectorXd& w, const DenseM& weights,
// 			const int& projection_dim)
// {
//   if (projection_dim == 0)
//     return;
  
//   // Assuming the first to the current projection_dim are the ones we want to be orthogonal to
//   VectorXd proj_sum(w.rows());
//   DenseM wt = w.transpose();
//   double norm;
  
//   proj_sum.setZero();
  
//   for (int i = 0; i < projection_dim; i++)
//     {
//       norm = weights.col(i).norm();
//       proj_sum = proj_sum
// 	+ weights.col(i) * ((wt * weights.col(i)) / (norm * norm));
//     }
  
//   w = (w - proj_sum);
// }


// *********************************
// Solve the optimization using the gradient decent on hinge loss

/********* template functions are implemented in the header
template<typename EigenType>
void solve_optimization(DenseM& weights, DenseM& lower_bounds,
			DenseM& upper_bounds, VectorXd& objective_val,
			EigenType& x, const VectorXd& y,
			const double C1_, const double C2_, bool resumed);
***********/

// ---------------------------------------
int main()
{

  srand (42782);
 
  //  DenseM weights(40000,1),lower_bounds(1000,1),upper_bounds(1000,1), x(10000,40000);
  //  VectorXd y(10000),objective_val;

  DenseM weights(467,1),lower_bounds(5,1),upper_bounds(5,1), x(281,467);
  VectorXd yVec(281),objective_val;

  param_struct params = set_default_params();
  weights.setRandom();
  lower_bounds.setZero();
  upper_bounds.setZero();
  x.setRandom();
  SparseM xs = x.sparseView();
  for (int i = 0; i < yVec.size(); i++)
    {
      //      y(i) = (i%1000)+1;
      yVec(i) = (i%5)+1;
    }
  SparseMb y = labelVec2Mat(yVec);
  
  // these calls are important so that the compiler instantiates the right templates
  solve_optimization(weights,lower_bounds,upper_bounds,objective_val,x,y,0,params);
  solve_optimization(weights,lower_bounds,upper_bounds,objective_val,xs,y,0,params);
  
  
  xs.conservativeResize(281,1123497);
  DenseM sweights (1123497,1);
  sweights.setRandom();
  solve_optimization(sweights,lower_bounds,upper_bounds,objective_val,xs,y,0,params);
  
}
