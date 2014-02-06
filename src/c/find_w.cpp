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
std::vector<int> get_classes(VectorXd y)
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
void rank_classes(std::vector<int>& cranks, VectorXd& l, VectorXd& u)
{
  VectorXd m = ((u + l) * .5);
  std::vector<int> indices(m.size());
  sort_index(m, indices);
  for (int i = 0; i < m.size(); i++)
    {
      cranks[indices[i]] = i;
    }
}


// *******************************
// Get the number of exampls in each class

void init_nc(VectorXi& nc, const VectorXd& y, const int noClasses)
{  
  int c;
  if (nc.rows() != noClasses) 
    {
      cerr << "init_nc has been called with vector nc of wrong size" << endl;
      exit(-1);
    }
  int n = y.rows();
  for (int k = 0; k < noClasses; k++)
    {
      nc(k)=0;
    }
  for (int i=0;i<n;i++)
    {
      c = (int) y[i] - 1; // Assuming all the class labels start from 1
      nc(c)++;
    }
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
  VectorXd y(281),objective_val;

  weights.setRandom();
  lower_bounds.setZero();
  upper_bounds.setZero();
  x.setRandom();
  SparseM xs = x.sparseView();
  for (int i = 0; i < y.size(); i++)
    {
      //      y(i) = (i%1000)+1;
      y(i) = (i%5)+1;
    }

  // these calls are important so that the compiler instantiates the right templates
  solve_optimization(weights,lower_bounds,upper_bounds,objective_val,x,y,10,1,0);
  solve_optimization(weights,lower_bounds,upper_bounds,objective_val,xs,y,10,1,0);
  
  
  xs.conservativeResize(281,1123497);
  DenseM sweights (1123497,1);
  sweights.setRandom();
  solve_optimization(sweights,lower_bounds,upper_bounds,objective_val,xs,y,10,1,0);
  
}
