/** \file
 * learn projections using only C++ */
#include "../find_w.h"
#include "utils.h"              // labelVec2Mat
#include "Eigen/Dense"
#include "Eigen/Sparse"
#ifdef _OPENMP
#include <omp.h>
#endif
#include <iostream>
#include <stdlib.h>
using namespace std;

int main(int,char**)
{

#ifdef _OPENMP
  cout<<" _OPENMP defined, omp_get_max_threads ... "<<omp_get_max_threads()<<endl;
#endif

  //  DenseM weights(40000,1),lower_bounds(1000,1),upper_bounds(1000,1), x(10000,40000);
  //  VectorXd y(10000),objective_val;
  param_struct params = set_default_params();
  params.no_projections = 4U;
  params.max_iter=100000U;              // default 1e6 takes 10-15 minutes
  int const rand_seed = 117;
  int const p = params.no_projections;
  int const d = 467U;      // x training data dimensionality
  int const k = 5U;        // number of classes

  // x training data and y class labels
  srand(rand_seed);
  DenseM x(281,d);
  VectorXd yVec(281);
  x.setRandom();
  cout<<" rand seed = "<<rand_seed<<" ,  x(100,100) = "<<x(100,100)<<endl;
  for (int i = 0; i < yVec.size(); i++) {
      //      y(i) = (i%1000)+1;
      yVec(i) = (i%k)+1;
  }
  SparseMb y = labelVec2Mat(yVec);

  DenseM weights(d,p), lower_bounds(k,p), upper_bounds(k,p);
  DenseM w_avg(d,p), l_avg(k,p), u_avg(k,p);
  VectorXd objective_val, o_avg;
  // Starting off a new calculation:
  weights.setRandom();
  lower_bounds.setZero();
  upper_bounds.setZero();
  w_avg.setRandom();    // ???
  l_avg.setZero();
  u_avg.setZero();

  cout<<"  pre-run call to rand() returns "<<rand()<<endl;
  // these calls are important so that the compiler instantiates the right templates
  solve_optimization(weights,lower_bounds,upper_bounds,objective_val
                     ,w_avg,l_avg,u_avg,o_avg
                     ,x,y,params);

  cout<<" post-run call to rand() returns "<<rand()<<endl;
#if 0
  // sparse case
  SparseM xs = x.sparseView();
  //solve_optimization(weights,lower_bounds,upper_bounds,objective_val,xs,y,params);
  xs.conservativeResize(281,1123497);
  DenseM sweights (1123497,1);
  sweights.setRandom();
  solve_optimization(sweights,lower_bounds,upper_bounds,objective_val,xs,y,params);
#endif
  
}

