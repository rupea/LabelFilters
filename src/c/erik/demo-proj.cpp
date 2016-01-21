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
#include <fstream>
using namespace std;

#ifndef USE_MCSOLVER
#define USE_MCSOLVER 1
#endif

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

#if ! USE_MCSOLVER
#if 0 // original method
  cout<<" demo-proj.cpp, original code, dense "<<endl;
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
#else                   // In fact, don't need to size or initialize
  cout<<" demo-proj.cpp, easy-init, dense "<<endl;
  DenseM weights, lower_bounds, upper_bounds;
  DenseM w_avg, l_avg, u_avg;
  VectorXd objective_val, o_avg;
#endif
  cout<<"  pre-run call to rand() returns "<<rand()<<endl;
  // these calls are important so that the compiler instantiates the right templates
  solve_optimization(weights,lower_bounds,upper_bounds,objective_val
                     ,w_avg,l_avg,u_avg,o_avg
                     ,x,y,params);
#else
  cout<<"  pre-run call to rand() returns "<<rand()<<endl;
  cout<<" demo-proj.cpp, MCsolver, dense "<<endl;
  MCsolver mc;
  mc.solve( x, y, &params );
  //MCsoln      & soln = mc.getSoln();
  //DenseM      & weights = soln.weights;
  //DenseM const& lower_bounds = soln.lower_bounds;
  //DenseM const& upper_bounds = soln.upper_bounds;
  //DenseM const& w_avg = soln.weights_avg;
  //DenseM const& l_avg = soln.lower_bounds_avg;
  //DenseM const& u_avg = soln.upper_bounds_avg;
  //VectorXd const& objective_val = soln.objective_val;
  //VectorXd const& objective_val_avg = soln.objective_val_avg;
#endif

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

#if 0 && USE_MCSOLVER
  if(1){
      cout<<" Saving to file 'proj.soln'"<<endl;
      try{
          ofstream ofs("proj.soln");
          soln.write( ofs, MCsoln::TEXT, MCsoln::SHORT );
          ofs.close();
      }catch(std::exception const& what){
          cout<<"OHOH! Error during text write of proj.soln"<<endl;
          throw(what);
      }
  }
  if(1){
      cout<<" Saving to file 'proj-bin.soln'"<<endl;
      try{
          ofstream ofs("proj-bin.soln");
          soln.write( ofs, MCsoln::BINARY, MCsoln::SHORT );
          ofs.close();
      }catch(std::exception const& what){
          cout<<"OHOH! Error during binary write of proj.soln"<<endl;
          throw(what);
      }
  }
#endif
  
}

