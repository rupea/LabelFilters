/** \file
 * learn projections using only C++ */
#include "../find_w.h"
#include "Eigen/Dense"
#include "Eigen/Sparse"
#include "utils.h"
#ifdef _OPENMP
#include <omp.h>
#endif
#include <iostream>
#include <stdlib.h>
#include <exception>
#include <cstdint>
using namespace std;

/** given a d-dimensional \c corner, and a list of d d-dimensional
 * edge vectors, return a skewed d-dimensional hyperrectangle \c rect.
 * - \c rect will have the 2^d corners of some skewed hyper-rectangle.
 * - For example, in 3-D if corner=[1,1,1] and
 * - edges=[[1,0,0],[0,2,0],[0,0,3]]
 * - then rect would be an unskewed rectangular prism with these vertices:
 *   - [1,1,1],
 *   - [2,1,1],[1,3,1],[1,1,4]   (single edge hop from corner)
 *   - [2,3,1],[2,1,4],[1,3,4]   (two edge hops from corner)
 *   - [2,3,4]                   (three edge hops from corner)
 *   - output size 8 x 3
 * - Why?
 *   - lower & upper bounds of projections of skewed hyper-rectangles are easy.
 *   - so can "guess" which projection axes should be found to have
 *     the lowest ranges, lack of overlap, etc.
 */
void hyperRect( VectorXd const& corner, DenseM const& edges, DenseM& rect )
{
    int const verbose=1;
    uint32_t const d = corner.size();                                 // dimensionality

    if(verbose>0){
        cout<<" edges.size()="<<edges.size()<<" corner.size()="<<corner.size()<<endl;
        cout<<" corner = "<<corner.transpose()<<endl;
        cout<<" edges  = ";
        for( uint32_t e=0U; e<d; ++e ) cout<<(e==0?"":"          ")<<edges.row(e)<<"\n";
    }
    assert( edges.cols() == d ); // every edge vector must have correct dimension
    assert( edges.rows() == d ); // we insist on full dimensionality object
    // i.e. no projection into some subplane (unless edges are not lin. ind.)

    uint32_t const v = floor( std::pow(2.0,double(corner.size())));   // vertices
    // NOTE: c++ integer powers?

    rect.resize(v,d);
    for(uint32_t r=0U; r<v; ++r){       // for all output vertices
        rect.row(r) = corner;
        for(uint32_t b=1U,e=0U; e<d; b<<=1,++e){// for edges[e] corresponding to bits b of r:
            if(verbose>1)cout<<" r="<<r<<" b="<<b<<" e="<<e<<endl;
            if( (r&b) != 0 ){                   //   if bit is set
                rect.row(r) += edges.row(e);    //     take hop along that edge
                //for(uint32_t i=0U; i<d; ++i)
                //    rect(r,i) += edges(e,i);  //     take hop along that edge
            }
        }
    }
}

int main(int,char**)
{

#ifdef _OPENMP
  cout<<" _OPENMP defined, omp_get_max_threads ... "<<omp_get_max_threads()<<endl;
#endif

  //  DenseM weights(40000,1),lower_bounds(1000,1),upper_bounds(1000,1), x(10000,40000);
  //  VectorXd y(10000),objective_val;
  const size_t nex = 3*8;                 // 3 cubes of 8 vertices each
  const size_t problem = 2;               // simplest problem is zero.
  assert( problem <= 2U );

  param_struct params = set_default_params();
  params.update_type    = MINIBATCH_SGD;
  params.batch_size     = nex;            // minibatch == # of examples (important)
#if 1 // for a quick run
  params.no_projections = 2U;             // There is only one optimum for problems 0,1,2 (increase to see something in 'top')
  params.max_iter       = nex*1000U;      // trivial problem, so insist on few iterations
#else // for run for a longer time (excessively long if you set OMP_NUM_THREADS > 1)
  params.no_projections = 5U;            // repeat "same" calc many times
  params.max_iter       = nex*100000U;    // trivial problem, so insist on few iterations
#endif
  params.report_epoch   = params.max_iter/10U; // limit the number of reports
  switch(problem){
    case(0):
        params.optimizeLU_epoch = nex*10U;      // good enough for convergence
        //params.update_type    = SAFE_SGD;     // ERROR: this trainer assumes x.row(i).norm() == 1
        //params.eta            = 1.0;          // default is 1.0
        //params.eta_type       = ETA_CONST;
        //params.min_eta        = 1.0;          // why does this have influence if ETA_CONST ?? there was a bug (now fixed)
        break;
    case(1):
        //params.optimizeLU_epoch = nex;
        params.optimizeLU_epoch = nex*10U;      // good enough for convergence
        break;
    case(2): // more difficult
        //params.max_iter       = nex*1000U;
        if(1){ // EITHER batch_size or optimizeLU_epoch MUST be one to converge to correct solution
            params.batch_size     = 1;          // REQUIRED (nex WILL NOT WORK)
            params.optimizeLU_epoch = nex;      // nex*10 didn't work
        }else{
            //params.batch_size     = nex;        //
            params.optimizeLU_epoch = 1;        // nex didn't work
        }

        //params.eta = 100;
        break;
  }
  int const rand_seed = 997787879;
  int const d = 3U;             // x training data dimensionality

  if(0){ // demo
      DenseM verts(8 /* 2^3 */, 3);
      VectorXd corner(3); corner<<0,0,0;
      DenseM edges(3,3); edges<<1,0,0,  0,1,0,  1,0,1;
      // unit cube skewed away from z-axis, toward x-axis
      hyperRect( corner, edges, verts );
      cout<< verts <<endl;
      return 0;
  }

  // x training data and y class labels
  DenseM x(nex,d);
  VectorXd yVec(nex);
  string x_msg;
  if(problem<=2)
  {
      uint32_t n=0U;                    // example number
      DenseM verts(8 /* 2^3 */, 3);
      VectorXd corner(3);
      DenseM edges(3,3);
      switch(problem){
          case(0): // solution is x-axis, (y- and z-axis both overlap)
              x_msg="unit cube, lengthened on z-axis";
              edges<<1,0,0,  0,1,0,  0,0,2;
              break;
          case(1): // soln is perp. to the skew dirn, so along (-1,0,1)
              x_msg="rectangular prism skewed away from z-axis toward x-axis";
              // 3 skyscrapers along x axis with 45 degree lean,
              // so MAX between-class margin is to project back along x=z
              edges<<1,0,0,  0,1,0,  100,0,100;
              break;
          case(2): // SAME soln as case 1, but this fails to converge to correct solution.
              x_msg="skyscrapers leaning 45-degrees along x=z";
              // with slightly shorter scyscrapers, (but still not making projection onto
              // x-axis the optimum), should have EXACTLY the same solution
              // ... BUT FAILS.
              edges<<0.1,0,0,  0,0.1,0,  sqrt(18.0),0,sqrt(18.0);
              break;
      }
      // first skewed rectangular prism
      corner<<0,0,0;
      hyperRect( corner, edges, verts );
      for(uint32_t i=0U; i<verts.rows(); ++i,++n){ cout<<" "<<n; cout.flush(); x.row(n) = verts.row(i); yVec(n)=0; }
      // second skewed rectaular prism
      corner<<3,0,0;
      hyperRect( corner, edges, verts );
      for(uint32_t i=0U; i<verts.rows(); ++i,++n){ cout<<" "<<n; cout.flush(); x.row(n) = verts.row(i); yVec(n)=1; }
      // third skewed rectangular prism
      corner<<6,0,0;
      hyperRect( corner, edges, verts );
      for(uint32_t i=0U; i<verts.rows(); ++i,++n){ cout<<" "<<n; cout.flush(); x.row(n) = verts.row(i); yVec(n)=2; }
  }
  SparseMb y = labelVec2Mat(yVec);

  // Starting off a new calculation:
  srand(rand_seed);
#if 0
  int const k = 5U;             // number of classes
  int const p = params.no_projections;
  //DenseM weights(d,p), lower_bounds(k,p), upper_bounds(k,p);
  //DenseM w_avg(d,p), l_avg(k,p), u_avg(k,p);
  //VectorXd objective_val, o_avg;
  weights.setRandom();
  lower_bounds.setZero();
  upper_bounds.setZero();
  w_avg.setRandom();    // ???
  l_avg.setZero();
  u_avg.setZero();
#else
  DenseM weights, lower_bounds, upper_bounds;
  DenseM w_avg, l_avg, u_avg;
  VectorXd objective_val, o_avg;
#endif

  cout<<"  pre-run call to rand() returns "<<rand()<<endl;
  // these calls are important so that the compiler instantiates the right templates
  solve_optimization(weights,lower_bounds,upper_bounds,objective_val
                     ,w_avg,l_avg,u_avg,o_avg
                     ,x,y,params);

  cout<<" post-run call to rand() returns "<<rand()<<endl;
  cout<<" quick demo of 3 translations of a "<<x_msg<<" along the x axis"<<endl;
  cout<<" projection matrix\n";
  //
  // Note that even though input examples were row-wise d-dimensional,
  //      the w-matrix is a bunch of column-wise vectors.
  //
  // w_avg is not produced by default -- same as w for default params.avg_epoch == 0
  //cout<<" w_avg = ";
  //for(int r=0U; r<w_avg.rows(); ++r) cout<<(r==0U?"":"         ")<<w_avg.row(r)<<"\n";
  //cout<<endl;
  cout<<" w     = ";
  for(int r=0U; r<weights.rows(); ++r) cout<<(r==0U?"":"         ")<<weights.row(r)<<"\n";
  cout<<endl;
  cout<<" lb    = ";
  for(int r=0U; r<lower_bounds.rows(); ++r) cout<<(r==0U?"":"         ")<<lower_bounds.row(r)<<"\n";
  cout<<endl;
  cout<<" ub    = ";
  for(int r=0U; r<upper_bounds.rows(); ++r) cout<<(r==0U?"":"         ")<<upper_bounds.row(r)<<"\n";
  cout<<endl;
  //
  // throw on wrong output
  //
  weights.colwise().normalize();
  switch( problem ){
    case(0):
        for(uint32_t i=0U; i<params.no_projections; ++i){
            if( fabs(weights(0,i)) < 0.99 || fabs(weights(1,i)) > 0.1 || fabs(weights(2,i)) > 0.1 ){
                throw std::runtime_error(" Incorrect solution for problem 0");
            }
        }
        break;
    case(1): ;// fall-through
    case(2):
        for(uint32_t i=0U; i<params.no_projections; ++i){
            if( fabs(weights(0,i)+weights(2,i)) > 0.1 || fabs(weights(1,i)) > 0.1 ){
                throw std::runtime_error(" Incorrect solution for problem 1");
            }
        }
        break;
  }
  cout<<" Solution was correct enough (GOOD)"<<endl;


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

