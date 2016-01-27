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
#include <fstream>
using namespace std;

#define TESTNUM 0
#define PROBLEM 2U
param_struct params = set_default_params();
int testnum = TESTNUM;
bool use_mcsolver = true;
std::string saveBasename = std::string("");
size_t problem = 2U;
string problem_msg;
int verbose = 0;

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


// parse arguments
int testparms(int argc, char**argv, param_struct & params) {
    problem = PROBLEM;
    testnum = TESTNUM;
    use_mcsolver = true;
    for(int a=1; a<argc; ++a){
        for(char* ca=argv[a]; *ca!='\0'; ++ca ){
            if(*ca == 'h'){
                cout<<" Options:"
                    <<"\n   pN          test problem N in [0,2] w/ base parameter settings [default=2]"
                    <<"\n   0-9A-Z      parameter modifiers, [default=0, no change from pN defaults]"
                    <<"\n   m           [default] mcsolver code"
                    <<"\n   o           original code"
                    <<"\n   s<string>   save file base name (only for 'm') [default=none]"
                    <<"\n   h           this help"
                    <<"\n   v           increase verbosity [default=0]"
                    <<"\n Convergence: p0 p1 p2 should converge"
                    <<"\n              p0 with 0-9A-Z should converge, but harder problems might not"
                    <<endl;
            }else if(isdigit(*ca))      testnum = *ca - '0';
            else if(isalpha(*ca) && isupper(*ca)) testnum = 10 + (*ca - 'A');
            else if(*ca == 'v')         ++verbose;
            else if(*ca == 'o')         use_mcsolver=false;
            else if(*ca == 'm')         use_mcsolver=true;
            else if(*ca == 'p')         problem = *++ca - '0';
            else if(*ca == 's'){
                saveBasename.assign(++ca);
                break;
            }
        }
    }
    if(problem>2U) problem=2U;
    return testnum;
}

// apply 0-9 parameter modifiers
void apply_testnum(param_struct & params)
{
    cout<<" demo-proj running testnum "<<testnum;
    switch( testnum ){
      case(0):
          cout<<" all parameters default";
          break;
      case(1):
          params.avg_epoch = 0;                 // default
          params.report_avg_epoch = 0;          // default
          params.report_epoch = 0;
          params.optimizeLU_epoch = 0;
          params.reorder_epoch = 0;
          cout<<" params.report_epoch="<<params.report_epoch
              <<" params.optimizeLU_epoch = "<<params.optimizeLU_epoch
              <<" params.reorder_epoch = "<<params.reorder_epoch;
          break;
      case(2):
          params.avg_epoch = 0;                 // default
          params.report_avg_epoch = 0;          // default
          params.report_epoch = 1000;           // default
          params.optimizeLU_epoch = 0;
          params.reorder_epoch = 0;
          cout<<" params.report_epoch="<<params.report_epoch
              <<" params.optimizeLU_epoch = "<<params.optimizeLU_epoch
              <<" params.reorder_epoch = "<<params.reorder_epoch;
          break;
      case(3):
          params.avg_epoch = 0;                 // default
          params.report_avg_epoch = 0;          // default
          params.report_epoch = 1000;           // default
          params.optimizeLU_epoch = 5500;
          params.reorder_epoch = 0;
          cout<<" params.report_epoch="<<params.report_epoch
              <<" params.optimizeLU_epoch = "<<params.optimizeLU_epoch
              <<" params.reorder_epoch = "<<params.reorder_epoch;
          break;
      case(4):
          params.avg_epoch = 0;                 // default
          params.report_avg_epoch = 0;          // default
          params.report_epoch = 2000;
          params.optimizeLU_epoch = 2500;
          params.reorder_epoch = 1000;
          cout<<" params.report_epoch="<<params.report_epoch
              <<" params.optimizeLU_epoch = "<<params.optimizeLU_epoch
              <<" params.reorder_epoch = "<<params.reorder_epoch;
          break;
      case(5):        // make nAccSortlu_avg increment (for sure)
          params.optimizeLU_epoch = 0;
          params.avg_epoch = 16000;
          cout<<" params.optimizeLU_epoch = "<<params.optimizeLU_epoch
              <<" params.avg_epoch = "<<params.avg_epoch;
          break;
      case(6):
          params.optimizeLU_epoch = 0;
          cout<<" params.optimizeLU_epoch = "<<params.optimizeLU_epoch;
          break;
      case(7):                // mcsolver does not run correctly
          //mcsolver OH? luPerm.nAccSortlu_avg not > 0 for t>=1200
          //mcsolver 'luPerm.ok_sortlu_avg' failed  at end of first dim (after t=100000)
          params.avg_epoch = 1200;
          cout<<" params.avg_epoch = "<<params.avg_epoch;
          break;
      case(8):                // original annd mcsolver BOTH do not run correctly
          //mcsolver OH? luPerm.nAccSortlu_avg not > 0 for t>=1200
          //mcsolver 'luPerm.ok_sortlu_avg' failed  at t=1400
          params.avg_epoch = 1200;
          params.report_avg_epoch = 1400;
          cout<<" params.avg_epoch = "<<params.avg_epoch;
          break;
      case(9):
          params.reorder_type = REORDER_PROJ_MEANS;
          cout<<" params.reorder_type = "<<tostring(params.reorder_type);
          break;
      case(10):
          params.reorder_type = REORDER_RANGE_MIDPOINTS;
          cout<<" params.reorder_type = "<<tostring(params.reorder_type);
          break;
      case(11):        // make nAccSortlu_avg increment (for sure)
          params.reorder_epoch = 1000;   // default
          params.optimizeLU_epoch = 0;
          params.report_avg_epoch = 9999;
          cout<<" params.optimizeLU_epoch = "<<params.optimizeLU_epoch
              <<" params.avg_epoch = "<<params.avg_epoch;
          break;
      case(12):        // make nAccSortlu_avg increment (for sure)
          params.reorder_epoch = 1000;   // default
          params.optimizeLU_epoch = 0;
          params.report_avg_epoch = params.report_epoch*2U;
          cout<<" params.optimizeLU_epoch = "<<params.optimizeLU_epoch
              <<" params.avg_epoch = "<<params.avg_epoch;
          break;
      default:
          cout<<" OHOH! testnum = "<<testnum<<", really?"<<endl;
          throw std::runtime_error(" UNKNOWN testnum (running defaults)");
    }
    cout<<endl;
}

/** partly destroys the solution by colwise renormalizing \c weights */
void check_solution( DenseM & weights, DenseM & lower_bounds, DenseM & upper_bounds )
{
    cout<<" post-run call to rand() returns "<<rand()<<endl;
    cout<<" quick demo of 3 translations of a "<<problem_msg<<" along the x axis"<<endl;
    cout<<" projection matrix\n";
    //
    // Note that even though input examples were row-wise d-dimensional,
    //      the w-matrix is a bunch of column-wise vectors.

    auto wn = weights.colwise().norm();
    cout<<" norms "; cout<<wn<<endl;
    weights.colwise().normalize();
    // w_avg is not produced by default -- same as w for default params.avg_epoch == 0
    //cout<<" w_avg = ";
    //for(int r=0U; r<w_avg.rows(); ++r) cout<<(r==0U?"":"         ")<<w_avg.row(r)<<"\n";
    //cout<<endl;
    cout<<" w(norm) ";
    for(int r=0U; r<weights.rows(); ++r) cout<<(r==0U?"":"           ")<<weights.row(r)<<"\n";
    cout<<endl;
    cout<<" lb    = ";
    for(int r=0U; r<lower_bounds.rows(); ++r) cout<<(r==0U?"":"           ")<<lower_bounds.row(r)<<"\n";
    cout<<endl;
    cout<<" ub    = ";
    for(int r=0U; r<upper_bounds.rows(); ++r){
        auto const& row_r = upper_bounds.row(r);
        cout<<(r==0U?"":"           ");
        for(uint32_t i=0; i<row_r.size(); ++i ){ cout<<" "<<row_r[i]; cout.flush(); }
        //cout<<row_r<<"\n"; cout.flush();
        cout<<endl;
    }
    cout<<endl;
    //
    // throw on wrong output
    //
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
    cout<<" demo-mcsolver ";
    cout<<(use_mcsolver? "m": "o");
    if( problem != PROBLEM ) cout<<"p"<<problem;
    if( testnum != TESTNUM ) cout<<testnum;
    cout<<" : Solution was correct enough (GOOD)"<<endl;
}

int main(int argc,char** argv)
{

#ifdef _OPENMP
    cout<<" _OPENMP defined, omp_get_max_threads ... "<<omp_get_max_threads()<<endl;
#endif
    //  DenseM weights(40000,1),lower_bounds(1000,1),upper_bounds(1000,1), x(10000,40000);
    //  VectorXd y(10000),objective_val;
    const size_t nex = 3*8;                 // 3 cubes of 8 vertices each
    params = set_default_params();
    testparms(argc,argv, params);
    assert( problem <= 2U );

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
    apply_testnum(params);
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
    if(problem<=2)
    {
        uint32_t n=0U;                    // example number
        DenseM verts(8 /* 2^3 */, 3);
        VectorXd corner(3);
        DenseM edges(3,3);
        switch(problem){
          case(0): // solution is x-axis, (y- and z-axis both overlap)
              problem_msg="unit cube, lengthened on z-axis";
              edges<<1,0,0,  0,1,0,  0,0,2;
              break;
          case(1): // soln is perp. to the skew dirn, so along (-1,0,1)
              problem_msg="rectangular prism skewed away from z-axis toward x-axis";
              // 3 skyscrapers along x axis with 45 degree lean,
              // so MAX between-class margin is to project back along x=z
              edges<<1,0,0,  0,1,0,  100,0,100;
              break;
          case(2): // SAME soln as case 1, but this fails to converge to correct solution.
              problem_msg="skyscrapers leaning 45-degrees along x=z";
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

    cout<<"  pre-run call to rand() returns "<<rand()<<endl;
    if( ! use_mcsolver ){
        cout<<" *** BEGIN SOLUTION *** original code: "<<problem_msg<<endl;
        // these calls are important so that the compiler instantiates the right templates
        DenseM weights, lower_bounds, upper_bounds;
        DenseM w_avg, l_avg, u_avg;
        VectorXd objective_val, o_avg;
        solve_optimization(weights,lower_bounds,upper_bounds,objective_val
                           ,w_avg,l_avg,u_avg,o_avg
                           ,x,y,params);
        check_solution( weights, lower_bounds, upper_bounds );  // throw on error
    }else{ // default: use new MCsolver code
        cout<<" *** BEGIN SOLUTION *** MCsolver code: "<<problem_msg<<endl;
        MCsolver mc;
        mc.solve( x, y, &params );
        MCsoln      & soln = mc.getSoln();
        DenseM      & weights = soln.weights;
        DenseM      & lower_bounds = soln.lower_bounds;
        DenseM      & upper_bounds = soln.upper_bounds;
        //DenseM const& w_avg = soln.weights_avg;
        //DenseM const& l_avg = soln.lower_bounds_avg;
        //DenseM const& u_avg = soln.upper_bounds_avg;
        //VectorXd const& objective_val = soln.objective_val;
        //VectorXd const& objective_val_avg = soln.objective_val_avg;
        if(1){
            cout<<"upper_bounds = "<<upper_bounds<<endl;
            check_solution( weights, lower_bounds, upper_bounds );  // throw on error
        }
        if( saveBasename.size() > 0U ){
            string saveTxt(saveBasename); saveTxt.append(".soln");
            cout<<" Saving to file "<<saveTxt<<endl;
            try{
                ofstream ofs(saveTxt);
                soln.write( ofs, MCsoln::TEXT, MCsoln::SHORT );
                ofs.close();
            }catch(std::exception const& e){
                cout<<"OHOH! Error during text write of demo.soln "<<e.what()<<endl;
                throw(e);
            }
            string saveBin(saveBasename); saveBin.append("-bin.soln");
            cout<<" Saving to file "<<saveBin<<endl;
            try{
                ofstream ofs(saveBin);
                soln.write( ofs, MCsoln::BINARY, MCsoln::SHORT );
                ofs.close();
            }catch(std::exception const& what){
                cout<<"OHOH! Error during binary write of demo.soln"<<endl;
                throw(what);
            }
        }
    }

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

