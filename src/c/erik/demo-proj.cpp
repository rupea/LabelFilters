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

#define TESTNUM 0
//#define PROBLEM 2U
param_struct params = set_default_params();
int testnum = TESTNUM;
bool use_mcsolver = true;
std::string saveBasename = std::string("");
//size_t problem = 2U;
string problem_msg("random data");
int verbose = 0;

// parse arguments
int testparms(int argc, char**argv, param_struct & params) {
    //problem = PROBLEM;
    testnum = TESTNUM;
    use_mcsolver = true;
    for(int a=1; a<argc; ++a){
        for(char* ca=argv[a]; *ca!='\0'; ++ca ){
            if(*ca == 'h'){
                cout<<" Options:"
                    //<<"\n   pN          test problem N in [0,2] w/ base parameter settings [default=2]"
                    <<"\n   0-9         parameter modifiers, [default=0, no change from pN defaults]"
                    <<"\n   m           [default] mcsolver code"
                    <<"\n   o           original code"
                    <<"\n   s<string>   save file base name (only for 'm') [default=none]"
                    <<"\n   h           this help"
                    <<"\n   v           increase verbosity [default=0]"
                    <<endl;
            }else if(isdigit(*ca))      testnum = *ca - '0';
            else if(*ca == 'v')         ++verbose;
            else if(*ca == 'o')         use_mcsolver=false;
            else if(*ca == 'm')         use_mcsolver=true;
            //else if(*ca == 'p')         problem = *++ca - '0';
            else if(*ca == 's'){
                saveBasename.assign(++ca);
                break;
            }
        }
    }
    //if(problem>2U) problem=2U;
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
          params.optimizeLU_epoch = 0;
          cout<<" params.optimizeLU_epoch = "<<params.optimizeLU_epoch;
          break;
      case(2):                // mcsolver does not run correctly
          //mcsolver OH? luPerm.nAccSortlu_avg not > 0 for t>=1200
          //mcsolver 'luPerm.ok_sortlu_avg' failed  at end of first dim (after t=100000)
          params.avg_epoch = 1200;    
          cout<<" params.avg_epoch = "<<params.avg_epoch;
          break;
      case(3):        // make nAccSortlu_avg increment (for sure)
          params.optimizeLU_epoch = 0;
          params.avg_epoch = 16000;
          cout<<" params.optimizeLU_epoch = "<<params.optimizeLU_epoch
              <<" params.avg_epoch = "<<params.avg_epoch;
          break;
      case(4):                // original annd mcsolver BOTH do not run correctly
          //mcsolver OH? luPerm.nAccSortlu_avg not > 0 for t>=1200
          //mcsolver 'luPerm.ok_sortlu_avg' failed  at t=1400
          params.avg_epoch = 1200;
          params.report_avg_epoch = 1400;
          cout<<" params.avg_epoch = "<<params.avg_epoch;
          break;
      case(5):
          params.reorder_type = REORDER_PROJ_MEANS;
          cout<<" params.reorder_type = "<<tostring(params.reorder_type);
          break;
      case(6):
          params.reorder_type = REORDER_RANGE_MIDPOINTS;
          cout<<" params.reorder_type = "<<tostring(params.reorder_type);
          break;
      case(7):        // make nAccSortlu_avg increment (for sure)
          params.reorder_epoch = 1000;   // default
          params.optimizeLU_epoch = 0;
          params.report_avg_epoch = 9999;
          cout<<" params.optimizeLU_epoch = "<<params.optimizeLU_epoch
              <<" params.avg_epoch = "<<params.avg_epoch;
          break;
      case(8):        // make nAccSortlu_avg increment (for sure)
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

int main(int argc,char** argv)
{

#ifdef _OPENMP
  cout<<" _OPENMP defined, omp_get_max_threads ... "<<omp_get_max_threads()<<endl;
#endif

  //  DenseM weights(40000,1),lower_bounds(1000,1),upper_bounds(1000,1), x(10000,40000);
  //  VectorXd y(10000),objective_val;
  params = set_default_params();
  testparms(argc,argv, params);
  params.no_projections = 4U;
  params.max_iter=100000U;              // default 1e6 takes 10-15 minutes
  apply_testnum(params);

  int const rand_seed = 117;
  int const p = params.no_projections;
  int const d = 12345U;    // x training data dimensionality
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

  if( ! use_mcsolver ){
      cout<<" *** BEGIN SOLUTION *** original code: "<<problem_msg<<endl;
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
  }else{
        cout<<" *** BEGIN SOLUTION *** MCsolver code "<<problem_msg<<endl;
        cout<<"  pre-run call to rand() returns "<<rand()<<endl;
        MCsolver mc;
        mc.solve( x, y, &params );
        MCsoln      & soln = mc.getSoln();
        //DenseM      & weights = soln.weights;
        //DenseM const& lower_bounds = soln.lower_bounds;
        //DenseM const& upper_bounds = soln.upper_bounds;
        //DenseM const& w_avg = soln.weights_avg;
        //DenseM const& l_avg = soln.lower_bounds_avg;
        //DenseM const& u_avg = soln.upper_bounds_avg;
        //VectorXd const& objective_val = soln.objective_val;
        //VectorXd const& objective_val_avg = soln.objective_val_avg;
        if( saveBasename.size() > 0U ){
            string saveTxt(saveBasename); saveTxt.append(".soln");
            cout<<" Saving to file "<<saveTxt<<endl;
            try{
                ofstream ofs(saveTxt);
                soln.write( ofs, MCsoln::TEXT, MCsoln::SHORT );
                ofs.close();
            }catch(std::exception const& what){
                cout<<"OHOH! Error during text write of demo.soln"<<endl;
                throw(what);
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

