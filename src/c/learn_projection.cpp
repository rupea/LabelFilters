#include <octave/oct.h> 
#include <octave/parse.h> 
#include <octave/oct-map.h>
//#include <octave/variables.h> 
#include <octave/builtin-defun-decls.h> // Fload
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
#include "printing.hh"
#include "utils.h"
#include "find_w.hh"
//#include "find_w_detail.hh"
#include "EigenOctave.h"


using Eigen::VectorXd;
using Eigen::VectorXi;

using namespace std;

// /* modeled on get_global_value in variables.cc; perhaps this 
//    should be there too */ 
// octave_value 
// get_value (const std::string& nm) 
// { 
//   octave_value retval; 

//   symbol_record *sr = curr_sym_tab->lookup (nm); 

//   if (sr) 
//     { 
//       octave_value sr_def = sr->def (); 

//       if (sr_def.is_undefined ()) 
//         error ("get_value: undefined symbol `%s'", nm.c_str ()); 
//       else 
//         retval = sr_def; 
//     } 
//   else 
//     error ("get_value: unknown symbol `%s'", nm.c_str ()); 

//   return retval; 
// } 


//extern DECLARE_FUN(load,args,nargout); 
int main(int argc, char * argv[])
{

#ifdef _OPENMP
  Eigen::initParallel();
  cout << "initialized Eigen parallel"<<endl;
#endif

  srand (438911);
 
  octave_value_list args; 
  //args(0)="~/Research/mcfilter/LSHTC-2014/data/LSHTC14train_minclass0_minfeat10_weighting_tfidf_normalization_row_trial1.mat"; 
  args(0)="~/Research/mcfilter/wiki10/data/wiki10_minclass0_minfeat1_coding_none_normalization_row.mat";
  //  args(0)="~/Research/mcfilter/ipc/ipc_full_db/data/ipc_minclass0_minfeat1_coding_none_normalization_row.mat"; 
  args(1)="x_tr"; 
  args(2)="y_tr"; 

  cout << "Loading file " << args(0).string_value() << " ... " <<endl;
  octave_value_list loaded = Fload(args, 1);
  //feval("load", args, 0); // no arguments returned 
  cout << "success" << endl; 

  cout << loaded(0).scalar_map_value().fieldnames()[0] << endl;

  cout << "Fetching variable (" << args(1).string_value() << ") from symbol table... " << endl; 
  octave_value x_tr = loaded(0).scalar_map_value().getfield(args(1).string_value()); //get_value("x_tr"); 
  // if (error_state) { 
  //   cout << "Error" << endl;
  //   exit (-1);
  // } 
  cout << "Fetching variable (" << args(2).string_value() << ") from symbol table... " << endl; 
  octave_value y_tr = loaded(0).scalar_map_value().getfield(args(2).string_value()); //get_value("y_tr"); 
  // if (error_state) { 
  //   cout << "Error" << endl;
  //   exit (-1);
  // } 

  
  VectorXd objective_val, objective_val_avg;
  SparseMb y;
  SparseMb smally;
  if (y_tr.is_sparse_type())
    {
      Sparse<bool> yArray = y_tr.sparse_bool_matrix_value(); 
      y = toEigenMat(yArray);      
    }
  else
    {      
      FloatNDArray yVector = y_tr.float_array_value(); // the label vector
      
      VectorXd yVec = toEigenVec(yVector);
  
      y = labelVec2Mat(yVec);
    }


  param_struct params = set_default_params();
  params.C2 = 0.5;
  params.C1 = y.cols() * 2;
  params.remove_constraints = true;
  params.max_iter = 1e+5;
  params.report_epoch = 1e+3; 
  params.reorder_epoch = 1e+2;
  params.update_type = SAFE_SGD;
  params.batch_size = 1;
  params.optimizeLU_epoch = params.max_iter;
  params.eta = 0.1;
  params.min_eta = 0;
  params.eta_type = ETA_3_4;
  params.avg_epoch = 30;
  params.class_samples = 0;
  params.reweight_lambda = REWEIGHT_ALL;
  params.num_threads = 8;
  params.no_projections = 1;

  // params.C2 = 100;
  // params.C1 = 200000;
  // params.remove_constraints = true;
  // params.max_iter = 1e+7;
  // params.report_epoch = 1e+6; 
  // params.reorder_epoch = 1e+6;
  // params.optimizeLU_epoch = params.max_iter;
  // params.update_type = SAFE_SGD;
  // params.batch_size = 1;
  // params.eta = 0.01;
  // params.min_eta = 0;
  // params.class_samples = 0;
  // params.reweight_lambda = 2;
  // params.num_threads = 1;
  // params.no_projections = 2;
  //smally = y.topLeftCorner(100000,y.cols());


  if(x_tr.is_sparse_type())
    {
      // Sparse data
      Sparse<double> xArray = x_tr.sparse_matrix_value();

      SparseM x = toEigenMat(xArray);

      xArray.~Sparse();

      size_t d = x.cols();
      size_t k = y.cols();
      
      //SparseM smallx = x.topLeftCorner(100000,d);

      DenseM w(d,2),l(k,2),u(k,2);
      w.setRandom();
      l.setZero();
      u.setZero();
      DenseM w_avg(d,2),l_avg(k,2),u_avg(k,2);
      w_avg.setRandom();
      l_avg.setZero();
      u_avg.setZero();
      
      solve_optimization(w, l, u, objective_val, w_avg, l_avg, u_avg, objective_val_avg, x, y, params);
    }
  else
    {
      // Dense data
      FloatNDArray xArray = x_tr.float_array_value();
      DenseM x = toEigenMat<DenseM>(xArray);

      size_t d = x.cols();
      size_t k = y.cols();
      
      DenseM w(d,2),l(k,2),u(k,2);
      w.setRandom();
      l.setZero();
      u.setZero();
      DenseM w_avg(d,2),l_avg(k,2),u_avg(k,2);
      w_avg.setRandom();
      l_avg.setZero();
      u_avg.setZero();
      
      solve_optimization(w, l, u, objective_val, w_avg, l_avg, u_avg, objective_val_avg, x, y, params);      
    }
  
  
}


# if 0

int main()
{

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
  solve_optimization(weights,lower_bounds,upper_bounds,objective_val,x,y,params);
  solve_optimization(weights,lower_bounds,upper_bounds,objective_val,xs,y,params);
  
  
  xs.conservativeResize(281,1123497);
  DenseM sweights (1123497,1);
  sweights.setRandom();
  solve_optimization(sweights,lower_bounds,upper_bounds,objective_val,xs,y,params);
  
}

#endif





