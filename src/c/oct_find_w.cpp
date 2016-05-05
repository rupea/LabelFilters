#include <octave/oct.h>
#include <octave/ov-struct.h>
#include <iostream>
#include <typeinfo>
#include "Eigen/Dense"
#include "Eigen/Sparse"
#include "find_w.hh"
#include "EigenOctave.h"

using namespace std;

void print_usage()
{
  cout << "[w l u obj_val w_last l_last u_last] = oct_find_w x y parameters w_init [l_init u_init w_last_init l_last_init u_last_init]" << endl;
  cout << "     w - the projection matrix" << endl;
  cout << "     l - the lower bound for each class and each projection" << endl;
  cout << "     u - the upper bound for each class and each projection" << endl;
  cout << "     obj_val - vector of objective values during learning" << endl;
  cout << "     w_last - the projection matrix from the last iteration (same as w if averaging is turned off)" << endl;
  cout << "     l_last - the lower bound for each class and each projection (same as l if averaging is turned off)" << endl;
  cout << "     u_last - the upper bound for each class and each projection (same as u if averaging is turned off)" << endl;
  cout << "     x - the data matrix (can be dense or sparse)" << endl;
  cout << "     y - a label vector (same size as rows(x)) with elements 1:noClasses" << endl;
  cout << "          or a sparse label matrix of size rows(x)*noClasses with y(i,j)=1 meaning that example i has class j" << endl;
  print_parameter_usage();
  
  cout << "     w_prev - previous w matrix. Used for filtering in case resume=true" << endl;
  cout << "     l_prev - previous l matrix. Used for filtering in case resume=true" << endl;
  cout << "     u_prev - previous u matrix. Used for filtering in case resume=true" << endl;
  cout << "     w_last_prev - previous w_last matrix. Not used, just returned as the first columns of w_last" << endl;
  cout << "     l_last_prev - previous l_last matrix. Not used, just returned as the first columns of l_last" << endl;
  cout << "     u_last_prev - previous u_last matrix. Not used, just returned as the first columns of u_last" << endl;  
  cout << " If resume true new projections will be trained. w_prev, l_prev and u_prev will be used as the first projections, and they will be used to filter the data if remove_constraint is true." << endl;
}


DEFUN_DLD (oct_find_w, args, nargout,
		"Interface to find_w; optimizes the objective to find w")
{

#ifdef _OPENMP
  Eigen::initParallel();
  cout << "initialized Eigen parallel"<<endl;
#endif  

  int nargin = args.length();
  if (nargin == 0)
    {
      print_usage();
      return octave_value_list(0);
    }
  
  param_struct params = set_default_params();
    
  octave_scalar_map parameters = args(2).scalar_map_value(); // the parameter
  octave_value tmp;
  if (! error_state)
    {
      tmp = parameters.contents("no_projections");
      if (tmp.is_defined())
	{
	  params.no_projections=tmp.int_value();
	}
      tmp = parameters.contents("C1");
      if (tmp.is_defined())
	{
	  params.C1=tmp.double_value();
	}
      tmp = parameters.contents("C2");
      if (tmp.is_defined())
	{
	  params.C2=tmp.double_value();
	}
      tmp = parameters.contents("max_iter");
      if (tmp.is_defined())
	{
	  params.max_iter=tmp.int_value();
	}
      tmp = parameters.contents("eta");
      if (tmp.is_defined())
	{
	  params.eta=tmp.double_value();
	}
      tmp = parameters.contents("seed");
      if (tmp.is_defined())
	{
	  params.seed=tmp.int_value();
	}
      tmp = parameters.contents("num_threads");
      if (tmp.is_defined())
	{
	  params.num_threads=tmp.int_value();
	}
      tmp = parameters.contents("resume");
      if (tmp.is_defined())
	{
	  params.resume=tmp.bool_value();
	}
      tmp = parameters.contents("reoptimize_LU");
      if (tmp.is_defined())
	{
	  params.reoptimize_LU=tmp.bool_value();
	}
      tmp = parameters.contents("class_samples");
      if (tmp.is_defined())
	{
	  params.class_samples=tmp.int_value();
	}
      tmp = parameters.contents("update_type");
      if (tmp.is_defined())
	{
	  if (tmp.is_numeric_type()){
	    if (tmp.int_value() > 1){
	      cerr << "ERROR: update_type value unrecognized" << endl;
	      exit(-4);
	    }	    	       
	    params.update_type=static_cast<enum Update_Type>(tmp.int_value());
	  }
	  else{
	    if (tmp.string_value() == "minibatch")
	      params.update_type = MINIBATCH_SGD;
	    else if (tmp.string_value() == "safe") 
	      {
		params.update_type = SAFE_SGD;
		params.batch_size = 1;
	      }
	    else 
	      {
		cerr << "ERROR: update_type value unrecognized" << endl;
		exit(-4);
	      }
	  }
	}
      tmp = parameters.contents("batch_size");
      if (tmp.is_defined())
	{
	  if (params.update_type == SAFE_SGD && tmp.int_value() != 1)
	    {
	      cerr << "ERROR: batch_size must be 1 with update_type = safe!" << endl;  
	      exit(-4);
	    }
	  params.batch_size=tmp.int_value();
	}
      tmp = parameters.contents("eps");
      if (tmp.is_defined())
	{
	  params.eps=tmp.double_value();
	}
      tmp = parameters.contents("eta_type");
      if (tmp.is_defined())
	{
	  if (tmp.is_numeric_type()){
	    if (tmp.int_value() > 3){
	      cerr << "ERROR: eta_type value unrecognized" << endl;
	      exit(-4);
	    }	    	       
	    params.eta_type=static_cast<enum Eta_Type>(tmp.int_value());
	  }
	  else{
	    if (tmp.string_value() == "const")
	      params.eta_type = ETA_CONST;
	    else if (tmp.string_value() == "sqrt") 
	      params.eta_type = ETA_SQRT;
	    else if (tmp.string_value() == "lin") 
	      params.eta_type = ETA_LIN;
	    else if (tmp.string_value() == "3_4")
	      params.eta_type = ETA_3_4;
	    else 
	      {
		cerr << "ERROR: eta_type value unrecognized" << endl;
		exit(-4);
	      }
	  }
	}
      tmp = parameters.contents("init_type");
      if (tmp.is_defined())
	{
	  if (tmp.is_numeric_type()){
	    if (tmp.int_value() > 3){
	      cerr << "ERROR: init_type value unrecognized" << endl;
	      exit(-4);
	    }	    	       
	    params.init_type=static_cast<enum Init_W_Type>(tmp.int_value());
	  }
	  else{
	    if (tmp.string_value() == "zero"){
	      params.init_type == INIT_ZERO;
	      params.init_norm = -1; // do not renormalize by default
	      params.init_orthogonal = false; // do not orthogonalize by default
	    }
	    else if (tmp.string_value() == "prev"){ 
	      params.init_type = INIT_PREV;
	      params.init_norm = -1; // do not renormalize by default
	      params.init_orthogonal = false; // do not orthogonalize by default
	    }
	    else if (tmp.string_value() == "random") 
	      params.init_type = INIT_RANDOM;
	    else if (tmp.string_value() == "diff")
	      params.init_type = INIT_DIFF;
	    else 
	      {
		cerr << "ERROR: init_type value unrecognized" << endl;
		exit(-4);
	      }
	  }
	}
      tmp = parameters.contents("init_norm");
      if (tmp.is_defined())
	{
	  params.init_norm=tmp.double_value();
	}      
      tmp = parameters.contents("init_orthogonal");
      if (tmp.is_defined())
	{
	  params.resume=tmp.bool_value();
	}
      tmp = parameters.contents("min_eta");
      if (tmp.is_defined())
	{
	  params.min_eta=tmp.double_value();
	}
      tmp = parameters.contents("avg_epoch");
      if (tmp.is_defined())
	{
	  params.avg_epoch=tmp.int_value();
	}
      tmp = parameters.contents("report_epoch");
      if (tmp.is_defined())
	{
	  params.report_epoch=tmp.int_value();
	}
      tmp = parameters.contents("report_avg_epoch");
      if (tmp.is_defined())
	{
	  params.report_avg_epoch=tmp.int_value();
	}
      tmp = parameters.contents("reorder_epoch");
      if (tmp.is_defined())
	{
	  params.reorder_epoch=tmp.int_value();
	}
      tmp = parameters.contents("reorder_type");
      if (tmp.is_defined())
	{
	  if (tmp.is_numeric_type()){
	    if (tmp.int_value() > 2){
	      cerr << "ERROR: reorder_type value unrecognized" << endl;
	      exit(-4);
	    }	    	       
	    params.reorder_type=static_cast<enum Reorder_Type>(tmp.int_value());
	  }
	  else{
	    if (tmp.string_value() == "avg_proj_means")
	      params.reorder_type = REORDER_AVG_PROJ_MEANS;
	    else if (tmp.string_value() == "proj_means") 
	      params.reorder_type = REORDER_PROJ_MEANS;
	    else if (tmp.string_value() == "range_midpoints") 
	      params.reorder_type = REORDER_RANGE_MIDPOINTS;
	    else 
	      {
		cerr << "ERROR: reorder_type value unrecognized" << endl;
		exit(-4);
	      }
	  }
	}
      tmp = parameters.contents("optimizeLU_epoch");
      if (tmp.is_defined())
	{
	  params.optimizeLU_epoch=tmp.int_value();
	}
      tmp = parameters.contents("remove_constraints");
      if (tmp.is_defined())
	{
	  params.remove_constraints=tmp.bool_value();
	}
      tmp = parameters.contents("remove_class_constraints");
      if (tmp.is_defined())
	{
	  params.remove_class_constraints=tmp.bool_value();
	}
      tmp = parameters.contents("reweight_lambda");
      if (tmp.is_defined())
	{
	  if (tmp.is_numeric_type()){
	    if (tmp.int_value() > 2){
	      cerr << "ERROR: reweight_lambda int value of " << tmp.int_value() << " unrecognized" << endl;
	      exit(-4);
	    }	    	       
	    params.reweight_lambda=static_cast<enum Reweight_Type>(tmp.int_value());
	  }else{
	    if (tmp.string_value() == "none")
	      params.reweight_lambda = REWEIGHT_NONE;
	    else if (tmp.string_value() == "lambda") 
	      params.reweight_lambda = REWEIGHT_LAMBDA;
	    else if (tmp.string_value() == "all") 
	      params.reweight_lambda = REWEIGHT_ALL;
	    else 
	      {
		cerr << "ERROR: reweight_lambda value " << tmp.string_value() << " unrecognized" << endl;
		exit(-4);
	      }
	  }
	  
	    
	}
      tmp = parameters.contents("ml_wt_by_nclasses");
      if (tmp.is_defined())
	{
	  params.ml_wt_by_nclasses=tmp.bool_value();
	}
      tmp = parameters.contents("ml_wt_class_by_nclasses");
      if (tmp.is_defined())
	{
	  params.ml_wt_class_by_nclasses=tmp.bool_value();
	}
#if GRADIENT_TEST
      tmp = parameters.contents("finite_diff_test_epoch");
      if (tmp.is_defined())
	{
	  params.finite_diff_test_epoch=tmp.int_value();
	}
      tmp = parameters.contents("no_finite_diff_tests");
      if (tmp.is_defined())
	{
	  params.no_finite_diff_tests=tmp.int_value();
	}
      tmp = parameters.contents("finite_diff_test_delta");
      if (tmp.is_defined())
	{
	  params.finite_diff_test_delta=tmp.double_value();
	}
#endif
    }


  
  DenseM w, l, u, w_avg, l_avg, u_avg;
  VectorXd objective_vals, objective_vals_avg;

  // get the parameters from previous runs

  if (nargin >= 4)
    {
      w_avg = toEigenMat<DenseM>(args(3).array_value()); // The weights from previous run
    }
  if (nargin >= 6)
    {
      l_avg = toEigenMat<DenseM>(args(4).array_value()); // the lower bounds from previous run
      u_avg = toEigenMat<DenseM>(args(5).array_value()); // the upper bounds from previous run
      assert(l_avg.cols() == w_avg.cols());
      assert(u_avg.cols() == w_avg.cols());
    }
  if (nargin >= 7)
    {
      objective_vals_avg = toEigenVec(args(6).array_value()); // the obj vals from previous run
    }
      
  if (nargin >= 8)
    {
      w = toEigenMat<DenseM>(args(7).array_value()); // The weights from previous run
      assert(w.cols() == w_avg.cols());
      assert(w.rows() == w_avg.rows());
    }
  else
    {
      // if w was not passed use w_avg for completeness
      w = w_avg;
    }

  if (nargin >= 10)
    {
      l = toEigenMat<DenseM>(args(8).array_value()); // the lower bounds from previous run
      u = toEigenMat<DenseM>(args(9).array_value()); // the upper bounds from previous run
      assert(l.cols() == w.cols());
      assert(u.cols() == w.cols());
    }
  else
    {
      // if l,u was not passed use l_avg,u_avg for completeness
      l = l_avg;
      u = u_avg;
    }
  
  if (nargin >= 11)
    {
      objective_vals = toEigenVec(args(10).array_value()); // the obj vals from previous run
    }
  else
    {
      // if objective_vals was not passed use objective_vals_avg for completeness
      objective_vals = objective_vals_avg;
    }
  
  if (params.resume && !params.reoptimize_LU && l_avg.cols()!=w_avg.cols())
    {
      cerr << "ERROR: asked to resume, but l_avg and u_avg do not match w_avg" << endl;
      exit(-4);
    }
 

  cout << "copying data starts ...\n";

  
  

  /* initialize random seed: */
  if (params.seed)
    {
      srand(params.seed);
    }
  else
    {
      srand (time(NULL));
    }

  SparseMb y;
  if (args(1).is_sparse_type())
    {
      Sparse<bool> yArray = args(1).sparse_bool_matrix_value(); 
      y = toEigenMat(yArray);
    }
  else
    {      
      FloatNDArray yVector = args(1).float_array_value(); // the label vector
      
      VectorXd yVec = toEigenVec(yVector);
  
      y = labelVec2Mat(yVec);
    }
      
  if(args(0).is_sparse_type())
    {
      // Sparse data
      SparseM x = toEigenMat(args(0).sparse_matrix_value());
      
      solve_optimization(w, l, u, objective_vals, w_avg, l_avg, u_avg, objective_vals_avg, x, y, params);
    }
  else
    {
      // Dense data
      FloatNDArray xArray = args(0).float_array_value();
      DenseM x = toEigenMat<DenseM>(xArray);

      solve_optimization(w, l, u, objective_vals, w_avg, l_avg, u_avg, objective_vals_avg, x, y, params);
    }

  octave_value_list retval(8);// return value
  retval(0) = toMatrix(w_avg);
  retval(1) = toMatrix(l_avg);
  retval(2) = toMatrix(u_avg);
  retval(3) = toMatrix(objective_vals_avg);
  retval(4) = toMatrix(w);
  retval(5) = toMatrix(l);
  retval(6) = toMatrix(u);
  retval(7) = toMatrix(objective_vals);
  return retval;
}
