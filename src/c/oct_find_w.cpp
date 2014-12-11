#include <octave/oct.h>
#include <octave/ov-struct.h>
#include <iostream>
#include <typeinfo>
#include "Eigen/Dense"
#include "Eigen/Sparse"
#include "find_w.h"
#include "EigenOctave.h"
//#include "parameter.h"

using Eigen::VectorXd;


void print_usage()
{
  cout << "[w l u obj_val w_last l_last u_last] = oct_find_w x y parameters w_init [l_init u_init]" << endl;
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
  cout << "     parameters - a structure with the optimization parameters. If a parmeter is not present the default is used" << endl;
  cout << "         Parameters (structure field names) are:" << endl;
  cout << "           no_projections - nubmer of projections to be learned [5]" << endl;
  cout << "           C1 - the penalty for an example being outside it's class bounary" << endl;
  cout << "           C2 - the penalty for an example being inside other class' boundary" << endl;
  cout << "           max_iter - maximum number of iterations [1e^6]" << endl;
  cout << "           batch_size - size of the minibatch [1000]" << endl;
  cout << "           avg_epoch - iteration to start averaging at. 0 for no averaging [0]" << endl;
  cout << "           reorder_epoch - number of iterations between class reorderings. 0 for no reordering of classes [1000]" << endl;
  cout << "           report_epochs - number of iterations between computation and report the objective value (can be expensive because obj is calculated on the entire training set). 0 for no reporting [1000]." << endl;
  cout << "           eta_type - the type of learning rate decay. One of: \"const\" (eta) \"sqrt\" (eta/sqrt(t)), \"lin\" (eta/(1+eta*lambda*t)), \"3_4\" (eta*(1+eta*lambda*t)^(-3/4)" << endl;
  cout << "           eta - the initial learning rate. The leraning rate is eta/sqrt(t) where t is the number of iterations [1]" << endl;
  cout << "           min_eta - the minimum value of the lerarning rate (i.e. lr will be max (eta/sqrt(t), min_eta)  [1e-4]" << endl;
  cout << "           remove_constraints - whether to remove the constraints for instances that fall outside the class boundaries in previous projections. [false] " << endl;
  cout << "           remove_class_constraints - whether to remove the constraints for examples that fell outside their own class boundaries in previous projections. [false] " << endl;
  cout << "           rank_by_mean - whether to rank the classes by the mean of the projected examples or by the midpoint of its [l,u] interval (i.e. (u-l)/2). [true]" << endl;
  cout << "           ml_wt_by_nclasses - whether to weight an example by the number of classes it belongs to when conssidering other class contraints. [false]" << endl;
  cout << "           ml_wt_class_by_nclasses - whether to weight an example by the number of classes it belongs to when conssidering its class contraints.[false]" << endl;
  cout << "           seed - random seed. 0 for time dependent seed. [0]" << endl;
  cout << "           num_threads - number of threads to run on. Negative value for architecture dependent maximum number of threads. [-1]" << endl;
  cout << "           finite_diff_test_epoch - number of iterations between testign the gradient with finite differences. 0 for no testing [0]" << endl;
  cout << "           no_finite_diff_tests - number of instances to perform the finite differences test at each testing round. The instances are randomly picked from the training set. [1]" << endl;
  cout << "           finite_diff_test_delta - the size of the finite difference. [1e-2]" << endl;
  cout << "     w_init - initial w vector" << endl;
  cout << "     l_init - initial lower bounds (optional)" << endl;
  cout << "     u_init - initial upper bounds (optional)" << endl;
  cout << " If l_init and u_init are specified, the class order will be based on l_init and u_init." << endl;
  cout << " If they are specified it is important that they are not random but rather values saved" << endl;
  cout << " from an earlier run. The indended use is to allow resuming the optmization if it had not" << endl;
  cout << " converged." << endl ;
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
      tmp = parameters.contents("avg_epoch");
      if (tmp.is_defined())
	{
	  params.avg_epoch=tmp.int_value();
	}
      tmp = parameters.contents("eps");
      if (tmp.is_defined())
	{
	  params.eps=tmp.double_value();
	}
      tmp = parameters.contents("eta_type");
      if (tmp.is_defined())
	{
	  if (tmp.string_value() == "const")
	    params.eta_type = ETA_CONST;
	  else if (tmp.string_value() == "sqrt") 
	    params.eta_type = ETA_SQRT;
	  else if (tmp.string_value() == "lin") 
	    params.eta_type = ETA_LIN;
	  else if (tmp.string_value() == "3/4")
	    params.eta_type = ETA_3_4;
	  else 
	    {
	      cerr << "ERROR: eta_type value unrecognized" << endl;
	      exit(-4);
	    }
	}
      tmp = parameters.contents("eta");
      if (tmp.is_defined())
	{
	  params.eta=tmp.double_value();
	}
      tmp = parameters.contents("min_eta");
      if (tmp.is_defined())
	{
	  params.min_eta=tmp.double_value();
	}
      tmp = parameters.contents("batch_size");
      if (tmp.is_defined())
	{
	  params.batch_size=tmp.int_value();
	}
      tmp = parameters.contents("report_epoch");
      if (tmp.is_defined())
	{
	  params.report_epoch=tmp.int_value();
	}
      tmp = parameters.contents("reorder_epoch");
      if (tmp.is_defined())
	{
	  params.reorder_epoch=tmp.int_value();
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
      tmp = parameters.contents("rank_by_mean");
      if (tmp.is_defined())
	{
	  params.rank_by_mean=tmp.bool_value();
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
    }

  FloatNDArray wArray = args(3).float_array_value(); // The initial weights

  bool resumed = false;
  FloatNDArray lArray,uArray;
  if (nargin == 6)
    {	    
      lArray = args(4).float_array_value(); // optional the initial lower bounds
      uArray = args(5).float_array_value(); // optional the initial upper bounds 
      resumed = true; 
    }

  cout << "copying data starts ...\n";

  DenseM w = toEigenMat<DenseM>(wArray);
  DenseM l,u, w_avg, l_avg, u_avg;
  
  if (resumed)
    {
      l = toEigenMat<DenseM>(lArray);
      u = toEigenMat<DenseM>(uArray);
    }
  
  VectorXd objective_vals;

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

      solve_optimization(w, l, u, w_avg, l_avg, u_avg, objective_vals, x, y, resumed, params);
    }
  else
    {
      // Dense data
      FloatNDArray xArray = args(0).float_array_value();
      DenseM x = toEigenMat<DenseM>(xArray);

      solve_optimization(w, l, u, w_avg, l_avg, u_avg, objective_vals, x, y, resumed, params);
    }

  octave_value_list retval(7);// return value
  retval(0) = toMatrix(w_avg);
  retval(1) = toMatrix(l_avg);
  retval(2) = toMatrix(u_avg);
  retval(3) = toMatrix(objective_vals);
  retval(4) = toMatrix(w);
  retval(5) = toMatrix(l);
  retval(6) = toMatrix(u);
  return retval;
}
