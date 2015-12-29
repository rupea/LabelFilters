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
  cout << "     parameters - a structure with the optimization parameters. If a parmeter is not present the default is used" << endl;
  cout << "         Parameters (structure field names) are:" << endl;
  cout << "           no_projections - nubmer of projections to be learned [5]" << endl;
  cout << "           C1 - the penalty for an example being outside it's class bounary" << endl;
  cout << "           C2 - the penalty for an example being inside other class' boundary" << endl;
  cout << "           max_iter - maximum number of iterations [1e^6]" << endl;
  cout << "           batch_size - size of the minibatch [1000]" << endl;
  cout << "           update_type - how to update w, L and U [minibatch]" << endl;
  cout << "                           minibatch - update w, L and U together using minibatch SGD" <<endl;
  cout << "                           safe - update w first without overshooting, then update L and U using projected gradient. batch_size will be set to 1" << endl;
  cout << "           avg_epoch - iteration to start averaging at. 0 for no averaging [0]" << endl;
  cout << "           reorder_epoch - number of iterations between class reorderings. 0 for no reordering of classes [1000]" << endl;
  cout << "           reorder_type - how to order the classes [avg_proj_mean]: " << endl;
  cout << "                           avg_proj_means reorder by the mean of the projection on the averaged w (if averaging has not started is the ame as proj_mean" << endl;
  cout << "                           proj_means reorder by the mean of the projection on the current w" << endl;
  cout << "                           range_midpoints reorder by the midpoint of the [l,u] interval (i.e. (u-l)/2)" << endl;
  cout << "           optimizeLU_epoch - number of iterations between full optimizations of  the lower and upper class boundaries. Expensive. 0 for no optimization [10000]" << endl;
  cout << "           report_epoch - number of iterations between computation and report the objective value (can be expensive because obj is calculated on the entire training set). 0 for no reporting [1000]." << endl;
  cout << "           report_avg_epoch - number of iterations between computation and report the objective value for the averaged w (this can be quite expensive if full optimization of LU is turned on, since it first fully optimize LU and then calculates the obj on the entire training set). 0 for no reporting [0]." << endl;
  cout << "           eta - the initial learning rate. The leraning rate is eta/sqrt(t) where t is the number of iterations [1]" << endl;
  cout << "           eta_type - the type of learning rate decay:[lin]" << endl;
  cout << "                        const (eta)" << endl;
  cout << "                        sqrt (eta/sqrt(t))" << endl;
  cout << "                        lin (eta/(1+eta*lambda*t))" << endl;
  cout << "                        3_4 (eta*(1+eta*lambda*t)^(-3/4)" << endl;
  cout << "           min_eta - the minimum value of the lerarning rate (i.e. lr will be max (eta/sqrt(t), min_eta)  [1e-4]" << endl;
  cout << "           remove_constraints - whether to remove the constraints for instances that fall outside the class boundaries in previous projections. [false] " << endl;
  cout << "           remove_class_constraints - whether to remove the constraints for examples that fell outside their own class boundaries in previous projections. [false] " << endl;
  cout << "           reweight_lambda - whether to diminish lambda and/or C1 as constraints are eliminated. 0 - do not diminish any, 1 - diminish lambda only, 2 - diminish lambda and C1 (increase C2) [1]." << endl;
  cout << "           ml_wt_by_nclasses - whether to weight an example by the number of classes it belongs to when conssidering other class contraints. [false]" << endl;
  cout << "           ml_wt_class_by_nclasses - whether to weight an example by the number of classes it belongs to when conssidering its class contraints.[false]" << endl;
  cout << "           seed - random seed. 0 for time dependent seed. [0]" << endl;
  cout << "           num_threads - number of threads to run on. Negative value for architecture dependent maximum number of threads. [-1]" << endl;
  cout << "           finite_diff_test_epoch - number of iterations between testign the gradient with finite differences. 0 for no testing [0]" << endl;
  cout << "           no_finite_diff_tests - number of instances to perform the finite differences test at each testing round. The instances are randomly picked from the training set. [1]" << endl;
  cout << "           finite_diff_test_delta - the size of the finite difference. [1e-2]" << endl;
  cout << "           resume - whether to continue with additional projections. Takes previous projections from w_prev l_prev and u_prev. [false]" << endl;
  cout << "           reoptimize_LU - optimize l and u for given projections w_prev. Implies resume is true (i.e. if no_projections > w_prev.cols() additional projections will be learned. [false]" << endl;
  cout << "           class_samples - the number of negative classes to sample for each example at each iteration. 0 to use all classes. [0]" << endl;
  
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
	  else if (tmp.string_value() == "3_4")
	    params.eta_type = ETA_3_4;
	  else 
	    {
	      cerr << "ERROR: eta_type value unrecognized" << endl;
	      exit(-4);
	    }
	}
      tmp = parameters.contents("update_type");
      if (tmp.is_defined())
	{
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
	  if (params.update_type == SAFE_SGD && tmp.int_value() != 1)
	    {
	      cerr << "ERROR: batch_size must be 1 with update_type = safe!" << endl;  
	      exit(-4);
	    }
	  params.batch_size=tmp.int_value();
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
	  params.reweight_lambda=tmp.int_value();
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
