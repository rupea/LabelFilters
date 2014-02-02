#include <octave/oct.h>
#include <iostream>
#include <typeinfo>
#include "Eigen/Dense"
#include "Eigen/Sparse"
#include "find_w.cpp"
#include "EigenOctave.cpp"

using Eigen::VectorXd;

void test(SparseM& x)
{
	std::cout << x.rows();
}

void test(DenseM& x)
{
	std::cout << x.rows();
}

void test(Eigen::EigenBase<SparseM >& x)
{
	test(x);
}

void test(Eigen::EigenBase<DenseM>& x)
{
	test(x);
}

void print_usage()
{
  cout << "oct_find_w x y C1 C2 w_init [l_init u_init]" << endl;
  cout << "     x - the data matrix (can be dense or sparse)" << endl;
  cout << "     y - the label vector (same size as rows(x))" << endl;
  cout << "     C1 - the penalty for an example being outside it's class bounary" << endl;
  cout << "     C2 - the penalty for an example being inside other class' boundary" << endl;
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

  int nargin = args.length();
  if (nargin == 0)
    {
      print_usage();
      return octave_value_list(0);
    }
  
  FloatNDArray yArray = args(1).float_array_value(); // the label vector
  FloatNDArray C1Array = args(2).float_array_value(); // C1 value
  FloatNDArray C2Array = args(3).float_array_value(); // C2 value
  FloatNDArray wArray = args(4).float_array_value(); // The initial weights

  bool resumed = false;
  FloatNDArray lArray,uArray;
  if (nargin == 7)
    {	    
      lArray = args(5).float_array_value(); // optional the initial lower bounds
      uArray = args(6).float_array_value(); // optional the initial upper bounds 
      resumed = true; 
    }

  cout << "copying data starts ...\n";

  VectorXd y = toEigenVec(yArray);
  double C1 = C1Array(0,0);
  double C2 = C2Array(0,0);
  DenseM w = toEigenMat(wArray);
  DenseM l,u;
  if (nargin == 7)
    {
      l = toEigenMat(lArray);
      u = toEigenMat(uArray);
    }
  
  VectorXd objective_vals;

  /* initialize random seed: */
  srand (time(NULL));

  if(args(0).is_sparse_type())
    {
      // Sparse data
      Sparse<double> xArray = args(0).sparse_matrix_value();

      SparseM x = toEigenMat(xArray);

      solve_optimization(w,l,u,objective_vals,x,y,C1,C2, resumed);
    }
  else
    {
      // Dense data
      FloatNDArray xArray = args(0).float_array_value();
      DenseM x = toEigenMat(xArray);

      solve_optimization(w,l,u,objective_vals,x,y,C1,C2, resumed);
    }

  octave_value_list retval(4);// return value

  retval(0) = toMatrix(w);
  retval(1) = toMatrix(l);
  retval(2) = toMatrix(u);
  retval(3) = toMatrix(objective_vals);
  return retval;
}
