#include <octave/oct.h>
#include <iostream>
#include <typeinfo>
#include "Eigen/Dense"
#include "Eigen/Sparse"
#include "find_w.cpp"
#include "EigenOctave.cpp"

using Eigen::MatrixXd;
using Eigen::VectorXd;

void test(SparseM& x)
{
	std::cout << x.rows();
}

void test(MatrixXd& x)
{
	std::cout << x.rows();
}

void test(Eigen::EigenBase<SparseM >& x)
{
	test(x);
}

void test(Eigen::EigenBase<MatrixXd>& x)
{
	test(x);
}

DEFUN_DLD (oct_find_w, args, nargout,
		"Interface to find_w; optimizes the objective to find w")
{
	FloatNDArray yArray = args(1).float_array_value();
	FloatNDArray C1Array = args(2).float_array_value();
	FloatNDArray C2Array = args(3).float_array_value();
	FloatNDArray wArray = args(4).float_array_value();

	cout << "copying data starts ...\n";

	VectorXd y = toEigenVec(yArray);
	double C1 = C1Array(0,0);
	double C2 = C2Array(0,0);
	MatrixXd w = toEigenMat(wArray);

	VectorXd objective_vals;
	MatrixXd l,u;

	if(args(0).is_sparse_type())
	{
		// Sparse data
		Sparse<double> xArray = args(0).sparse_matrix_value();
		SparseM w_gradient(1,xArray.cols());

		SparseM x = toEigenMat(xArray);

		solve_optimization< SparseM >(w,l,u,objective_vals,x,y,C1,C2,w_gradient);
	}
	else
	{
		// Dense data
		FloatNDArray xArray = args(0).float_array_value();
		MatrixXd x = toEigenMat(xArray);
		MatrixXd w_gradient(1,xArray.cols());

		solve_optimization< MatrixXd >(w,l,u,objective_vals,x,y,C1,C2,w_gradient);
	}

	octave_value_list retval(4);// return value

	retval(0) = toMatrix(w);
	retval(1) = toMatrix(l);
	retval(2) = toMatrix(u);
	retval(3) = toMatrix(objective_vals);
	return retval;
}
