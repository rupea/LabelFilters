#include <octave/oct.h>
#include <iostream>
#include "Eigen/Dense"
#include "Eigen/Sparse"
#include "find_w.h"
#include "EigenOctave.h"

using Eigen::MatrixXd;
using Eigen::VectorXd;

// type definitions -- should all be done in one place

typedef Eigen::SparseMatrix<double, Eigen::RowMajor> SparseM;


DEFUN_DLD (oct_calc_objective, args, nargout,
	   "Calculates objective value of the optimization based on given w")
{

  FloatNDArray yArray = args(1).float_array_value();
  FloatNDArray wArray = args(2).float_array_value();
  FloatNDArray lArray = args(3).float_array_value();
  FloatNDArray uArray = args(4).float_array_value();
  FloatNDArray label_orderArray = args(5).float_array_value(); // be careful. The code assumes that this is an int vector.
  FloatNDArray C1Array = args(6).float_array_value();
  FloatNDArray C2Array = args(7).float_array_value();


  VectorXd y = toEigenVec(yArray);
  double C1  = C1Array(0,0);
  double C2  = C2Array(0,0);
  VectorXd w = toEigenVec(wArray);
  VectorXd label_order = toEigenVec(label_orderArray);
  VectorXd l = toEigenVec(lArray);
  VectorXd u = toEigenVec(uArray);
  
  double ret;

  if (args(0).is_sparse_type())
    {
      //Sparse data
      Sparse<double> xArray = args(0).sparse_matrix_value();
      SparseM x = toEigenMat(xArray);
      ret = calculate_objective_hinge <SparseM> (w,x,y,l,u,label_order,C1,C2) ;
    }
  else
    {
      //Dense data     
      FloatNDArray xArray = args(0).float_array_value();
      MatrixXd x = toEigenMat(xArray);
      ret = calculate_objective_hinge <MatrixXd> (w,x,y,l,u,label_order,C1,C2) ;
    }


  // nor really necessary to go through MatrixXd
  //  MatrixXd d(1,1);  
  // d(0,0)=ret;

  octave_value_list retval(1);
  retval(0) = ret; //toMatrix(d);

  return retval;
}
