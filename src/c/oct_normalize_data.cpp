#include <octave/oct.h>
#include <iostream>
#include "Eigen/Dense"
#include "Eigen/Sparse"
#include "normalize.h"
#include "EigenOctave.h"

using Eigen::MatrixXd;

DEFUN_DLD (oct_normalize_data, args, nargout,
    "returns the normalized data")
  {
    octave_value_list retval(1);

    DenseM opt = toEigenMat<DenseM>(args(1).float_array_value());

    if(args(0).is_sparse_type())
      {
        Sparse<double> xArray = args(0).sparse_matrix_value();
        SparseM x = toEigenMat(xArray);

        if(opt(0,0)==1)
          {
	    cout << "normalize";fflush(stdout);
            normalize_row(x);
          }
        else
          {
	    cout << "normalize_col";fflush(stdout);
	    normalize_col(x);
          }

        retval(0) = toMatrix(x);
      }
    else
      {
        FloatNDArray xArray = args(0).float_array_value();
        DenseM x = toEigenMat<DenseM>(xArray);
        normalize_col(x);
        retval(0) = toMatrix(x);
      }

    return retval;
  }
