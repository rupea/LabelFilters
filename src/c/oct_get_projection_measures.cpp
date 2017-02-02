#include <octave/oct.h>
#include <octave/parse.h>
#include <octave/ov-struct.h>
//#include <octave/builtin-defun-decls.h>
//#include <octave/oct-map.h>
#include <iostream>
//#include <typeinfo>
#include <time.h>
#include <boost/numeric/conversion/bounds.hpp>
#include <boost/limits.hpp>
#include "Eigen/Dense"
#include "Eigen/Sparse"
#include "EigenOctave.h"
//#include "predict.h"
#include "utils.h"
#include "evaluate.h"

void print_usage()
{
  cout << "[nrTrueActive, nrActive, nrTrue] = oct_get_projection_measures(x, y, w, l, u)" << endl;
  cout << "     x - the data matrix (can be dense or sparse)" << endl;
  cout << "     y - a label vector (same size as rows(x)) with elements 1:noClasses" << endl;
  cout << "          or a sparse label matrix of size rows(x)*noClasses with y(i,j)=1 meaning that example i has class j" << endl;
  cout << "     w - the projection vectors" << endl;
  cout << "     l - the lower bounds" << endl;
  cout << "     u - the upper bounds" << endl;
}


DEFUN_DLD (oct_get_projection_measures,  args, nargout,
		"Interface to get various measure about the projection")
{

#ifdef _OPENMP
  Eigen::initParallel();
  cout << "initialized Eigen parallel"<<endl;
#endif  

  int nargin = args.length();
  if (nargin != 5)
    {
      print_usage();
      return octave_value_list(0);
    }
  
  DenseColM wmat = toEigenMat<DenseColM>(args(2).array_value());
  DenseColM lmat = toEigenMat<DenseColM>(args(3).array_value());
  DenseColM umat = toEigenMat<DenseColM>(args(4).array_value());


  // should do these via options
  bool verbose = true;

  SparseMb y;
  if (args(1).is_sparse_type())
    {
      Sparse<bool> yArray = args(1).sparse_bool_matrix_value(); 
      y = toEigenMat(yArray);
    }
  else
    {      
      FloatNDArray yVector = args(1).array_value(); // the label vector
      
      Eigen::VectorXd yVec = toEigenVec(yVector);
  
      y = labelVec2Mat(yVec);
    }

  assert(lmat.rows() == y.cols());
  assert(umat.rows() == y.cols());
  
  VectorXsz nrTrueActive(y.cols());
  VectorXsz nrActive(y.cols());
  VectorXsz nrTrue(y.cols());
  

  if(args(0).is_sparse_type())
    {
      // Sparse data
      SparseM x = toEigenMat(args(0).sparse_matrix_value());

      get_projection_measures(x,y,wmat, lmat, umat, verbose, nrTrueActive, nrActive, nrTrue);      
    }
  else
    {
      // Dense data
      DenseM x = toEigenMat<DenseM>(args(0).array_value());
      get_projection_measures(x,y,wmat, lmat, umat, verbose, nrTrueActive, nrActive, nrTrue);      
    }
  
  octave_value_list retval(3);
  retval(0) = toInt64Array(nrTrueActive);
  retval(1) = toInt64Array(nrActive);
  retval(2) = toInt64Array(nrTrue);
  
  return retval;
}
