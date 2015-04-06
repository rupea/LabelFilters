#include <octave/oct.h>
#include <octave/parse.h>
#include <octave/ov-struct.h>
//#include <octave/builtin-defun-decls.h>
//#include <octave/oct-map.h>
#include <iostream>
#include <typeinfo>
#include "Eigen/Dense"
#include "Eigen/Sparse"
#include "EigenOctave.h"
#include "predict.h"

void print_usage()
{
  cout << "oct_predict_data x models w l u" << endl;
  cout << "     x - the data matrix (can be dense or sparse)" << endl;
  cout << "     models - a cell array with the ova weight vectors (does not support bias for now)" << endl;
  cout << "              the weight vectors are in floats to save memory" << endl;
  cout << "     w - the projection vectors" << endl;
  cout << "     l - the lower bounds" << endl;
  cout << "     u - the upper bounds" << endl;
}




DEFUN_DLD (oct_predict_data, args, nargout,
		"Interface to make predictions on a set of vectors")
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
  
  DenseColM wmat = toEigenMat<DenseColM>(args(2).array_value());
  DenseColM lmat = toEigenMat<DenseColM>(args(3).array_value());
  DenseColM umat = toEigenMat<DenseColM>(args(4).array_value());
  DenseM projections;

  
  octave_value_list arg;
  arg(0) = args(1); // the file name
  arg(1) = "svm_models_final";
  
  cout << "Loading file " << arg(0).string_value() << " ... " <<endl;
  //  octave_value_list loaded = Fload(arg, 1);  
  octave_value_list loaded = feval("load", arg, 1);  
  cout << "Done loading" << endl;
  cout << loaded(0).scalar_map_value().fieldnames()[0] << endl;
    
  
  DenseColMf ovaW;
  toEigenMat(ovaW, loaded(0).scalar_map_value().getfield(arg(1).string_value()).cell_value());
  
  loaded.clear(); // clear memory

  ActiveDataSet* active;
  PredictionSet* predictions;
  size_t nact;
  vector<size_t> no_active(wmat.cols());
  if(args(0).is_sparse_type())
    {
      // Sparse data
      SparseM x = toEigenMat(args(0).sparse_matrix_value());
      
      active = getactive (no_active, x, wmat, lmat, umat);
      predictions = predict(x, ovaW, active, nact); 
    }
  else
    {
      // Dense data
      DenseM x = toEigenMat<DenseM>(args(0).float_array_value());
      active = getactive (no_active, x, wmat, lmat, umat);
      predictions = predict(x, ovaW, active, nact); 
    }

  SparseM* m = predictions->toSparseM();
  octave_value_list retval(1);
  retval(0) = toMatrix(*m);
  
  //cleanup. maybe these should be done in Active and Prediction classes
  delete m;
  delete predictions;
  for(ActiveDataSet::iterator actit = active->begin(); actit !=active->end();actit++)
    {
      delete (*actit);
    }
  delete active;

  return retval;
}

