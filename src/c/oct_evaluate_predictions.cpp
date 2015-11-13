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
#include "predict.h"
#include "utils.h"
#include "evaluate.h"

void print_usage()
{
  cout << "oct_evaluate_predictions preds y" << endl;
  cout << "     preds - the preditions matrix n_instances x n_classes (dense or sparse)" << endl;
  cout << "     y - a label vector (same size as rows(x)) with elements 1:noClasses" << endl;
  cout << "          or a sparse label matrix of size rows(x)*noClasses with y(i,j)=1 meaning that example i has class j" << endl;
}


DEFUN_DLD (oct_evaluate_predictions, args, nargout,
		"Interface to evaluate the performance of a prediction set")
{

// #ifdef _OPENMP
//   Eigen::initParallel();
//   cout << "initialized Eigen parallel"<<endl;
// #endif  

  int nargin = args.length();
  if (nargin == 0)
    {
      print_usage();
      return octave_value_list(0);
    }
  
  
  // should do these via options
  bool verbose = true;
  predtype thresh; //threshold to use for classification
  int k=1; //return at least one predictions for threshold metrics

  SparseMb y;
  if (args(1).is_sparse_type())
    {
      Sparse<bool> yArray = args(1).sparse_bool_matrix_value(); 
      y = toEigenMat(yArray);
      // multilabel problems. Use a threshold of 0 for classification 
      // if no prediction is above 0, return the class with the highest predictions
      // should get this info in the parameters
      thresh = 0.0; 
      k=1;
    }
  else
    {      
      FloatNDArray yVector = args(1).array_value(); // the label vector
      
      VectorXd yVec = toEigenVec(yVector);
  
      y = labelVec2Mat(yVec);
      // multiclass data 
      // the class with the highest output will be the prediction
      thresh = boost::numeric::bounds<predtype>::highest();
      k=1;
    }
  
  double MicroF1, MacroF1, MacroF1_2, MicroPrecision, MacroPrecision, MicroRecall, MacroRecall, Top1, Top5, Top10, Prec1, Prec5, Prec10;

  PredictionSet* preds;
  
  if (verbose)
    {
      cout << "Loading Predictions ... " << endl;
    }

  if(args(0).is_sparse_type())
    {
      // Sparse predictions      
      preds = new PredictionSet(toEigenMat(args(0).sparse_matrix_value()));
    }
  else
    {
      // Dense predictions
      preds = new PredictionSet(toEigenMat<DenseM>(args(0).array_value()));
    }
  
  if (verbose)
    {
      cout << "Evaluating ... " << endl;
    }
  preds->ThreshMetrics(MicroF1, MacroF1, MacroF1_2, MicroPrecision, MacroPrecision, MicroRecall, MacroRecall, y, thresh, k);
  preds->TopMetrics(Prec1, Top1, Prec5, Top5, Prec10, Top10, y);
      
 
  octave_scalar_map perfs;
  perfs.assign("MicroF1", MicroF1);
  perfs.assign("MacroF1", MacroF1);
  perfs.assign("MacroF1_2", MacroF1_2);
  perfs.assign("MicroPrecision", MicroPrecision);
  perfs.assign("MacroPrecision", MacroPrecision);
  perfs.assign("MicroRecall", MicroRecall);
  perfs.assign("MacroRecall", MacroRecall);
  perfs.assign("Top1", Top1);
  perfs.assign("Prec1", Prec1);
  perfs.assign("Top5", Top5);
  perfs.assign("Prec5", Prec5);
  perfs.assign("Top10", Top10);
  perfs.assign("Prec10", Prec10);

  octave_value_list retval(1);
  retval(0) = perfs;
  return retval;
}
