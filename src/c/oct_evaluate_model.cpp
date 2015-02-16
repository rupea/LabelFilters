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
  cout << "oct_evaluate_model x y ova_models w l u" << endl;
  cout << "     x - the data matrix (can be dense or sparse)" << endl;
  cout << "     y - a label vector (same size as rows(x)) with elements 1:noClasses" << endl;
  cout << "          or a sparse label matrix of size rows(x)*noClasses with y(i,j)=1 meaning that example i has class j" << endl;
  cout << "     ova_models_file - the file with a cell array with the ova models in a variable named svm_models_final" << endl;
  cout << "              the weight vectors are in floats to save memory" << endl;
  cout << "     w - the projection vectors" << endl;
  cout << "     l - the lower bounds" << endl;
  cout << "     u - the upper bounds" << endl;
}


DEFUN_DLD (oct_evaluate_model, args, nargout,
		"Interface to evaluate ova model with and without projection")
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
  
  DenseColM wmat = toEigenMat<DenseColM>(args(3).float_array_value());
  DenseColM lmat1 = toEigenMat<DenseColM>(args(4).float_array_value());
  DenseColM umat1 = toEigenMat<DenseColM>(args(5).float_array_value());
  DenseM projections;

  
  octave_value_list arg;
  arg(0) = args(2); // the file name
  arg(1) = "svm_models_final";
  
  cout << "Loading file " << arg(0).string_value() << " ... " <<endl;
  //  octave_value_list loaded = Fload(arg, 1);  
  octave_value_list* loaded = new octave_value_list;
  *loaded = feval("load", arg, 1);  
  cout << "Done loading" << endl;
  cout << (*loaded)(0).scalar_map_value().fieldnames()[0] << endl;
    
  
  DenseColMf ovaW;
  toEigenMat(ovaW, (*loaded)(0).scalar_map_value().getfield(arg(1).string_value()).cell_value());
  
  cout << lmat1.rows() << "   " <<  lmat1.cols() << endl;
  DenseM lmat = lmat1.topRows(ovaW.cols());
  DenseM umat = umat1.topRows(ovaW.cols());
  cout << lmat.rows() << "   " <<  lmat.cols() << endl;

  cout << "size of prediction " << sizeof(prediction) << endl;
  cout << "size of predvec " << sizeof(PredVec) << endl;

  assert(lmat.rows() == ovaW.cols());
  assert(umat.rows() == ovaW.cols());

  delete loaded; // clear memory

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
      FloatNDArray yVector = args(1).float_array_value(); // the label vector
      
      VectorXd yVec = toEigenVec(yVector);
  
      y = labelVec2Mat(yVec);
      // multiclass data 
      // the class with the highest output will be the prediction
      thresh = boost::numeric::bounds<predtype>::highest();
      k=1;
    }
  
  size_t nact, nact_noproj;
  double MicroF1, MacroF1, MacroF1_2, MicroPrecision, MacroPrecision, MicroRecall, MacroRecall, Top1, Top5, Top10, Prec1, Prec5, Prec10, act_prc,time;
  double MicroF1_noproj, MacroF1_noproj, MacroF1_2_noproj, MicroPrecision_noproj, MacroPrecision_noproj, MicroRecall_noproj, MacroRecall_noproj, Top1_noproj, Top5_noproj, Top10_noproj, Prec1_noproj, Prec5_noproj, Prec10_noproj, act_prc_noproj, time_noproj;

  if(args(0).is_sparse_type())
    {
      // Sparse data
      SparseM x = toEigenMat(args(0).sparse_matrix_value());

      evaluate_projection(x,y,ovaW,wmat,lmat,umat,thresh,k,"filtered",verbose,
			  MicroF1, MacroF1, MacroF1_2, MicroPrecision,
			  MacroPrecision, MicroRecall, MacroRecall,
			  Top1, Top5, Top10, Prec1, Prec5, Prec10, nact, act_prc, time);
      
      if (nargout > 1)
	{
	  evaluate_full(x,y,ovaW,thresh,k,"not filtered",verbose,
			MicroF1_noproj, MacroF1_noproj, MacroF1_2_noproj, 
			MicroPrecision_noproj, MacroPrecision_noproj, 
			MicroRecall_noproj, MacroRecall_noproj, 
			Top1_noproj, Top5_noproj, Top10_noproj, 
			Prec1_noproj, Prec5_noproj, Prec10_noproj,
			nact_noproj, act_prc_noproj, time_noproj);
	}      
    }
  else
    {
      // Dense data
      DenseM x = toEigenMat<DenseM>(args(0).float_array_value());

      evaluate_projection(x,y,ovaW,wmat,lmat,umat,thresh,k,"filtered",verbose,
			  MicroF1, MacroF1, MacroF1_2, MicroPrecision,
			  MacroPrecision, MicroRecall, MacroRecall,
			  Top1, Top5, Top10, Prec1, Prec5, Prec10, nact, act_prc, time);
      
      if (nargout > 1)
	{
	  evaluate_full(x,y,ovaW,thresh,k,"not filtered",verbose,
			MicroF1_noproj, MacroF1_noproj, MacroF1_2_noproj, 
			MicroPrecision_noproj, MacroPrecision_noproj, 
			MicroRecall_noproj, MacroRecall_noproj, 
			Top1_noproj, Top5_noproj, Top10_noproj, 
			Prec1_noproj, Prec5_noproj, Prec10_noproj,
			nact_noproj, act_prc_noproj, time_noproj);
	}
    }
  

  octave_scalar_map perfs_noproj;
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
  perfs.assign("nactive", nact);
  perfs.assign("percent_active", act_prc);
  perfs.assign("time", time);

  if (nargout > 1)
    {
      
      perfs_noproj.assign("MicroF1", MicroF1_noproj);
      perfs_noproj.assign("MacroF1", MacroF1_noproj);
      perfs_noproj.assign("MacroF1_2", MacroF1_2_noproj);
      perfs_noproj.assign("MicroPrecision", MicroPrecision_noproj);
      perfs_noproj.assign("MacroPrecision", MacroPrecision_noproj);
      perfs_noproj.assign("MicroRecall", MicroRecall_noproj);
      perfs_noproj.assign("MacroRecall", MacroRecall_noproj);
      perfs_noproj.assign("Top1", Top1_noproj);
      perfs_noproj.assign("Prec1", Prec1_noproj);
      perfs_noproj.assign("Top5", Top5_noproj);
      perfs_noproj.assign("Prec5", Prec5_noproj);
      perfs_noproj.assign("Top10", Top10_noproj);
      perfs_noproj.assign("Prec10", Prec10_noproj);
      perfs_noproj.assign("nactive", static_cast<double>(y.rows()*y.cols()));
      perfs_noproj.assign("percent_active", 1.0);
      perfs_noproj.assign("time", time_noproj);
    }
  
  //cleanup. maybe these should be done in Active and Prediction classes    

  if (nargout > 1)
    {
      octave_value_list retval(2);
      retval(0) = perfs;
      retval(1) = perfs_noproj;      
      return retval;
    }
  else
    {
      octave_value_list retval(1);
      retval(0) = perfs;
      return retval;
    }
}
