#include <octave/oct.h> 
#include <octave/parse.h> 
#include <octave/oct-map.h>
//#include <octave/variables.h> 
#include <octave/builtin-defun-decls.h>
#include <octave/octave.h>
#include <octave/parse.h>
#include <octave/toplev.h>
#include <iostream>
#include <vector>
#include <stdio.h>
#include <typeinfo>
#include <math.h>
#include <stdlib.h>
#include <boost/program_options.hpp>
#include "Eigen/Dense"
#include "Eigen/Sparse"
#include "constants.h"
#include "typedefs.h"
#include "EigenOctave.h"
#include "evaluate.h"

using Eigen::VectorXd;
using Eigen::VectorXi;

using namespace std;
namespace po = boost::program_options;



void parse_options(po::variables_map& vm, int argc, char* argv[])
{
  po::options_description opt("Options");
  opt.add_options()
    ("help", "Displays help message")
    ("verbose,v", "Display status messages")
    ("projection_file,p", po::value<string>(),".mat file with the learned projection parameters (w, min_proj, max_proj)")
    ("threshold,t", po::value<predtype>(), "Threshold for predictions. By default it is not used in multiclass problems and it is 0.0 in multilabel problems")
    ("top,k", po::value<int>()->default_value(1), "Minimum number of classes to pbe predicted positive. When threshold is not used, or not enough predictions are above the threshold, the classes with highest predicted values are used. Default 1.")
    ("full", "Evaluate the performance without projections");

  po::options_description hidden_opt;
  hidden_opt.add_options()
    ("data_file", po::value<string>(), ".mat file with the test data  (x_te, y_te)")
    ("ova_file", po::value<string>(), ".mat file with the ova models in a cell array. Only works with liblinear matlab models for now");    

  po::positional_options_description pd;
  pd.add("data_file",1).add("ova_file",1);

  po::options_description all_opt;
  all_opt.add(opt).add(hidden_opt);   

  po::store(po::command_line_parser(argc,argv).options(all_opt).positional(pd).run(),vm);
  po::notify(vm);

  if(vm.count("help"))
    {
      cerr << opt;
      exit(0);
    }
  if(!vm.count("data_file"))
    {
      cerr << "No data file supplied" << endl;
      cerr << opt;
      exit(-1);
    }
  if(!vm.count("ova_file"))
    {
      cerr << "No ova file supplied" << endl;
      cerr << opt;
      exit(-1);
    }
}




// TO DO: put protections when files are not available or the right 
// variables are not in them.
// now it crashes badly with a seg fault and can corrupt other processes
int main(int argc, char * argv[])
{ 
  po::variables_map vm;
  parse_options(vm, argc, argv);


  bool verbose = vm.count("verbose")?true:false;

#ifdef _OPENMP
  Eigen::initParallel();
  if (verbose)
    {
      cout << "initialized Eigen parallel"<<endl;
    }
#endif

  // need to initialize the octave interpreter or else loading
  // ascii files results in segfault
  string_vector oct_arg(2);
  oct_arg(0) = "embeded";
  oct_arg(1) = "-q"; 

  if (!octave_main(2, oct_arg.c_str_vec(), 1))
    {
      cerr << "Error initiallizing octave" << endl;
      exit(-2);
    }

  octave_value_list args; 
  args(0)=vm["data_file"].as<string>();
  args(1)="x_te"; 
  args(2)="y_te"; 

  if (verbose)
    {
      cout << "Loading data file " << args(0).string_value() << " ... " <<endl;
    }
  octave_value_list loaded = Fload(args, 1);
  //feval("load", args, 0); // no arguments returned 
  if (verbose)
    {
      cout << "success" << endl; 
    }
  octave_value x_te = loaded(0).scalar_map_value().getfield(args(1).string_value()); 
  octave_value y_te = loaded(0).scalar_map_value().getfield(args(2).string_value());
  args.clear();
  loaded.clear();
  
  bool do_full = vm.count("full")?true:false;
  bool do_projection;
  DenseColM wmat, lmat, umat;
  if (vm.count("projection_file"))
    {
      do_projection = true; 
      args(0) = vm["projection_file"].as<string>(); // the projection filename 
      args(1) = "w";
      args(2) = "min_proj";
      args(3) = "max_proj";
      if (verbose)
	{
	  cout << "Loading file " << args(0).string_value() << " ... " <<endl;
	}
      loaded = Fload(args, 1);
      //feval("load", args, 0); // no arguments returned 
      if (verbose)
	{
	  cout << "success" << endl; 
	}
      wmat = toEigenMat<DenseColM>(loaded(0).scalar_map_value().getfield(args(1).string_value()).float_array_value());
      lmat = toEigenMat<DenseColM>(loaded(0).scalar_map_value().getfield(args(2).string_value()).float_array_value());
      umat = toEigenMat<DenseColM>(loaded(0).scalar_map_value().getfield(args(3).string_value()).float_array_value());
      args.clear();
      loaded.clear();
    }
  
  args(0) = vm["ova_file"].as<string>(); // ova file name
  args(1) = "svm_models_final";  
  if (verbose)
    {
      cout << "Loading file " << args(0).string_value() << " ... " <<endl;
    }
  loaded = Fload(args, 1);  
  if (verbose)
    {
      cout << "success" << endl;
    }

  DenseColMf ovaW;
  toEigenMat(ovaW, loaded(0).scalar_map_value().getfield(args(1).string_value()).cell_value());
  
  loaded.clear();

  Fclear();

  if (do_projection)
    {
      assert(lmat.rows() == ovaW.cols());
      assert(umat.rows() == ovaW.cols());
    }

  predtype thresh; //threshold to use for classification
  int k=vm["top"].as<int>(); //return at least one predictions for threshold metrics
  
  SparseMb y;
  if (y_te.is_sparse_type())
    {
      y = toEigenMat(y_te.sparse_bool_matrix_value());
      // multilabel problems. Use a threshold of 0 for classification 
      // if no prediction is above 0, return the class with the highest predictions
      // should get this info in the parameters
      if (!vm.count("threshold"))
	{
	  thresh = 0.0; 
	}
      else 
	{
	  thresh = vm["threshold"].as<predtype>();
	}
    }
  else
    {      
      VectorXd yVec = toEigenVec(y_te.float_array_value());
  
      y = labelVec2Mat(yVec);
      // multiclass data 
      // the class with the highest output will be the prediction
      if (!vm.count("threshold"))
	{
	  thresh = std::numeric_limits<predtype>::max();
	}
      else 
	{
	  thresh = vm["threshold"].as<predtype>();
	}
    }

  if(x_te.is_sparse_type())
    {
      // Sparse data
      SparseM x = toEigenMat(x_te.sparse_matrix_value());

      // size_t reducedsize = 10000;
      // SparseM smallx = x.topLeftCorner(reducedsize,x.cols());
      // SparseMb smally = y.topLeftCorner(reducedsize, y.cols());
      // x=smallx;
      // y=smally;

      if (do_projection)
	{
	  evaluate_projection(x,y,ovaW,wmat,lmat,umat,thresh,k,verbose);
	}      
      if (do_full)
	{
	  evaluate_full(x,y,ovaW,thresh,k,verbose);
	}
    }
  else
    {
      // Dense data
      DenseM x = toEigenMat<DenseM>(x_te.float_array_value());

      if (do_projection)
	{
	  evaluate_projection(x,y,ovaW,wmat,lmat,umat,thresh,k,verbose);
	}      
      if (do_full)
	{
	  evaluate_full(x,y,ovaW,thresh,k,verbose);
	}
    }
  clean_up_and_exit(0);
  
}
