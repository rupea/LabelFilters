#include <octave/oct.h> 
#include <octave/parse.h> 
#include <octave/oct-map.h>
//#include <octave/variables.h> 
#include <octave/builtin-defun-decls.h>
#include <octave/octave.h>
#include <octave/parse.h>
#include <octave/toplev.h>
#include <iostream>
#include <ostream>
#include <fstream>
#include <vector>
#include <stdio.h>
//#include <typeinfo>
#include <math.h>
#include <stdlib.h>
#include <boost/numeric/conversion/bounds.hpp>
#include <boost/limits.hpp>
#include <boost/program_options.hpp>
#include "Eigen/Dense"
#include "Eigen/Sparse"
#include "constants.h"
#include "typedefs.h"
#include "EigenOctave.h"
#include "EigenIO.h"
#include "KMeans.h"
#include "LPSR.h"
#include "normalize.h"
#include "evaluate.hh"  // template impls

using Eigen::VectorXd;
using Eigen::VectorXi;

using namespace std;
namespace po = boost::program_options;

void print_usage(po::options_description opt)
{
  cerr << endl;
  cerr << "USAGE: run_kmeans [options] data_file ova_file" << endl << endl;
  cerr << "  data_file: .mat file with the data  (x_tr, y_tr, x_te)" << endl;
  cerr << "  ova_file : .mat file with the ova models in  binary file with the weights of the linear ova" << endl;
  cerr << "                models stored as 32 bit floats concatenated in the order of the classes." << endl<< endl;
  cerr << opt;
}

void parse_options(po::variables_map& vm, int argc, char* argv[])
{
  po::options_description opt("Options");
  opt.add_options()
    ("help", "Displays help message")
    ("verbose,v", "Display status messages")    
    ("clusters,c",po::value<int>()->default_value(256),"Number of clusters [256].")
    ("classes_per_cluster,C",po::value<int>()->default_value(1000),"Numbet of active classes per cluster [1000].")    
    ("iterations,i",po::value<int>()->default_value(100),"Maximum nubmer of iterations [100].")
    //    ("spherical,p", "Perform sferical K-means (i.e. project the data and the centers on the unit sphere).")
    ("seed,s",po::value<int>()->default_value(0),"Random seed. 0 for using TIME [0].") 
    ("out_file,o", po::value<string>(), "Output file. If not specified prints to stdout")
    ("model_file,m", po::value<string>(), "Model file. If it exists, LPSR model is loaded from this file. If it does not exist, a new model is trained and saved to this file.")
    ("chunks", po::value<int>()->default_value(1), "Number of chunks to split the ova file in. Used to deal with ova matrices that are too large to fit in memory. If chunks > 1 the chunks are reloaded for each projection file which leads to long load times. If chunks > 1, ova_format must be binary")
    ("threshold,t", po::value<predtype>(), "Threshold for predictions. By default it is not used in multiclass problems and it is 0.0 in multilabel problems")
    ("top,k", po::value<int>()->default_value(1), "Minimum number of classes to pbe predicted positive. When threshold is not used, or not enough predictions are above the threshold, the classes with highest predicted values are used. Default 1.")
    ("num_threads", po::value<int>()->default_value(0), "Number of threads to run on. 0 for using all available threads.[0]");
  
  po::options_description hidden_opt("Arguments");
  hidden_opt.add_options()
    ("data_file", po::value<string>(), ".mat file with the train and test data  (x_tr,y_tr,x_te, y_te)")
    ("ova_file", po::value<string>(), ".mat file with the ova models in a binary file with the weights of the linear ova models concatenated in the order of the classes.");

  po::positional_options_description pd;
  pd.add("data_file",1).add("ova_file",1);

  po::options_description all_opt;
  all_opt.add(opt).add(hidden_opt);   

  po::store(po::command_line_parser(argc,argv).options(all_opt).positional(pd).run(),vm);
  po::notify(vm);

  if(vm.count("help"))
    {
      print_usage(opt);
      exit(0);
    }

  if (vm["clusters"].as<int>() < 1)
    {
      cerr << endl;
      cerr << "ERROR: Number of clusters smaller than 1" << endl;
      print_usage(opt);
      exit(-1);
    }

  if (vm["chunks"].as<int>() <= 0)
    {
      cerr << endl;
      cerr << "ERROR: Number of chunks smaller than 1" << endl;
      print_usage(opt);
      exit(-1);
    }

  if(!vm.count("data_file"))
    {
      cerr << endl;
      cerr << "ERROR:No data file supplied" << endl;
      print_usage(opt);
      exit(-1);
    }

  if(!vm.count("ova_file"))
    {
      cerr << endl;
      cerr << "ERROR:No ova file supplied" << endl;
      print_usage(opt);
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
  //  bool spherical = vm.count("spherical")?true:false;
  bool spherical= true; // only shperical k-means is implemented at this time. 
  bool validation = vm.count("validation")?true:false;
  int num_threads = vm["num_threads"].as<int>();

#ifdef _OPENMP
  if (num_threads < 1)
    {
      omp_set_num_threads(omp_get_max_threads());
    }
  else
    {
      omp_set_num_threads(num_threads);
    }
  if (num_threads != 1)
    {
      Eigen::initParallel();
      if (verbose)
	{
	  cout << "initialized Eigen parallel"<<endl;
	}
    }
#endif

  ofstream outf;
  if (vm.count("out_file"))
    {
      outf.open(vm["out_file"].as<string>().c_str());
      if (!outf.is_open())
	{
	  cerr << "Error opening the output file " << vm["out_file"].as<string>() << endl;
	  exit(-1);
	}
    }
  ostream& out = vm.count("out_file")?outf:cout;

  
  int c=vm["clusters"].as<int>(); //number of clusters
  int iterations = vm["iterations"].as<int>(); // number of iterations for K-means
  int C=vm["classes_per_cluster"].as<int>(); //number of classes per cluster

  predtype thresh; //threshold to use for classification
  int k=vm["top"].as<int>(); //return at least one predictions for threshold metrics

  /* initialize random seed: */
  int seed = vm["seed"].as<int>();
  if (seed)
    {
      srand(seed);
    }
  else
    {
      srand(time(NULL));
    }


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

  octave_value_list args(5); 
  args(0)=vm["data_file"].as<string>();
  args(1)="x_tr"; 
  args(2)="y_tr"; 
  args(3)="x_te"; 
  args(4)="y_te"; 

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
  octave_value x_train = loaded(0).scalar_map_value().getfield(args(1).string_value()); 
  octave_value y_train = loaded(0).scalar_map_value().getfield(args(2).string_value());
  octave_value x_test = loaded(0).scalar_map_value().getfield(args(3).string_value()); 
  octave_value y_test = loaded(0).scalar_map_value().getfield(args(4).string_value());
  args.clear();
  loaded.clear();
  
  SparseMb y_tr,y_te;
  if (y_train.is_sparse_type())
    {
      y_tr = toEigenMat(y_train.sparse_bool_matrix_value());
      y_te = toEigenMat(y_test.sparse_bool_matrix_value());
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
      VectorXd yVec;
      yVec = toEigenVec(y_train.array_value());
      y_tr = labelVec2Mat(yVec);
      yVec = toEigenVec(y_test.array_value());
      y_te = labelVec2Mat(yVec);
      // multiclass data 
      // the class with the highest output will be the prediction
      if (!vm.count("threshold"))
	{
	  thresh = boost::numeric::bounds<predtype>::highest();
	}
      else 
	{
	  thresh = vm["threshold"].as<predtype>();
	}
    }

  DenseColMf ovaW;
  int chunks = vm["chunks"].as<int>();
  assert(chunks > 0);

  size_t dim;
  if(x_train.is_sparse_type())
    {
      // Sparse data
      dim = x_train.sparse_matrix_value().cols();
    }
  else
    {
      //Dense data
      dim = x_train.array_value().cols();
    }

  DenseColM centers;
  ActiveDataSet* active_classes=NULL;
  bool train_model = true;
  bool save_model = false;
  if (vm.count("model_file"))
    {
      if (fexists(vm["model_file"].as<string>().c_str()))
	{
	  if (verbose)
	    {
	      cout << "Loading LPSR model from " << vm["model_file"].as<string>() << endl;
	    }
	  if (!load_LPSR_model(vm["model_file"].as<string>().c_str(), centers, active_classes))
	    {
	      train_model = false;
	      assert(centers.rows() == dim);
	      assert(centres.cols() == active_classes->size());
	      assert(active_classes->front()->size() == y_tr.cols());
	      if (vm.count("clusters"))
		{
		  assert(vm["clusters"].as<int>() == centers.cols());
		}
	      else
		{
		  c = centers.cols();
		}
	    }
	  else
	    {
	      cerr << "Loading of LPSR model from " << vm["model_file"].as<string>() << " has failed. Retraining the model" << endl;
	    }
	}
      if (train_model)
	{
	  save_model=true;
	}
    }
  if (train_model)
    {
      // the model has not been loaded succesfully from a file
      centers.resize(dim,c);
    }
      
  char buffer[1000]="";
  sprintf(buffer, "K-means c = %d, C = %d",c,C); 
  string evalname(buffer);

  if(x_train.is_sparse_type())
    {
      // Sparse data
      SparseM x;
      if (train_model)
	{
	  x = toEigenMat(x_train.sparse_matrix_value());
	  if (spherical)
	    {
	      normalize_row(x);
	    }
	  if (verbose)
	    {
	      cout << "Train LPSR model ... " << endl;
	    }
	  train_LPSR(centers,active_classes,x,y_tr,C,iterations,spherical,verbose);      
	  if(verbose)
	    {
	      cout << "Done training LPSR model." << endl;
	    }
	}
      if (save_model)
	{
	  if (verbose) 
	    {
	      cout << "Saving LPSR model to " << vm["model_file"].as<string>() << endl;
	    }
	  save_LPSR_model(vm["model_file"].as<string>().c_str(),centers, *active_classes);	  
	}
      
      if(verbose)
	{
	  cout << "Evaluating LPSR model..." << endl;      
	}
      x = toEigenMat(x_test.sparse_matrix_value());
      if (spherical)
	{
	  normalize_row(x);
	}
      evaluate_LPSR_chunks(x, y_te, vm["ova_file"].as<string>(), chunks, centers, *active_classes,
			   thresh, k, evalname, validation, spherical, verbose, out);
    }
  else
    {
      // Dense data
      DenseM x;
      if (train_model)
	{
	  x = toEigenMat<DenseM>(x_train.array_value());
	  if (spherical)
	    {
	      x.rowwise().normalize();
	    }
	  if (verbose)
	    {
	      cout << "Train LPSR model ... " << endl;
	    }
	  train_LPSR(centers,active_classes,x,y_tr,C,iterations,spherical,verbose);      
	  if(verbose)
	    {
	      cout << "Done training LPSR model." << endl;
	    }
	}
      if (save_model)
	{
	  if (verbose) 
	    {
	      cout << "Saving LPSR model to " << vm["model_file"].as<string>() << endl;
	    }
	  save_LPSR_model(vm["model_file"].as<string>().c_str(),centers, *active_classes);	  
	}
      
      if(verbose)
	{
	  cout << "Evaluating LPSR model..." << endl;      
	}
      
      x = toEigenMat<DenseM>(x_test.array_value());
      if (spherical)
	{
	  x.rowwise().normalize();
	}
      evaluate_LPSR_chunks(x, y_te, vm["ova_file"].as<string>(), chunks, centers,*active_classes,
			   thresh, k, evalname, validation, spherical, verbose, out);
      
    }
  if (vm.count("out_file"))
    {
      outf.close();
    }
  
  if (active_classes)
    {
      free_ActiveDataSet(active_classes);
    }
  clean_up_and_exit(0);  
}
