#include "constants.h"
#include "typedefs.h"
#include "EigenOctave.h"
#include "EigenIO.h"
#include "evaluate.h"
#include "utils.h"

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

using Eigen::VectorXd;
using Eigen::VectorXi;

using namespace std;
namespace po = boost::program_options;

void print_usage(po::options_description opt)
{
  cerr << endl;
  cerr << "USAGE: evaluate_projection [options] data_file ova_file" << endl << endl;
  cerr << "  data_file: .mat file with the test data  (x_te, y_te)" << endl;
  cerr << "  ova_file : .mat file with the ova models in a cell array" <<endl;
  cerr << "                (only works with liblinear matlab models for now)" << endl;
  cerr << "             or a binary file with the weights of the linear ova" << endl;
  cerr << "                models concatenated in the order of the classes." << endl<< endl;
  cerr << opt;
}

void parse_options(po::variables_map& vm, int argc, char* argv[])
{
  po::options_description opt("Options");
  opt.add_options()
    ("help", "Displays help message")
    ("verbose,v", "Display status messages")
    ("validation", "Use the first half of the test set as a validation set")
    ("allproj", "Evaluate the performance for all intermediate projections")
    ("projection_files,p", po::value<std::vector<string> >()->multitoken(),".mat file with the learned projection parameters (w, min_proj, max_proj). This is a multitoken option so it must be ended with '--' when all projection files have been specified.")
    ("threshold,t", po::value<predtype>(), "Threshold for predictions. By default it is not used in multiclass problems and it is 0.0 in multilabel problems")
    ("top,k", po::value<int>()->default_value(1), "Minimum number of classes to pbe predicted positive. When threshold is not used, or not enough predictions are above the threshold, the classes with highest predicted values are used. Default 1.")
    ("full", "Evaluate the performance without projections")
    ("distributed", po::value<string>()->default_value("None"), "Whether it is ran on a distributed fasion. Possible values are: \"None\"")
    ("ova_format", po::value<string>()->default_value("binary"), "Format of the file with the ova models. One of \"cellarray\" or \"binary\"")
    ("out_file,o", po::value<string>(), "Output file. If not specified prints to stdout")
    ("chunks", po::value<int>()->default_value(1), "Number of chunks to split the ova file in. Used to deal with ova matrices that are too large to fit in memory. If chunks > 1 the chunks are reloaded for each projection file which leads to long load times. If chunks > 1, ova_format must be binary")
    ("num_threads", po::value<int>()->default_value(0), "Number of threads to run on. 0 for using all available threads.");
  
  po::options_description hidden_opt("Arguments");
  hidden_opt.add_options()
    ("data_file", po::value<string>(), ".mat file with the test data  (x_te, y_te)")
    ("ova_file", po::value<string>(), ".mat file with the ova models in a cell array (only works with liblinear matlab models for now) or a binary file with the weights of the linear ova models concatenated in the order of the classes.");

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

  if (vm.count("distributed"))
    {
      if (vm["distributed"].as<string>() != "None")
	{
	  cerr << endl;
	  cerr << "ERROR:Argument to distributed unrecognized" << endl;
	  print_usage(opt);
	  exit(-1);
	}
    }

  if (vm.count("ova_format"))
    {
      if (vm["ova_format"].as<string>() !="binary" && vm["ova_format"].as<string>() != "cellarray")
	{
	  cerr << endl;
	  cerr << "ERROR:Argument to ova_format unrecognized" << endl;
	  print_usage(opt);
	  exit(-1);
	}
    }

  if (vm["chunks"].as<int>() <= 0)
    {
      cerr << endl;
      cerr << "ERROR: Number of chunks smaller than 1" << endl;
      print_usage(opt);
      exit(-1);
    }

  if (vm["chunks"].as<int>() > 1 && vm["ova_format"].as<string>() !="binary")
    {
      cerr << endl;
      cerr << "ERROR: Multiple chunks can only be used in conjunction with binary ova_format" << endl;
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

  bool validation = vm.count("validation")?true:false;
  bool allproj = vm.count("allproj")?true:false;

  int num_threads = vm["num_threads"].as<int>();
#ifdef _OPENMP
  if (num_threads < 1)
    omp_set_num_threads(omp_get_max_threads());
  else
    omp_set_num_threads(num_threads);
  Eigen::initParallel();
  if (verbose)
    {
      cout << "initialized Eigen parallel"<<endl;
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

  bool use_dstorm = false;
  if (vm.count("distributed"))
    {
      if (vm["distributed"].as<string>() == "dstorm")
	{
	  use_dstorm = true;
	}
      else if (vm["distributed"].as<string>() == "None")
	{
	  use_dstorm = false;
	}
      else
	{
	  // should not happen since it was checked when parsing options. 
	  cerr << "Argument to distributed unrecognized" << endl;
	  exit(-1);
	}
    }
      	    

  predtype thresh; //threshold to use for classification
  int k=vm["top"].as<int>(); //return at least one predictions for threshold metrics
  

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
      VectorXd yVec = toEigenVec(y_te.array_value());
  
      y = labelVec2Mat(yVec);
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
  if (chunks == 1)
    {
      size_t noClasses = y.cols();
      size_t dim;
      if(x_te.is_sparse_type())
	{
	  // Sparse data
	  dim = x_te.sparse_matrix_value().cols();
	}
      else
	{
	  //Dense data
	  dim = x_te.array_value().cols();
	}
      
      if (vm["ova_format"].as<string>() == "cellarray")
	{
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
	  toEigenMat(ovaW, loaded(0).scalar_map_value().getfield(args(1).string_value()).cell_value());
	  
	  args.clear();
	  loaded.clear();
	  Fclear();
	}
      else if (vm["ova_format"].as<string>() == "binary")
	{
	  read_binary(vm["ova_file"].as<string>().c_str(), ovaW, dim, noClasses);
	}
      else
	{ 
	  cerr << "Unrecognized format for the ova file" << endl;
	  exit(-1);
	}
    }

  bool do_full = vm.count("full")?true:false;
  bool do_projection=false;
  DenseColM wmat, lmat, umat;
  std::vector<string> proj_files;
  if (vm.count("projection_files"))
    {
      proj_files = vm["projection_files"].as<std::vector<string> >();
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
      
      for (std::vector<string>::iterator pit = proj_files.begin(); pit !=proj_files.end(); ++pit)
	{
	  cerr << "***********" << *pit << "************" << endl;
	  load_projections(wmat, lmat, umat, *pit, verbose);
	  if (chunks == 1)
	    {
	      evaluate_projection(x, y, ovaW, &wmat, &lmat, &umat, thresh, k, *pit, validation, allproj, verbose, out);
	    } 
	  else
	    {
	      evaluate_projection_chunks(x, y, vm["ova_file"].as<string>(), chunks, &wmat, &lmat, &umat, thresh, k, *pit, validation, allproj, verbose, out);
	    }	  
	}      
      if (do_full)
	{
	  if (chunks == 1)
	    {
	      evaluate_projection(x, y, ovaW, NULL, NULL, NULL, thresh, k, "full", validation, false, verbose, out);
	    }
	  else
	    {
	      evaluate_projection_chunks(x, y, vm["ova_file"].as<string>(), chunks, NULL, NULL, NULL, thresh, k, "full", validation, false,  verbose, out);
	    }
	}
    }
  else
    {
      // Dense data
      DenseM x = toEigenMat<DenseM>(x_te.array_value());

      for (std::vector<string>::iterator pit = proj_files.begin(); pit !=proj_files.end(); ++pit)
	{
	  cerr << "***********" << *pit << "************" << endl;
	  load_projections(wmat, lmat,umat,*pit,verbose);
	  if (chunks == 1)
	    {
	      evaluate_projection(x, y, ovaW, &wmat, &lmat, &umat, thresh, k, *pit, validation, allproj, verbose, out);
	    } 
	  else
	    {
	      evaluate_projection_chunks(x, y, vm["ova_file"].as<string>(), chunks, &wmat, &lmat, &umat, thresh, k, *pit, validation, allproj, verbose, out);
	    }	  
	}      
      if (do_full)
	{
	  if (chunks == 1)
	    {
	      evaluate_projection(x, y, ovaW, NULL, NULL, NULL, thresh, k, "full", validation, false, verbose, out);
	    }
	  else
	    {
	      evaluate_projection_chunks(x, y, vm["ova_file"].as<string>(), chunks, NULL, NULL, NULL, thresh, k, "full", validation, false, verbose, out);
	    }
	}
    }
  if (vm.count("out_file"))
    {
      outf.close();
    }
  clean_up_and_exit(0);  
}
