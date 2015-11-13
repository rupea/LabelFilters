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
#include "normalize.h"

using Eigen::VectorXd;
using Eigen::VectorXi;

using namespace std;
namespace po = boost::program_options;

void print_usage(po::options_description opt)
{
  cerr << endl;
  cerr << "USAGE: run_kmeans [options] data_file" << endl << endl;
  cerr << "  data_file: .mat file with the data  (x_te, y_te)" << endl;
  cerr << opt;
}

void parse_options(po::variables_map& vm, int argc, char* argv[])
{
  po::options_description opt("Options");
  opt.add_options()
    ("help", "Displays help message")
    ("verbose,v", "Display status messages")    
    ("clusters,k",po::value<int>()->default_value(2),"Number of clusters [2].")
    ("iterations,i",po::value<int>()->default_value(100),"Maximum nubmer of iterations [100].")
    //    ("spherical,p", "Perform sferical K-means (i.e. project the data and the centers on the unit sphere).")
    ("seed,s",po::value<int>()->default_value(0),"Random seed. 0 for using TIME [0].") 
    ("num_threads", po::value<int>()->default_value(0), "Number of threads to run on. 0 for using all available threads.[0]");
  
  po::options_description hidden_opt("Arguments");
  hidden_opt.add_options()
    ("data_file", po::value<string>(), ".mat file with the test data  (x_te, y_te)");

  po::positional_options_description pd;
  pd.add("data_file",1);

  po::options_description all_opt;
  all_opt.add(opt).add(hidden_opt);   

  po::store(po::command_line_parser(argc,argv).options(all_opt).positional(pd).run(),vm);
  po::notify(vm);

  if(vm.count("help"))
    {
      print_usage(opt);
      exit(0);
    }

  if (vm["clusters"].as<int>() <= 1)
    {
      cerr << endl;
      cerr << "ERROR: Number of clusters smaller than 2" << endl;
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

  
  int k=vm["clusters"].as<int>(); //number of clusters
  int iterations = vm["iterations"].as<int>(); // number of iterations

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

  octave_value_list args(2); 
  args(0)=vm["data_file"].as<string>();
  args(1)="x_tr"; 
  //args(2)="y_tr"; 

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
  octave_value x_tr = loaded(0).scalar_map_value().getfield(args(1).string_value()); 
  //  octave_value y_tr = loaded(0).scalar_map_value().getfield(args(2).string_value());
  args.clear();
  loaded.clear();
  

  size_t dim,n;
  if(x_tr.is_sparse_type())
    {
      // Sparse data
      dim = x_tr.sparse_matrix_value().cols();
      n = x_tr.sparse_matrix_value().rows();
    }
  else
    {
      //Dense data
      dim = x_tr.array_value().cols();
      n = x_tr.array_value().rows();
    }

  DenseColM centers(dim,k);
  VectorXi assignments(n);
  if(x_tr.is_sparse_type())
    {
      // Sparse data
      SparseM x = toEigenMat(x_tr.sparse_matrix_value());
      if (spherical)
	{
	  normalize_row(x);
	}
      run_kmeans(centers,assignments,x,iterations,spherical);
    }
  else
    {
      // Dense data
      DenseM x = toEigenMat<DenseM>(x_tr.array_value());
      if (spherical)
	{
	  x.rowwise().normalize();
	}
      run_kmeans(centers,assignments,x,iterations,spherical);      
    }
  if (vm.count("out_file"))
    {
      outf.close();
    }
  clean_up_and_exit(0);  
}
