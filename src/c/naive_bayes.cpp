#include <octave/oct.h> 
#include <octave/parse.h> 
#include <octave/oct-map.h>
#include <octave/builtin-defun-decls.h> // Fload
#include <octave/octave.h>
#include <octave/toplev.h>
#include <iostream>
#include <ostream>
#include <fstream>
#include <boost/program_options.hpp>


#include "Eigen/Dense"
#include "Eigen/Sparse"
#include "typedefs.h"
#include "EigenOctave.h"
#include "utils.h"
#include "naive_bayes.hh"

using namespace std;
namespace po = boost::program_options;

void print_usage(po::options_description opt)
{
  cerr << endl;
  cerr << "USAGE: naive_bayes [options] data_file model_file" << endl << endl;
  cerr << "  data_file: .mat file with the test data  (x_tr, y_tr)" << endl;
  cerr << "  model_file : file to write the model to in binary format." <<endl;
  cerr << opt;
}

void parse_options(po::variables_map& vm, int argc, char* argv[])
{
  po::options_description opt("Options");
  opt.add_options()    
    ("help", "Displays help message")
    ("verbose,v", "Display status messages")
    ("alpha,a", po::value<ovaCoeffType>()->default_value(1), "Laplace smoothing constant");
  
  po::options_description hidden_opt("Arguments");
  hidden_opt.add_options()
    ("data_file", po::value<string>(), ".mat file with the test data  (x_te, y_te)")
    ("model_file", po::value<string>(), "file to write the model in binary format.");

  po::positional_options_description pd;
  pd.add("data_file",1).add("model_file",1);

  po::options_description all_opt;
  all_opt.add(opt).add(hidden_opt);

  po::store(po::command_line_parser(argc,argv).options(all_opt).positional(pd).run(),vm);
  po::notify(vm);

  if(vm.count("help"))
    {
      print_usage(opt);
      exit(0);
    }

  if(!vm.count("data_file"))
    {
      cerr << endl;
      cerr << "ERROR:No data file supplied" << endl;
      print_usage(opt);
      exit(-1);
    }
  if(!vm.count("model_file"))
    {
      cerr << endl;
      cerr << "ERROR:No model file supplied" << endl;
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

  ofstream model_out;
  model_out.open(vm["model_file"].as<string>().c_str());
  if (!model_out.is_open())
    {
      cerr << "Error opening the model file " << vm["model_file"].as<string>() << endl;
      exit(-1);
    }

  ovaCoeffType a = vm["alpha"].as<ovaCoeffType>();

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

  
  octave_value x_tr,y_tr;
    
  octave_value_list args(3); 
  args(0)=vm["data_file"].as<string>();
  args(1)="x_tr"; 
  args(2)="y_tr"; 

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
  x_tr = loaded(0).scalar_map_value().getfield(args(1).string_value()); 
  y_tr = loaded(0).scalar_map_value().getfield(args(2).string_value());
  args.clear();
  loaded.clear();

  SparseMb y;
  if (y_tr.is_sparse_type())
    {
      y = toEigenMat(y_tr.sparse_bool_matrix_value());
    }
  else
    {      
      y = labelVec2Mat(toEigenVec(y_tr.array_value()));
    }
  
  if(x_tr.is_sparse_type())
    {
      SparseM x = toEigenMat(x_tr.sparse_matrix_value());
      train_NB(x,y,a,model_out);
    }
  else
    {
      DenseM x = toEigenMat<DenseM, NDArray>(x_tr.array_value());
      train_NB(x,y,a,model_out);
    }

  model_out.close();
  clean_up_and_exit(0);  

  
}
