#include <octave/oct.h>
#include <octave/ov-struct.h>
#include <iostream>
#include <typeinfo>
#include "Eigen/Dense"
#include "Eigen/Sparse"
#include "KMeans.h"
#include "EigenOctave.h"
#include "normalize.h"

using Eigen::VectorXd;


void print_usage()
{
  cout << "[centroids assignments] = oct_kmeans x params" << endl;
  cout << "     x - the data matrix (can be dense or sparse)" << endl;
  cout << "     parameters - a structure with the k-means parameters. If a parmeter is not present the default is used" << endl;
  cout << "         Parameters (structure field names) are:" << endl;
  cout << "           k - nubmer of clusters [5]" << endl;
  cout << "           iterations - the maximum number of EM iterations [100]" << endl;
  cout << "           spherical - whether to perform sferical K-means (i.e. project the data and the centers on the unit sphere)." << endl;
  cout << "           seed - the random seed (0 to use time as a seed) [0]" << endl;
  cout << "           num_threads - the number of threads. 0 for using all threads. [0]" << endl; 
  cout << "           verbose - display progress messages. [0]" << endl;
}


DEFUN_DLD (oct_kmeans, args, nargout,
		"Interface to kmeans; performs kmeans")
{




  int nargin = args.length();
  if (nargin < 2)
    {
      print_usage();
      return octave_value_list(0);
    }

  int k = 5;
  int seed = 0;
  int iterations = 100;			       
  bool spherical = true; 
  int num_threads = 0;
  bool verbose = false;

  octave_scalar_map parameters = args(1).scalar_map_value(); // the parameter
  octave_value tmp;
  if (! error_state)
    {
      tmp = parameters.contents("k");
      if (tmp.is_defined())
	{
	  k = tmp.int_value();
	}
      tmp = parameters.contents("iterations");
      if (tmp.is_defined())
	{
	  iterations = tmp.int_value();
	}
      tmp = parameters.contents("seed");
      if (tmp.is_defined())
	{
	  seed = tmp.int_value();
	}
      tmp = parameters.contents("num_threads");
      if (tmp.is_defined())
	{
	  num_threads = tmp.int_value();
	}
      tmp = parameters.contents("verbose");
      if (tmp.is_defined())
	{
	  verbose = tmp.bool_value();
	}
    }

  /* initialize random seed: */
  if (seed)
    {
      srand(seed);
    }
  else
    {
      srand(time(NULL));
    }


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

  DenseColM centers;
  VectorXi assignments;
  
  if(args(0).is_sparse_type())
    {
      // Sparse data
      SparseM x = toEigenMat(args(0).sparse_matrix_value());      
      centers.resize(x.cols(),k);
      assignments.resize(x.rows());
      if (spherical)
	{
	  normalize_row(x);
	}
      run_kmeans(centers,assignments,x,iterations,spherical);
    }
  else
    {
      // Dense data
      FloatNDArray xArray = args(0).float_array_value();
      DenseM x = toEigenMat<DenseM>(xArray);
      centers.resize(x.cols(),k);
      assignments.resize(x.rows());
      if (spherical)
	{
	  x.rowwise().normalize();
	}
      run_kmeans(centers,assignments,x,iterations,spherical);      
    }

  octave_value_list retval(2);// return value
  retval(0) = toMatrix(centers);
  retval(1) = toIntArray(assignments);
  return retval;
}

