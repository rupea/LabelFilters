#include <iostream>
#include <math.h>
#include "Eigen/Dense"
#include "Eigen/Sparse"
#include "constants.h"
#include "typedefs.h"
#include "printing.h"
#include "normalize.h"

using namespace std;



// *************************
// Normalize data : centers and makes sure the variance is one
void normalize_col(SparseM& mat)
{
  using std::isnan;     // perhaps more efficient (constexpr)
  if (mat.rows() < 2)
    return;

  double v = 0, m = 0, delta;

  cout << "normalizing sparse matrix with " << mat.rows() << " rows and "
       << mat.cols() << "cols ... " <<endl;

  // Lets first compute the mean and the square of the values and center each column
  for (int k = 0; k < mat.outerSize(); ++k)
    {
      m = 0;
      v = 0;

      print_progress("normalizing ...", k, mat.outerSize());

      // online variance and mean calculation
      for (SparseM::InnerIterator it(mat, k); it; ++it)
	{
	  delta = it.value() - m;
	  m = m + delta / (it.row() + 1);
	  v = v + delta * (it.value() - m);
	}

      v = sqrt(v / (mat.rows() - 1));

      if (v < 0 || isnan(v))
	continue;

      for (SparseM::InnerIterator it(mat, k); it; ++it)
	{
	  it.valueRef() -= m;
	  if (isnan(it.value()))
	    it.valueRef() = 0;

	  if (!isnan(it.value() / v))
	    it.valueRef() = it.value() / v;

	}
    }

  cout << "done ... data is normalized ... \n";
}

void normalize_col(DenseM& mat)
{
  if (mat.rows() < 2)
    return;

  if (PRINT_M)
    cout << "normalizing sparse matrix with " << mat.rows() << " rows and "
	 << mat.cols() << "cols ..." << endl;

  mat = mat.rowwise() - mat.colwise().mean();		// first center the data
  DenseM v = mat.colwise().squaredNorm() / (mat.rows() - 1); // compute the sqruare of standard deviation
  for (int i = 0; i < mat.cols(); i++)
    {
      print_progress("normalizing ...", i, mat.cols());

      mat.col(i) = mat.col(i) / sqrt(v(i)); // devide each column by the standard deviation
    }

  cout << "done ... data is normalized ... "<<endl;
}


void normalize_row(SparseM &mat)
{
  if (mat.rows() < 2)
    return;

  if (PRINT_M)
    cout << "normalizing sparse matrix with " << mat.rows() << " rows and "
	 << mat.cols() << "cols ... \n"<<endl;

  double norm;
	
  for (int i = 0; i < mat.rows(); i++)
    {
      norm = mat.row(i).norm();
      if (norm != 0)
	{
	  mat.row(i) /= norm;
	}
    }
	
  cout << "done ... data is normalized ... "<<endl;
}
