#include "normalize.h"
// [ejk] make this guy reasonable standalone...
//#include "Eigen/Dense"
//#include "Eigen/Sparse"
//#include "constants.h"
#include "typedefs.h"
//#include "printing.h"
#include <iostream>
//#include <math.h>
//
#if !defined(PRINT_M)
#define PRINT_M 0
#endif

using namespace std;


// *************************
// Normalize data : centers and makes sure the variance is one
void normalize_row_remove_mean(SparseM& mat)
{
#if 0 // no longer needed
  using std::isnan;     // perhaps more efficient (constexpr)
  if (mat.rows() < 2)   // OUCH! why not at least remove the mean?
    return;
#endif

  double v = 0, m = 0, delta;

  cout << "normalizing sparse matrix with " << mat.rows() << " rows and "
       << mat.cols() << "cols ... " <<endl;

  // Lets first compute the mean and the square of the values and center each column
  for (int k = 0; k < mat.outerSize(); ++k)
    {
      m = 0;
      v = 0;

      //print_progress("normalizing ...", k, mat.outerSize());

      // online variance and mean calculation -- THIS IS JUST PLAIN ** WRONG ** [ejk]
#if 0
      for (SparseM::InnerIterator it(mat, k); it; ++it)
	{
	  delta = it.value() - m;
	  m = m + delta / (it.row() + 1);       // [ejk] OUCH!
	  v = v + delta * (it.value() - m);
	}
#else
      size_t nnz=0U;
      for (SparseM::InnerIterator it(mat, k); it; ++it) {
          delta = it.value() - m;
          ++nnz;
          m += delta / nnz;
          v += delta * (it.value() - m);
          cout<<" it="<<it.value()<<" m,v="<<m<<","<<v;
      }
#endif

#if 0 // and this is also not robust at all (but correct if no problems) [ejk]
      v = sqrt(v / (mat.rows() - 1));
      cout<<" outer k="<<k<<" m="<<m<<" v="<<v<<endl;
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
#else // Better: Just don't add any *new* NaN values.
      if( mat.rows() < 2 || v < 1.e-10) v=1.0;  // get value for
      else v = sqrt((mat.rows()-1)/v);          // 1.0/stdev, if possible
      cout<<" v^{-1}="<<v;
      for (SparseM::InnerIterator it(mat, k); it; ++it) {
          it.valueRef() = (it.value() - m) * v; // remove mean & stdev
      }
      cout<<endl;

#endif
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
      //print_progress("normalizing ...", i, mat.cols());

      mat.col(i) = mat.col(i) / sqrt(v(i)); // devide each column by the standard deviation
    }

  cout << "done ... data is normalized ... "<<endl;
}


/** renamed because it does not remove mean at all */
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

void normalize_col(SparseM &mat){
    throw std::runtime_error(" normalize_col(SparseM&) TBD - convert to DenseM, or write it");
}
