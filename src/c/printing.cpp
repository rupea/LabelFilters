#include <iostream>
#include <stdio.h>
#include "Eigen/Dense"
#include "Eigen/Sparse"
#include "typedefs.h"
#include "printing.h"

using Eigen::VectorXd;
using namespace std;

// *******************************
// Prints the progress bar
void print_progress(string s, int t, int max_t)
{
  double p = ((double) ((double) t * 100.0)) / (double) max_t;
  int percent = (int) p;

  string str = "\r" + s + "=";
  for (int i = 0; i < (int) percent / 10; i++)
    {
      str = str + "==";
    }
  
  int c = 60;
  char buff[c];
  sprintf(buff,
	  " > (%d\%) @%d                                                 ",
	  percent, t);
  str = str + buff;
  
  cout << str;
  
  if (percent == 100)
    {
      cout << std::endl;
    }
}

void print_mat_size(const Eigen::VectorXd& mat)
{
  cout << "(" << mat.size() << ")";
}

/********* template functions are implemented in the header
template<typename EigenType>
void print_mat_size(const EigenType& mat);
**********/

void print_report(const SparseM& x)
{
  int nnz = x.nonZeros();
  cout << "x:non-zeros: " << nnz << ", avg. nnz/row: " << nnz / x.rows();
}

void print_report(const DenseM& x)
{
}

/********* template functions are implemented in the header
template<typename EigenType>
void print_report(const int& projection_dim, const int& batch_size,
		  const int& noClasses, const int& C1, const int& C2, const int& w_size,
		  const EigenType& x);
*******************/
