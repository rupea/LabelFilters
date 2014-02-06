#ifndef __PRINTING_H
#define __PRINTING_H

#include "typedefs.h"

using Eigen::VectorXd;
using namespace std;

// *******************************
// Prints the progress bar
void print_progress(string s, int t, int max_t);

void print_mat_size(const Eigen::VectorXd& mat);

template<typename EigenType>
void print_mat_size(const EigenType& mat)
{
  cout << "(" << mat.rows() << ", " << mat.cols() << ")";
}

void print_report(const SparseM& x);

void print_report(const DenseM& x);


template<typename EigenType>
void print_report(const int& projection_dim, const int& batch_size,
		  const int& noClasses, const int& C1, const int& C2, const int& w_size,
		  const EigenType& x)
{
  cout << "projection_dim: " << projection_dim << ", batch_size: "
       << batch_size << ", noClasses: " << noClasses << ", C1: " << C1
       << ", C2: " << C2 << ", size w: " << w_size << ", ";
  print_report(x);
  cout << "\n-----------------------------\n";

}
#endif