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
  
  int c = 1000;
  char buff[c];
  sprintf(buff,
	  " > (%d%%) @%d                     ",
	  percent, t);
  str = str + buff;
  
  cout << str;
  
  if (percent == 100)
    {
      cout << std::endl;
    }
}

/********* template functions are implemented in the header
template<typename EigenType>
void print_mat_size(const EigenType& mat);
**********/

std::string print_report(const DenseM& x)
{
  return std::string();
}

void print_report(const int projection_dim, const int batch_size,
                  const int noClasses, const double C1, const double C2, const double lambda, const int w_size,
                  std::string x_report) //const EigenType& x)
{
    using namespace std;
    cout << "projection_dim: " << projection_dim << ", batch_size: "
        << batch_size << ", noClasses: " << noClasses << ", C1: " << C1
        << ", C2: " << C2 << ", lambda: " << lambda << ", size w: " << w_size;
    if(x_report.size()) cout<< ", "<<x_report; // print_report(x);
    cout << "\n-----------------------------\n";

}

/********* template functions are implemented in the header
template<typename EigenType>
void print_report(const int& projection_dim, const int& batch_size,
		  const int& noClasses, const int& C1, const int& C2, const int& w_size,
		  const EigenType& x);
*******************/
