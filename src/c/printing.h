#ifndef __PRINTING_H
#define __PRINTING_H

#include "typedefs.h"
#include <string>
//#include <iosfwd>
#include <iostream>

//using Eigen::VectorXd;
//using namespace std;

namespace detail {
    template<typename T> inline std::ostream& io_txt( std::ostream& os, T const& x, char const* ws="\n" );
    template<typename T> inline std::istream& io_txt( std::istream& is, T& x );
    template<typename T> inline std::ostream& io_bin( std::ostream& os, T const& x );
    template<typename T> inline std::istream& io_bin( std::istream& is, T& x );

    // specializations
    //   strings as length + blob (no intervening space)
    template<> std::ostream& io_txt( std::ostream& os, std::string const& x, char const* /*ws="\n"*/ );
    template<> std::istream& io_txt( std::istream& is, std::string& x );
    template<> std::ostream& io_bin( std::ostream& os, std::string const& x );
    template<> std::istream& io_bin( std::istream& is, std::string& x );
}

// *******************************
// Prints the progress bar
void print_progress(std::string s, int t, int max_t);

void print_mat_size(const Eigen::VectorXd& mat);

template<typename EigenType>
void print_mat_size(const EigenType& mat)
{
  using namespace std;
  cout << "(" << mat.rows() << ", " << mat.cols() << ")";
}

void print_report(const SparseM& x);

void print_report(const DenseM& x);


template<typename EigenType>
void print_report(const int projection_dim, const int batch_size,
		  const int noClasses, const double C1, const double C2, const double lambda, const int w_size,
		  const EigenType& x)
{
  using namespace std;
  cout << "projection_dim: " << projection_dim << ", batch_size: "
       << batch_size << ", noClasses: " << noClasses << ", C1: " << C1
       << ", C2: " << C2 << ", lambda: " << lambda << ", size w: " << w_size << ", ";
  print_report(x);
  cout << "\n-----------------------------\n";

}
#endif
