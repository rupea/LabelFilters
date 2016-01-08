#ifndef __PRINTING_H
#define __PRINTING_H
/** \file
 * IO helpers -- not endian-safe, but please use these with portably-sized types.
 */

#include "typedefs.h"
#include <boost/dynamic_bitset.hpp>
#include <string>
#include <iosfwd>

namespace detail {
    template<typename T> inline std::ostream& io_txt( std::ostream& os, T const& x, char const* ws="\n" );
    template<typename T> inline std::istream& io_txt( std::istream& is, T      & x );
    template<typename T> inline std::ostream& io_bin( std::ostream& os, T const& x );
    template<typename T> inline std::istream& io_bin( std::istream& is, T      & x );

    // specializations
    //   strings as length + blob (no intervening space)
    template<> std::ostream& io_txt( std::ostream& os, std::string const& x, char const* /*ws="\n"*/ );
    template<> std::istream& io_txt( std::istream& is, std::string& x );
    template<> std::ostream& io_bin( std::ostream& os, std::string const& x );
    template<> std::istream& io_bin( std::istream& is, std::string& x );

#define TBITSET template<typename Block, typename Alloc>
#define BITSET  boost::dynamic_bitset<Block,Alloc>
    // dynamic_bitset has >>, << so default io_txt should be just fine
    // BUT boost << does not skipws, so...
    //TBITSET std::ostream& io_txt( std::ostream& os, BITSET const& x, char const* ws="\n" );
    //TBITSET std::istream& io_txt( std::istream& is, BITSET      & x );
    TBITSET std::ostream& io_bin( std::ostream& os, BITSET const& x );
    TBITSET std::istream& io_bin( std::istream& is, BITSET      & x );
#undef BITSET
#undef TBITSET
}

// from EigenIO.h -- actually this is generic, not related to Eigen
// TODO portable types
template <typename Block, typename Alloc>
  void save_bitvector(std::ostream& out, const boost::dynamic_bitset<Block, Alloc>& bs);

template <typename Block, typename Alloc>
  int load_bitvector(std::istream& in, boost::dynamic_bitset<Block, Alloc>& bs);

/** Prints the progress bar */
void print_progress(std::string s, int t, int max_t);

/** "(rows,cols)" -- ok for Vector or Matrix EigenType */
template<typename EigenType> inline void print_mat_size(const EigenType& mat);

void print_report(const SparseM& x);

void print_report(const DenseM& x);

template<typename EigenType> inline
void print_report(const int projection_dim, const int batch_size,
		  const int noClasses, const double C1, const double C2, const double lambda, const int w_size,
		  const EigenType& x);
#endif
