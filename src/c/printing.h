#ifndef __PRINTING_H
#define __PRINTING_H
/** \file
 * IO helpers -- not endian-safe, but please use these with portably-sized types.
 */

#include "typedefs.h"
#include "boolmatrix.h"
#include <boost/dynamic_bitset.hpp>
#include <string>
#include <iosfwd>

namespace detail {
    template<typename T> inline std::ostream& io_txt( std::ostream& os, T const& x, char const* ws="\n" );
    template<typename T> inline std::istream& io_txt( std::istream& is, T      & x );
    template<typename T> inline std::ostream& io_bin( std::ostream& os, T const& x );
    template<typename T> inline std::istream& io_bin( std::istream& is, T      & x );

    inline std::ostream& io_bin( std::ostream& os, void const* p, size_t bytes );
    inline std::istream& io_bin( std::istream& is, void      * p, size_t bytes );

    // specializations
    //   strings as length + blob (no intervening space)
    template<> std::ostream& io_txt( std::ostream& os, std::string const& x, char const* /*ws="\n"*/ );
    template<> std::istream& io_txt( std::istream& is, std::string& x );
    template<> std::ostream& io_bin( std::ostream& os, std::string const& x );
    template<> std::istream& io_bin( std::istream& is, std::string& x );

    // boost::dynamic_bitset
#define TBITSET template<typename Block, typename Alloc>
#define BITSET  boost::dynamic_bitset<Block,Alloc>
    // io_txt defaults are OK for dynamic_bitset
    TBITSET std::ostream& io_bin( std::ostream& os, BITSET const& x );
    TBITSET std::istream& io_bin( std::istream& is, BITSET      & x );
#undef BITSET
#undef TBITSET

    /** boolmatrix i/o.
     * - boolmatrix is NOT dynamically sized, so on input, the "dimensions"
     *   MUST already be known. So:
     *   - [*] DO NOT store rows, cols ...
     *   - or STORE them and use throw on mismatch. */
    template<> std::ostream& io_txt( std::ostream& os, boolmatrix const& x, char const* ws/*="\n"*/ );
    template<> std::istream& io_txt( std::istream& is, boolmatrix& x );
    template<> std::ostream& io_bin( std::ostream& os, boolmatrix const& x );
    template<> std::istream& io_bin( std::istream& is, boolmatrix& x );

    // Eigen DENSE matrix/vector/array
    // Note: RAW data i/o only -- transpose flags etc. will not be stored) Also see
    // http://stackoverflow.com/questions/22725867/eigen-type-deduction-in-template-specialization-of-base-class
#define TMATRIX template<typename Derived>
#define MATRIX  Eigen::PlainObjectBase< Derived >
    TMATRIX std::ostream& eigen_io_bin( std::ostream& os, MATRIX const& x );
    TMATRIX std::istream& eigen_io_bin( std::istream& is, MATRIX      & x );
#undef MATRIX
#undef TMATRIX
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
