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

    /** \name dynamic_bitset i/o
     * \b beware that boost and io_txt do BIG-ENDIAN output, </em>msb first</em> */
    //@{
#define TBITSET template<typename Block, typename Alloc>
#define BITSET  boost::dynamic_bitset<Block,Alloc>
    // boost provides a big-endian operator<< and operator>>, so default impl works
    //TBITSET std::ostream& io_txt( std::ostream& os, BITSET const& x, char const* ws="\n" );
    //TBITSET std::istream& io_txt( std::istream& is, BITSET      & x );
    TBITSET std::ostream& io_bin( std::ostream& os, BITSET const& x );
    TBITSET std::istream& io_bin( std::istream& is, BITSET      & x );
#undef BITSET
#undef TBITSET
    //@}

    /** \name std::array<T,N> i/o
     * specialized for array<char,N> to remove intervening ws. */
    //@{
#define TARRAY template<class T, std::size_t N>
    // std::array (fixed-size array, so size N is not part of stream)
    TARRAY std::ostream& io_txt( std::ostream& os, std::array<T,N> const& x, char const* ws="\n" );
    TARRAY std::istream& io_txt( std::istream& is, std::array<T,N>      & x );
    TARRAY std::ostream& io_bin( std::ostream& os, std::array<T,N> const& x );
    TARRAY std::istream& io_bin( std::istream& is, std::array<T,N>      & x );
#undef TARRAY
    //@}

    /** \name boolmatrix i/o.
     * - boolmatrix is NOT dynamically sized, so on input, the "dimensions"
     *   MUST already be known. So:
     *   - [*] DO NOT store rows, cols --> trivial equiv. of i/o with boolmatrix::[c]base()
     *   - or STORE them and use throw on mismatch. */
    //@{
    template<> std::ostream& io_txt( std::ostream& os, boolmatrix const& x, char const* ws/*="\n"*/ );
    template<> std::istream& io_txt( std::istream& is, boolmatrix& x );
    template<> std::ostream& io_bin( std::ostream& os, boolmatrix const& x );
    template<> std::istream& io_bin( std::istream& is, boolmatrix& x );
    //@}

    // Eigen DENSE matrix/vector/array
    // Note: RAW data i/o only -- transpose flags etc. will not be stored) Also see
    // http://stackoverflow.com/questions/22725867/eigen-type-deduction-in-template-specialization-of-base-class
    /** \name Eigen Dense I/O
     * - Eigen I/O for matrices [vectors] always with prepended row,col dimension.
     * - If dimensions known before-hand \em could have a different set of I/O routines
     * - Vectors get stored as single-column matrix
     * - txt i/o prepends matrix dimension (unlike operators <<, >>)
     * - \b Unsupported: binary i/o is coerced to/from float
     * - \b Unsupported: special flags for matrix (sym, etc.)
     * - \b Unsupported: retaining matrix properties (sym, row-wise, ...)
     */
    //@{
#define TMATRIX template<typename Derived>
#define MATRIX  Eigen::PlainObjectBase< Derived >
    TMATRIX std::ostream& eigen_io_bin( std::ostream& os, MATRIX const& x );
    TMATRIX std::istream& eigen_io_bin( std::istream& is, MATRIX      & x );
    TMATRIX std::ostream& eigen_io_txt( std::ostream& os, MATRIX const& x, char const *ws="\n" );
    TMATRIX std::istream& eigen_io_txt( std::istream& is, MATRIX      & x );
#undef MATRIX
#undef TMATRIX
    //@}
    /** \name Eigen Sparse I/O
     * - binary i/o is coerced to/from float
     * - output via Outer/InnerIterator (compressed) + innerNonZeroPtr (uncompressed)
     * - input always forms a compressed matrix
     * - \b Unsupported: special types of sparse matrix (sym,...)
     * - \b Unsupported: retaining matrix properties (sym, row-wise, ...)
     * \todo Eigen Sparse Idx i/o --> reduced byte size (in printing.hh)
     * \todo Eigen Sparse bool values are irrelevant (once pruned), so files can be very short
     */
    //@{
#define TMATRIX template<typename Scalar, int Options, typename Index>
#define MATRIX  Eigen::SparseMatrix< Scalar, Options, Index >
    TMATRIX std::ostream& eigen_io_bin( std::ostream& os, MATRIX const& x );
    TMATRIX std::istream& eigen_io_bin( std::istream& is, MATRIX      & x );
    TMATRIX std::ostream& eigen_io_txt( std::ostream& os, MATRIX const& x, char const *ws="\n" );
    TMATRIX std::istream& eigen_io_txt( std::istream& is, MATRIX      & x );
    template<int Options, typename Index> // bool override:
    std::istream& eigen_io_bin( std::istream& is, Eigen::SparseMatrix<bool,Options,Index> const& x );
    template<int Options, typename Index> // bool override:
    std::ostream& eigen_io_bin( std::ostream& os, Eigen::SparseMatrix<bool,Options,Index> const& x );
#undef MATRIX
#undef TMATRIX
    //@}
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

template< typename DERIVED >
std::string print_report(Eigen::SparseMatrixBase<DERIVED> const& x);     ///< nnz for sparse matrix

std::string print_report(const DenseM& x);      ///< empty string

void print_report(const int projection_dim, const int batch_size,
		  const int noClasses, const double C1, const double C2, const double lambda, const int w_size,
		  std::string x_report);
#endif
