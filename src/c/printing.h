/*  Copyright (C) 2017 NEC Laboratories America, Inc. ("NECLA"). All rights reserved.
 *
 * This source code is licensed under the license found in the LICENSE file in
 * the root directory of this source tree. An additional grant of patent rights
 * can be found in the PATENTS file in the same directory.
 */
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

class Roaring;

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

    /** \name Roaring i/o */
    //@{
    std::ostream& io_txt( std::ostream& os, Roaring const& x, char const* ws = "\n" );
    std::istream& io_txt( std::istream& is, Roaring      & x );
    std::ostream& io_txt( std::ostream& os, std::vector<Roaring> const& x, char const* ws = "\n" );
    std::istream& io_txt( std::istream& os, std::vector<Roaring>      & x );
    std::ostream& io_bin( std::ostream& os, Roaring const& x );
    std::istream& io_bin( std::istream& is, Roaring      & x );
    std::ostream& io_bin( std::ostream& os, std::vector<Roaring> const& x );
    std::istream& io_bin( std::istream& os, std::vector<Roaring>      & x );
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
    std::ostream& eigen_io_bin( std::ostream& os, Eigen::SparseMatrix<bool,Options,Index> const& x );
    template<int Options, typename Index> // bool override:
    std::istream& eigen_io_bin( std::istream& is, Eigen::SparseMatrix<bool,Options,Index>      & x );
#undef MATRIX
#undef TMATRIX
    extern std::array<char,4> magicSparseMbBin; ///< "SMbb"
    extern std::array<char,4> magicSparseMbTxt; ///< "SMbt"
    /** Compressed SparseMb output, with magic header bytes and only-true values.
     * \throw if x has any false values or if x is uncompressed. */
    std::ostream& eigen_io_txtbool( std::ostream& os, SparseMb const& x );
    /** read a SparseMb matrix of only-true values into compressed \c x.
     * \throw if bad magic header bytes. */
    std::istream& eigen_io_txtbool( std::istream& is, SparseMb      & x );
    /** Compressed SparseMb output, with magic header bytes and only-true values.
     * \throw if x has any false values or if x is uncompressed. */
    std::ostream& eigen_io_binbool( std::ostream& os, SparseMb const& x );
    /** read a SparseMb matrix of only-true values into compressed \c x.
     * \throw if bad magic header bytes. */
    std::istream& eigen_io_binbool( std::istream& is, SparseMb      & x );
    //@}

    /** input only - read text libsvm-format. \throw on error. */
    template< typename X_REAL >
    std::istream& eigen_read_libsvm( std::istream& is,
                                     typename Eigen::SparseMatrix<X_REAL,Eigen::RowMajor> &x,
                                     Eigen::SparseMatrix<bool,Eigen::RowMajor> &y );

}//detail::

// from EigenIO.h -- actually this is generic, not related to Eigen
// TODO portable types
template <typename Block, typename Alloc>
  void save_bitvector(std::ostream& out, const boost::dynamic_bitset<Block, Alloc>& bs);

template <typename Block, typename Alloc>
  int load_bitvector(std::istream& in, boost::dynamic_bitset<Block, Alloc>& bs);

void dumpFeasible(std::ostream& os
		  , std::vector<boost::dynamic_bitset<>> const& vbs
		  , bool denseFmt=false);
  

/// \name misc pretty printing
//@{
/** Prints the progress bar */
void print_progress(std::string s, int t, int max_t);

/** alt matrix dimension printer with an operator<<, as [MxN] */
struct PrettyDimensions {
    friend std::ostream& operator<<(std::ostream& os, PrettyDimensions const& pd);
    static uint_least8_t const maxDim=3U;
    size_t dims[maxDim];
    uint_least8_t dim;
};

template<typename EigenType> PrettyDimensions prettyDims( EigenType const& x );

std::ostream& operator<<(std::ostream& os, PrettyDimensions const& pd);

template< typename DERIVED >
std::string print_report(Eigen::SparseMatrixBase<DERIVED> const& x);     ///< nnz for sparse matrix

template< typename EigenType > inline
std::string print_report(EigenType const& x);      ///< empty string

void print_report(const int projection_dim, const int batch_size,
		  const int noClasses, const double C1, const double C2, const double lambda, const int w_size,
		  std::string x_report);
//@}

#endif
