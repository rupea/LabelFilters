#ifndef __EIGENOCTAVE_H
#define __EIGENOCTAVE_H

#include "typedefs.h"   // pulls in Eigen Dense and Sparse types

#include <octave/oct.h>
// NOTE: octave has its own matrix and array types, apart from Eigen:
//    int32NDArray Matrix SparseMatrix ...

#include <stdexcept>

// TODO most of these have NOTHING to do with octave (move to EigenUtil.h or EigenIO.h ?)

/** this DOES use octave */
void load_projections(DenseColM& wmat, DenseColM& lmat, DenseColM& umat, const std::string& filename, DenseColM::Scalar bias, bool verbose);

inline void write_projections(DenseColM& wmat, DenseColM& lmat, DenseColM& umat, const std::string& filename, bool verbose = false)
{
    throw std::runtime_error("Unimplemented: write_projections");
}

inline void read_projections(DenseColM& wmat, DenseColM& lmat, DenseColM& umat, const std::string& filename, bool verbose = false)
{
    throw std::runtime_error("Unimplemented: write_projections");
}


Matrix       toMatrix(const DenseM& data);
SparseMatrix toMatrix(const SparseM &mat);

int32NDArray toIntArray(const Eigen::VectorXi& eigenVec);
int64NDArray toInt64Array(const VectorXsz& eigenVec);

Eigen::VectorXd toEigenVec(const FloatNDArray& data);

template<typename DenseMatType, typename OctDenseMatType> DenseMatType toEigenMat(OctDenseMatType const& data, typename DenseMatType::Scalar bias=0);
//template<typename DenseMatType> DenseMatType                           toEigenMat(const NDArray& data);
template<typename Scalar> Eigen::SparseMatrix<Scalar, Eigen::RowMajor> toEigenMat(const Sparse<Scalar>& data, Scalar bias = 0);
/** converts data from a cell array of vectors of the same length to
 * a eigen dense matrix, with each vector representing a column of the
 * matrix. Assumes the vectors are row vectors (maybe this can be relaxed)
 * to do: include checks to make sure data is in the right format
 * The result will be very large so we might want to avoid copying it. */
void toEigenMat(DenseColMf& m, const Cell& data);

// ------------------------- inline template impls ----------------------

template<typename DenseMatType, typename OctDenseMatType> inline 
  DenseMatType toEigenMat(OctDenseMatType const& data, typename DenseMatType::Scalar bias) {
  dim_vector datasize = data.dims();

  size_t rows = datasize(0);
  size_t cols = datasize(1);
  if(bias)
    {
      cols++;
    }
  DenseMatType m(rows,cols);
  for (size_t i = 0; i < datasize(0); i++) {
    for (size_t j = 0; j < datasize(1); j++) {
      m(i, j) = data(i, j);
    }
  }
  // add a constant entry at the last column to serve as a bias
  if (bias)
    {      
      for (size_t i=0 ; i<datasize(0); i++)
	{
	  m(i,datasize(1)) = bias;
	}
    }
  return m;
}

/* template<typename DenseMatType> inline  */
/* DenseMatType toEigenMat(const NDArray& data) { */
/*   dim_vector datasize = data.dims(); */

/*   DenseMatType m(datasize(0), datasize(1)); */
/*   for (int i = 0; i < datasize(0); i++) { */
/*     for (int j = 0; j < datasize(1); j++) { */
/*       m(i, j) = data(i, j); */
/*     } */
/*   } */
/*   return m; */
/* } */


template<typename Scalar> inline
Eigen::SparseMatrix<Scalar, Eigen::RowMajor> toEigenMat(const Sparse<Scalar>& data, Scalar bias) {
  using namespace std;
  dim_vector datasize = data.dims();
  
  std::cout << data.rows() << ", " << data.cols() << std::endl;
  
  size_t nr = data.rows();
  size_t nc = data.cols();
  size_t nnz = data.nnz();
  Scalar d;

  std::vector< Eigen::Triplet<Scalar> > tripletList;
  tripletList.reserve(nnz);
  
  for (octave_idx_type j = 0; j < nc; j++) {
    for (octave_idx_type i = data.cidx(j); i < data.cidx(j + 1); i++) {
      d = data.data(i);
      if (d != 0) {
	tripletList.push_back(Eigen::Triplet<Scalar> ((int) data.ridx(i), (int) j, d));
      }
    }
  }
  // add a constant entry at the last column to serve as an intercept
  if (bias) 
    {
      for (size_t i = 0; i < nr; i++)
	{
	  tripletList.push_back(Eigen::Triplet<Scalar> (i, nc, bias));
	}
      nc++;
    }
  cout << nc << endl;
  Eigen::SparseMatrix<Scalar, Eigen::RowMajor>  m(nr, nc);
  m.setFromTriplets(tripletList.begin(), tripletList.end());
  cout << "data is read to Eigen ... " << endl;
  return m;
};


#endif
