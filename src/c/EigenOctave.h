#ifndef __EIGENOCTAVE_H
#define __EIGENOCTAVE_H
#include "typedefs.h"

using Eigen::VectorXd;
using Eigen::VectorXi;

using namespace std;


void toEigenMat(DenseColMf& m, const Cell& data);

// templetize this and maybe use MatrixBase to avoid code dupplication
template<typename DenseMatType>
DenseMatType toEigenMat(const FloatNDArray& data) {
  dim_vector datasize = data.dims();

  DenseMatType m(datasize(0), datasize(1));
  for (int i = 0; i < datasize(0); i++) {
    for (int j = 0; j < datasize(1); j++) {
      m(i, j) = data(i, j);
    }
  }
  return m;
}


template<typename Scalar>
Eigen::SparseMatrix<Scalar, Eigen::RowMajor> toEigenMat(const Sparse<Scalar>& data) {
  dim_vector datasize = data.dims();
  
  std::cout << data.rows() << ", " << data.cols() << std::endl;
  
  int nr = data.rows();
  int nc = data.cols();
  int nnz = data.nnz();
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

  Eigen::SparseMatrix<Scalar, Eigen::RowMajor>  m(nr, nc);
  m.setFromTriplets(tripletList.begin(), tripletList.end());
  cout << "data is read to Eigen ... \n";
  return m;
};



VectorXd toEigenVec(const FloatNDArray& data);

Matrix toMatrix(const DenseM& data);

SparseMatrix toMatrix(const SparseM &mat);

#endif
