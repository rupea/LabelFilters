#ifndef __EIGENOCTAVE_H
#define __EIGENOCTAVE_H
#include "typedefs.h"

using Eigen::VectorXd;
using Eigen::VectorXi;

using namespace std;


DenseM toEigenMat(const FloatNDArray& data);

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



VectorXd toEigenVec(FloatNDArray data);

Matrix toMatrix(DenseM data);

SparseMatrix toMatrix(const SparseM &mat);

#endif
