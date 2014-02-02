#include <octave/oct.h>
#include <iostream>
#include "Eigen/Dense"
#include "Eigen/Sparse"

using Eigen::VectorXd;

typedef Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> DenseM;
typedef Eigen::SparseMatrix<double,Eigen::RowMajor>  SparseM;


DenseM toEigenMat(const FloatNDArray& data) {
  dim_vector datasize = data.dims();

  DenseM m(datasize(0), datasize(1));
  for (int i = 0; i < datasize(0); i++) {
    for (int j = 0; j < datasize(1); j++) {
      m(i, j) = data(i, j);
    }
  }

  return m;
}

SparseM toEigenMat(const Sparse<double>& data) {
  dim_vector datasize = data.dims();
  
  std::cout << data.rows() << ", " << data.cols() << std::endl;
  
  int nr = data.rows();
  int nc = data.cols();
  int nnz = data.nnz();
  double d;

  std::vector< Eigen::Triplet<double> > tripletList;
  tripletList.reserve(nnz);
  
  for (octave_idx_type j = 0; j < nc; j++) {
    for (octave_idx_type i = data.cidx(j); i < data.cidx(j + 1); i++) {
      d = data.data(i);
      if (d != 0) {
	tripletList.push_back(Eigen::Triplet<double> ((int) data.ridx(i), (int) j, d));
      }
    }
  }

  SparseM m(nr, nc);
  m.setFromTriplets(tripletList.begin(), tripletList.end());
  cout << "data is read to Eigen ... \n";
  return m;
}

VectorXd toEigenVec(FloatNDArray data) {
  dim_vector datasize = data.dims();
  int dim = datasize(0);
  int idx = 0;
  if (datasize(1) > dim) {
    dim = datasize(1);
    idx = 1;
  }
  VectorXd v(dim);
  for (int i = 0; i < dim; i++) {
    if (idx == 0)
      v[i] = data(i, 0);
    else
      v[i] = data(0, i);
  }

  return v;
}

Matrix toMatrix(DenseM data) {
  Matrix m(data.rows(), data.cols());

  for (int i = 0; i < data.rows(); i++) {
    for (int j = 0; j < data.cols(); j++) {
      m(i, j) = data(i, j);
    }
  }

  return m;
}

SparseMatrix toMatrix(const SparseM &mat) {
  Array<octave_idx_type> rows(dim_vector(mat.nonZeros(),1)),cols(dim_vector(mat.nonZeros(),1));
  Array<double> vals(dim_vector(mat.nonZeros(),1));
	
  int i = 0;
  for (int k = 0; k < mat.outerSize(); ++k) {
    for (SparseM::InnerIterator it(mat, k); it; ++it) {
      rows(i)=it.row();
      cols(i)=it.col();
      vals(i)=it.value();
      i++;
    }
  }
  
  SparseMatrix m(vals,rows,cols,mat.rows(),mat.cols(),true,mat.nonZeros());
  return m;

}
