#include <octave/oct.h>
#include <octave/Cell.h>
#include <octave/ov-struct.h>
#include <iostream>
#include "Eigen/Dense"
#include "Eigen/Sparse"
#include "typedefs.h"
#include "EigenOctave.h"

using Eigen::VectorXd;

// converts data from a cell array of vectors of the same length to
// a eigen dense matrix, with each vector representing a column of the
// matrix. Assumes the vectors are row vectors (maybe this can be relaxed)
// to do: include checks to make sure data is in the right format
// The result will be very large so we might want to avoid copying it. 

void toEigenMat(DenseColMf& m, const Cell& data){
  cout << "Reading Cell data into Eigen.." << endl;
  octave_idx_type cols = data.numel();
  assert(cols > 0);
  FloatNDArray coldata;
  octave_idx_type rows = data(0).scalar_map_value().contents("w").float_array_value().length();
  m.resize(rows, cols);
  for (size_t j = 0; j<cols; j++)
    {
      coldata = data(j).scalar_map_value().contents("w").float_array_value();
      assert(coldata.ndims() == 1);
      assert(coldata.length() == rows);
      for (size_t i = 0; i<rows; i++)
	{
	  m(i,j) = coldata(i);
	}
    }
  cout << "Done reading ..." << endl;
}

VectorXd toEigenVec(const FloatNDArray& data) {
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

int32NDArray toIntArray(const VectorXi& eigenVec) 
{
  int32NDArray v(dim_vector(eigenVec.size(),1));
  for (size_t i = 0; i < eigenVec.size(); i++) {
    v(i) = eigenVec(i);
  }
  return v;
}

int64NDArray toInt64Array(const VectorXsz& eigenVec) 
{
  int64NDArray v(dim_vector(eigenVec.size(),1));
  for (size_t i = 0; i < eigenVec.size(); i++) {
    v(i) = eigenVec(i);
  }
  return v;
}

Matrix toMatrix(const DenseM& data) 
{
  Matrix m(data.rows(), data.cols());
  for (int i = 0; i < data.rows(); i++) {
    for (int j = 0; j < data.cols(); j++) {
      m(i, j) = data(i, j);
    }
  }
  return m;
}

SparseMatrix toMatrix(const SparseM &mat) {
  assert(mat.nonZeros() <= dim_vector::dim_max());
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
