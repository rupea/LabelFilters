#ifndef __EIGENOCTAVE_H
#define __EIGENOCTAVE_H
#include "typedefs.h"

using Eigen::VectorXd;
using Eigen::VectorXi;

using namespace std;


DenseM toEigenMat(const FloatNDArray& data);

SparseM toEigenMat(const Sparse<double>& data);

VectorXd toEigenVec(FloatNDArray data);

Matrix toMatrix(DenseM data);

SparseMatrix toMatrix(const SparseM &mat);

#endif
