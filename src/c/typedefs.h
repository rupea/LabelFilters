#ifndef __TYPEDEFS_H
#define __TYPEDEFS_H

typedef Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> DenseM;
typedef Eigen::Matrix< bool, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor > MatrixXb;
typedef Eigen::SparseMatrix<double,Eigen::RowMajor>  SparseM;

#endif
