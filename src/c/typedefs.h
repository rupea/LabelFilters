#ifndef __TYPEDEFS_H
#define __TYPEDEFS_H

typedef Eigen::Matrix<size_t, Eigen::Dynamic, 1> VectorXsz;
typedef Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> DenseM;
typedef Eigen::Matrix< bool, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor > MatrixXb;
typedef Eigen::SparseMatrix<bool, Eigen::RowMajor> SparseMb;
typedef Eigen::SparseMatrix<double,Eigen::RowMajor>  SparseM;

#endif
