#ifndef __TYPEDEFS_H
#define __TYPEDEFS_H

#include "Eigen/Dense"
#include "Eigen/Sparse"
#include <boost/dynamic_bitset.hpp>

using   Eigen::VectorXd;
using   Eigen::VectorXi;
//using   Eigen::RowVectorXd;
//using   Eigen::ColVectorXd;
typedef Eigen::Matrix<size_t, Eigen::Dynamic, 1> VectorXsz;
typedef Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> DenseM;
typedef Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::ColMajor> DenseColM;
//use float to save space
typedef Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> DenseMf;
typedef Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::ColMajor> DenseColMf;
typedef Eigen::Matrix< bool, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor > MatrixXb;
typedef Eigen::SparseMatrix<bool, Eigen::RowMajor> SparseMb;
typedef Eigen::SparseMatrix<double,Eigen::RowMajor>  SparseM;

typedef std::vector<boost::dynamic_bitset<>*> ActiveDataSet;

#endif
