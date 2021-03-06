/*  Copyright (C) 2017 NEC Laboratories America, Inc. ("NECLA"). All rights reserved.
 *
 * This source code is licensed under the license found in the LICENSE file in
 * the root directory of this source tree. An additional grant of patent rights
 * can be found in the PATENTS file in the same directory.
 */
#ifndef __TYPEDEFS_H
#define __TYPEDEFS_H

#include "Eigen/Dense"
#include "Eigen/Sparse"
#include <omp.h>                        // omp_get_max_threads

// forward declarations
class Roaring;

typedef Eigen::Matrix<size_t, Eigen::Dynamic, 1> VectorXsz;
typedef Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> DenseM;
typedef Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::ColMajor> DenseColM;
typedef Eigen::Map<DenseM> ExtDenseM;
typedef Eigen::Map<DenseM const> ExtConstDenseM;

//use float to save space
typedef Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> DenseMf;
typedef Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::ColMajor> DenseColMf;
typedef Eigen::Matrix< bool, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor > MatrixXb;
typedef Eigen::SparseMatrix<bool, Eigen::RowMajor> SparseMb;
typedef Eigen::SparseMatrix<double,Eigen::RowMajor>  SparseM;
typedef Eigen::Map<Eigen::SparseMatrix<double, Eigen::RowMajor>> ExtSparseM;
typedef Eigen::Map<Eigen::SparseMatrix<double, Eigen::RowMajor>> ExtConstSparseM;

typedef std::vector<Roaring> ActiveSet;

typedef Eigen::SparseMatrix<float,Eigen::RowMajor> SparseMf;


//use float to save space
typedef float ovaCoeffType;
typedef float predtype;


typedef Eigen::Matrix<ovaCoeffType, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> ovaDenseRowM;
typedef Eigen::Matrix<ovaCoeffType, Eigen::Dynamic, Eigen::Dynamic, Eigen::ColMajor> ovaDenseColM;
typedef Eigen::Matrix<ovaCoeffType, 1, Eigen::Dynamic, Eigen::RowMajor> ovaDenseRowV;
typedef Eigen::Matrix<ovaCoeffType, Eigen::Dynamic, 1, Eigen::ColMajor> ovaDenseColV;
typedef Eigen::SparseMatrix<ovaCoeffType, Eigen::RowMajor, int64_t> ovaSparseRowM;

typedef Eigen::SparseMatrix<ovaCoeffType, Eigen::ColMajor, int64_t> ovaSparseColM;


#endif
