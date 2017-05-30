#ifndef __TYPEDEFS_H
#define __TYPEDEFS_H

#include "Eigen/Dense"
#include "Eigen/Sparse"
#include <boost/dynamic_bitset.hpp>
#include <boost/variant.hpp>
#include <omp.h>                        // omp_get_max_threads

//using   Eigen::VectorXd;
//using   Eigen::VectorXi;
//using   Eigen::RowVectorXd;
//using   Eigen::ColVectorXd;
typedef Eigen::Matrix<size_t, Eigen::Dynamic, 1> VectorXsz;
typedef Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> DenseM;
typedef Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::ColMajor> DenseColM;
typedef Eigen::Map<DenseM> ExtDenseM;
typedef Eigen::Map<DenseM const> ExtConstDenseM;
//typedef Eigen::Map<Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>> ExtConstDenseM;
//typedef Eigen::Map<Eigen::Matrix<double, -1, -1, 1, -1, -1> const, 0, Eigen::Stride<0, 0> > ExtConstDenseM;
//use float to save space
typedef Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> DenseMf;
typedef Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::ColMajor> DenseColMf;
typedef Eigen::Matrix< bool, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor > MatrixXb;
typedef Eigen::SparseMatrix<bool, Eigen::RowMajor> SparseMb;
typedef Eigen::SparseMatrix<double,Eigen::RowMajor>  SparseM;
typedef Eigen::MappedSparseMatrix<double, Eigen::RowMajor> ExtSparseM;
// typedef Eigen::MappedSparseMatrix<double const, Eigen::RowMajor> ExtConstSparseM; // Eigen has incomplete support!
typedef Eigen::MappedSparseMatrix<double, Eigen::RowMajor> ExtConstSparseM;

typedef std::vector<boost::dynamic_bitset<>*> ActiveDataSet;

typedef Eigen::SparseMatrix<float,Eigen::RowMajor> SparseMf;


//use float to save space
typedef float ovaCoeffType; 
typedef Eigen::Matrix<ovaCoeffType, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> ovaDenseRowM;
typedef Eigen::Matrix<ovaCoeffType, Eigen::Dynamic, Eigen::Dynamic, Eigen::ColMajor> ovaDenseColM;
typedef Eigen::Matrix<ovaCoeffType, 1, Eigen::Dynamic, Eigen::RowMajor> ovaDenseRowV;
typedef Eigen::Matrix<ovaCoeffType, Eigen::Dynamic, 1, Eigen::ColMajor> ovaDenseColV;
typedef Eigen::SparseMatrix<ovaCoeffType, Eigen::RowMajor, int64_t> ovaSparseRowM;
typedef Eigen::SparseMatrix<ovaCoeffType, Eigen::ColMajor, int64_t> ovaSparseColM;


typedef  boost::variant<ovaDenseColM, ovaSparseColM> ovaModel;



#endif
