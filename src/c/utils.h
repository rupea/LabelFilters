#ifndef __UTILS_H
#define __UTILS_H

#include "typedefs.h"
#include <exception>
#include <iostream>

#define OUTWIDE( OSTREAM, WIDTH, STUFF ) do{ std::ostringstream oss; oss<<STUFF; OSTREAM<<setw(WIDTH)<<oss.str(); }while(0)



// *********************************
// functions and structures for sorting and keeping indices

template<typename IntType>
void sort_index( Eigen::VectorXd const& m, std::vector<IntType>& cranks)
{
  if( cranks.size() != static_cast<size_t>(m.size()) )
    throw std::runtime_error("ERROR: sort_index(vec,ranks): vec and ranks sizes must match");
  if( m.size() == 0U )
    throw std::runtime_error("ERROR: sort_index( m, cranks ) with m.size==0 !");
  std::iota(cranks.begin(), cranks.end(), IntType(0));
  std::sort(cranks.begin(), cranks.end(), [&m](int const i, int const j)
            {return m[i] < m[j];} );
};


//**********************************
// x.row(i) will result in a dense vector even if x is sparse. 
// Functions below are performing addition and dot products with a 
// row of a sparse matrix efficiently
// these functions should be not needed once eigen is updated to have 
// better support for sparse matrices. 

template <typename EigenType>
void addInnerVector (Eigen::Ref<Eigen::VectorXd> addto, const EigenType& addfrom, size_t outerIndex)
{
  typename EigenType::InnerIterator iter(addfrom,outerIndex); 
  for  (; iter; ++iter) 
    {
      addto(iter.index())+=iter.value();
    }  
}

template <typename Scalar1, typename IndexType1,  typename Scalar2, typename IndexType2> inline
  double DotProductInnerVector (const Eigen::SparseMatrix<Scalar1, Eigen::RowMajor, IndexType1>& rowmat, const Eigen::Index row, const Eigen::SparseMatrix<Scalar2, Eigen::ColMajor, IndexType2>& colmat, const Eigen::Index col)
{
  assert(rowmat.cols()==colmat.rows());
  typename Eigen::SparseMatrix<Scalar1, Eigen::RowMajor, IndexType1>::InnerIterator iter1(rowmat,row);
  typename Eigen::SparseMatrix<Scalar2, Eigen::ColMajor, IndexType2>::InnerIterator iter2(colmat,col);
  double ans = 0.0;
  while (iter1 && iter2)
    {
      if (iter1.index() == iter2.index())
  	{
  	  ans+=iter1.value()*iter2.value();
  	  ++iter1;
  	  ++iter2;
  	}
      else if (iter1.index() < iter2.index())
  	{
  	  ++iter1;
  	}
      else
  	{
  	  ++iter2;
  	}
    }
  return ans;
}

template <typename Scalar1, typename Scalar2, typename IndexType2> inline
  double DotProductInnerVector (const Eigen::Matrix<Scalar1, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>& rowmat, const Eigen::Index row, const Eigen::SparseMatrix<Scalar2, Eigen::ColMajor, IndexType2>& colmat, const Eigen::Index col)
{
  assert(rowmat.cols()==colmat.rows());
  typename Eigen::SparseMatrix<Scalar2, Eigen::ColMajor, IndexType2>::InnerIterator iter2(colmat,col);
  double ans = 0.0;
  for  (; iter2; ++iter2) 
    {
      ans += rowmat.coeff(row,iter2.index())*iter2.value();
    }  
  return ans;
}

template <typename Scalar1, typename IndexType1, typename Scalar2>
  double DotProductInnerVector (const Eigen::SparseMatrix<Scalar1, Eigen::RowMajor, IndexType1>& rowmat, const Eigen::Index row, const Eigen::Matrix<Scalar2, Eigen::Dynamic, Eigen::Dynamic, Eigen::ColMajor>& colmat, const Eigen::Index col)
{
  assert(rowmat.cols()==colmat.rows());
  typename Eigen::SparseMatrix<Scalar1, Eigen::RowMajor, IndexType1>::InnerIterator iter1(rowmat,row);
  double ans = 0.0;
  for  (; iter1; ++iter1) 
    {
      double c = colmat.coeff(iter1.index(),col);
      ans += c*iter1.value();
    }  
  return ans;
}

template <typename Scalar1, typename Scalar2>
  double DotProductInnerVector (const Eigen::Matrix<Scalar1, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>& rowmat, const Eigen::Index row, const Eigen::Matrix<Scalar2, Eigen::Dynamic, Eigen::Dynamic, Eigen::ColMajor>& colmat, const Eigen::Index col)
{
  assert(rowmat.cols()==colmat.rows());
  //should do this casting when we know the types. Define templates with a single scalar. 
  double ans = rowmat.row(row)*colmat.col(col).template cast<Scalar1>();
  return ans;
}

template <typename RowMatType, typename ColMatType>
  void DotProductInnerVector (Eigen::Ref<Eigen::VectorXd> result, const RowMatType& rowmat, Eigen::Index row, const ColMatType& colmat)
{
  assert(result.size() == colmat.cols());
  
  for (Eigen::Index col=0;col<colmat.cols();++col)
    {
      result(col)=DotProductInnerVector(rowmat, row, colmat ,col);
    }
}

// ************************************
// Convert a label vector to a label matrix
// Assumes that the label vector contains labels from 1 to noClasses

SparseMb labelVec2Mat (const Eigen::VectorXd& yVec);


#endif
