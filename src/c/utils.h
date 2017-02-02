#ifndef __UTILS_H
#define __UTILS_H

#include "typedefs.h"
#include <exception>
#include <iostream>

//#include <sstream>
//#include <iomanip>
#define OUTWIDE( OSTREAM, WIDTH, STUFF ) do{ std::ostringstream oss; oss<<STUFF; OSTREAM<<setw(WIDTH)<<oss.str(); }while(0)

//using Eigen::VectorXd;


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
#if 0 // plainer implementation
struct Cmp {
Cmp( Eigen::VectorXd const& m ) : m(m) {}
  bool operator()( int const i, int const j ){
    //return m.coeff(i) < m.coeff(j);
    return m(i) < m(j);   // with checking
  }
  Eigen::VectorXd const& m;
};
for(size_t i=0U; i<cranks.size(); ++i)
  cranks[i] = i;
std::cout<<" copy..."<<std::endl;
Eigen::VectorXd n(m);
std::cout<<" std::sort ... "<<std::endl; std::cout.flush();
Cmp cmp(n);
std::sort(cranks.begin(), cranks.end(), cmp );
#endif


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

template <typename Scalar1, typename Scalar2> inline
  double DotProductInnerVector (const Eigen::SparseMatrix<Scalar1, Eigen::RowMajor>& rowmat, const Eigen::Index row, const Eigen::SparseMatrix<Scalar2, Eigen::ColMajor>& colmat, const Eigen::Index col)
{
  assert(rowmat.cols()==colmat.rows());
  typename Eigen::SparseMatrix<Scalar1, Eigen::RowMajor>::InnerIterator iter1(rowmat,row);
  typename Eigen::SparseMatrix<Scalar2, Eigen::ColMajor>::InnerIterator iter2(colmat,col);
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

template <typename Scalar1, typename Scalar2> inline
  double DotProductInnerVector (const Eigen::Matrix<Scalar1, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>& rowmat, const Eigen::Index row, const Eigen::SparseMatrix<Scalar2, Eigen::ColMajor>& colmat, const Eigen::Index col)
{
  assert(rowmat.cols()==colmat.rows());
  typename Eigen::SparseMatrix<Scalar2, Eigen::ColMajor>::InnerIterator iter2(colmat,col);
  double ans = 0.0;
  for  (; iter2; ++iter2) 
    {
      ans += rowmat.coeff(row,iter2.index())*iter2.value();
    }  
  return ans;
}

template <typename Scalar1, typename Scalar2>
  double DotProductInnerVector (const Eigen::SparseMatrix<Scalar1, Eigen::RowMajor>& rowmat, const Eigen::Index row, const Eigen::Matrix<Scalar2, Eigen::Dynamic, Eigen::Dynamic, Eigen::ColMajor>& colmat, const Eigen::Index col)
{
  assert(rowmat.cols()==colmat.rows());
  typename Eigen::SparseMatrix<Scalar1, Eigen::RowMajor>::InnerIterator iter1(rowmat,row);
  double ans = 0.0;
  for  (; iter1; ++iter1) 
    {
      ans += colmat.coeff(iter1.index(),col)*iter1.value();
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
    
/* template <typename EigenType> */
/* double DotProductInnerVector (const Eigen::Ref<const Eigen::VectorXf>& vec, const EigenType& mat, size_t outerIndex) */
/* { */
/*   assert(vec.size() == mat.innerSize()); */
/*   double val = 0.0; */
/*   typename EigenType::InnerIterator iter(mat,outerIndex);  */
/*   for  (; iter; ++iter)  */
/*     { */
/*       val += vec(iter.index())*iter.value(); */
/*     }   */
/*   return val; */
/* } */

/* template <typename EigenType> */
/* void DotProductInnerVector (Eigen::Ref<Eigen::VectorXd> result, const Eigen::Ref<const DenseColMf>& mat1, const EigenType& mat2, size_t outerIndex) */
/* { */
/*   assert(result.size() == mat1.cols()); */

/*   for (size_t i=0;i<mat1.cols();++i) */
/*     { */
/*       result(i)=DotProductInnerVector(mat1.col(i),mat2,outerIndex); */
/*     } */
/* } */


// ************************************
// Convert a label vector to a label matrix
// Assumes that the label vector contains labels from 1 to noClasses

SparseMb labelVec2Mat (const Eigen::VectorXd& yVec);


// check if a file exists and is ready for reading

bool fexists(const char *filename);

/** delete an ActiveDataSet to free up memory. \post \c active==nullptr. */
void free_ActiveDataSet(ActiveDataSet*& active);

#endif
