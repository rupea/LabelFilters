#ifndef __UTILS_H
#define __UTILS_H

#include "typedefs.h"

using Eigen::VectorXd;


// *********************************
// functions and structures for sorting and keeping indeces
// Should implement bound checking but it is faster this way.
template<typename IntType>
struct IndexComparator
{
  const VectorXd* v;
  IndexComparator(const VectorXd* m)
  {
    v = m;
  }
  bool operator()(IntType i, IntType j)
  {
    return (v->coeff(i) < v->coeff(j));
  }
};

template<typename IntType>
void sort_index(const VectorXd& m, std::vector<IntType>& cranks)
{
  for (IntType i = 0; i < m.size(); i++)
    {
      cranks[i] = i;
    }
  IndexComparator<IntType> cmp(&m);
  std::sort(cranks.begin(), cranks.end(), cmp);
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

template <typename EigenType>
double DotProductInnerVector (const Eigen::Ref<const Eigen::VectorXf>& vec, const EigenType& mat, size_t outerIndex)
{
#ifndef NDEBUG
  assert(vec.size() == mat.innerSize());
#endif
  double val = 0.0;
  typename EigenType::InnerIterator iter(mat,outerIndex); 
  for  (; iter; ++iter) 
    {
      val += vec(iter.index())*iter.value();
    }  
  return val;
}

#if 0
template <typename EigenType>
void DotProductInnerVector (Eigen::Ref<Eigen::RowVectorXd> result, const Eigen::Ref<const DenseColM>& mat1, const EigenType& mat2, size_t outerIndex)
{
#ifndef NDEBUG
  assert(result.size() == mat1.cols());
#endif
  size_t i;
  result.setZero();
  typename EigenType::InnerIterator iter(mat2,outerIndex); 
  for  (; iter; ++iter) 
    {
      result+=iter.value()*mat1.row(iter.index());
    }
}
#endif 
#if 1
template <typename EigenType>
void DotProductInnerVector (Eigen::Ref<Eigen::VectorXd> result, const Eigen::Ref<const DenseColMf>& mat1, const EigenType& mat2, size_t outerIndex)
{
#ifndef NDEBUG
  assert(result.size() == mat1.cols());
#endif
  size_t i;
  for (i=0;i<mat1.cols();++i)
    {
      result(i)=DotProductInnerVector(mat1.col(i),mat2,outerIndex);
    }
}
#endif


// ************************************
// Convert a label vector to a label matrix
// Assumes that the label vector contains labels from 1 to noClasses

SparseMb labelVec2Mat (const VectorXd& yVec);


// check if a file exists and is ready for reading

bool fexists(const char *filename);

// delete and ActiveDataSet to free up memory

void free_ActiveDataSet(ActiveDataSet*& active);

#endif
