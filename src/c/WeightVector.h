/*  Copyright (C) 2017 NEC Laboratories America, Inc. ("NECLA"). All rights reserved.
 *
 * This source code is licensed under the license found in the LICENSE file in
 * the root directory of this source tree. An additional grant of patent rights
 * can be found in the PATENTS file in the same directory.
 */
#ifndef __WEIGHTVECTOR_H
#define __WEIGHTVECTOR_H

#include "utils.h"
#include "constants.h" // MCTHREADS
using Eigen::VectorXd;
using Eigen::VectorXi;

class WeightVector
{
 private:
#if __cplusplus >= 201103L
  static double constexpr MIN_SCALE = 1e-4; // min scale value, for numerical stability
  static double constexpr MAX_SCALE = 1e+4; // max scale value, for numerical stability
  static double constexpr MAX_BETA  = 1e+4; // max beta  value, for numerical stability
#else
  static double const MIN_SCALE = 1e-4; // min scale value, for numerical stability
  static double const MAX_SCALE = 1e+4; // max scale value, for numerical stability
  static double const MAX_BETA  = 1e+4; // max beta  value, for numerical stability
#endif
  VectorXd my_weights;  // current weights of the last iteration
  double my_scale; // scalar to multiply the weights with. Makes
                   // the gradient update to the weight vector due
                   //  to the L2 norm very fast
  double my_norm_sq; // the square of the L2 norm of the vector. It is
                     // computed incrementally
                     // to avoid costly computations when it is needed.

  VectorXd my_A; // current averaged weights
  double my_alpha; // for averaged gradient
  double my_beta;  // for averaged gradient
  size_t my_avg_t;  // the round of averaging

 public:
  /** non-zero after averaging has started and \c norm_avg() and \c toVectorXd_avg and \c project_avg become useful */
  inline size_t getAvg_t() const {return my_avg_t;}

  WeightVector()
    {
      my_weights=VectorXd();
      my_scale = 1.0;
      my_norm_sq = 0;
      my_A = VectorXd();
	my_A.setZero();
      my_avg_t = 0;
      my_beta = 1;
      my_alpha = 1;
    }

  // constructor from a dense vector.
  WeightVector( VectorXd const& w)
      : my_weights(w)
        , my_scale(1.0)
        , my_norm_sq( w.squaredNorm() )
        , my_A(w.size())
        , my_alpha(1.0)
        , my_beta(1.0)
        , my_avg_t(0U)
    {
        my_A.setZero(); // absolutely nec. (valgrind)
    }
  template<typename Derived>
  void init( Eigen::MatrixBase<Derived> const& src )
  {
      my_scale = 1.0;
      my_weights = src;
      my_norm_sq = my_weights.squaredNorm();
      my_A = VectorXd(src.size());
      my_A.setZero();
      my_avg_t = 0;
      my_beta = 1;
      my_alpha = 1;
  }
  //construct a vector of a fixed size and initialize it with zero
  WeightVector(const int size)
    {
      my_norm_sq = 0.0;
      my_weights = VectorXd(size);
      my_weights.setZero();
      my_scale = 1.0;
      my_A = VectorXd(size);
      my_A.setZero();
      my_avg_t = 0;
      my_beta = 1;
      my_alpha = 1;
    }


  inline void scale(const double s)
  {
    if (s == 0.0) // reset everything to zero, including the average
      {
	my_scale = 1.0;
	my_norm_sq = 0.0;
	my_weights.setZero();
	my_A.setZero();
	my_beta = 1.0;
	my_alpha = 1.0;
	my_avg_t = 0;
      }
    else
      {
	my_scale *= s;
	my_norm_sq *= s*s;
	if (my_avg_t == 0)
	  {
	    my_alpha = my_scale;
	  }
	if (my_scale < MIN_SCALE)
	  {
	    reset_scale();
	  }
	if (my_scale > MAX_SCALE)
	  {
	    reset_scale();
	  }
      }
  }

  inline  void update_alpha_beta()
  {
    my_avg_t++;
    my_beta *= my_avg_t*1.0/(my_avg_t - 1);
    my_alpha += my_beta*my_scale/my_avg_t;
    if (my_beta > MAX_BETA)
      {
	reset_beta();
      }
  }

  inline void reset_alpha()
  {
    // reset A and alpha to avoid numerical instability
    if (my_avg_t > 0)
      {
	my_A += my_alpha*my_weights;
	my_alpha = 0;
      }
    else
      {
	my_alpha = my_scale;
      }
  }

  inline void reset_scale()
  {
    reset_alpha();
    my_weights*=my_scale;
    my_alpha /= my_scale; // my_alpha is set to 0 by reset_alpha if averaging is on. It will be set to my_scale if averaging is off.
    my_scale = 1.0;
  }

  // reset beta if it gets too large
  inline void reset_beta()
  {
    my_A /= my_beta;
    my_alpha /= my_beta;
    my_beta = 1;
  }

  inline void toVectorXd(VectorXd& v) const
    {
      v = my_weights*my_scale;
    }
  /// some complicate type (decltype in C++1, auto return type deduction in C++14, later)
  typedef decltype(my_weights*my_scale) VecExprType;
  /// avoid copies with m.col(i) = w.getVec()
  inline VecExprType getVec()
  {
      return my_weights*my_scale;
  }

  inline void toVectorXd_avg(VectorXd& v) const
    {
      v = (my_A + my_weights*my_alpha)*(1.0/my_beta);
    }

  /// avoid copies with m.col(i) = w.getVec()
  typedef decltype((my_A + my_weights*my_alpha) * (1.0/my_beta)) VecAvgExprType;
  inline VecAvgExprType getVecAvg()
  {
    //if averaging has not started (my_avg_t == 0) this is the same as getVec()
    return (my_A + my_weights*my_alpha) * (1.0/my_beta);
  }

  inline double norm() const
  {
    return sqrt(my_norm_sq);
  }

  inline double norm_avg() const
  {
    return (my_A + my_weights*my_alpha).norm()/my_beta;
  }

  inline int size() const
  {
    return my_weights.size();
  }

  // template functions must be defined in the header

  // updates the current weight only, not the average
  // should not be called if the average has been updated (i.e. my_avg_t > 1)
  template<typename EigenType>
    void batch_gradient_update(const EigenType& x, const VectorXsz& index, const VectorXd& gradient, double lambda, double eta)
    {
      assert(x.cols()==my_weights.size());
      assert(my_avg_t == 0);
      // update for the reglarizer
      scale(1.0-lambda*eta);
      size_t batch_size = index.size();
      double eta1 = eta/batch_size;
      for (size_t idx = 0; idx < batch_size; idx++)
	{
	  double g = gradient.coeff(idx);
	  if ( g != 0 )
	    {
	      gradient_update_nochecks(x,index.coeff(idx), g * eta1);
	    }
	}
    }

  // updates the current weight only, not the average
  // should not be called if the average has been updated (i.e. my_avg_t > 1)
  // special function when batch size = 1
  template<typename EigenType>
    void batch_gradient_update(const EigenType& x, size_t index, double gradient, double lambda, double eta)
    {
      assert(x.cols()==my_weights.size());
      assert(my_avg_t == 0);
      // update for the reglarizer
      scale(1.0-lambda*eta);
      if ( gradient != 0 )
	{
	  gradient_update_nochecks(x, index, gradient * eta);
	}
    }

  // updates the current weight and the average
  // should not be called once the averaging should start
  // should have a different function for dense examples vectors
  // as one could only update the average once per batch rather than at
  // every iteration.
  // in fact shoud have a different class for dense examples
  template<typename EigenType>
    void batch_gradient_update_avg(const EigenType& x, const VectorXsz& index, const VectorXd& gradient, double lambda, double eta)
    {
      if (my_avg_t == 0)
	{
	  // first time calling the averaging,
	  // is the same as simply updating the gradient
	  batch_gradient_update(x,index,gradient,lambda,eta);
	  my_avg_t++;
	}
      else
	{
	  assert(x.cols()==my_weights.size());
	  // update for the reglarizer
	  scale(1.0-lambda*eta);
	  size_t batch_size = index.size();
	  double eta1 = eta/batch_size;
	  for (size_t idx = 0; idx < batch_size; idx++)
	    {
	      double g = gradient.coeff(idx);
	      if ( g != 0 )
		{
		  gradient_update_avg_nochecks(x,index.coeff(idx), g * eta1);
		}
	    }
	  update_alpha_beta();
	}
    }


  // special function for cases where batch_size = 1
  // updates the current weight and the average
  // should not be called once the averaging should start
  // should have a different function for dense examples vectors
  // as one could only update the average once per batch rather than at
  // every iteration.
  // in fact shoud have a different class for dense examples
  template<typename EigenType>
    void batch_gradient_update_avg(const EigenType& x, size_t index, double gradient, double lambda, double eta)
    {
      if (my_avg_t == 0)
	{
	  // first time calling the averaging,
	  // is the same as simply updating the gradient
	  batch_gradient_update(x,index,gradient,lambda,eta);
	  my_avg_t++;
	}
      else
	{
	  assert(x.cols()==my_weights.size());
	  // update for the reglarizer
	  scale(1.0-lambda*eta);
	  if ( gradient != 0 )
	    {
	      gradient_update_avg_nochecks(x, index, gradient * eta);
	    }
	  update_alpha_beta();
	}
    }

  // updates the current weight only, not the average
  // should not be called if the average has been updated (i.e. my_avg_t > 1)
  // should check if my_avg_t > 1, but won't do it here to eliminate an operation

  // we could have a separate function for dense vectors that automatically resets the scale
  // but we do this to keep things simple for now.
  template<typename EigenType>
    inline void gradient_update(const EigenType& x, const size_t row, const double eta)
    {
      // avoid boudary checks inside the loop.
      // check that sizes match here.
      assert(x.cols()==my_weights.size());
      assert(my_avg_t == 0);
      gradient_update_nochecks(x,row,eta);
    }

  // we could have a separate function for dense vectors that automatically resets the scale
  // but we do this to keep things simple for now.
  template<typename EigenType>
    inline void gradient_update_avg(const EigenType& x, const size_t row, const double eta)
    {
      // avoid boudary checks inside the loop.
      // check that sizes match here.
      assert(x.cols()==my_weights.size());
      gradient_update_avg_nochecks(x,row,eta);
    }

  template<typename EigenType> inline double project_row(const EigenType& x, const int row) const
    {
      //return my_scale*DotProductInnerVector(my_weights,x,row);
      //return my_scale*((x.row(row)*my_weights)(0,0));
      return x.row(row).dot(my_weights) * my_scale;
    }

  /** A very minor speed increase... */
  template<typename _Scalar, int _Options, typename _Index> inline // Eigen does NOT ||ize, so...
      double project_row_sparse(const Eigen::SparseMatrix<_Scalar,_Options,_Index>& x, const int row) const
      {
          //return my_scale*DotProductInnerVector(my_weights,x,row);
          //return my_scale*((x.row(row)*my_weights)(0,0));
          return x.row(row).dot(my_weights) * my_scale;
      }

  //
  // ------------------- project( VectorXd& proj, EigenType const& x ) ------------
  //
  template<typename EigenType> inline void project(VectorXd& proj, EigenType const& x) const {
      proj = (x*my_weights)*my_scale;
  }
  template<typename _Scalar, int _Options, typename _Index> inline // Eigen does NOT ||ize, so...
    void project(VectorXd& proj,
		 Eigen::SparseMatrix<_Scalar,_Options,_Index> const& x) const {
    proj.resize(x.rows());
#pragma omp parallel for schedule(guided,256)
    for(size_t i=0U; i<x.rows(); ++i){
      //proj.coeffRef(i) = project_row( x, i );
      //proj.coeffRef(i) = x.row(i) .dot(my_weights) * my_scale;
      proj.coeffRef(i) = project_row_sparse( x, i );
    }
  }
  
  template<typename Scalar, int _Flags, typename _Index> inline // Eigen does NOT ||ize, so...
    void project(VectorXd& proj,
		 Eigen::MappedSparseMatrix<Scalar,_Flags,_Index> const& x) const {
#pragma omp parallel for schedule(static,4096)
    for(size_t i=0U; i<x.rows(); ++i){
      proj.coeffRef(i) = project_row( x, i );
    }
  }
  // -------------------------------------------------------------------------------

  template<typename EigenType> inline double project_row_avg(const EigenType& x, const int row) const
    {
      //return (DotProductInnerVector(my_A,x,row) + my_alpha*DotProductInnerVector(my_weights,x,row))/my_beta;
      return (x.row(row)*my_A + my_alpha*(x.row(row)*my_weights))(0,0)/my_beta;
    }


  template<typename EigenType> inline void project_avg(VectorXd& proj, const EigenType& x) const
    {
      proj = (x*my_A + (x*my_weights)*my_alpha)/my_beta;
    }


 private:
  // have these functions private because they do no error checking
  /** 18% faster than sparse version */
  template<typename DERIVED>
    void gradient_update_nochecks(Eigen::DenseBase<DERIVED> const& x, const size_t row, const double eta)
    {
      my_weights -= x.row(row).transpose() * (eta/my_scale);
      my_norm_sq = my_weights.squaredNorm() * my_scale*my_scale;
    }
  template<typename DERIVED>
    void gradient_update_nochecks(Eigen::SparseCompressedBase<DERIVED> const& x, const size_t row, const double eta)
    {
      typename DERIVED::InnerIterator it(x, row);       // lookup issue for MappedSparseMatrix?
      double norm_update = 0;
      double eta1 = eta/my_scale;
      for (; it; ++it )
      {
          int col = it.col();
          double val = my_weights.coeff(col);
          norm_update -= val*val;
          val -= (it.value() * eta1);
          norm_update += val*val;
          my_weights.coeffRef(col) = val;
      }
      my_norm_sq += norm_update*my_scale*my_scale;
  }

  // we could have a separate function for dense vectors that automatically resets the scale
  // but we do this to keep things simple for now.
  template<typename EigenType>
    void gradient_update_avg_nochecks(const EigenType& x, const size_t row, const double eta)
    {
      typename EigenType::InnerIterator it(x, row);
      double norm_update = 0;
      double eta1 = eta/my_scale;
      double eta_A = eta1 * my_alpha;
      for (; it; ++it )
	{
	  int col = it.col();
	  double val = my_weights.coeff(col);
	  norm_update -= val*val;
	  val -= (it.value() * eta1);
	  norm_update += val*val;
	  my_weights.coeffRef(col) = val;
	  my_A.coeffRef(col) += it.value() * eta_A;
	}
      my_norm_sq += norm_update*my_scale*my_scale;
    }


};

#endif
