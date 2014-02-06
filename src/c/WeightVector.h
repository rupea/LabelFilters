#ifndef __WEIGHTVECTOR_H
#define __WEIGHTVECTOR_H

using Eigen::VectorXd;

class WeightVector
{
 private: 
  const static double MIN_SCALE = 1e-4; // minimum value of the scale scalar, to 
                                // avoid numerical instability. 
  const static double MAX_SCALE = 1e+4; // minimum value of the scale scalar, to 
                                // avoid numerical instability. 

  VectorXd my_weights;  // weights of the matrix
  double my_scale; // scalar to multiply the weights with. Makes 
                   // the gradient update to the weight vector due
                   //  to the L2 norm very fast
  double my_norm_sq; // the square of the L2 norm of the vector. It is 
                     // computed incrementally
                     // to avoid costly computations when it is needed. 
  
 public: 
  WeightVector()
    {
      my_weights=VectorXd();
      my_scale = 1.0;
      my_norm_sq = 0;
    }
  // constructor from a dense vector. 
  WeightVector(const VectorXd& w)
    {
      double norm = w.norm();      
      my_norm_sq = norm*norm;
      my_scale = 1.0;
      my_weights = w;
    }
  //construct a vector of a fixed size and initialize it with zero
  WeightVector(const int size)
    {
      my_norm_sq = 0.0;
      my_weights = VectorXd(size);
      my_scale = 1.0;
    }

  
  inline void scale(const double s)
  {
    if (s == 0.0)
      {
	my_scale = 1.0;
	my_norm_sq = 0.0;
	my_weights.setZero();
      }
    else
      {
	my_scale *= s;
	my_norm_sq *= s*s;
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
  void reset_scale()
  {
    my_weights*=my_scale;
    my_scale = 1.0;
  }

  inline void toVectorXd(VectorXd& v) const
    {
      v = my_weights*my_scale;
    }
  
  inline double norm() const
  {
    return sqrt(my_norm_sq);
  }  

  inline int size() const
  {
    return my_weights.size();
  }

  // template functions must be defined in the header 

  // we could have a separate function for dense vectors that automatically resets the scale
  // but we do this to keep things simple for now. 
  template<typename EigenType> 
    void gradient_update(const EigenType& x, const int row, const double eta)
    {      
      // avoid boudary checks inside the loop. 
      // check that sizes match here.
      assert(x.cols()==my_weights.size());
      typename EigenType::InnerIterator it(x, row);
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

  template<typename EigenType> inline double project_row(const EigenType& x, const int row) const
    {
      return my_scale*((x.row(row)*my_weights)(0,0));
    }

  template<typename EigenType> inline void project(VectorXd& proj, const EigenType& x) const
    {
      proj = (x*my_weights)*my_scale;
    }
};

#endif
