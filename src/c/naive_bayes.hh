#ifndef NAIVE_BAYES_HH
#define NAIVE_BAYES_HH

#include <iostream>
#include <ostream>
#include <fstream>
#include "typedefs.h"

template <typename EigenType> inline 
void train_NB(const EigenType& x, const SparseMb& y, ovaCoeffType a=1,  std::ostream& model_out=std::cout)
{
  size_t d = x.cols();
  size_t n = x.rows();				
  ovaDenseRowV t_given_c(d);
  size_t nc = 0;
  ovaCoeffType p_c  = 0;
  ovaCoeffType total_w = 0;
  for (size_t c = 0; c < y.cols(); c++)
    {
      t_given_c.setZero();
      nc = 0;
      for (size_t i = 0; i < n; i++)
	{	
	  if (y.coeff(i,c))
	    {
	      nc++;
	      t_given_c += x.row(i).template cast<ovaCoeffType>();
	    }
	}
      t_given_c.array() += a;  
      total_w  = t_given_c.sum();

      t_given_c = t_given_c.array().log() - log(total_w);

      model_out.write((char*)t_given_c.data(),t_given_c.size()*sizeof(ovaCoeffType));
      p_c = log(nc) - log(n);
      model_out.write((char*)&p_c, sizeof(ovaCoeffType));
    }
}

#endif // NAIVE_BAYES_HH
