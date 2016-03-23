#include "find_w_detail.h"
#include "boost/iterator/counting_iterator.hpp"
//#include <boost/numeric/conversion/bounds.hpp>  // boost::numeric::bounds<T>

#define __restricted /* __restricted seems to be an error */

/** 0 --> Alex's original dev-branch version
 * 1 --> my version */
#define OPTIMIZE_LU_VERSION 0

using namespace std;

//*****************************************
// function used by optimizeLU
// grad are stored in order of the ranked classes
// to minimize cash misses and false sharing

static void getBoundGrad (VectorXd& __restricted grad, VectorXd& __restricted bound,
                          const size_t idx, const size_t allproj_idx,
                          const std::vector<int>& __restricted sorted_class,
                          const int sc_start, const int sc_end,
                          const std::vector<int>& __restricted classes,
                          const double start_update, const double other_weight,
                          const VectorXd& __restricted allproj,
                          const bool none_filtered, const boolmatrix& __restricted filtered)
{
  std::vector<int>::const_iterator class_iter = std::lower_bound(classes.begin(), classes.end(), sc_start);
  double update = start_update + (class_iter - classes.begin())*other_weight;
// #pragma omp critical
//   {
//     cout << "1  " << idx << "   " << allproj_idx << "   " << sc_start << "   " << sc_end << "    " << class_iter - classes.begin() << "  " << update << "   " << other_weight << endl;
//   }
  for (int sc = sc_start; sc < sc_end; sc++)
    {
      if (class_iter != classes.end() && sc == *class_iter)
	{
	  // example is of this class
	  update += other_weight;
	  class_iter++;
	  continue;
	}
      //const double gsc = grad.coeff(sc);
      // if (gsc >= 0 )
      // 	{
      // 	  const int cp = sorted_class[sc];
      // 	  if (none_filtered || !(filtered.get(idx,cp)))
      const int cp = sorted_class[sc];
      if (grad.coeff(sc) >= 0 && (none_filtered || !(filtered.get(idx,cp))))
	{
	  // const double ngsc = gsc - update;
	  // grad.coeffRef(sc) = ngsc;
	  grad.coeffRef(sc) -= update;
	  //	      if (ngsc < 0)
	  if (grad.coeff(sc) < 0)
	    {
	      bound.coeffRef(cp) = allproj.coeff(allproj_idx);
	    }
	}
      //}
    }
// #pragma omp critical
//   {
//     cout << "2  " << idx << "   " << allproj_idx << "   "<< sc_start << "   " << sc_end << "    " << classes.end() - class_iter << "  " << update << "   " << endl;
//   }


  // for (std::vector<int>::const_iterator sc = sorted_class.begin() + sc_start; sc != sorted_class.begin() + sc_end; sc++)
  //   {
  //     if (class_iter != classes.end() && *sc == sorted_class[*class_iter])
  // 	{
  // 	  // example is of this class
  // 	  update += other_weight;
  // 	  class_iter++;
  // 	  continue;
  // 	}
  //     if (grad.coeff(*sc) >= 0 && (none_filtered || !(filtered.get(idx,*sc))))
  // 	{
  // 	  grad.coeffRef(*sc) -= update;
  // 	  if (grad.coeff(*sc) < 0)
  // 	    {
  // 	      bound.coeffRef(*sc) = allproj.coeff(allproj_idx);
  // 	    }
  // 	}
  //   }

}

#if OPTIMIZE_LU_VERSION == 0 // Alex's original version

// *****************************************************
// get the optimal values for lower and upper bounds given 
// a projection and the class order
// computationally expensive so it should be done sparingly 
void optimizeLU(VectorXd&l, VectorXd&u, 
		const VectorXd& projection, const SparseMb& y, 
		const vector<int>& class_order, const vector<int>& sorted_class,
		const VectorXd& wc, const VectorXi& nclasses,
		const boolmatrix& filtered,
		const double C1, const double C2,
		const param_struct& params,
		bool print)
{
  size_t n = projection.size();
  size_t noClasses = y.cols();
  VectorXd allproj(2*n);
  allproj << (projection.array() - 1), (projection.array() + 1);

  bool none_filtered = filtered.count()==0;
  std::vector<size_t> indices(2*n);
  sort_index(allproj, indices);
  int min_chunk_size = 10000;
#ifdef _OPENMP
  int max_n_chunks = omp_get_max_threads();
#else
  int max_n_chunks = 1;
#endif 

#pragma omp parallel default(shared) shared(l, u, allproj, indices,none_filtered, filtered, n, noClasses, y, class_order, sorted_class, wc, nclasses, params, min_chunk_size, max_n_chunks)
  {
#pragma omp single
    {
      double class_weight, other_weight;      
      std::vector<int> classes;
      classes.reserve(nclasses.maxCoeff());
      // calculate the optimal value for upper bounds
      // iterate from the beginning 

      // grad are stored in order of the ranked classes
      // to minimize cash misses and false sharing       
      VectorXd grad(noClasses);
      double classweight;
      for (int sc = 0; sc <noClasses; sc++) 
	{
	  // this does not work if the remove_class_constraints is true (i.e. true classes can be filtered)!!
	  classweight = wc.coeff(sorted_class[sc]);
	  if (classweight == 0)
	    {
	      // there are no examples of this class
	      // so just put l and u to 0 (l is set below)
	      u.coeffRef(sorted_class[sc]) = 0;
	      grad.coeffRef(sc) = -1; 
	    }
	  else
	    {
	      grad.coeffRef(sc) = C1*classweight;
	    }	  
	}
      //      VectorXd grad = C1*wc;

      for (std::vector<size_t>::const_iterator i = indices.begin(); i != indices.end(); i++)
	{
	  bool plus = false;
	  size_t idx = *i;
	  if (idx >= n)
	    {
	      plus = true;
	      idx -= n;
	    }
	  
	  if (plus)
	    {
	      // only the upper bounds of the classes of this example are affected
	      class_weight = C1;
	      if (params.ml_wt_class_by_nclasses)
		{
		  class_weight /= nclasses.coeff(idx);
		}
	      for (SparseMb::InnerIterator it(y,idx); it; ++it)
		{
		  if (it.value())
		    {
		      int cs = it.col();
		      int sc = class_order[cs];
		      // should check for filtered (in case classes were filtered)
		      // but things brake above if remove_class_constraints is true
		      if (grad.coeff(sc) >= 0 )
			{
			  grad.coeffRef(sc) -= class_weight;
			  if (grad.coeff(sc) <= 0)
			    {
			      // the upper bound for the last class
			      // will be at the end of the last
			      // example. If we make it at +inf then
			      // it will create problems if the order
			      // of the classes ever changes without
			      // optimizing the LU bounds 
			      u.coeffRef(cs) = allproj.coeff(*i);
			    }
			}
		    }
		}
	    }
	  else
	    {
	      // only the classes ranked lower than the classes of this example are affected
	      other_weight = C2;
	      if (params.ml_wt_by_nclasses)
		{
		  other_weight /= nclasses.coeff(idx);
		}
	      
	      // how many classes of the curent instance should be ranked higher 
	      //  times the weight of each
	      //  if each class has its own weight will need to
	      //  be calculated below (or have it precomputed for each example 
	      //  as a corresponding wclasses to nclasses to be wclasses the same as wc 
	      //  corresponds to nc
	      double right_update = other_weight * nclasses.coeff(idx); 
	      
	      // calling y.coeff is expensive so get the classes in the ranked order here
	      classes.resize(0);
	      for (SparseMb::InnerIterator it(y,idx); it; ++it)
		{
		  if (it.value())
		    {
		      classes.push_back(class_order[it.col()]);
		    }
		}
	      std::sort(classes.begin(),classes.end()); 
	      // we  update the upper bounds. 
	      // if a class has higher rank than the highest rank class of this example
	      // it's upper bound will not be influenced by this example 
	      //	      int class_end = sorted_class[classes.back()]; // make sure classes is not empty	  

	      if (classes.back() == 0)
		{
		  continue;
		}
#ifdef _OPENMP
	      // make sure there is enough work to do to paralelize this
	      int n_chunks = classes.back()/min_chunk_size + 1;
	      n_chunks = n_chunks < max_n_chunks?n_chunks:max_n_chunks;
	      int chunk_size = classes.back()/n_chunks;
	      int remaining = classes.back()%n_chunks;
	      for (int chunk=0; chunk < n_chunks; chunk++)
		{
#pragma omp task default(shared) firstprivate(chunk) shared(grad, u, idx, i, sorted_class, classes, right_update, other_weight, allproj, none_filtered, filtered, chunk_size, remaining)
		  {
		    int sc_start = chunk*chunk_size + (chunk<remaining?chunk:remaining);
		    int sc_incr = chunk_size + (chunk<remaining);
		    // #pragma omp critical
		    // {
		    //   cout << "0   " << idx << "   " << *i << "   " << sc_start << "   " << sc_incr << "    " << chunk << "   " << classes.back() << endl;
		    // }
		    getBoundGrad(grad, u, idx, *i, sorted_class, sc_start, sc_start + sc_incr, classes, right_update, -other_weight, allproj, none_filtered, filtered);
		  }
		}
#pragma omp taskwait
#else // if not _OPENMP
	      getBoundGrad(grad, u, idx, *i, sorted_class, 0,classes.back(),classes,right_update,-other_weight,allproj,none_filtered,filtered);
#endif // _OPENMP	   
	    }
	}
    }

#pragma omp single
    {
      double class_weight, other_weight;      
      std::vector<int> classes;
      classes.reserve(nclasses.maxCoeff());    
      // calculate the optimal value for upper bounds
      // iterate from the end 

      // grad are stored in order of the ranked classes
      // to minimize cash misses and false sharing       
      VectorXd grad(noClasses);
      double classweight;
      for (int sc = 0; sc <noClasses; sc++) 
	{
	  // this does not work if the remove_class_constraints is true (i.e. true classes can be filtered)!!
	  classweight = wc.coeff(sorted_class[sc]);
	  if (classweight == 0)
	    {
	      // there are no examples of this class
	      // so just put l and u to 0
	      l.coeffRef(sorted_class[sc]) = 0.0;
	      grad.coeffRef(sc) = -1; 
	    }
	  else
	    {
	      grad.coeffRef(sc) = C1*classweight;
	    }
	}	  

      //      VectorXd grad = C1*wc;
      for (std::vector<size_t>::const_reverse_iterator i = indices.rbegin(); i != indices.rend(); i++)
	{
	  bool plus = false;
	  size_t idx = *i;
	  if (idx >= n)
	    {
	      plus = true;
	      idx -= n;
	    }
	  
	  if (!plus)
	    {
	      // only the lower bounds of the classes of this example are affected
	      class_weight = C1;
	      if (params.ml_wt_class_by_nclasses)
		{
		  class_weight /= nclasses.coeff(idx);
		}
	      for (SparseMb::InnerIterator it(y,idx); it; ++it)
		{
		  if (it.value())
		    {
		      int cs = it.col();
		      int sc = class_order[cs];
		      if (grad.coeff(sc) >= 0 )
			{
			  grad.coeffRef(sc) -= class_weight;
			  if (grad.coeff(sc) <= 0 )
			    {
			      // the lower bound for the last class
			      // will be at the end of the last
			      // example. If we make it at +inf then
			      // it will create problems if the order
			      // of the classes ever changes without
			      // optimizing the LU bounds 
			      l.coeffRef(cs) = allproj.coeff(*i);
			    }
			}
		    }
		}
	    }
	  else
	    {
	      // only the classes ranked higher than the classes of this example are affected
	      other_weight = C2;
	      if (params.ml_wt_by_nclasses)
		{
		  other_weight /= nclasses.coeff(idx);
		}
	      
	      
	      // calling y.coeff is expensive so get the classes in the ranked order here
	      classes.resize(0);
	      for (SparseMb::InnerIterator it(y,idx); it; ++it)
		{
		  if (it.value())
		    {
		      classes.push_back(class_order[it.col()]);
		    }
		}
	      std::sort(classes.begin(),classes.end()); 
	      // we  update the lower bounds. 
	      // if a class has lower rank than the lowest rank class of this example
	      // it's lower bound will not be influenced by this example 
	      //   int class_end = sorted_class[classes.front()]; // make sure classes is not empty

	      int n_active = noClasses - classes.front() - 1;
	      if (n_active == 0) 
		{
		  continue;
		}
#ifdef _OPENMP	   
	      // make sure there is enough work to do to paralelize this
	      int n_chunks = n_active/min_chunk_size + 1;
	      n_chunks = n_chunks < max_n_chunks?n_chunks:max_n_chunks;
	      int chunk_size = n_active/n_chunks;
	      int remaining = n_active%n_chunks;
	      for (int chunk=0; chunk < n_chunks; chunk++)
		{
#pragma omp task default(shared) firstprivate(chunk) shared(grad, l, idx, i, sorted_class, classes, other_weight, allproj, none_filtered, filtered, chunk_size, remaining)
		  {
		    int sc_start = classes.front() + 1 + chunk*chunk_size + (chunk<remaining?chunk:remaining);
		    int sc_incr = chunk_size + (chunk<remaining);
		    getBoundGrad(grad, l, idx, *i, sorted_class, sc_start, sc_start + sc_incr, classes, 0.0, other_weight, allproj, none_filtered, filtered);
		  }
		}
#pragma omp taskwait
#else // if not _OPENMP
	      getBoundGrad(grad, l, idx, *i, sorted_class, classes.front() + 1,noClasses,classes, 0.0, other_weight, allproj, none_filtered, filtered);
#endif // _OPENMP	   	      
	    }
	}
    }
  }

#if 0
    
#pragma omp parallel sections default(none) shared(l, u, allproj, indices,none_filtered, filtered, n, noClasses, y, class_order, sorted_class, wc, nclasses, params)
  {
#pragma omp section
    {
      double class_weight, other_weight;      
      std::vector<int> classes;
      classes.reserve(nclasses.maxCoeff());
      // calculate the optimal value for upper bounds
      // iterate from the beginning 
      VectorXd grad = C1*wc;
      for (std::vector<size_t>::const_iterator i = indices.begin(); i != indices.end(); i++)
	{
	  bool plus = false;
	  size_t idx = *i;
	  if (idx >= n)
	    {
	      plus = true;
	      idx -= n;
	    }
	  
	  if (plus)
	    {
	      // only the upper bounds of the classes of this example are affected
	      class_weight = C1;
	      if (params.ml_wt_class_by_nclasses)
		{
		  class_weight /= nclasses.coeff(idx);
		}
	      for (SparseMb::InnerIterator it(y,idx); it; ++it)
		{
		  if (it.value())
		    {
		      int cs = it.col();
		      if (grad.coeff(cs) >= 0 )
			{
			  grad.coeffRef(cs) -= class_weight;
			  if (grad.coeff(cs) < 0)
			    {
			      u.coeffRef(cs) = allproj.coeff(*i);
			    }
			}
		    }
		}
	    }
	  else
	    {
	      // only the classes ranked lower than the classes of this example are affected
	      other_weight = C2;
	      if (params.ml_wt_by_nclasses)
		{
		  other_weight /= nclasses.coeff(idx);
		}
	      
	      // how many classes of the curent instance should be ranked higher 
	      //  times the weight of each
	      //  if each class has its own weight will need to
	      //  be calculated below (or have it precomputed for each example 
	      //  as a corresponding wclasses to nclasses to be wclasses the same as wc 
	      //  corresponds to nc
	      double right_update = other_weight * nclasses.coeff(idx); 
	      
	      // calling y.coeff is expensive so get the classes in the ranked order here
	      classes.resize(0);
	      for (SparseMb::InnerIterator it(y,idx); it; ++it)
		{
		  if (it.value())
		    {
		      classes.push_back(class_order[it.col()]);
		    }
		}
	      std::sort(classes.begin(),classes.end()); 
	      // we  update the upper bounds. 
	      // if a class has higher rank than the highest rank class of this example
	      // it's upper bound will not be influenced by this example 
	      //	      int class_end = sorted_class[classes.back()]; // make sure classes is not empty	  
	      std::vector<int>::const_iterator class_iter = classes.begin();
	      for (std::vector<int>::const_iterator sc=sorted_class.begin(); *sc != class_end; sc++)
		{
		  if (*sc == sorted_class[*class_iter])		
		    {
		      // example has this class
		      right_update -= other_weight;
		      class_iter++;
		      continue;
		    }		  
		  if (grad.coeff(*sc) >= 0 && (none_filtered || !(filtered.get(idx,*sc))))
		    {			      
		      grad.coeffRef(*sc) -= right_update;
		      if (grad.coeff(*sc) < 0)
			{
			  u.coeffRef(*sc) = allproj.coeff(*i);
			}
		    }
		}
	    }	      
	}
      // set the upper bound of the highest ranked class to be infinity 
      // should be careful with this if not ranking by means!
      u.coeffRef(sorted_class.back()) = boost::numeric::bounds<double>::highest();
    }
  
#pragma omp section  
    {
      double class_weight, other_weight;      
      std::vector<int> classes;
      classes.reserve(nclasses.maxCoeff());    
      // calculate the optimal value for upper bounds
      // iterate from the end 
      VectorXd grad = C1*wc;
      for (std::vector<size_t>::const_reverse_iterator i = indices.rbegin(); i != indices.rend(); i++)
	{
	  bool plus = false;
	  size_t idx = *i;
	  if (idx >= n)
	    {
	      plus = true;
	      idx -= n;
	    }
	  
	  if (!plus)
	    {
	      // only the upper bounds of the classes of this example are affected
	      class_weight = C1;
	      if (params.ml_wt_class_by_nclasses)
		{
		  class_weight /= nclasses.coeff(idx);
		}
	      for (SparseMb::InnerIterator it(y,idx); it; ++it)
		{
		  if (it.value())
		    {
		      int cs = it.col();
		      if (grad.coeff(cs) >= 0 )
			{
			  grad.coeffRef(cs) -= class_weight;
			  if (grad.coeff(cs) < 0)
			    {
			      l.coeffRef(cs) = allproj.coeff(*i);
			    }
			}
		    }
		}
	    }
	  else
	    {
	      // only the classes ranked higher than the classes of this example are affected
	      other_weight = C2;
	      if (params.ml_wt_by_nclasses)
		{
		  other_weight /= nclasses.coeff(idx);
		}
	      
	      // how many classes of the curent instance should be ranked lower 
	      //  times the weight of each
	      // we are starting from the end, so all classes are ranked lower
	      //  if each class has its own weight will need to
	      //  be calculated below (or have it precomputed for each example 
	      //  as a corresponding wclasses to nclasses to be wclasses the same as wc 
	      //  corresponds to nc
	      double left_update = other_weight * nclasses.coeff(idx); 
	      
	      // calling y.coeff is expensive so get the classes in the ranked order here
	      classes.resize(0);
	      for (SparseMb::InnerIterator it(y,idx); it; ++it)
		{
		  if (it.value())
		    {
		      classes.push_back(class_order[it.col()]);
		    }
		}
	      std::sort(classes.begin(),classes.end()); 
	      // we  update the lower bounds. 
	      // if a class has lower rank than the lowest rank class of this example
	      // it's lower bound will not be influenced by this example 
	      int class_end = sorted_class[classes.front()]; // make sure classes is not empty
	      
	      std::vector<int>::const_reverse_iterator class_iter = classes.rbegin();
	      for (std::vector<int>::const_reverse_iterator sc=sorted_class.rbegin(); *sc != class_end; sc++)
		{
		  if (*sc == sorted_class[*class_iter])
		    {
		      // example is of this class
		      left_update -= other_weight;
		      class_iter++;
		      continue;
		    }		  
		  if (grad.coeff(*sc) >= 0 && (none_filtered || !(filtered.get(idx,*sc))))
		    {			      
		      grad.coeffRef(*sc) -= left_update;
		      if (grad.coeff(*sc) < 0)
			{
			  l.coeffRef(*sc) = allproj.coeff(*i);
			}
		    }
		}
	    }
	}
      // set the lower bound of the lowest ranked class to be -infinity 
      // should be careful with this if not ranking by means!
      l.coeffRef(sorted_class.front()) = boost::numeric::bounds<double>::lowest();
    }
  }

#endif 

#if 0

#  pragma omp parallel for default(shared) shared(l, u)
  for (int cs = 0; cs < y.cols(); cs++)
    {    
      // get the optimial value of the upperbound for class cs
      int cs2;
      double grad;
      if (class_order[cs] == noClasses - 1) 
	{
	  // the upper bound for the last ranked class is infinity
	  u.coeffRef(cs) = boost::numeric::bounds<double>::highest();
	  if (print)
	    {
	      cout << cs << " grad U start " << 0.00 << endl;
	      cout << cs << "  grad U end  " << grad << endl;
	      cout << cs << " opt U " << u.coeff(cs) << endl;
	    }	  
	}
      else
	{
	  // start from ub = -infinity 
	  grad = C1*wc.coeff(cs);
	  if (print)
	    {
	      cout << cs << "  grad U start  " << grad << endl;
	    }
 	  for (std::vector<size_t>::iterator i = indices.begin(); i != indices.end(); i++)
	    {
	      bool plus = false;
	      size_t idx = *i;
	      if (idx >= n)
		{
		  plus = true;
		  idx -= n;
		}

	      class_weight = C1;
	      other_weight = C2;
	      if (params.ml_wt_by_nclasses)
		{
		  other_weight /= nclasses.coeff(idx);
		}
	      if (params.ml_wt_class_by_nclasses)
		{
		  class_weight /= nclasses.coeff(idx);
		}

	      for (SparseMb::InnerIterator it(y,idx); it; ++it)
		{
		  if (it.value())
		    {
		      cs2 = it.col();
		      if (cs2 == cs && plus)
			{
			  grad -= class_weight;
			}
		      else
			{
			  if (class_order[cs2] > class_order[cs] && !plus && (none_filtered || !(filtered.get(idx,cs))))
			    {
			      grad -= other_weight;
			    }
			}
		    }	      
		}
	      if (grad < 0) // protect against very small doubles? 
		{
		  u.coeffRef(cs) = allproj.coeff(*i);
		  if (print)
		    {
		      cout << cs << "  grad U end  " << grad << endl;
		      cout << cs << " opt U " << u.coeff(cs) << endl;
		    }
		  break;
		}
	    }
	}
      // get the optimial value of the lower bound  for class cs
      if (class_order[cs] == 0) 
	{
	  // the lower bound for the first ranked class is -infinity
	  l.coeffRef(cs) = boost::numeric::bounds<double>::lowest();
	  if (print)
	    {
	      cout << cs << " grad L start " << 0.00 << endl;
	      cout << cs << "  grad L end  " << grad << endl;
	      cout << cs << " opt L " << l.coeff(cs) << endl;
	    }	  
	}
      else
	{
	  // start from lb = infinity and move bakwards 	  
	  grad = C1*wc.coeff(cs);	  
	  if (print)
	    {
	      cout << cs << "  grad L start  " << grad << endl;
	    }
	  for (std::vector<size_t>::reverse_iterator i = indices.rbegin(); i != indices.rend(); i++)
	    {
	      bool plus = false;
	      size_t idx = *i;
	      if (idx >= n)
		{
		  plus = true;
		  idx -= n;
		}

	      class_weight = C1;
	      other_weight = C2;
	      if (params.ml_wt_by_nclasses)
		{
		  other_weight /= nclasses.coeff(idx);
		}
	      if (params.ml_wt_class_by_nclasses)
		{
		  class_weight /= nclasses.coeff(idx);
		}

	      for (SparseMb::InnerIterator it(y,idx); it; ++it)
		{
		  if (it.value())
		    {
		      cs2 = it.col();
		      if (cs2 == cs && !plus)
			{
			  grad -= class_weight;
			}
		      else
			{
			  if (class_order[cs2] < class_order[cs] && plus  && (none_filtered || !(filtered.get(idx,cs))))
			    {
			      grad -= other_weight;
			    }
			}
		    }	      
		}
	      if (grad < 0)
		{
		  l.coeffRef(cs) = allproj.coeff(*i);
		  if (print)
		    {
		      cout << cs << "  grad L end  " << grad << endl;
		      cout << cs << " opt L " << l.coeff(cs) << endl;
		    }
		  break;
		}
	    }
	}
    }      
#endif

      // gradL = ;
      // for (i = 2*n-1; i>=0; i--)
      // 	{
      // 	  bool plus = false;
      // 	  size_t idx = indices[n];
      // 	  if (idx >= n)
      // 	    {
      // 	      plus = true;
      // 	      idx -= n;
      // 	    }
      // 	  for (SparseMb::Iterator it(y,idx); it; ++it)
      // 	    {
      // 	      if (it.value())
      // 		{
      // 		  cs2 = it.col();
      // 		  if (cs2 == cs && !plus)
      // 			  gradU -= C1;
      // 			}
      // 		      else
      // 			{
      // 			  gradL += C1;
      // 			}
      // 		    }
      // 		  else
      // 		    {
      // 		      if (class_order[cs2] < class_order[cs] && plus)
      // 			{
      // 			  gradL -= C2;
      // 			}
      // 		      if (class_order[cs2] > class_order[cs] && !plus)
      // 			{
      // 			  gradU += C2;
      // 			}
      // 		    }
      // 	       	}	      
      // 	    }
      // 	  if (gradU > 0)
      // 	    {
      // 	      l.coeffRef(c) = allproj.coeff(indices[i]);
      // 	      break;
      // 	    }
      // 	}
}


#elif OPTIMIZE_LU_VERSION == 1          // Erik's version

// *****************************************************
// get the optimal values for lower and upper bounds given
// a projection and the class order
// computationally expensive so it should be done sparingly
void optimizeLU(VectorXd& l, VectorXd& u,
		const VectorXd& projection, const SparseMb& y,
		const vector<int>& class_order, const vector<int>& sorted_class,
		const VectorXd& wc, const VectorXi& nclasses,
		const boolmatrix& filtered,
		const double C1, const double C2,
		const param_struct& params,
		bool print)
{
  size_t n = projection.size();
  size_t noClasses = y.cols();
  VectorXd allproj(2*n);
  //allproj.setZero();            // valgrind issues ???
  allproj << (projection.array() - 1), (projection.array() + 1);

  bool none_filtered = filtered.count()==0;             // XXX ouch ???
  std::vector<size_t> indices(allproj.size());
  if(1){
      sort_index(allproj, indices);  // valgrind complaints ??? XXX
  }else{ // incorrect?
      std::vector<size_t> indices( boost::counting_iterator<size_t>(0),
                                   boost::counting_iterator<size_t>(n+n));
      std::sort(indices.begin(), indices.end(),
                [&allproj]( size_t const i, size_t const j )-> bool
                { return allproj[i] < allproj[j]; });
  }

#ifdef _OPENMP
  int min_chunk_size = 10000;
  int max_n_chunks = omp_get_max_threads();
#else
  //int max_n_chunks = 1;
#endif

#if MCTHREADS
#pragma omp parallel default(shared) shared(l, u, allproj, indices,none_filtered, filtered, n, noClasses, y, class_order, sorted_class, wc, nclasses, params, min_chunk_size, max_n_chunks)
#endif
  {
#if MCTHREADS
#pragma omp single
#endif
    {
      double class_weight, other_weight;
      std::vector<int> classes;
      classes.reserve(nclasses.maxCoeff());
      // calculate the optimal value for upper bounds
      // iterate from the beginning

      // grad are stored in order of the ranked classes
      // to minimize cache misses and false sharing
      VectorXd grad(noClasses);
      double classweight;
      for (size_t sc = 0; sc <noClasses; sc++) {
          // this does not work if the remove_class_constraints is true (i.e. true classes can be filtered)!!
          classweight = wc.coeff(sorted_class[sc]);
          if (classweight == 0) { // there are no examples of this class
              // so just put l and u to 0 (l is set below)
              u.coeffRef(sorted_class[sc]) = 0;
              grad.coeffRef(sc) = -1;
          } else {
              grad.coeffRef(sc) = C1*classweight;
          }
      }
      //      VectorXd grad = C1*wc;

      for (std::vector<size_t>::const_iterator i = indices.begin(); i != indices.end(); i++) {
          bool plus = false;
          size_t idx = *i;
          if (idx >= n) { plus = true; idx -= n; }

          if (plus) {
              // only the upper bounds of the classes of this example are affected
              class_weight = C1;
              if (params.ml_wt_class_by_nclasses) {
                  class_weight /= nclasses.coeff(idx);
              }
              for (SparseMb::InnerIterator it(y,idx); it; ++it) {
                  if (it.value()) {
                      int cs = it.col();                // raw [unsorted] class
                      int sc = class_order[cs];         // sorted class number
                      // should check for filtered (in case classes were filtered)
                      // but things break above if remove_class_constraints is true
                      if (grad.coeff(sc) >= 0 ) {
                          grad.coeffRef(sc) -= class_weight;
                          if (grad.coeff(sc) <= 0) {
                              // the upper bound for the last class will be at the end of 
                              // the last example. If we make it at +inf then it will create
                              // problems if the order of the classes ever changes without
                              // optimizing the LU bounds.
                              u.coeffRef(cs) = allproj.coeff(*i);
                              // Loop is over example projections ordered by
                              // increasing upper bound, so end up with the highest
                              // projection for this class in 'u'.
                          }
                      }
                  }
              }
          }else{
              // only the classes ranked lower than the classes of this example are affected
              other_weight = C2;
              if (params.ml_wt_by_nclasses) {
                  other_weight /= nclasses.coeff(idx);
              }

              // how many classes of the curent instance should be ranked higher
              //  times the weight of each
              //  if each class has its own weight will need to
              //  be calculated below (or have it precomputed for each example
              //  as a corresponding wclasses to nclasses to be wclasses the same as wc
              //  corresponds to nc
              double right_update = other_weight * nclasses.coeff(idx);

              // calling y.coeff is expensive so get the classes in the ranked order here
              classes.resize(0);
              for (SparseMb::InnerIterator it(y,idx); it; ++it) {
                  if (it.value()) {
                      classes.push_back(class_order[it.col()]);
                  }
              }
              if (classes.size() == 0)
                  continue;

              std::sort(classes.begin(),classes.end());
              // we  update the upper bounds.
              // if a class has higher rank than the highest rank class of this example
              // it's upper bound will not be influenced by this example
              //	      int class_end = sorted_class[classes.back()]; // make sure classes is not empty

#if MCTHREADS
#ifdef _OPENMP
#endif
              // make sure there is enough work to do to paralelize this
              int n_chunks = classes.back()/min_chunk_size + 1;
              n_chunks = n_chunks < max_n_chunks?n_chunks:max_n_chunks;
              int chunk_size = classes.back()/n_chunks;
              int remaining = classes.back()%n_chunks;
              for (int chunk=0; chunk < n_chunks; chunk++)
              {
#if MCTHREADS
#pragma omp task default(shared) firstprivate(chunk) shared(grad, u, idx, i, sorted_class, classes, right_update, other_weight, allproj, none_filtered, filtered, chunk_size, remaining)
#endif
                  {
                      int sc_start = chunk*chunk_size + (chunk<remaining?chunk:remaining);
                      int sc_incr = chunk_size + (chunk<remaining);
                      // #pragma omp critical
                      // {
                      //   cout << "0   " << idx << "   " << *i << "   " << sc_start << "   " << sc_incr << "    " << chunk << "   " << classes.back() << endl;
                      // }
                      getBoundGrad(grad, u, idx, *i, sorted_class, sc_start, sc_start + sc_incr, classes, right_update, -other_weight, allproj, none_filtered, filtered);
                  }
              }
#if MCTHREADS
#pragma omp taskwait
#endif
#else // if not _OPENMP
              getBoundGrad(grad, u, idx, *i, sorted_class, 0,classes.back(),classes,right_update,-other_weight,allproj,none_filtered,filtered);
#endif // _OPENMP
          }
      }
    }

#if MCTHREADS
#pragma omp single
#endif
    {
      double class_weight, other_weight;
      std::vector<int> classes;
      classes.reserve(nclasses.maxCoeff());
      // calculate the optimal value for upper bounds
      // iterate from the end

      // grad are stored in order of the ranked classes
      // to minimize cash misses and false sharing
      VectorXd grad(noClasses);
      double classweight;
      for (size_t sc = 0; sc <noClasses; sc++)
	{
	  // this does not work if the remove_class_constraints is true (i.e. true classes can be filtered)!!
	  classweight = wc.coeff(sorted_class[sc]);
	  if (classweight == 0)
	    {
	      // there are no examples of this class
	      // so just put l and u to 0
	      l.coeffRef(sorted_class[sc]) = 0.0;
	      grad.coeffRef(sc) = -1;
	    }
	  else
	    {
	      grad.coeffRef(sc) = C1*classweight;
	    }
	}

      //      VectorXd grad = C1*wc;
      for (std::vector<size_t>::const_reverse_iterator i = indices.rbegin(); i != indices.rend(); i++)
	{
	  bool plus = false;
	  size_t idx = *i;
	  if (idx >= n)
	    {
	      plus = true;
	      idx -= n;
	    }

	  if (!plus)
	    {
	      // only the lower bounds of the classes of this example are affected
	      class_weight = C1;
	      if (params.ml_wt_class_by_nclasses)
		{
		  class_weight /= nclasses.coeff(idx);
		}
	      for (SparseMb::InnerIterator it(y,idx); it; ++it)
		{
		  if (it.value())
		    {
		      int cs = it.col();
		      int sc = class_order[cs];
		      if (grad.coeff(sc) >= 0 )
			{
			  grad.coeffRef(sc) -= class_weight;
			  if (grad.coeff(sc) <= 0 )
			    {
			      // the lower bound for the last class will be at the end of the
                              // last [reverse] example. If we make it at +inf then it will
                              // create problems if the order of the classes ever changes
                              // without optimizing the LU bounds.
			      l.coeffRef(cs) = allproj.coeff(*i);
			    }
			}
		    }
		}
	    }
	  else
	    {
	      // only the classes ranked higher than the classes of this example are affected
	      other_weight = C2;
	      if (params.ml_wt_by_nclasses)
		{
		  other_weight /= nclasses.coeff(idx);
		}


	      // calling y.coeff is expensive so get the classes in the ranked order here
	      classes.resize(0);
	      for (SparseMb::InnerIterator it(y,idx); it; ++it)
		  if (it.value())
		      classes.push_back(class_order[it.col()]);
              if(classes.size() == 0U)
                  continue;
	      std::sort(classes.begin(),classes.end());
	      // we  update the lower bounds.
	      // if a class has lower rank than the lowest rank class of this example
	      // it's lower bound will not be influenced by this example
	      //   int class_end = sorted_class[classes.front()]; // make sure classes is not empty

	      int n_active = noClasses - classes.front() - 1;
	      if (n_active == 0)
		{
		  continue;
		}
#ifdef _OPENMP
	      // make sure there is enough work to do to paralelize this
	      int n_chunks = n_active/min_chunk_size + 1;
	      n_chunks = n_chunks < max_n_chunks?n_chunks:max_n_chunks;
	      int chunk_size = n_active/n_chunks;
	      int remaining = n_active%n_chunks;
	      for (int chunk=0; chunk < n_chunks; chunk++)
		{
#if MCTHREADS
#pragma omp task default(shared) firstprivate(chunk) shared(grad, l, idx, i, sorted_class, classes, other_weight, allproj, none_filtered, filtered, chunk_size, remaining)
#endif
		  {
		    int sc_start = classes.front() + 1 + chunk*chunk_size + (chunk<remaining?chunk:remaining);
		    int sc_incr = chunk_size + (chunk<remaining);
		    getBoundGrad(grad, l, idx, *i, sorted_class, sc_start, sc_start + sc_incr, classes, 0.0, other_weight, allproj, none_filtered, filtered);
		  }
		}
#if MCTHREADS
#pragma omp taskwait
#endif
#else // if not _OPENMP
	      getBoundGrad(grad, l, idx, *i, sorted_class, classes.front() + 1,noClasses,classes, 0.0, other_weight, allproj, none_filtered, filtered);
#endif // _OPENMP
	    }
	}
    }
  }

#if 0

#pragma omp parallel sections default(none) shared(l, u, allproj, indices,none_filtered, filtered, n, noClasses, y, class_order, sorted_class, wc, nclasses, params)
  {
#pragma omp section
    {
      double class_weight, other_weight;
      std::vector<int> classes;
      classes.reserve(nclasses.maxCoeff());
      // calculate the optimal value for upper bounds
      // iterate from the beginning
      VectorXd grad = C1*wc;
      for (std::vector<size_t>::const_iterator i = indices.begin(); i != indices.end(); i++)
	{
	  bool plus = false;
	  size_t idx = *i;
	  if (idx >= n)
	    {
	      plus = true;
	      idx -= n;
	    }

	  if (plus)
	    {
	      // only the upper bounds of the classes of this example are affected
	      class_weight = C1;
	      if (params.ml_wt_class_by_nclasses)
		{
		  class_weight /= nclasses.coeff(idx);
		}
	      for (SparseMb::InnerIterator it(y,idx); it; ++it)
		{
		  if (it.value())
		    {
		      int cs = it.col();
		      if (grad.coeff(cs) >= 0 )
			{
			  grad.coeffRef(cs) -= class_weight;
			  if (grad.coeff(cs) < 0)
			    {
			      u.coeffRef(cs) = allproj.coeff(*i);
			    }
			}
		    }
		}
	    }
	  else
	    {
	      // only the classes ranked lower than the classes of this example are affected
	      other_weight = C2;
	      if (params.ml_wt_by_nclasses)
		{
		  other_weight /= nclasses.coeff(idx);
		}

	      // how many classes of the curent instance should be ranked higher
	      //  times the weight of each
	      //  if each class has its own weight will need to
	      //  be calculated below (or have it precomputed for each example
	      //  as a corresponding wclasses to nclasses to be wclasses the same as wc
	      //  corresponds to nc
	      double right_update = other_weight * nclasses.coeff(idx);

	      // calling y.coeff is expensive so get the classes in the ranked order here
	      classes.resize(0);
	      for (SparseMb::InnerIterator it(y,idx); it; ++it)
		{
		  if (it.value())
		    {
		      classes.push_back(class_order[it.col()]);
		    }
		}
	      std::sort(classes.begin(),classes.end());
	      // we  update the upper bounds.
	      // if a class has higher rank than the highest rank class of this example
	      // it's upper bound will not be influenced by this example
	      //	      int class_end = sorted_class[classes.back()]; // make sure classes is not empty
	      std::vector<int>::const_iterator class_iter = classes.begin();
	      for (std::vector<int>::const_iterator sc=sorted_class.begin(); *sc != class_end; sc++)
		{
		  if (*sc == sorted_class[*class_iter])
		    {
		      // example has this class
		      right_update -= other_weight;
		      class_iter++;
		      continue;
		    }
		  if (grad.coeff(*sc) >= 0 && (none_filtered || !(filtered.get(idx,*sc))))
		    {
		      grad.coeffRef(*sc) -= right_update;
		      if (grad.coeff(*sc) < 0)
			{
			  u.coeffRef(*sc) = allproj.coeff(*i);
			}
		    }
		}
	    }
	}
      // set the upper bound of the highest ranked class to be infinity
      // should be careful with this if not ranking by means!
      u.coeffRef(sorted_class.back()) = boost::numeric::bounds<double>::highest();
    }

#pragma omp section
    {
      double class_weight, other_weight;
      std::vector<int> classes;
      classes.reserve(nclasses.maxCoeff());
      // calculate the optimal value for upper bounds
      // iterate from the end
      VectorXd grad = C1*wc;
      for (std::vector<size_t>::const_reverse_iterator i = indices.rbegin(); i != indices.rend(); i++)
	{
	  bool plus = false;
	  size_t idx = *i;
	  if (idx >= n)
	    {
	      plus = true;
	      idx -= n;
	    }

	  if (!plus)
	    {
	      // only the upper bounds of the classes of this example are affected
	      class_weight = C1;
	      if (params.ml_wt_class_by_nclasses)
		{
		  class_weight /= nclasses.coeff(idx);
		}
	      for (SparseMb::InnerIterator it(y,idx); it; ++it)
		{
		  if (it.value())
		    {
		      int cs = it.col();
		      if (grad.coeff(cs) >= 0 )
			{
			  grad.coeffRef(cs) -= class_weight;
			  if (grad.coeff(cs) < 0)
			    {
			      l.coeffRef(cs) = allproj.coeff(*i);
			    }
			}
		    }
		}
	    }
	  else
	    {
	      // only the classes ranked higher than the classes of this example are affected
	      other_weight = C2;
	      if (params.ml_wt_by_nclasses)
		{
		  other_weight /= nclasses.coeff(idx);
		}

	      // how many classes of the curent instance should be ranked lower
	      //  times the weight of each
	      // we are starting from the end, so all classes are ranked lower
	      //  if each class has its own weight will need to
	      //  be calculated below (or have it precomputed for each example
	      //  as a corresponding wclasses to nclasses to be wclasses the same as wc
	      //  corresponds to nc
	      double left_update = other_weight * nclasses.coeff(idx);

	      // calling y.coeff is expensive so get the classes in the ranked order here
	      classes.resize(0);
	      for (SparseMb::InnerIterator it(y,idx); it; ++it)
		{
		  if (it.value())
		    {
		      classes.push_back(class_order[it.col()]);
		    }
		}
	      std::sort(classes.begin(),classes.end());
	      // we  update the lower bounds.
	      // if a class has lower rank than the lowest rank class of this example
	      // it's lower bound will not be influenced by this example
	      int class_end = sorted_class[classes.front()]; // make sure classes is not empty

	      std::vector<int>::const_reverse_iterator class_iter = classes.rbegin();
	      for (std::vector<int>::const_reverse_iterator sc=sorted_class.rbegin(); *sc != class_end; sc++)
		{
		  if (*sc == sorted_class[*class_iter])
		    {
		      // example is of this class
		      left_update -= other_weight;
		      class_iter++;
		      continue;
		    }
		  if (grad.coeff(*sc) >= 0 && (none_filtered || !(filtered.get(idx,*sc))))
		    {
		      grad.coeffRef(*sc) -= left_update;
		      if (grad.coeff(*sc) < 0)
			{
			  l.coeffRef(*sc) = allproj.coeff(*i);
			}
		    }
		}
	    }
	}
      // set the lower bound of the lowest ranked class to be -infinity
      // should be careful with this if not ranking by means!
      l.coeffRef(sorted_class.front()) = boost::numeric::bounds<double>::lowest();
    }
  }

#endif

#if 0

#  pragma omp parallel for default(shared) shared(l, u)
  for (int cs = 0; cs < y.cols(); cs++)
    {
      // get the optimial value of the upperbound for class cs
      int cs2;
      double grad;
      if (class_order[cs] == noClasses - 1)
	{
	  // the upper bound for the last ranked class is infinity
	  u.coeffRef(cs) = boost::numeric::bounds<double>::highest();
	  if (print)
	    {
	      cout << cs << " grad U start " << 0.00 << endl;
	      cout << cs << "  grad U end  " << grad << endl;
	      cout << cs << " opt U " << u.coeff(cs) << endl;
	    }
	}
      else
	{
	  // start from ub = -infinity
	  grad = C1*wc.coeff(cs);
	  if (print)
	    {
	      cout << cs << "  grad U start  " << grad << endl;
	    }
 	  for (std::vector<size_t>::iterator i = indices.begin(); i != indices.end(); i++)
	    {
	      bool plus = false;
	      size_t idx = *i;
	      if (idx >= n)
		{
		  plus = true;
		  idx -= n;
		}

	      class_weight = C1;
	      other_weight = C2;
	      if (params.ml_wt_by_nclasses)
		{
		  other_weight /= nclasses.coeff(idx);
		}
	      if (params.ml_wt_class_by_nclasses)
		{
		  class_weight /= nclasses.coeff(idx);
		}

	      for (SparseMb::InnerIterator it(y,idx); it; ++it)
		{
		  if (it.value())
		    {
		      cs2 = it.col();
		      if (cs2 == cs && plus)
			{
			  grad -= class_weight;
			}
		      else
			{
			  if (class_order[cs2] > class_order[cs] && !plus && (none_filtered || !(filtered.get(idx,cs))))
			    {
			      grad -= other_weight;
			    }
			}
		    }
		}
	      if (grad < 0) // protect against very small doubles?
		{
		  u.coeffRef(cs) = allproj.coeff(*i);
		  if (print)
		    {
		      cout << cs << "  grad U end  " << grad << endl;
		      cout << cs << " opt U " << u.coeff(cs) << endl;
		    }
		  break;
		}
	    }
	}
      // get the optimial value of the lower bound  for class cs
      if (class_order[cs] == 0)
	{
	  // the lower bound for the first ranked class is -infinity
	  l.coeffRef(cs) = boost::numeric::bounds<double>::lowest();
	  if (print)
	    {
	      cout << cs << " grad L start " << 0.00 << endl;
	      cout << cs << "  grad L end  " << grad << endl;
	      cout << cs << " opt L " << l.coeff(cs) << endl;
	    }
	}
      else
	{
	  // start from lb = infinity and move bakwards
	  grad = C1*wc.coeff(cs);
	  if (print)
	    {
	      cout << cs << "  grad L start  " << grad << endl;
	    }
	  for (std::vector<size_t>::reverse_iterator i = indices.rbegin(); i != indices.rend(); i++)
	    {
	      bool plus = false;
	      size_t idx = *i;
	      if (idx >= n)
		{
		  plus = true;
		  idx -= n;
		}

	      class_weight = C1;
	      other_weight = C2;
	      if (params.ml_wt_by_nclasses)
		{
		  other_weight /= nclasses.coeff(idx);
		}
	      if (params.ml_wt_class_by_nclasses)
		{
		  class_weight /= nclasses.coeff(idx);
		}

	      for (SparseMb::InnerIterator it(y,idx); it; ++it)
		{
		  if (it.value())
		    {
		      cs2 = it.col();
		      if (cs2 == cs && !plus)
			{
			  grad -= class_weight;
			}
		      else
			{
			  if (class_order[cs2] < class_order[cs] && plus  && (none_filtered || !(filtered.get(idx,cs))))
			    {
			      grad -= other_weight;
			    }
			}
		    }
		}
	      if (grad < 0)
		{
		  l.coeffRef(cs) = allproj.coeff(*i);
		  if (print)
		    {
		      cout << cs << "  grad L end  " << grad << endl;
		      cout << cs << " opt L " << l.coeff(cs) << endl;
		    }
		  break;
		}
	    }
	}
    }
#endif

      // gradL = ;
      // for (i = 2*n-1; i>=0; i--)
      // 	{
      // 	  bool plus = false;
      // 	  size_t idx = indices[n];
      // 	  if (idx >= n)
      // 	    {
      // 	      plus = true;
      // 	      idx -= n;
      // 	    }
      // 	  for (SparseMb::Iterator it(y,idx); it; ++it)
      // 	    {
      // 	      if (it.value())
      // 		{
      // 		  cs2 = it.col();
      // 		  if (cs2 == cs && !plus)
      // 			  gradU -= C1;
      // 			}
      // 		      else
      // 			{
      // 			  gradL += C1;
      // 			}
      // 		    }
      // 		  else
      // 		    {
      // 		      if (class_order[cs2] < class_order[cs] && plus)
      // 			{
      // 			  gradL -= C2;
      // 			}
      // 		      if (class_order[cs2] > class_order[cs] && !plus)
      // 			{
      // 			  gradU += C2;
      // 			}
      // 		    }
      // 	       	}
      // 	    }
      // 	  if (gradU > 0)
      // 	    {
      // 	      l.coeffRef(c) = allproj.coeff(indices[i]);
      // 	      break;
      // 	    }
      // 	}
}


#else
#error "OPTIMIZE_LU_VERSION should be 0 (Alex's original) or 1 (Erik's version)"
#endif
