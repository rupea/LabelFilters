#include <iostream>
#include <vector>
#include <stdio.h>
#include <typeinfo>
#include <math.h>
#include <stdlib.h>
#include "Eigen/Dense"
#include "Eigen/Sparse"
#include "constants.h"
#include "typedefs.h"
#include "WeightVector.h"
#include "printing.h"
#include "utils.h"
#include "find_w.h"

using Eigen::VectorXd;
using Eigen::VectorXi;

using namespace std;



// ******************************
// Convert to a STD vetor from Eigen Vector
void toVector(std::vector<int>& to, const VectorXd& from)
{
  for (int i = 0; i < from.size(); i++)
    {
      to.push_back((int) from(i));
    }
}


// ********************************
// Get unique values in the class vector -> classes

std::vector<int> get_classes(VectorXd& y)
{
  std::vector<int> v;
  for (int i = 0; i < y.rows(); i++)
    {
      if (std::find(v.begin(), v.end(), y[i]) == v.end()) // if the label does not exist
	{
	  v.push_back((int) y[i]);
	}
    }
  return v;
}


// *********************************
// Ranks the classes to build the switches
void rank_classes(std::vector<int>& indices, std::vector<int>& cranks, const VectorXd& sortkey)
{
  sort_index(sortkey, indices);
  for (int i = 0; i < sortkey.size(); i++)
    {
      cranks[indices[i]] = i;
    }
}

// **********************************************
// get l and u in the original class order

void get_lu (VectorXd& l, VectorXd& u, const VectorXd& sortedLU, const vector<int>& sorted_class)
{
  vector<int>::const_iterator sorted_class_iter;
  const double* sortedLU_iter;
  int cp;
  for (sorted_class_iter = sorted_class.begin(),sortedLU_iter=sortedLU.data(); sorted_class_iter != sorted_class.end(); sorted_class_iter++)
    {
      cp = *sorted_class_iter;
      l.coeffRef(cp) = *(sortedLU_iter++);
      u.coeffRef(cp) = *(sortedLU_iter++);
    }      
}

// **********************************
// sort l and u in the new class order

void get_sortedLU(VectorXd& sortedLU, const VectorXd& l, const VectorXd& u, const vector<int>& sorted_class)
{
  for (int i = 0; i < sorted_class.size(); i++)
    {
      sortedLU.coeffRef(2*i) = l.coeff(sorted_class[i]);
      sortedLU.coeffRef(2*i+1) = u.coeff(sorted_class[i]);
    }
}

// *******************************
// Get the number of exampls in each class

void init_nc(VectorXi& nc, VectorXi& nclasses, const SparseMb& y)
{  
  int noClasses = y.cols();
  if (nc.size() != noClasses) 
    {
      cerr << "init_nc has been called with vector nc of wrong size" << endl;
      exit(-1);
    }
  int n = y.rows();  
  if (nclasses.size() != n) 
    {
      cerr << "init_nc has been called with vector nclasses of wrong size" << endl;
      exit(-1);
    }
  for (int k = 0; k < noClasses; k++)
    {
      nc(k)=0;
    }
  for (int i=0;i<n;i++)
    {
      nclasses[i]=0;
      for (SparseMb::InnerIterator it(y,i);it;++it)
	{
	  if (it.value())
	    {
	      nc(it.col())++;
	      nclasses(it.row())++;
	    }
	}
    }
}

//*****************************************
// Update the filtered constraints

void update_filtered(boolmatrix& filtered, const VectorXd& projection,  
		     const VectorXd& l, const VectorXd& u, const SparseMb& y, 
		     const bool filter_class)
{
  int noClasses = y.cols();			
  int c;
  for (size_t i = 0; i < projection.size(); i++)
    {      
      double proj = projection.coeff(i);
      SparseMb::InnerIterator it(y,i);
      while ( it && !it.value() ) ++it;
      c=it?it.col():noClasses;
      for (int cp = 0; cp < noClasses; cp++)
	{
	  if ( filter_class || cp != c )
	    {
	      bool val = (proj<l.coeff(cp))||(proj>u.coeff(cp))?true:false;
	      if (val) 
		{
		  filtered.set(i,cp); 
		}
	      //no_filtered += filtered[i][cp] = filtered[i][cp] || (projection.coeff(i)<l.coeff(cp))||(projection.coeff(i)>u.coeff(cp))?true:false;
	    }
	  if ( cp == c )
	    {
	      ++it;
	      while ( it && !it.value() ) ++it;      
	      c=it?it.col():noClasses;
	    }
	} 
    }
}

  

// ***********************************************
// calculate the multipliers (for the w gradient update)
// and the gradients for l and u updates 
// on a subset of classes and instances

void compute_gradients (VectorXd& multipliers , VectorXd& sortedLU_gradient, 
			size_t idx_start, size_t idx_end, 
			int sc_start, int sc_end,
			const VectorXd& proj, const VectorXsz& index,
			const SparseMb& y, const VectorXi& nclasses, 
			int maxclasses, 
			const vector<int>& sorted_class,
			const vector<int>& class_order, 
			const VectorXd& sortedLU,
			const boolmatrix& filtered,
			double C1, double C2,
			const param_struct& params )
{  
  int sc, cp;
  int noClasses = y.cols();
  size_t idx, i;
  size_t no_filtered = filtered.count();
  double ml_wt = 1.0;
  double ml_wt_class = 1.0;
  double class_weight, other_weight, left_update, right_update;
  double tmp;
  vector<int> classes;
  vector<int>::iterator class_iter;
  int left_classes, right_classes;
  double *multipliers_iter, *sortedLU_gradient_iter;
  const double *sortedLU_iter;

  // initialize the multiplier and sortedLU_gradient arrays
  multipliers.setZero(idx_end-idx_start);
  sortedLU_gradient.setZero(2*(sc_end-sc_start));
  classes.reserve(maxclasses+1);
  
  multipliers_iter = multipliers.data();
  for (idx = idx_start; idx < idx_end; idx++)// batch_size will be equal to n for complete GD
    {		
      tmp = proj.coeff(idx);
      i=index.coeff(idx);

      #ifdef PRINTI
      cout<< idx << "    " <<  i << endl;
      #endif

      if (params.ml_wt_by_nclasses)
	{
	  ml_wt = 1/nclasses[i];
	}
      if (params.ml_wt_class_by_nclasses)
	{
	  ml_wt_class = 1/nclasses[i];
	}		
      class_weight = ml_wt_class * C1;
      other_weight = ml_wt * C2;
      
      left_classes = 0; //number of classes to the left of the current one
      left_update = 0; // left_classes * other_weight
      right_classes = nclasses[i]; //number of classes to the right of the current one		  
      right_update = other_weight * right_classes;
      
      // calling y.coeff is expensive so get the classes here
      classes.resize(0);
      for (SparseMb::InnerIterator it(y,i); it; ++it)
	{
	  if (it.value())
	    {
	      int c = class_order[it.col()];
	      if ( c < sc_start ) 
		{
		  left_classes++;
		  left_update += other_weight;
		  right_classes--;
		  right_update -= other_weight;
		}
	      else if (c < sc_end)
		{
		  classes.push_back(c);
		}
	    }
	}
      classes.push_back(sc_end); // this will always be the last 
      std::sort(classes.begin(),classes.end());

      sc=sc_start;
      class_iter = classes.begin();
      sortedLU_iter = sortedLU.data() + 2*sc_start;
      sortedLU_gradient_iter = sortedLU_gradient.data();
      while (sc < sc_end)
	{
	  while(sc < *class_iter)
	    {
	      // while example is not of class cp
	      cp = sorted_class[sc]; 			  
	      if (no_filtered == 0 || !(filtered.get(i,cp))) 
		{			      
		  if (left_classes && ((1 - *sortedLU_iter + tmp) > 0)) // I3 Condition w*x > l(cp) - 1
		    {
#ifdef PRINTI
		      {
			cout << "I3 : " << idx << ", " << i << endl;
		      }
#endif
		      *multipliers_iter += left_update; // use the iterator for multiplier too ?
		      *sortedLU_gradient_iter += left_update;
		      //l_gradient.coeffRef(cp) -= other_weight*left_classes;
		    }
		  sortedLU_iter++;
		  sortedLU_gradient_iter++;
		  //if (right_classes && hinge_loss(tmp - u.coeff(cp)) > 0) //  I4 Condition
		  if (right_classes && ((1 - tmp + *sortedLU_iter) > 0)) //  I4 Condition  w*x < u(cp) + 1
		    {
#ifdef PRINTI
		      {
			cout << "I4 : " << idx << ", " << i<< endl;
		      }
#endif
		      *multipliers_iter -= right_update;
		      *sortedLU_gradient_iter -= right_update;
				  //u_gradient.coeffRef(cp) += other_weight*right_classes;
		    }
		  sortedLU_iter++;			      
		  sortedLU_gradient_iter++;
		}
	      else
		{
		  sortedLU_iter += 2; //the iterator needs to be incremeted even if the class is filtered
		  sortedLU_gradient_iter += 2; //the iterator needs to be incremeted even if the class is filtered
		}
	      sc++;
	    }
	  if (sc < sc_end) // test if we are done
	    {
	      // example has class cp
	      cp = sorted_class[sc]; 			  
	      if (!params.remove_class_constraints || no_filtered == 0 || !(filtered.get(i,cp))) 
		{			      
		  if ((1 - tmp + *(sortedLU_iter++)) > 0)// I1 Condition  w*x < l(c)+1
		    {
#ifdef PRINTI
		      {
			cout << "I1 : " << idx << ", " << i<< endl;
		      }
#endif
		      *multipliers_iter -= class_weight;
		      *sortedLU_gradient_iter -= class_weight;
		      //l_gradient.coeffRef(cp) += class_weight;
		    } // end if
		  
		  //if (hinge_loss(-tmp + u.coeff(cp)) > 0)//  I2 Condition
		  sortedLU_gradient_iter++;
		  
		  if ((1 + tmp - *(sortedLU_iter++)) > 0)//  I2 Condition  w*x > u(c)-1
		    {
#ifdef PRINTI
		      {
			cout << "I2 : " << idx << ", " << i << endl;
		      }
#endif
		      *multipliers_iter += class_weight;
		      *sortedLU_gradient_iter += class_weight;
		      //u_gradient.coeffRef(cp) -= class_weight;
		    } // end if			      
		  sortedLU_gradient_iter++;
		}
	      else
		{
		  sortedLU_iter +=2; //the iterator needs to be incremeted even if the class is filtered
		  sortedLU_gradient_iter +=2; //the iterator needs to be incremeted even if the class is filtered
		}
	      //update the left and right classes;
	      left_classes++;
	      left_update += other_weight;
	      right_classes--;
	      right_update -= other_weight;
	      ++class_iter;
	      sc++;
	    }
	} // while(sc<noClasses)
      multipliers_iter++;
    }  // end for idx (second)
}

// *****************************************
// Function to calculate the objective for one example
// this almost duplicates the function compute_objective
// the two functions should be unified
// this functions is easier to use with the finite_diff_test because
// it does not require the entire projection vector 
double calculate_ex_objective_hinge(size_t i, double proj, const SparseMb& y,
				    const VectorXi& nclasses,
				    const std::vector<int>& sorted_class,
				    const std::vector<int>& class_order,
				    const VectorXd& sortedLU,
				    const boolmatrix& filtered,
				    bool none_filtered,
				    double C1, double C2,
				    const param_struct& params)
{
  double obj_val=0;
  int noClasses = y.cols();
  double ml_wt,ml_wt_class;
  double class_weight, other_weight;
  std::vector<int> classes;
  std::vector<int>::iterator class_iter;
  classes.reserve(nclasses.coeff(i)+1);
  ml_wt = 1.0;
  ml_wt_class = 1.0;
  int left_classes, right_classes;
  double left_weight, right_weight;
  int sc,cp;
  const double* sortedLU_iter;
  
  if (params.ml_wt_by_nclasses)
    {
      ml_wt = 1/nclasses[i];
    }
  if (params.ml_wt_class_by_nclasses)
    {
      ml_wt_class = 1/nclasses[i];
    }
  class_weight = ml_wt_class * C1;
  other_weight = ml_wt * C2;
  
  left_classes = 0; //number of classes to the left of the current one
  left_weight = 0; // left_classes * other_weight
  right_classes = nclasses[i]; //number of classes to the right of the current one
  right_weight = other_weight * right_classes;
  
  // calling y.coeff is expensive so get the classes here
  classes.resize(0);
  for (SparseMb::InnerIterator it(y,i); it; ++it)
    {
      if (it.value())
	{
	  classes.push_back(class_order[it.col()]);
	}
    }
  classes.push_back(noClasses); // this will always be the last
  std::sort(classes.begin(),classes.end());
  
  sc=0;
  class_iter = classes.begin();
  sortedLU_iter=sortedLU.data();
  while (sc < noClasses)
    {
      while(sc < *class_iter)
	{
	  // while example is not of class cp
	  cp = sorted_class[sc];
	  if (none_filtered || !(filtered.get(i,cp)))
	    {
	      obj_val += left_classes?left_weight * hinge_loss(*sortedLU_iter - proj):0
		+ right_classes?right_weight * hinge_loss(proj - *(sortedLU_iter+1)):0;
	      
	      //obj_val += left_classes?left_weight * hinge_loss(l.coeff(cp) - proj):0
	      //+ right_classes?right_weight * hinge_loss(proj - u.coeff(cp)):0;
	    }
	  sc++;
	  sortedLU_iter+=2;
	}
      if (sc < noClasses) // test if we are done
	{
	  // example has class cp
	  cp = sorted_class[sc];
	  if (none_filtered == 1 || !(filtered.get(i,cp)))
	    {
	      obj_val += (class_weight
			  * (hinge_loss(proj - *sortedLU_iter)
			     + hinge_loss(*(sortedLU_iter+1) - proj)));
	      //obj_val += (class_weight
	      //* (hinge_loss(proj - l.coeff(cp))
	      //+ hinge_loss(u.coeff(cp) - proj)));		  		  
	    }
	  left_classes++;
	  right_classes--;
	  left_weight += other_weight;
	  right_weight -= other_weight;
	  ++class_iter;
	  sc++;
	  sortedLU_iter+=2;
	}
    }
  return obj_val;
}

// ***********************************
// calculates the objective value for a subset of instances and classes

double compute_objective(const VectorXd& projection, const SparseMb& y,
			 const VectorXi& nclasses, int maxclasses,
			 size_t i_start, size_t i_end, 
			 int sc_start, int sc_end, 
			 const vector<int>& sorted_class, 
			 const vector<int>& class_order,
			 const VectorXd& sortedLU,
			 const boolmatrix& filtered,
			 double C1, double C2,
			 const param_struct& params) 
{
  double tmp;
  int sc, cp;
  int left_classes, right_classes;
  double left_weight, right_weight, other_weight, class_weight;
  std::vector<int> classes;
  std::vector<int>::iterator class_iter;
  size_t no_filtered = filtered.count();
  classes.reserve(maxclasses+1);
  double obj_val = 0.0;
  double ml_wt = 1.0;
  double ml_wt_class = 1.0;
  const double* sortedLU_iter;
  for (size_t i = i_start; i < i_end; i++)
    {
      tmp = projection.coeff(i);
     
      if (params.ml_wt_by_nclasses)
	{
	  ml_wt = 1/nclasses[i];
	}
      if (params.ml_wt_class_by_nclasses)
	{
	  ml_wt_class = 1/nclasses[i];
	}
      class_weight = ml_wt_class * C1;
      other_weight = ml_wt * C2;
      
      left_classes = 0; //number of classes to the left of the current one
      left_weight = 0; // left_classes * other_weight
      right_classes = nclasses[i]; //number of classes to the right of the current one
      right_weight = other_weight * right_classes;
      
      // calling y.coeff is expensive so get the classes here
      classes.resize(0);
      for (SparseMb::InnerIterator it(y,i); it; ++it)
	{
	  int c = class_order[it.col()];
	  if ( c < sc_start ) 
	    {
	      left_classes++;
	      left_weight += other_weight;
	      right_classes--;
	      right_weight -= other_weight;
	    }
	  else if (c < sc_end)
	    {
	      classes.push_back(c);
	    }
	}
      classes.push_back(sc_end); // this will always be the last
      std::sort(classes.begin(),classes.end());
      
      sc=sc_start;
      class_iter = classes.begin();
      sortedLU_iter = sortedLU.data() + 2*sc_start;
      while (sc < sc_end)
	{
	  while(sc < *class_iter)
	    {
	      // while example is not of class cp
	      cp = sorted_class[sc];
	      if (no_filtered == 0 || !(filtered.get(i,cp)))
		{
		  obj_val += left_classes?left_weight * hinge_loss(*sortedLU_iter - tmp):0
		    + right_classes?right_weight * hinge_loss(tmp - *(sortedLU_iter+1)):0;

		  //obj_val += left_classes?left_weight * hinge_loss(l.coeff(cp) - tmp):0
		  //+ right_classes?right_weight * hinge_loss(tmp - u.coeff(cp)):0;
		}
	      sc++;
	      sortedLU_iter+=2;
	    }
	  if (sc < sc_end) // test if we are done
	    {
	      // example has class cp
	      cp = sorted_class[sc];
	      if (!params.remove_class_constraints || no_filtered == 0 || !(filtered.get(i,cp)))
		{
		  obj_val += (class_weight
			      * (hinge_loss(tmp - *sortedLU_iter)
				 + hinge_loss(*(sortedLU_iter+1) - tmp)));
		  //obj_val += (class_weight
		  //* (hinge_loss(tmp - l.coeff(cp))
		  //+ hinge_loss(u.coeff(cp) - tmp)));		  
		}
	      left_classes++;
	      right_classes--;
	      left_weight += other_weight;
	      right_weight -= other_weight;
	      ++class_iter;
	      sc++;
	      sortedLU_iter+=2;
	    }
	}
    }
  return obj_val;
}

// *******************************
// Calculates the objective function
#if 1
double calculate_objective_hinge(const VectorXd& projection, const SparseMb& y,
				 const VectorXi& nclasses,
                                 const std::vector<int>& sorted_class,
                                 const std::vector<int>& class_order,
				 const double norm, const VectorXd& sortedLU,
				 const boolmatrix& filtered,
				 double lambda, double C1, double C2,
				 const param_struct& params)
{
  const int noClasses = y.cols();
  double obj_val = 0;
  //  size_t no_filtered = filtered.count();
  bool any_filtered = filtered.count()==0;
  int maxclasses = nclasses.maxCoeff();
#pragma omp parallel for default(shared) reduction(+:obj_val)
  for (size_t i = 0; i < projection.size(); i++)
    {
      obj_val += calculate_ex_objective_hinge(i, projection.coeff(i),  y,
					      nclasses,
					      sorted_class,class_order,
					      sortedLU, filtered,
					      any_filtered,
					      C1, C2, params);      
    }
  obj_val += .5 * lambda * norm * norm;
  return obj_val;
}
#endif 



#if 0
double calculate_objective_hinge(const VectorXd& projection, const SparseMb& y,
				 const VectorXi& nclasses,
                                 const std::vector<int>& sorted_class,
                                 const std::vector<int>& class_order,
				 const double norm, const VectorXd& sortedLU,
                                 //const vector<bool> *filtered,
				 const boolmatrix& filtered,
				 double lambda, double C1, double C2,
				 const param_struct& params)
{
  const int noClasses = y.cols();
  double obj_val;
  int maxclasses = nclasses.maxCoeff(); 
  // how to split the work for gradient update iterations

#ifdef _OPENMP
  int total_chunks = 32*10;//omp_get_max_threads();
  int sc_chunks = total_chunks;// floor(sqrt(total_chunks));
  int i_chunks = total_chunks/sc_chunks;
  sc_chunks = total_chunks/i_chunks;
  //  omp_set_num_threads(total_chunks);
#else
  int i_chunks = 1;
  int sc_chunks = 1;
#endif 
  int sc_chunk_size = noClasses/sc_chunks;
  int sc_remaining = noClasses % sc_chunks;
  size_t i_chunk_size = projection.size()/i_chunks;
  size_t i_remaining = projection.size() % i_chunks;

# pragma omp parallel for  default(shared) collapse(2) reduction(+:obj_val)
  for (int i_chunk = 0; i_chunk < i_chunks; i_chunk++)
    for (int sc_chunk = 0; sc_chunk < sc_chunks; sc_chunk++)
      {
	// the first chunks will have an extra iteration 
	size_t i_start = i_chunk*i_chunk_size + (i_chunk<i_remaining?i_chunk:i_remaining);
	size_t i_incr = i_chunk_size + (i_chunk<i_remaining);
	// the first chunks will have an extra iteration 
	int sc_start = sc_chunk*sc_chunk_size + (sc_chunk<sc_remaining?sc_chunk:sc_remaining);
	int sc_incr = sc_chunk_size + (sc_chunk<sc_remaining);
	obj_val += compute_objective(projection,y,nclasses,maxclasses,
				     i_start, i_start+i_incr,
				     sc_start, sc_start+sc_incr,
				     sorted_class,class_order,
				     sortedLU,
				     filtered,
				     C1,C2,params);
      }
  obj_val += .5 * lambda * norm*norm;
  return obj_val;
}

#endif

double calculate_objective_hinge(const VectorXd& projection, const SparseMb& y,
				 const VectorXi& nclasses,
                                 const std::vector<int>& sorted_class, 
                                 const std::vector<int>& class_order, 
				 const double norm, const VectorXd& sortedLU, 
				 double lambda, double C1, double C2,
				 const param_struct& params)
{
  const int noClasses = y.cols();
  const int n = projection.size();
  boolmatrix filtered(n,noClasses);
  return calculate_objective_hinge(projection, y,nclasses, sorted_class, class_order, norm, sortedLU, filtered, lambda, C1, C2, params);
}




// ************************
// function to set eta for each iteration

double set_eta(const param_struct& params, size_t t, double lambda)
{
  double eta_t;
  switch (params.eta_type)
    {
    case ETA_CONST: 
      eta_t = params.eta;
    case ETA_SQRT:
      eta_t = params.eta/sqrt(t);
      break;
    case ETA_LIN:
      eta_t = params.eta/(1+params.eta*lambda*t);
      break;
    case ETA_3_4:
      eta_t = params.eta/pow(1+params.eta*lambda*t,3*1.0/4);
      break;
    default:
      cerr << "Eta option unknown" << endl;
      exit(-3);
    }
  if (eta_t < params.min_eta)
    {
      eta_t = params.min_eta;
    }
  return eta_t;
}

// ********************************
// Compute the means of the classes of the projected data
void proj_means(VectorXd& means, const VectorXi& nc,
		const VectorXd& projection, const SparseMb& y)
{
  int noClasses = y.cols();
  size_t n = projection.size();
  size_t c,i,k;
  means.resize(noClasses);
  means.setZero();
  for (i=0;i<n;i++)
    {
      for (SparseMb::InnerIterator it(y,i);it;++it)
	{	
	  if (it.value())
	    {
	      c = it.col();
	      means(c)+=projection.coeff(i);
	    }
	}
    }
  for (k = 0; k < noClasses; k++)
    {
      means(k) /= nc(k);
    }
}





// ********************************
// Initializes the lower and upper bound

/********* template functions are implemented in the header
template<typename EigenType>
void init_lu(VectorXd& l, VectorXd& u, VectorXd& means, const VectorXi& nc, const VectorXd& w,
	     EigenType& x, const VectorXd& y,
	     const int noClasses);
****************/

// *************************************
// need to reimplement this funciton to work with (inside) the WeightVector class
// this might be a costly operation that might be not needed
// we'll implement this when we get there

// // ******************************
// // Projection to a new vector that is orthogonal to the rest
// // It is basically Gram-Schmidt Orthogonalization
// void project_orthogonal(VectorXd& w, const DenseM& weights,
// 			const int& projection_dim)
// {
//   if (projection_dim == 0)
//     return;
  
//   // Assuming the first to the current projection_dim are the ones we want to be orthogonal to
//   VectorXd proj_sum(w.rows());
//   DenseM wt = w.transpose();
//   double norm;
  
//   proj_sum.setZero();
  
//   for (int i = 0; i < projection_dim; i++)
//     {
//       norm = weights.col(i).norm();
//       proj_sum = proj_sum
// 	+ weights.col(i) * ((wt * weights.col(i)) / (norm * norm));
//     }
  
//   w = (w - proj_sum);
// }


// *********************************
// Solve the optimization using the gradient decent on hinge loss

/********* template functions are implemented in the header
template<typename EigenType>
void solve_optimization(DenseM& weights, DenseM& lower_bounds,
			DenseM& upper_bounds, VectorXd& objective_val,
			EigenType& x, const VectorXd& y,
			const double C1_, const double C2_, bool resumed);
***********/

// ---------------------------------------
