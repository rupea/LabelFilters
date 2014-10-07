#ifndef __FIND_W_H
#define __FIND_W_H

#include "constants.h"
#include "typedefs.h"
#include "WeightVector.h"
#include "printing.h"
#include "parameter.h"

#ifdef PROFILE
#include <gperftools/profiler.h>
#endif

using Eigen::VectorXd;
using Eigen::VectorXi;

using namespace std;


// *******************************
// The hinge loss -- implemented here to get rid of compiler warnings
inline double hinge_loss(double val)
{
  return 1 - ((val<1)?val:1.0);
}


// ******************************
// Convert to a STD vetor from Eigen Vector
void toVector(std::vector<int>& to, const VectorXd& from);


// *******************************
// Calculates the objective function

template<typename EigenType>
double calculate_objective_hinge(const WeightVector& w,
				 const EigenType& x, const SparseMb& y,
				 const VectorXi& nclasses,
				 const VectorXd& l, const VectorXd& u, 
                                 const std::vector<int>& sorted_class, 
                                 const MatrixXb& filtered,
				 bool ml_wt_by_nclasses, bool ml_wt_class_by_nclasses,
				 double lambda, double C1, double C2)
{
  int noClasses = y.cols();
  double obj_val = w.norm();
  obj_val = .5 * lambda * obj_val * obj_val;
  double ml_wt,ml_wt_class;
  int left_classes, right_classes;
  int sc,cp;
  VectorXd projection;
  w.project(projection,x);
  for (int i = 0; i < x.rows(); i++)
    {
      //      c = (int) y.coeff(i) - 1; // again the label is started from 1
      
      
      ml_wt = 1.0;
      ml_wt_class = 1.0;
      if (ml_wt_by_nclasses)
	{
	  ml_wt = 1/nclasses[i];
	}
      if (ml_wt_class_by_nclasses)
	{
	  ml_wt_class = 1/nclasses[i];
	}		    
      left_classes = 0; //number of classes to the left of the current one
      right_classes = nclasses[i]; //number of classes to the right of the current one
      for (sc = 0; sc < noClasses; sc++)		    
	{
	  // traverse the classes in order of their desired ranks
	  cp = sorted_class[sc]; 	  
	  if (!filtered.coeff(i,cp))
	    {
	      if(y.coeff(i,cp))
		{
		  //example has class cp
		  obj_val += (ml_wt_class 
			      * (C1 
				 * (hinge_loss(projection.coeff(i) - l.coeff(cp)) 
				    + hinge_loss(-projection.coeff(i) + u.coeff(cp)))));
		  left_classes++;
		  right_classes--;
		}
	      else
		{
		  //example does not have class cp
		  obj_val += (ml_wt 
			      * (C2
				 * (left_classes * hinge_loss(-projection.coeff(i) + l.coeff(cp))
				    + right_classes * hinge_loss(projection.coeff(i) - u.coeff(cp)))));
		}
	    }		  
	}
    }	
  return obj_val;
}

template<typename EigenType>
double calculate_objective_hinge(const WeightVector& w,
				 const EigenType& x, const SparseMb& y,
				 const VectorXi& nclass,
				 const VectorXd& l, const VectorXd& u, 
                                 const std::vector<int>& sorted_class, 
				 double lambda, double C1, double C2)
{
  MatrixXb filtered = MatrixXb::Zero(x.rows(),sorted_class.size());
  return calculate_objective_hinge(w,x,y,nclass,l,u,sorted_class,filtered,lambda,C1,C2);
}

template<typename EigenType>
double calculate_objective_hinge(const WeightVector& w,
				 const EigenType& x, const SparseMb& y,
				 const VectorXi& nclass,
				 const VectorXd& l, const VectorXd& u, 
				 const VectorXd& sorted_class,
				 double lambda, double C1, double C2)
{
  if (PRINT_O)
    cout << "calc objective: ";

  std::vector<int> v(sorted_class.size());
  toVector(v, sorted_class);
  double d = calculate_objective_hinge(w, x, y, nclass, l, u, v, lambda, C1, C2);

  if (PRINT_O)
    cout << d << endl;

  return d;
}

// ************************************
// Convert a label vector to a label matrix
// Assumes that the label vector contains labels from 1 to noClasses

SparseMb labelVec2Mat (const VectorXd& yVec);


// ********************************
// Get unique values in the class vector -> classes
std::vector<int> get_classes(VectorXd y);

// *********************************
// functions and structures for sorting and keeping indeces
struct IndexComparator;

void sort_index(VectorXd& m, std::vector<int>& cranks);

// *********************************
// Ranks the classes to build the switches
void rank_classes(std::vector<int>& order, std::vector<int>& cranks, VectorXd& l, VectorXd& u);


// *******************************
// Get the number of exampls in each class

void init_nc(VectorXi& nc, VectorXi& nclasses, const SparseMb& y);

// ********************************
// Initializes the lower and upper bound
template<typename EigenType>
void init_lu(VectorXd& l, VectorXd& u, VectorXd& means, const VectorXi& nc,
	     const WeightVector& w,
	     EigenType& x, const SparseMb& y)
{
  int noClasses = y.cols();
  int n = x.rows();
  int c,i,k;
  double pr;
  means.resize(noClasses);
  means.setZero();
  for (k = 0; k < noClasses; k++)
    {
      l(k)=std::numeric_limits<double>::max();
      u(k)=std::numeric_limits<double>::min();	      
    }
  VectorXd projection;
  w.project(projection,x);
  for (i=0;i<n;i++)
    {
      for (SparseMb::InnerIterator it(y,i);it;++it)
	{	
	  if (it.value())
	    {
	      c = it.col();
	      pr = projection.coeff(i);
	      means(c)+=pr;

	      l(c)=pr<l(c)?pr:l(c);
	      u(c)=pr>u(c)?pr:u(c);
	    }
	}
    }
  for (k = 0; k < noClasses; k++)
    {
      means(k) /= nc(k);
    }
}

// ********************************
// Compute the means of the classes of the projected data
template<typename EigenType>
void proj_means(VectorXd& means, const VectorXi& nc, const WeightVector& w,
	     const EigenType& x, const SparseMb& y)
{
  int noClasses = y.cols();
  int n = x.rows();
  int c,i,k;
  means.resize(noClasses);
  means.setZero();
  VectorXd projection;
  double pr;
  w.project(projection,x);
  for (i=0;i<n;i++)
    {
      for (SparseMb::InnerIterator it(y,i);it;++it)
	{	
	  if (it.value())
	    {
	      c = it.col();
	      pr = projection.coeff(i);
	      means(c)+=pr;
	    }
	}
    }
  for (k = 0; k < noClasses; k++)
    {
      means(k) /= nc(k);
    }
}

template<typename EigenType>
void update_filtered(MatrixXb& filtered, const WeightVector& w, const VectorXd& l, const VectorXd& u, const EigenType& x, const SparseMb& y, bool filter_class)
{
  VectorXd projection;
  int noClasses = y.cols();
  w.project(projection,x);
  for (int i = 0; i < x.rows(); i++)
    {      
      for (int cp = 0; cp < noClasses; cp++)
	{
	  if ( filter_class || !y.coeff(i,cp) )
	    {
	      filtered.coeffRef(i,cp) = filtered.coeffRef(i,cp) || (projection.coeff(i)<l.coeff(cp))||(projection.coeff(i)>u.coeff(cp))?true:false;
	    }
	}
    }
}

// function to calculate the difference vector beween the mean vectors of two classes

template<typename EigenType>
  void difference_means(VectorXd& difference, const EigenType& x, const SparseMb& y, const VectorXi& nc, const int c1, const int c2)
{
  int d = x.cols();
  int n = x.rows();
  difference.resize(d);
  difference.setZero();
  for (int row=0;row<n;row++)
    {
      if (y.coeff(row,c1))
	{
	  typename EigenType::InnerIterator it(x,row);
	  for (; it; ++it)
	    {
	      difference.coeffRef(it.col())+=(it.value()/nc(c1));
	    }
	}
      if (y.coeff(row,c2))
	{
	  typename EigenType::InnerIterator it(x,row);
	  for (; it; ++it)
	    {
	      difference.coeffRef(it.col())-=(it.value()/nc(c2));
	    }
	}
    }
}

// ******************************
// Projection to a new vector that is orthogonal to the rest
// It is basically Gram-Schmidt Orthogonalization
// *************************************
// need to reimplement this funciton to work with (inside) the WeightVector class
// this might be a costly operation that might be not needed
// we'll implement this when we get there

// void project_orthogonal(VectorXd& w, const DenseM& weights,
// 			const int& projection_dim);

// *********************************
// Solve the optimization using the gradient decent on hinge loss

template<typename EigenType>
void solve_optimization(DenseM& weights, DenseM& lower_bounds,
			DenseM& upper_bounds, VectorXd& objective_val,
			EigenType& x, const SparseMb& y,
			bool resumed, const param_struct& params)

{
  #ifdef PROFILE
  ProfilerStart("find_w.profile");
  #endif

  double lambda = 1.0/params.C2;
  double C1 = params.C1/params.C2;
  double C2 = 1.0;
  const	int no_projections = weights.cols();
  cout << "no_projections: " << no_projections << endl;
  const int n = x.rows();  // number of samples in double
  const int batch_size = (params.batch_size < 1 || params.batch_size > n) ? (int) n : params.batch_size;
  int d = x.cols();
  //std::vector<int> classes = get_classes(y);
  cout << "size x: " << x.rows() << " rows and " << x.cols() << " columns.\n";
  cout << "size y: " << y.rows() << " rows and " << y.cols() << " columns.\n";

  const int noClasses = y.cols();
  WeightVector w;
  VectorXd l(noClasses),u(noClasses);
  VectorXd means(noClasses); // used for initialization of the class order vector;
  VectorXi nc(noClasses); // the number of examples in each class 
  VectorXi nclasses(n); // the number of examples in each class 
  double eta_t, tmp, sj;
  int cp;// current class and the other classes
  int obj_idx = 0;
  bool order_changed = 1;
  VectorXd l_gradient(noClasses), u_gradient(noClasses);
  VectorXd proj(batch_size);
  VectorXi index(batch_size);
  double multiplier;
  // in the multilabel case each example will have an impact proportinal
  // to the number of classes it belongs to. ml_wt and ml_wt_class
  // allows weighting that impact when updating params for the other classes
  // respectively its own class. 
  double ml_wt, ml_wt_class;
  int sc;
  int left_classes, right_classes;
  unsigned int t = 1, i=0, k=0,idx=0;
  char iter_str[30];
  for(i=0; i<30; i++) iter_str[i]=' ';
  std::vector<int> sorted_class(noClasses), class_order(noClasses), prev_class_order(noClasses);// used as the switch
       
  lower_bounds.resize(noClasses, no_projections);
  upper_bounds.resize(noClasses, no_projections);
  objective_val.resize(1000 + (no_projections * params.max_iter * params.max_reorder / params.report_epoch));

  init_nc(nc, nclasses, y);
  MatrixXb filtered(n,noClasses);
  filtered.setZero(n,noClasses);
  VectorXd difference(d);
  long int total_constraints = n*noClasses - (1-params.remove_class_constraints)*nc.sum();

  for(int projection_dim=0; projection_dim < no_projections; projection_dim++)
    {
      
      if ( projection_dim == 0 )
	{
	  w = WeightVector(weights.col(projection_dim));
	}
      else
	{
	  int c1 = ((int) rand()) % noClasses;
	  int c2 = ((int) rand()) % noClasses;
	  if (c1 == c2)
	    {
	      c2=(c1+1)%noClasses;
	    }
	  difference_means(difference,x,y,nc,c1,c2);
	  w = WeightVector(difference);
	}
      
      // w.setRandom(); // initialize to a random value
      if (!resumed)
	{
	  //initialize the class_order vector by sorting the means of the projections of each class. Use l to store the means.
	  init_lu(l,u,means,nc,w,x,y);
	  rank_classes(sorted_class, class_order,means,means);
	}
      else 
	{	  
	  l = lower_bounds.col(projection_dim);
	  u = upper_bounds.col(projection_dim);
	  rank_classes(sorted_class, class_order, l, u);
	}
	      
      order_changed = 1;

      print_report<EigenType>(projection_dim,batch_size, noClasses,C1,C2,lambda,w.size(),x);
      t = 1;
      // staring optimization
      for (int iter = 0; iter < params.max_reorder && order_changed == 1; iter++)
	{
	  snprintf(iter_str,30, "Iteration %d > ",iter+1);

	  // init the optimization specific parameters
	  std::copy(class_order.begin(),class_order.end(), prev_class_order.begin());
		    
	  while (t < params.max_iter)
	    {
	      t++;
			    
	      // setting eta
	      eta_t = params.eta / sqrt(t);
	      if(eta_t < params.min_eta)
		{
		  eta_t = params.min_eta;
		}
	      
	      if( params.report_epoch && ((t-2) % params.report_epoch == 0) )
		{
		  print_progress(iter_str, t, params.max_iter);
		  objective_val[obj_idx++] = calculate_objective_hinge(w, x, y, nclasses, l, u, sorted_class, filtered, params.ml_wt_by_nclasses, params.ml_wt_class_by_nclasses, lambda, C1, C2); // save the objective value
		  if(PRINT_O)
		    {
		      cout << "objective_val[" << t << "]: " << objective_val[obj_idx-1] << " "<< w.norm() << endl;
		    }
		}

	      if (params.reorder_epoch && (t % params.reorder_epoch == 0))
		{
		  if (params.rank_by_mean)
		    {
		      proj_means(means, nc, w, x, y);
		      rank_classes(sorted_class, class_order, means, means);
		    }
		  else
		    {
		      rank_classes(sorted_class, class_order, l, u);// ranking classes			       
		    }
		}
	      
	      l_gradient.setZero();
	      u_gradient.setZero();
	      
	      

	      // first compute all the projections so that we can update w directly

	      for (idx = 0; idx < batch_size; idx++)// batch_size will be equal to n for complete GD
		{
		  if(batch_size < n)
		    {
		      i = ((int) rand()) % n;
		    }
		  else
		    {
		      i=idx;
		    }
		  
		  proj.coeffRef(idx) = w.project_row(x,i);
		  index.coeffRef(idx)=i;
		}
	      
	      // now we can update w directly
	      // update for the reglarizer
	      w.scale(1.0-lambda*eta_t);
	      
	      for (idx = 0; idx < batch_size; idx++)// batch_size will be equal to n for complete GD
		{		
		  tmp = proj.coeff(idx);
		  i=index.coeff(idx);
				
		  multiplier = 0.0;
		  
		  if(PRINT_I)
		    {	
		      cout << i << endl;
		    }							       		    
		  ml_wt = 1.0;
		  ml_wt_class = 1.0;
		  if (params.ml_wt_by_nclasses)
		    {
		      ml_wt = 1/nclasses[i];
		    }
		  if (params.ml_wt_class_by_nclasses)
		    {
		      ml_wt_class = 1/nclasses[i];
		    }		    
		  left_classes = 0; //number of classes to the left of the current one
		  right_classes = nclasses[i]; //number of classes to the right of the current one
		  for (sc = 0; sc < noClasses; sc++)		    
		    {
		      // traverse the classes in order of their desired ranks
		      cp = sorted_class[sc]; 
		      if (!filtered.coeff(i,cp)) 
			{
			  if (!y.coeff(i,cp))
			    {
			      // if example i does not have class cp 
			      if (left_classes && hinge_loss(-tmp + l.coeff(cp)) > 0) // I3 Condition
				{
				  if(PRINT_I)
				    {
				      cout << "I3 : " << idx << ", " << i << endl;
				    }
				  multiplier += ml_wt*left_classes*C2;
				  l_gradient.coeffRef(cp) -= ml_wt*left_classes*C2;
				}
			      
			      if (right_classes && hinge_loss(tmp - u.coeff(cp)) > 0) //  I4 Condition
				{
				  if(PRINT_I)
				    {
				      cout << "I4 : " << idx << ", " << i<< endl;
				    }
				  multiplier -= ml_wt*right_classes*C2;
				  u_gradient.coeffRef(cp) += ml_wt*right_classes*C2;
				}
			    } // end if example does not have class cp
			  else
			    {
			      //this example has class cp
			      if (hinge_loss(tmp - l.coeff(cp)) > 0)// I1 Condition
				{
				  if(PRINT_I)
				    {
				      cout << "I1 : " << idx << ", " << i<< endl;
				    }
				  multiplier -= ml_wt_class*C1;
				  l_gradient.coeffRef(cp) += ml_wt_class*C1;
				} // end if
			      
			      if (hinge_loss(-tmp + u.coeff(cp)) > 0)//  I2 Condition
				{
				  if(PRINT_I)
				    {
				      cout << "I2 : " << idx << ", " << i << endl;
				    }
				  multiplier += ml_wt_class*C1;
				  u_gradient.coeffRef(cp) -= ml_wt_class*C1;
				} // end if
			      //update the left and right classes;
			      left_classes++;
			      right_classes--;			 
			    } // end if example has class cp
			}
		    } // end for sc

		  if (multiplier != 0)
		    {
		      w.gradient_update(x,i,(multiplier*eta_t)/batch_size);
		    }
		} // end for idx (second)
	      
	      // update the lower and upper bounds
	      multiplier = eta_t * 1.0 / batch_size;
	      l -= ( l_gradient * multiplier );
	      u -= ( u_gradient * multiplier );
	      
	      /// not implemented yet
	      // if(true)
	      // 	{
	      // 	  // perform orthogonal projection
	      // 	  project_orthogonal(w,weights,projection_dim);
	      // 	}
	      
	      if(PRINT_T==1)
		{
		  double obj = obj_idx >= 1 ? objective_val[obj_idx-1] : 0;
		  cout << "t: " << t << ", obj:" << objective_val[obj_idx-1]
		       << ", l:" << l.transpose() << ", u:" << u.transpose()
		       << ", cur_norm: " << w.norm() << endl;
		} // end if print
	      
	    } // end while t
	  
	  
	  // Let's check if s changed
	  // check if the orders are the same
	  order_changed = 0;
	  // check if the class_order are still the same
	  rank_classes(sorted_class, class_order, l, u);// ranking classes				
	  // check that the ranks are the same 
	  for(int c = 0; c < noClasses; c++)
	    {
	      if (class_order[c] != prev_class_order[c])
		{
		  order_changed = 1;
		  break;
		}
	    }
			
	  if(PRINT_T==1)
	    {
	      double obj = obj_idx >= 1 ? objective_val[obj_idx-1] : 0;
	      cout << "\nt: " << t << ", obj:" << obj
		   << ", l:" << l.transpose() << ", u:" << u.transpose()
		   << ", cur_norm: " << w.norm() << endl;
	    } // end if print
			
	  cout << "\r>> " << iter+1 << ": Done in " << t
	       << " iterations ... with w.norm(): " << w.norm() << endl;
			
	} // end for iter
      
      VectorXd vect;
      w.toVectorXd(vect);
      weights.col(projection_dim) = vect;
      lower_bounds.col(projection_dim) = l;
      upper_bounds.col(projection_dim) = u;
      
      if (params.remove_constraints)
	{
	  update_filtered(filtered, w, l, u, x, y, params.remove_class_constraints );
	  long int no_filtered = (filtered.cast<int>()).sum();
	  cout << "Filtered " << no_filtered << " out of " << total_constraints << endl;
	  // work on this. This is just a crude approximation.
	  // now every example - class pair introduces nclass(example) constraints
	  // if weighting is done, the number is different
	  // eliminating one example -class pair removes nclass(exmple) potential
	  // if the class not among the classes of the example
	  long int no_remaining = total_constraints - no_filtered;
	  lambda = no_remaining*1.0/(total_constraints*params.C2);
	}
      
      //      C2*=((n-1)*noClasses)*1.0/no_remaining;
      //C1*=((n-1)*noClasses)*1.0/no_remaining;

      
      
    } // end for projection_dim
	
  cout << "\n---------------\n" << endl;
	
  objective_val[obj_idx++] = calculate_objective_hinge(w, x, y, nclasses, l, u,
						       sorted_class, filtered,
						       params.ml_wt_by_nclasses, params.ml_wt_class_by_nclasses,
						       lambda, C1, C2);// save the objective value
  objective_val.conservativeResize(obj_idx);
  
  #ifdef PROFILE
  ProfilerStop();
  #endif
}

#endif
