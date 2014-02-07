#ifndef __FIND_W_H
#define __FIND_W_H

#include "constants.h"
#include "typedefs.h"
#include "WeightVector.h"
#include "printing.h"

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
				 const EigenType& x, const VectorXd& y,
				 const VectorXd& l, const VectorXd& u, const std::vector<int>& class_order,
				 double C1, double C2)
{

  double obj_val = w.norm();
  obj_val = .5 * obj_val * obj_val;
  int c;
  double sj;
  VectorXd projection;
  w.project(projection,x);
  for (int i = 0; i < x.rows(); i++)
    {
      c = (int) y.coeff(i) - 1; // again the label is started from 1
      obj_val += (C1
		  * (hinge_loss(projection.coeff(i) - l.coeff(c)) + hinge_loss(-projection.coeff(i) + u.coeff(c))));
	  
      for (int cp = 0; cp < l.size(); cp++)
	{
	  if (c != cp)
	    {
	      sj = (class_order[c] < class_order[cp]) ? 1 : 0;
	      obj_val += (C2
			  * (sj * hinge_loss(-projection.coeff(i) + l.coeff(cp))
			     + (1 - sj) * hinge_loss(projection.coeff(i) - u.coeff(cp))));
	    }
	}
	  
    }	
  return obj_val;
}

template<typename EigenType>
double calculate_objective_hinge(const WeightVector& w,
				 const EigenType& x, const VectorXd& y,
				 const VectorXd& l, const VectorXd& u, const VectorXd& class_order, double C1,
				 double C2)
{
  if (PRINT_O)
    cout << "calc objective: ";

  std::vector<int> v(class_order.size());
  toVector(v, class_order);
  double d = calculate_objective_hinge(w, x, y, l, u, v, C1, C2);

  if (PRINT_O)
    cout << d << endl;

  return d;
}

// ********************************
// Get unique values in the class vector -> classes
std::vector<int> get_classes(VectorXd y);

// *********************************
// functions and structures for sorting and keeping indeces
struct IndexComparator;

void sort_index(VectorXd& m, std::vector<int>& cranks);

// *********************************
// Ranks the classes to build the switches
void rank_classes(std::vector<int>& cranks, VectorXd& l, VectorXd& u);


// *******************************
// Get the number of exampls in each class

void init_nc(VectorXi& nc, const VectorXd& y, const int noClasses);

// ********************************
// Initializes the lower and upper bound
template<typename EigenType>
void init_lu(VectorXd& l, VectorXd& u, VectorXd& means, const VectorXi& nc, const WeightVector& w,
	     EigenType& x, const VectorXd& y,
	     const int noClasses)
{
  int n = x.rows();
  int c,i,k;
  for (k = 0; k < noClasses; k++)
    {
      means(k)=0;
      l(k)=std::numeric_limits<double>::max();
      u(k)=std::numeric_limits<double>::min();	      
    }
  VectorXd projection;
  w.project(projection,x);
  for (i=0;i<n;i++)
    {
      c = (int) y[i] - 1; // Assuming all the class labels start from 1
      double pr = projection.coeff(i);
      means(c)+=pr;
      l(c)=pr<l(c)?pr:l(c);
      u(c)=pr>u(c)?pr:u(c);
    }
  for (k = 0; k < noClasses; k++)
    {
      means(k) /= nc(k);
    }
}

template<typename EigenType>
void update_filtered(MatrixXb& filtered, const WeightVector& w, const VectorXd& l, const VectorXd& u, const EigenType& x, const VectorXd& y, bool filter_class)
{
  VectorXd projection;
  w.project(projection,x);
  for (int i = 0; i < x.rows(); i++)
    {
      int c = (int) y.coeff(i) - 1; // again the label is started from 1
      for (int cp = 0; cp < l.size(); cp++)
	{
	  if (c != cp || filter_class)
	    {
	      filtered.coeffRef(i,cp) = filtered.coeffRef(i,cp) || (projection.coeff(i)<l.coeff(cp))||(projection.coeff(i)>u.coeff(cp))?true:false;
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
			EigenType& x, const VectorXd& y,
			const double C1_, const double C2_, bool resumed)

{
  #ifdef PROFILE
  ProfilerStart("find_w.profile");
  #endif

  double lambda = 1.0/C2_;
  double C1 = C1_/C2_;
  double C2 = 1.0;
  const	int no_projections = weights.cols();
  cout << "no_projections: " << no_projections << endl;
  const int n = x.rows();
  bool filter_class = FILTER_CLASS;
  const int batch_size = (STOCHASTIC_BATCH_SIZE < 1 || STOCHASTIC_BATCH_SIZE > n) ? n : STOCHASTIC_BATCH_SIZE;
  int d = x.cols();
  std::vector<int> classes = get_classes(y);
  cout << "size x: " << x.rows() << " rows and " << x.cols() << " columns.\n";
  cout << "size y: " << y.rows() << " rows and " << y.cols() << " columns.\n";

  int noClasses = classes.size();
  WeightVector w;
  VectorXd l(noClasses),u(noClasses);
  VectorXd means(noClasses); // used for initialization of the class order vector;
  VectorXi nc(noClasses); // the number of examples in each class 
  double eta_t, tmp, sj;
  int c, cp;// current class and the other classes
  int obj_idx = 0, cp_count,cp1_count,cp0_count;
  bool order_changed = 1;
  VectorXd l_gradient(noClasses), u_gradient(noClasses);
  VectorXd proj(batch_size);
  VectorXi index(batch_size);
  double multiplier;
 
  unsigned int t = 1, i=0, k=0,idx=0;
  char iter_str[30];
  for(i=0; i<30; i++) iter_str[i]=' ';
  std::vector<int> class_order(noClasses), prev_class_order(noClasses);// mid points in the bound; used as the switch
       
  lower_bounds.resize(noClasses, no_projections);
  upper_bounds.resize(noClasses, no_projections);
  objective_val.resize(2 + (no_projections * OPT_MAX_ITER * OPT_MAX_REORDERING / OPT_EPOCH));

  init_nc(nc, y, noClasses);
  MatrixXb filtered(n,noClasses);
  
  // can't do it this way because the initial projections won't be orthogonal
  // this will create some problems when we initialize the calss order vector
  // we'll need to change this. 
  for(int projection_dim=0; projection_dim < no_projections; projection_dim++)
    {
	 
      w = WeightVector(weights.col(projection_dim));
      
      // w.setRandom(); // initialize to a random value
      if (!resumed)
	{
	  //initialize the class_order vector by sorting the means of the projections of each class. Use l to store the means.
	  init_lu(l,u,means,nc,w,x,y,noClasses);
	  rank_classes(class_order,means,means);
	}
      else 
	{	  
	  l = lower_bounds.col(projection_dim);
	  u = upper_bounds.col(projection_dim);
	  rank_classes(class_order, l, u);
	}
	      
      order_changed = 1;

      print_report<EigenType>(projection_dim,batch_size, noClasses,C1,C2,w.size(),x);

      // staring optimization
      for (int iter = 0; iter < OPT_MAX_REORDERING && order_changed == 1; iter++)
	{
	  snprintf(iter_str,30, "Iteration %d > ",iter+1);

	  // init the optimization specific parameters
	  std::copy(class_order.begin(),class_order.end(), prev_class_order.begin());
		    
	  cout << "\n Class_Order: ";
		    
	  for (k = 0; k < noClasses; k++)
	    {
	      cout << class_order[k] << "  ";
	    }
	  cout << endl;
		    
	  t = 1;
		    
	  while (t < OPT_MAX_ITER)
	    {
	      t++;
			    
	      // setting eta
	      eta_t = 1.0 / sqrt(t);
	      if(eta_t < 1e-4)// (t < OPT_MAX_IT		ER / 10){
		{
		  eta_t = 1e-4;
		}
			    
	      if(t % OPT_EPOCH == 0)
		{
		  print_progress(iter_str, t, OPT_MAX_ITER);
		  objective_val[obj_idx++] = calculate_objective_hinge(w, x, y, l, u, class_order, C1, C2); // save the objective value
		  if(PRINT_O)
		    {
		      cout << "objective_val[" << t << "]: " << objective_val[obj_idx-1] << " "<< w.norm() << endl;
		    }
		}
			    
	      // find the number of samples from other classes
	      l_gradient.setZero();
	      u_gradient.setZero();
	      
	      

	      // first compute all the projections so that we can update w directly

	      for (idx = 0; idx < batch_size; idx++)// batch_size will be equal to n for complete GD
		{
		  if(STOCHASTIC_BATCH_SIZE > 0)
		    {
		      i = ((int) rand()) % ((int)n);
		    }
		  else
		    {
		      i=idx;
		    }

		  proj.coeffRef(idx) = w.project_row(x,i);// (x.row(i)*w)(0,0);
		  index.coeffRef(idx)=i;
		}

	      // now we can update w directly
	      // update for the reglarizer
	      w.scale(1.0-lambda*eta_t);
		
	      for (idx = 0; idx < batch_size; idx++)// batch_size will be equal to n for complete GD
		{		
		  tmp = proj.coeff(idx);
		  i=index.coeff(idx);
		  c = (int) y[i] - 1; // Assuming all the class labels start from 1
				
		  multiplier = 0.0;
		  
		  if(PRINT_I)
		    {	
		      cout << i << ", c: " << c << endl;
		    }
				
				
		  if ((hinge_loss(tmp - l.coeff(c)) > 0) && !filtered.coeff(i,c))// I1 Condition
		    {
		      if(PRINT_I)
			{
			  cout << "I1 : " << idx << ", " << i<< endl;
			}
		      multiplier -= C1;
		      l_gradient.coeffRef(c) += C1;
		    } // end if
				
		  if ((hinge_loss(-tmp + u.coeff(c)) > 0) && !filtered.coeff(i,c))//  I2 Condition
		    {
		      if(PRINT_I)
			{
			  cout << "I2 : " << idx << ", " << i << endl;
			}
		      multiplier +=C1;
		      u_gradient.coeffRef(c) -= C1;
		    } // end if
				
		  for (cp = 0; cp < noClasses; cp++)
		    {
		      if (cp != c && !filtered.coeff(i,cp))
			{
			  sj = (class_order[c] < class_order[cp]) ? 1 : 0;
			  if (sj == 1 && hinge_loss(-tmp + l.coeff(cp)) > 0) // I3 Condition
			    {
			      if(PRINT_I)
				{
				  cout << "I3 : " << idx << ", " << i << endl;
				}
			      multiplier +=C2;
			      l_gradient.coeffRef(cp) -= C2;
			    }
					
			  if (sj == 0 && hinge_loss(tmp - u.coeff(cp)) > 0) //  I4 Condition
			    {
			      if(PRINT_I)
				{
				  cout << "I4 : " << idx << ", " << i<< endl;
				}
			      multiplier -= C2;
			      u_gradient.coeffRef(cp) += C2;
			    }
			} // end if cp != c
		    } // end for cp

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
	  rank_classes(class_order, l, u);// ranking classes				
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
	       << " iterations ... with w.norm(): " << w.norm();
			
	} // end for iter
      
      VectorXd vect;
      w.toVectorXd(vect);
      weights.col(projection_dim) = vect;
      lower_bounds.col(projection_dim) = l;
      upper_bounds.col(projection_dim) = u;
      
      update_filtered(filtered, w, l, u, x, y, filter_class);

		      
    } // end for projection_dim
	
  cout << "\n---------------\n" << endl;
	
  objective_val[obj_idx++] = calculate_objective_hinge(w, x, y, l, u,
						       class_order, C1, C2);// save the objective value
  objective_val = objective_val.head(obj_idx);
  
  #ifdef PROFILE
  ProfilerStop();
  #endif
}

#endif
