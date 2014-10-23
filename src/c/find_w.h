#ifndef __FIND_W_H
#define __FIND_W_H

#include "constants.h"
#include "typedefs.h"
#include "WeightVector.h"
#include "printing.h"
#include "parameter.h"
#include "boolmatrix.h"

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
  return ((val<1)?(1-val):0);
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
                                 const std::vector<int>& class_order, 
                                 //const vector<bool> *filtered,
				 const boolmatrix& filtered,
				 bool ml_wt_by_nclasses, bool ml_wt_class_by_nclasses,
				 double lambda, double C1, double C2)
{
  const int noClasses = y.cols();
  double obj_val = w.norm();
  obj_val = .5 * lambda * obj_val * obj_val;
  double ml_wt,ml_wt_class;
  int left_classes, right_classes;
  double left_weight, right_weight;
  int sc,cp;
  VectorXd projection;
  double tmp, class_weight, other_weight;
  std::vector<int> classes;
  std::vector<int>::iterator class_iter;
  size_t no_filtered = filtered.count();
  classes.reserve(nclasses.maxCoeff()+1);
  w.project(projection,x);
  ml_wt = 1.0;
  ml_wt_class = 1.0;
  for (size_t i = 0; i < x.rows(); i++)
    {
      tmp = projection.coeff(i);
     
      if (ml_wt_by_nclasses)
	{
	  ml_wt = 1/nclasses[i];
	}
      if (ml_wt_class_by_nclasses)
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
      while (sc < noClasses)
	{
	  while(sc < *class_iter)
	    {
	      // while example is not of class cp
	      cp = sorted_class[sc]; 			  
	      if (no_filtered == 0 || !(filtered.get(i,cp))) 
		{			      
		  obj_val += left_classes?left_weight * hinge_loss(l.coeff(cp) - tmp):0
		    + right_classes?right_weight * hinge_loss(tmp - u.coeff(cp)):0;
		}
	      sc++;
	    }
	  if (sc < noClasses) // test if we are done
	    {
	      // example has class cp
	      cp = sorted_class[sc]; 			  
	      if (no_filtered == 0 || !(filtered.get(i,cp))) 
		{			      
		  obj_val += (class_weight 
			      * (hinge_loss(tmp - l.coeff(cp)) 
				 + hinge_loss(u.coeff(cp) - tmp)));
		  
		}
	      left_classes++;
	      right_classes--;
	      left_weight += other_weight;
	      right_weight -= other_weight;
	      ++class_iter;
	      sc++;
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
                                 const std::vector<int>& class_order, 
				 double lambda, double C1, double C2)
{
  const int noClasses = y.cols();
  const int n = x.rows();
  boolmatrix filtered(n,noClasses);
  /* vector<bool> *filtered = new vector<bool>[n]; */
  /* for (size_t i=0; i<n ; i++) */
  /*   { */
  /*     filtered[i]=vector<bool>(noClasses, false); */
  /*   } */
  return calculate_objective_hinge(w,x,y,nclass,l,u,sorted_class,class_order,filtered,lambda,C1,C2);
}

/* template<typename EigenType> */
/* double calculate_objective_hinge(const WeightVector& w, */
/* 				 const EigenType& x, const SparseMb& y, */
/* 				 const VectorXi& nclass, */
/* 				 const VectorXd& l, const VectorXd& u,  */
/* 				 const VectorXd& sorted_class, */
/* 				 const VectorXd& class_order, */
/* 				 double lambda, double C1, double C2) */
/* { */
/*   if (PRINT_O) */
/*     cout << "calc objective: "; */

/*   std::vector<int> v(sorted_class.size()); */
/*   toVector(v, sorted_class); */
/*   double d = calculate_objective_hinge(w, x, y, nclass, l, u, v, lambda, C1, C2); */

/*   if (PRINT_O) */
/*     cout << d << endl; */

/*   return d; */
/* } */

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
void rank_classes(std::vector<int>& order, std::vector<int>& cranks, const VectorXd& sortKey, std::vector<double>& sortedLU, const VectorXd& l, const VectorXd& u);


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
  size_t n = x.rows();
  size_t c,i,k;
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
  size_t n = x.rows();
  size_t c,i,k;
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
void update_filtered(boolmatrix& filtered, const WeightVector& w, const VectorXd& l, const VectorXd& u, const EigenType& x, const SparseMb& y, const bool filter_class)
//size_t update_filtered(vector<bool> *filtered, const WeightVector& w, const VectorXd& l, const VectorXd& u, const EigenType& x, const SparseMb& y, bool filter_class)
{
  VectorXd projection;
  int noClasses = y.cols();			
  int c;
  w.project(projection,x);
  for (size_t i = 0; i < x.rows(); i++)
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

// function to calculate the difference vector beween the mean vectors of two classes

template<typename EigenType>
  void difference_means(VectorXd& difference, const EigenType& x, const SparseMb& y, const VectorXi& nc, const int c1, const int c2)
{
  size_t d = x.cols();
  size_t n = x.rows();
  difference.resize(d);
  difference.setZero();
  for (size_t row=0;row<n;row++)
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
// 			const size_t& projection_dim);

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
  const size_t n = x.rows(); 
  const size_t batch_size = (params.batch_size < 1 || params.batch_size > n) ? (size_t) n : params.batch_size;
  const size_t d = x.cols();
  //std::vector<int> classes = get_classes(y);
  cout << "size x: " << x.rows() << " rows and " << x.cols() << " columns.\n";
  cout << "size y: " << y.rows() << " rows and " << y.cols() << " columns.\n";

  const int noClasses = y.cols();
  WeightVector w;
  VectorXd l(noClasses),u(noClasses);
  std::vector<double> sortedLU, sortedLUnew; // used to improve cache performance
  std::vector<double>::iterator sortedLU_iter, sortedLUnew_iter;
  VectorXd means(noClasses); // used for initialization of the class order vector;
  VectorXi nc(noClasses); // the number of examples in each class 
  VectorXi nclasses(n); // the number of examples in each class 
  double eta_t, tmp, sj;
  int cp;// current class and the other classes
  size_t obj_idx = 0;
  bool order_changed = 1;
  VectorXd l_gradient(noClasses), u_gradient(noClasses);
  VectorXd proj(batch_size);
  VectorXi index(batch_size);
  double multiplier;
  // in the multilabel case each example will have an impact proportinal
  // to the number of classes it belongs to. ml_wt and ml_wt_class
  // allows weighting that impact when updating params for the other classes
  // respectively its own class. 
  double ml_wt, ml_wt_class, class_weight, other_weight;
  int sc;
  int left_classes, right_classes;
  double left_update, right_update;
  size_t  i=0, k=0,idx=0;
  unsigned long t = 1;
  char iter_str[30];
  for(i=0; i<30; i++) iter_str[i]=' ';
  std::vector<int> sorted_class(noClasses), class_order(noClasses), prev_class_order(noClasses);// used as the switch
  std::vector<int> classes;
  std::vector<int>::iterator class_iter, sorted_class_iter;
  
  lower_bounds.resize(noClasses, no_projections);
  upper_bounds.resize(noClasses, no_projections);
  if (params.report_epoch > 0)
    {
      objective_val.resize(1000 + (no_projections * params.max_iter * params.max_reorder / params.report_epoch));
    }

  init_nc(nc, nclasses, y);
  
  classes.reserve(nclasses.maxCoeff()+1);
  sortedLU.reserve(2*noClasses);
  sortedLUnew.reserve(2*noClasses);

  //keep track of which classes have been elimninated for a particular example
  boolmatrix filtered(n,noClasses);
  /* vector<bool> *filtered = new vector<bool>[n]; */
  /* for (size_t i=0; i<n ; i++) */
  /*   { */
  /*     filtered[i]=vector<bool>(noClasses, false); */
  /*   } */
  //MatrixXb filtered(n,noClasses);  
  //filtered.setZero(n,noClasses);
  VectorXd difference(d);
  long int total_constraints = n*noClasses - (1-params.remove_class_constraints)*nc.sum();
  size_t no_filtered=0;

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
	  rank_classes(sorted_class, class_order, means,sortedLU,l,u);
	}
      else 
	{	  
	  l = lower_bounds.col(projection_dim);
	  u = upper_bounds.col(projection_dim);
	  if (params.rank_by_mean)
	    {
	      proj_means(means, nc, w, x, y);
	      rank_classes(sorted_class, class_order, means, sortedLU, l, u);
	    }
	  else
	    {
	      rank_classes(sorted_class, class_order, (l+u)*.5, sortedLU, l, u);// ranking classes			       
	    }
	}
      sortedLUnew = sortedLU;

      order_changed = 1;

      print_report<EigenType>(projection_dim,batch_size, noClasses,C1,C2,lambda,w.size(),x);

      // staring optimization
      for (int iter = 0; iter < params.max_reorder && order_changed == 1; iter++)
	{

	  // init the optimization specific parameters
	  std::copy(class_order.begin(),class_order.end(), prev_class_order.begin());
	  t = 1;
		    
	  while (t < params.max_iter)
	    {
	      t++;
			    
	      if (t % 1000 == 0)
		{
		  snprintf(iter_str,30, "Iteration %d > ",iter+1);
		  print_progress(iter_str, t, params.max_iter);
		  fflush(stdout);
		}
	      // setting eta
	      eta_t = params.eta / sqrt(t);
	      if(eta_t < params.min_eta)
		{
		  eta_t = params.min_eta;
		}
	      
	      if( params.report_epoch && params.report_epoch > 0 && (t % params.report_epoch == 0) )
		{
		  // get the current l and u in the original class order
		  for (sorted_class_iter = sorted_class.begin(),sortedLU_iter=sortedLU.begin(); sorted_class_iter != sorted_class.end(); sorted_class_iter++)
		    {
		      cp=*sorted_class_iter;
		      l.coeffRef(cp) = *(sortedLU_iter++);
		      u.coeffRef(cp) = *(sortedLU_iter++);
		    }
		  objective_val[obj_idx++] = calculate_objective_hinge(w, x, y, nclasses, l, u, sorted_class, class_order, filtered, params.ml_wt_by_nclasses, params.ml_wt_class_by_nclasses, lambda, C1, C2); // save the objective value
		  if(PRINT_O)
		    {
		      cout << "objective_val[" << t << "]: " << objective_val[obj_idx-1] << " "<< w.norm() << endl;
		    }
		}

	      if (params.reorder_epoch && (t % params.reorder_epoch == 0))
		{
		  // get the current l and u in the original class order
		  for (sorted_class_iter = sorted_class.begin(),sortedLU_iter=sortedLU.begin(); sorted_class_iter != sorted_class.end(); sorted_class_iter++)
		    {
		      cp = *sorted_class_iter;
		      l.coeffRef(cp) = *(sortedLU_iter++);
		      u.coeffRef(cp) = *(sortedLU_iter++);
		    }
		  if (params.rank_by_mean)
		    {
		      proj_means(means, nc, w, x, y);
		      rank_classes(sorted_class, class_order, means, sortedLU, l, u);
		    }
		  else
		    {
		      rank_classes(sorted_class, class_order, (l+u)*.5, sortedLU, l, u);// ranking classes			       
		    }
		  sortedLUnew = sortedLU;
		}
	      
	      //l_gradient.setZero();
	      //u_gradient.setZero();
	      
	      

	      // first compute all the projections so that we can update w directly
	      for (idx = 0; idx < batch_size; idx++)// batch_size will be equal to n for complete GD
		{
		  if(batch_size < n)
		    {
		      i = ((size_t) rand()) % n;
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
		  
		  #ifdef PRINTI
		    cout<< i << endl;
                  #endif

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
		  class_weight = ml_wt_class * C1 * eta_t / batch_size;
		  other_weight = ml_wt * C2 * eta_t / batch_size;
    
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
			  classes.push_back(class_order[it.col()]);
			}
		    }
		  classes.push_back(noClasses); // this will always be the last 
		  std::sort(classes.begin(),classes.end());

		  sc=0;
		  class_iter = classes.begin();
		  sortedLU_iter = sortedLU.begin();
		  sortedLUnew_iter = sortedLUnew.begin();
		  while (sc < noClasses)
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
				   multiplier += left_update;
				   *sortedLUnew_iter += left_update;
				  //l_gradient.coeffRef(cp) -= other_weight*left_classes;
				}
			      sortedLU_iter++;
			      sortedLUnew_iter++;
			      //if (right_classes && hinge_loss(tmp - u.coeff(cp)) > 0) //  I4 Condition
			      if (right_classes && ((1 - tmp + *sortedLU_iter) > 0)) //  I4 Condition  w*x < u(cp) + 1
				{
				  #ifdef PRINTI
				    {
				      cout << "I4 : " << idx << ", " << i<< endl;
				    }
				  #endif
				  multiplier -= right_update;
				  *sortedLUnew_iter -= right_update;
				  //u_gradient.coeffRef(cp) += other_weight*right_classes;
				}
			      sortedLU_iter++;			      
			      sortedLUnew_iter++;
			    }
			  else
			    {
			      sortedLU_iter += 2; //the iterator needs to be incremeted even if the class is filtered
			      sortedLUnew_iter += 2; //the iterator needs to be incremeted even if the class is filtered
			    }
			  sc++;
			}
		      if (sc < noClasses) // test if we are done
			{
			  // example has class cp
			  cp = sorted_class[sc]; 			  
			  if (no_filtered == 0 || !(filtered.get(i,cp))) 
			    {			      
			      if ((1 - tmp + *(sortedLU_iter++)) > 0)// I1 Condition  w*x < l(c)+1
				{
				  #ifdef PRINTI
				    {
				      cout << "I1 : " << idx << ", " << i<< endl;
				    }
				  #endif
				  multiplier -= class_weight;
				  *sortedLUnew_iter -= class_weight;
				  //l_gradient.coeffRef(cp) += class_weight;
				} // end if
			      
			      //if (hinge_loss(-tmp + u.coeff(cp)) > 0)//  I2 Condition
			      sortedLUnew_iter++;
			      
			      if ((1 + tmp - *(sortedLU_iter++)) > 0)//  I2 Condition  w*x > u(c)-1
				{
				  #ifdef PRINTI
				    {
				      cout << "I2 : " << idx << ", " << i << endl;
				    }
				  #endif
				  multiplier += class_weight;
				  *sortedLUnew_iter += class_weight;
				  //u_gradient.coeffRef(cp) -= class_weight;
				} // end if			      
			      sortedLUnew_iter++;
			    }
			  else
			    {
			      sortedLU_iter +=2; //the iterator needs to be incremeted even if the class is filtered
			      sortedLUnew_iter +=2; //the iterator needs to be incremeted even if the class is filtered
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
		  
		  /* for (sc = 0; sc < noClasses; sc++)		     */
		  /*   { */
		  /*     // traverse the classes in order of their desired ranks */
		  /*     cp = sorted_class[sc];  */
		  /*     // if (!(filtered[i][cp]))  */
		  /*     if (!(filtered.get(i,cp)))  */
		  /* 	{ */
		  /* 	  if (!y.coeff(i,cp)) */
		  /* 	    { */
		  /* 	      // if example i does not have class cp  */
		  /* 	      //if (left_classes && hinge_loss(l.coeff(cp) - tmp) > 0) // I3 Condition */
		  /* 	      if (left_classes && ((1 - l.coeff(cp) + tmp) > 0)) // I3 Condition */
		  /* 		{ */
		  /* 		  #ifdef PRINTI */
		  /* 		    { */
		  /* 		      cout << "I3 : " << idx << ", " << i << endl; */
		  /* 		    } */
                  /*                  #endif */
		  /* 		  multiplier += other_weight*left_classes; */
		  /* 		  l_gradient.coeffRef(cp) -= other_weight*left_classes; */
		  /* 		} */
			      
		  /* 	      //if (right_classes && hinge_loss(tmp - u.coeff(cp)) > 0) //  I4 Condition */
		  /* 	      if (right_classes && ((1 - tmp + u.coeff(cp)) > 0)) //  I4 Condition */
		  /* 		{ */
		  /* 		  #ifdef PRINTI */
		  /* 		    { */
		  /* 		      cout << "I4 : " << idx << ", " << i<< endl; */
		  /* 		    } */
		  /* 		  #endif */
		  /* 		  multiplier -= other_weight*right_classes; */
		  /* 		  u_gradient.coeffRef(cp) += other_weight*right_classes; */
		  /* 		} */
		  /* 	    } // end if example does not have class cp */
		  /* 	  else */
		  /* 	    { */
		  /* 	      //this example has class cp */
		  /* 	      //if (hinge_loss(tmp - l.coeff(cp)) > 0)// I1 Condition */
		  /* 	      if ((1 - tmp + l.coeff(cp)) > 0)// I1 Condition */
		  /* 		{ */
		  /* 		  #ifdef PRINTI */
		  /* 		    { */
		  /* 		      cout << "I1 : " << idx << ", " << i<< endl; */
		  /* 		    } */
		  /* 		  #endif */
		  /* 		  multiplier -= class_weight; */
		  /* 		  l_gradient.coeffRef(cp) += class_weight; */
		  /* 		} // end if */
			      
		  /* 	      //if (hinge_loss(-tmp + u.coeff(cp)) > 0)//  I2 Condition */
		  /* 	      if ((1 + tmp - u.coeff(cp)) > 0)//  I2 Condition */
		  /* 		{ */
		  /* 		  #ifdef PRINTI */
		  /* 		    { */
		  /* 		      cout << "I2 : " << idx << ", " << i << endl; */
		  /* 		    } */
		  /* 		  #endif */
		  /* 		  multiplier += class_weight; */
		  /* 		  u_gradient.coeffRef(cp) -= class_weight; */
		  /* 		} // end if */
		  /* 	      //update the left and right classes; */

		  /* 	      // this was wrongggggggggg should be outside the filter  */
		  /* 	      left_classes++; */
		  /* 	      right_classes--;			  */
		  /* 	    } // end if example has class cp */
		  /* 	} */
		  /*   } // end for sc */

		  if (multiplier != 0)
		    {
		      w.gradient_update(x,i,multiplier);
		    }
		} // end for idx (second)
	      
	      // update the lower and upper bounds
	      sortedLU = sortedLUnew; 
	      //multiplier = eta_t * 1.0 / batch_size;
	      //l -= ( l_gradient * multiplier );
	      //u -= ( u_gradient * multiplier );
	      
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
	  // get the current l and u in the original class order
	  for (sorted_class_iter = sorted_class.begin(),sortedLU_iter=sortedLU.begin();sorted_class_iter != sorted_class.end();sorted_class_iter++)
	    {
	      cp=*sorted_class_iter;
	      l.coeffRef(cp) = *(sortedLU_iter++);
	      u.coeffRef(cp) = *(sortedLU_iter++);
	    }
	  if (params.rank_by_mean)
	    {
	      proj_means(means, nc, w, x, y);
	      rank_classes(sorted_class, class_order, means, sortedLU, l, u);
	    }
	  else
	    {
	      rank_classes(sorted_class, class_order, (l+u)*.5, sortedLU, l, u);// ranking classes			       
	    }
	  sortedLUnew = sortedLU;

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
	  update_filtered(filtered, w, l, u, x, y, params.remove_class_constraints);
	  no_filtered = filtered.count();
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
  
  if (params.report_epoch > 0)
    {
      // get the current l and u in the original class order
      for (sorted_class_iter = sorted_class.begin(),sortedLU_iter=sortedLU.begin(); sorted_class_iter != sorted_class.end(); sorted_class_iter++)
	{
	  cp=*sorted_class_iter;
	  l.coeffRef(cp) = *(sortedLU_iter++);
	  u.coeffRef(cp) = *(sortedLU_iter++);
	}      
      objective_val[obj_idx++] = calculate_objective_hinge(w, x, y, nclasses, l, u,
							   sorted_class, class_order,
							   filtered,
							   params.ml_wt_by_nclasses, params.ml_wt_class_by_nclasses,
							   lambda, C1, C2);// save the objective value
    }
  objective_val.conservativeResize(obj_idx);
  
  #ifdef PROFILE
  ProfilerStop();
  #endif
}

#endif
