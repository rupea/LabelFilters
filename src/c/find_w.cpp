#include <iostream>
#include <vector>
#include <stdio.h>
#include <typeinfo>
#include <math.h>
#include <stdlib.h>
#include "Eigen/Dense"
#include "Eigen/Sparse"

#ifdef PROFILE
#include <gperftools/profiler.h>
#endif

using Eigen::MatrixXd;
using Eigen::VectorXd;
using Eigen::VectorXi;

using namespace std;

// type definitions
typedef Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> DenseM;
typedef Eigen::SparseMatrix<double, Eigen::RowMajor> SparseM;

// ***********  Constant values used
const unsigned int OPT_MAX_ITER = 1e4; 	// Maximum number of iterations
const int OPT_MAX_REORDERING = 10; // maximum time the ordering of switches have to be changed
const double OPT_EPSILON = 1e-4; // optimization epsilon: how different the update for w is
const int PRINT_T = 0;                 	// print values in each iteration
const int PRINT_O = 1;                // print objective function in each epoch
const int PRINT_M = 0;                 	// print matrix operations
const int PRINT_I = 0;                 	// print Conditions in each iteration
const int STOCHASTIC_BATCH_SIZE = 100; // perform stochastic gradient with this batchsize; if -1 then complete GD
const int OPT_EPOCH = 1000;
// ********************************

// *******************************
// Prints the progress bar
void print_progress(string s, int t, int max_t)
{
  double p = ((double) ((double) t * 100.0)) / (double) max_t;
  int percent = (int) p;

  if (t % 10 == 0 || t == 0)
    {
      string str = "\r" + s + "=";
      for (int i = 0; i < (int) percent / 10; i++)
	{
	  str = str + "==";
	}
	    
      int c = 60;
      char buff[c];
      sprintf(buff,
	      " > (%d\%) @%d                                                 ",
	      percent, t);
      str = str + buff;
	    	    
      cout << str;
    }
	
  if (percent == 100)
    {
      cout << std::endl;
    }
}

void print_mat_size(const Eigen::VectorXd& mat)
{
  cout << "(" << mat.size() << ")";
}

void print_mat_size(const DenseM& mat)
{
  cout << "(" << mat.rows() << ", " << mat.cols() << ")";
}

void print_mat_size(const SparseM& mat)
{
  cout << "(" << mat.rows() << ", " << mat.cols() << ")";
}

void print_report(const SparseM& x)
{
  int nnz = x.nonZeros();
  cout << "x:non-zeros: " << nnz << ", avg. nnz/row: " << nnz / x.rows();
}

void print_report(const DenseM& x)
{
}

template<typename EigenType>
void print_report(const int& projection_dim, const int& batch_size,
		  const int& noClasses, const int& C1, const int& C2, const int& w_size,
		  const EigenType& x)
{
  cout << "projection_dim: " << projection_dim << ", batch_size: "
       << batch_size << ", noClasses: " << noClasses << ", C1: " << C1
       << ", C2: " << C2 << ", size w: " << w_size << ", ";
  print_report(x);
  cout << "\n-----------------------------\n";

}

// *******************************
// The hinge loss
inline double hinge_loss(double val)
{
  return 1 - ((val<1)?val:1.0);
}


// ******************************
// Convert to a STD vetor from Eigen Vector
void toVector(std::vector<int>& to, const VectorXd& from)
{
  for (int i = 0; i < from.size(); i++)
    {
      to.push_back((int) from(i));
    }
}


// *************************
// Normalize data : centers and makes sure the variance is one
void normalize_col(SparseM& mat)
{
  if (mat.rows() < 2)
    return;

  double v = 0, m = 0, delta;

  cout << "normalizing sparse matrix with " << mat.rows() << " rows and "
       << mat.cols() << "cols ... \n";

  // Lets first compute the mean and the square of the values and center each column
  for (int k = 0; k < mat.outerSize(); ++k)
    {
      m = 0;
      v = 0;

      print_progress("normalizing ...", k, mat.outerSize());

      // online variance and mean calculation
      for (SparseM::InnerIterator it(mat, k); it; ++it)
	{
	  delta = it.value() - m;
	  m = m + delta / (it.row() + 1);
	  v = v + delta * (it.value() - m);
	}

      v = sqrt(v / (mat.rows() - 1));

      if (v < 0 || isnan(v))
	continue;

      for (SparseM::InnerIterator it(mat, k); it; ++it)
	{
	  it.valueRef() -= m;
	  if (isnan(it.value()))
	    it.valueRef() = 0;

	  if (!isnan(it.value() / v))
	    it.valueRef() = it.value() / v;

	}
    }

  cout << "done ... data is normalized ... \n";
}

void normalize(SparseM &mat)
{
  if (mat.rows() < 2)
    return;

  if (PRINT_M)
    cout << "normalizing sparse matrix with " << mat.rows() << " rows and "
	 << mat.cols() << "cols ... \n";

  double norm;
	
  for (int i = 0; i < mat.rows(); i++)
    {
      norm = mat.row(i).norm();
      if (norm != 0)
	{
	  mat.row(i) /= norm;
	}
    }
	
  cout << "done ... data is normalized ... \n";
  fflush (stdout);
}

void normalize(DenseM& mat)
{
  if (mat.rows() < 2)
    return;

  if (PRINT_M)
    cout << "normalizing sparse matrix with " << mat.rows() << " rows and "
	 << mat.cols() << "cols ... \n";

  mat = mat.rowwise() - mat.colwise().mean();		// first center the data
  DenseM v = mat.colwise().squaredNorm() / (mat.rows() - 1); // compute the sqruare of standard deviation
  for (int i = 0; i < mat.cols(); i++)
    {
      print_progress("normalizing ...", i, mat.cols());

      mat.col(i) = mat.col(i) / sqrt(v(i)); // devide each column by the standard deviation
    }

  cout << "done ... data is normalized ... \n";
}

// *******************************
// Calculates the objective function
template<typename EigenType>
double calculate_objective_hinge(const VectorXd& w,
				 const EigenType& x, const VectorXd& y,
				 const VectorXd& l, const VectorXd& u, const std::vector<int>& class_order,
				 double C1, double C2)
{

  double obj_val = w.norm();
  obj_val = .5 * obj_val * obj_val;
  int c;
  double sj;
  VectorXd projection = x * w;
  //VectorXd projection;
  //prod(projection, x, w);
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
double calculate_objective_hinge(const VectorXd& w,
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
std::vector<int> get_classes(VectorXd y)
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
// functions and structures for sorting and keeping indeces
struct IndexComparator
{
  VectorXd v;
  IndexComparator(VectorXd& m)
  {
    v = m;
  }
  bool operator()(int i, int j)
  {
    return (v(i) < v(j));
  }
};

void sort_index(VectorXd& m, std::vector<int>& cranks)
{
  for (int i = 0; i < m.size(); i++)
    {
      cranks[i] = i;
    }
  IndexComparator cmp(m);
  std::sort(cranks.begin(), cranks.end(), cmp);

}

// *********************************
// Ranks the classes to build the switches
void rank_classes(std::vector<int>& cranks, VectorXd& l, VectorXd& u)
{
  VectorXd m = ((u + l) * .5);
  std::vector<int> indices(m.size());
  sort_index(m, indices);
  for (int i = 0; i < m.size(); i++)
    {
      cranks[indices[i]] = i;
    }
}


// *******************************
// Get the number of exampls in each class

void init_nc(VectorXi& nc, const VectorXd& y, const int noClasses)
{  
  int c;
  if (nc.rows() != noClasses) 
    {
      cerr << "init_nc has been called with vector nc of wrong size" << endl;
      exit(-1);
    }
  int n = y.rows();
  for (int k = 0; k < noClasses; k++)
    {
      nc(k)=0;
    }
  for (int i=0;i<n;i++)
    {
      c = (int) y[i] - 1; // Assuming all the class labels start from 1
      nc(c)++;
    }
}

// ********************************
// Initializes the lower and upper bound
template<typename EigenType>
void init_lu(VectorXd& l, VectorXd& u, VectorXd& means, const VectorXi& nc, const VectorXd& w,
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
  VectorXd projection = x * w;
  //VectorXd projection;
  //prod(projection, x, w);
  for (i=0;i<n;i++)
    {
      c = (int) y[i] - 1; // Assuming all the class labels start from 1
      means(c)+=projection(i);
      l(c)=projection(i)<l(c)?projection(i):l(c);
      u(c)=projection(i)>u(c)?projection(i):u(c);
    }
  for (k = 0; k < noClasses; k++)
    {
      means(k) /= nc(k);
    }
}

// ******************************
// Projection to a new vector that is orthogonal to the rest
// It is basically Gram-Schmidt Orthogonalization
void project_orthogonal(VectorXd& w, const DenseM& weights,
			const int& projection_dim)
{
  if (projection_dim == 0)
    return;
  
  // Assuming the first to the current projection_dim are the ones we want to be orthogonal to
  VectorXd proj_sum(w.rows());
  DenseM wt = w.transpose();
  double norm;
  
  proj_sum.setZero();
  
  for (int i = 0; i < projection_dim; i++)
    {
      norm = weights.col(i).norm();
      proj_sum = proj_sum
	+ weights.col(i) * ((wt * weights.col(i)) / (norm * norm));
    }
  
  w = (w - proj_sum);
}

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
  const double n = x.rows();  // number of samples in double
  const int batch_size = (STOCHASTIC_BATCH_SIZE < 1 || STOCHASTIC_BATCH_SIZE > n) ? (int) n : STOCHASTIC_BATCH_SIZE;
  int d = x.cols();
  std::vector<int> classes = get_classes(y);
  cout << "size x: " << x.rows() << " rows and " << x.cols() << " columns.\n";
  cout << "size y: " << y.rows() << " rows and " << y.cols() << " columns.\n";

  int noClasses = classes.size();
  VectorXd w(d),l(noClasses),u(noClasses),w_prev(d);
  VectorXd means(noClasses); // used for initialization of the class order vector;
  VectorXi nc(noClasses); // the number of examples in each class 
  double eta_t, tmp, sj;
  double prev_norm = 0, cur_norm = 1;// current and previous changes in the norm of w ...
  //used as stopping point if it doesn't change
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
  
  // can't do it this way because the initial projections won't be orthogonal
  // this will create some problems when we initialize the calss order vector
  // we'll need to change this. 
  for(int projection_dim=0; projection_dim < no_projections; projection_dim++)
    {
	 
      prev_norm = 0, cur_norm = 1;
      w = weights.col(projection_dim);

      // w.setRandom(); // initialize to a random value
      w_prev = w*2;
      
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
	  cur_norm = OPT_EPSILON+1;// if s is changed make sure we go through another iteration
		    
	  while (t < OPT_MAX_ITER)// && (cur_norm > OPT_EPSILON))
	    {
	      t++;
	      w_prev = w;
			    
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

		  proj.coeffRef(idx) = (x.row(i)*w)(0,0);
		  index.coeffRef(idx)=i;
		}

	      // now we can update w directly
	      // update for the reglarizer
	      w-=lambda*w*eta_t;
		
	      for (idx = 0; idx < batch_size; idx++)// batch_size will be equal to n for complete GD
		{		
		  c = (int) y[index.coeff(idx)] - 1; // Assuming all the class labels start from 1
				
		  multiplier = 0.0;
		  
		  if(PRINT_I)
		    {	
		      cout << index.coeff(i) << ", c: " << c << endl;
		    }
				
		  tmp = proj(idx);
				
		  if (hinge_loss(tmp - l.coeff(c)) > 0)// I1 Condition
		    {
		      if(PRINT_I)
			{
			  cout << "I1 : " << idx << ", " << i<< endl;
			}
		      multiplier -= C1;
		      l_gradient.coeffRef(c) += C1;
		    } // end if
				
		  if (hinge_loss(-tmp + u.coeff(c)) > 0)//  I2 Condition
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
		      if (cp != c)
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
		      multiplier = (multiplier*eta_t)/batch_size;
		      w -= x.row(index.coeff(idx)).transpose()*multiplier;
		    }
		} // end for idx (second)
			    
	      // update the lower and upper bounds
	      multiplier = eta_t * 1.0 / batch_size;
	      l -= ( l_gradient * multiplier );
	      u -= ( u_gradient * multiplier );
			    
	      if(true)
		{
		  // perform orthogonal projection
		  project_orthogonal(w,weights,projection_dim);
		}
			    
	      cur_norm = (w - w_prev).norm();
			    
	      if(PRINT_T==1)
		{
		  double obj = obj_idx >= 1 ? objective_val[obj_idx-1] : 0;
		  cout << "t: " << t << ", obj:" << objective_val[obj_idx-1]
		       <<", w: " << w.transpose()
		       << ", l:" << l.transpose() << ", u:" << u.transpose()
		       << ", cur_norm: " << cur_norm << endl;
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
		   << ", w: " << w.transpose()
		   << ", l:" << l.transpose() << ", u:" << u.transpose()
		   << ", cur_norm: " << cur_norm << endl;
	    } // end if print
			
	  cout << "\r>> " << iter+1 << ": Done in " << t
	       << " iterations ... with (w-w_prev).norm(): " << cur_norm;
			
	  if(w.size() < 5)
	    cout << ", w:" << w.transpose();
	} // end for iter
		
      weights.col(projection_dim) = w;
      lower_bounds.col(projection_dim) = l;
      upper_bounds.col(projection_dim) = u;
    } // end for projection_dim
	
  cout << "\n---------------\n" << endl;
	
  objective_val[obj_idx++] = calculate_objective_hinge(w, x, y, l, u,
						       class_order, C1, C2);// save the objective value
  objective_val = objective_val.head(obj_idx);

  cout << "w norm at the end: "<< weights.col(0).norm() << endl;

  #ifdef PROFILE
  ProfilerStop();
  #endif
}

// ---------------------------------------
int main()
{

  srand (42782);
  
  DenseM weights(400,1),lower_bounds(1000,1),upper_bounds(1000,1), x(10000,400);
  VectorXd y(10000),objective_val;

  weights.setRandom();
  lower_bounds.setZero();
  upper_bounds.setZero();
  x.setRandom();
  SparseM xs = x.sparseView();
  for (int i = 0; i < y.size(); i++)
    {
      y(i) = (i%1000)+1;
    }

  solve_optimization(weights,lower_bounds,upper_bounds,objective_val,x,y,10,1,0);
  //solve_optimization(weights,lower_bounds,upper_bounds,objective_val,xs,y,10,1,0);

}
