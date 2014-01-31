#include <iostream>
#include <vector>
#include <stdio.h>
#include <typeinfo>
#include <math.h>
#include <stdlib.h>
#include "Eigen/Dense"
#include "Eigen/Sparse"

using Eigen::MatrixXd;
using Eigen::VectorXd;
using Eigen::VectorXi;

using namespace std;

// type definitions

typedef Eigen::SparseMatrix<double, Eigen::RowMajor> SparseM;

// ***********  Constant values used
const unsigned int OPT_MAX_ITER = 1e5; 	// Maximum number of iterations
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

void print_mat_size(const Eigen::MatrixXd& mat)
{
  cout << "(" << mat.rows() << ", " << mat.cols() << ")";
}

void print_mat_size(const SparseM& mat)
{
  cout << "(" << mat.rows() << ", " << mat.cols() << ")";
}

void print_report(const Eigen::EigenBase<SparseM>& x)
{
  int nnz = static_cast<SparseM>(x.derived()).nonZeros();
  cout << "x:non-zeros: " << nnz << ", avg. nnz/row: " << nnz / x.rows();
}

void print_report(const Eigen::EigenBase<MatrixXd>& x)
{
}

template<typename EigenType>
void print_report(const int& projection_dim, const int& batch_size,
		  const int& noClasses, const int& C1, const int& C2, const int& w_size,
		  const Eigen::EigenBase<EigenType>& x)
{
  cout << "projection_dim: " << projection_dim << ", batch_size: "
       << batch_size << ", noClasses: " << noClasses << ", C1: " << C1
       << ", C2: " << C2 << ", size w: " << w_size << ", ";
  print_report(x);
  cout << "\n-----------------------------\n";

}

// *********************************
// init values useful for faster gradient update
// currently just a dummy function.
void init_gradient_internal(const Eigen::EigenBase<SparseM>& x,
			    Eigen::EigenBase<SparseM>& w_gradient)
{
}

void init_gradient_internal(const Eigen::EigenBase<MatrixXd>& x,
			    Eigen::EigenBase<MatrixXd>& w_gradient)
{
}

template<typename EigenType>
void init_gradient(const Eigen::EigenBase<EigenType>& x,
		   Eigen::EigenBase<EigenType>& w_gradient)
{
  init_gradient_internal(x, w_gradient);
}

// *******************************
// The hinge loss
double hinge_loss(double val)
{
  val = 1 - val;
  if (val > 0)
    return val;

  return 0;
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

std::vector<int> toVector(const VectorXd from)
{
  std::vector<int> to(from.size());

  toVector(to, from);
}

// ***********************
// Some basic operations

double rowprod(const MatrixXd& x, const int i, const VectorXd& w)
{
  return x.row(i) * w;
}

double rowprod(const Eigen::EigenBase<MatrixXd>& x, const int i,
	       const VectorXd& w)
{
  return x.derived().row(i) * w;
}

double rowprod(const SparseM& x, const int& i, const VectorXd& w)
{
  return (x.row(i)*w)(0,0);

}

double rowprod(const Eigen::EigenBase<SparseM>& x, const int& i,
	       const VectorXd& w)
{
  return rowprod(x.derived(),i,w);
}

//  assumes that w is a row vector
void assign_add(Eigen::EigenBase<MatrixXd>& w,
		const Eigen::EigenBase<MatrixXd>& x, const int& i, const double& C)
{
  (w.derived()) += (x.derived().row(i) * C);
}

// assumes that w is a row vector
void assign_add(SparseM &w, const SparseM &x, const int i, const double C)
{
  w += (x.row(i)*C);
}

// assumes that w is a row vector
void assign_add(Eigen::EigenBase<SparseM>& w,
		const Eigen::EigenBase<SparseM>& x, const int i, const double C)
{
  assign_add(w.derived(),x.derived(),i,C);
}

void prod(VectorXd& projection, const MatrixXd& x, const VectorXd& w)
{
  projection = x * w;
}

void prod(VectorXd& projection, const Eigen::EigenBase<MatrixXd>& x, const VectorXd& w)
{
  projection = x.derived() * w;
}

void prod(VectorXd& projection, const SparseM& x, const VectorXd& w)
{
  projection = x * w;
}

void prod(VectorXd& projection, const Eigen::EigenBase<SparseM> & x, const VectorXd& w)
{
  projection = (x.derived()) * w;
}

void prod(Eigen::EigenBase<MatrixXd> & w, const double scalar)
{
  w.derived() *= scalar;
}

void prod (SparseM &w, const double scalar)
{
  w*=scalar;
}

void prod(Eigen::EigenBase<SparseM> & w, const double scalar)
{
  prod(w.derived(),scalar);
}

void setZero(Eigen::EigenBase<MatrixXd>& w)
{
  ((MatrixXd*) &w.derived())->setZero();
}

void setZero(Eigen::EigenBase<SparseM>& w)
{
  ((SparseM*) &w.derived())->setZero();
}

// *************************
// update the gradient; gradient is already divided by n if necessary
void update_gradient(VectorXd& w, const Eigen::EigenBase<MatrixXd>& w_gradient,
		     const double& eta)
{
  w -= (w_gradient.derived().transpose() * eta);
}

// gradient is already divided by n if necessary
void update_gradient(VectorXd& w, const Eigen::EigenBase<SparseM>& w_gradient,
		     const double& eta)
{
  SparseM* mat = (SparseM*) &w_gradient.derived();
  
  for (int k = 0; k < mat->outerSize(); ++k)
    {
      SparseM::InnerIterator it(*mat, k);
      for (; it; ++it)
	{
	  w(it.col()) -= (it.value() * eta);
	}
    }
}

// *************************
// Normalize data : centers and makes sure the variance is one
void normalize_col(SparseM & mat)
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

void normalize(MatrixXd & mat)
{
  if (mat.rows() < 2)
    return;

  if (PRINT_M)
    cout << "normalizing sparse matrix with " << mat.rows() << " rows and "
	 << mat.cols() << "cols ... \n";

  mat = mat.rowwise() - mat.colwise().mean();		// first center the data
  MatrixXd v = mat.colwise().squaredNorm() / (mat.rows() - 1); // compute the sqruare of standard deviation
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
				 const Eigen::EigenBase<EigenType>& x, const VectorXd& y,
				 const VectorXd& l, const VectorXd& u, const std::vector<int>& class_order,
				 double C1, double C2)
{

  double obj_val = w.norm();
  obj_val = .5 * obj_val * obj_val;
  int c;
  double sj;
  VectorXd projection;
  prod(projection, x, w);
  for (int i = 0; i < x.rows(); i++)
    {
      c = (int) y[i] - 1; // again the label is started from 1
      obj_val += (C1
		  * (hinge_loss(projection(i) - l[c]) + hinge_loss(-projection(i) + u[c])));
	  
      for (int cp = 0; cp < l.size(); cp++)
	{
	  if (c != cp)
	    {
	      sj = (class_order[c] < class_order[cp]) ? 1 : 0;
	      obj_val += (C2
			  * (sj * hinge_loss(-projection(i) + l[cp])
			     + (1 - sj) * hinge_loss(projection(i) - u[cp])));
	    }
	}
	  
    }	
  return obj_val;
}

template<typename EigenType>
double calculate_objective_hinge(const VectorXd& w,
				 const Eigen::EigenBase<EigenType>& x, const VectorXd& y,
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
  //	VectorXd m = l; //  + ((u + l) * .5);
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
	     const Eigen::EigenBase<EigenType>& x, const VectorXd& y,
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
  prod(projection, x, w);
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
void project_orthogonal(VectorXd& w, const MatrixXd& weights,
			const int& projection_dim)
{
  if (projection_dim == 0)
    return;
  
  // Assuming the first to the current projection_dim are the ones we want to be orthogonal to
  VectorXd proj_sum(w.rows());
  MatrixXd wt = w.transpose();
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
void solve_optimization(MatrixXd& weights, MatrixXd& lower_bounds,
			MatrixXd& upper_bounds, VectorXd& objective_val,
			const Eigen::EigenBase<EigenType>& x, const VectorXd& y,
			const double C1_, const double C2_,
			Eigen::EigenBase<EigenType>& w_gradient, bool resumed)

{

  /* initialize random seed: */
  double lambda = 1.0/C2_;
  double C1 = C1_/C2_;
  double C2 = 1.0;
  srand (time(NULL));

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
  unsigned int t = 1, i=0, k=0;
  char iter_str[30];
  for(i=0; i<30; i++) iter_str[i]=' ';
  std::vector<int> class_order(noClasses), prev_class_order(noClasses);// mid points in the bound; used as the switch
       
  lower_bounds.resize(noClasses, no_projections);
  upper_bounds.resize(noClasses, no_projections);
  objective_val.resize(2 + (no_projections * OPT_MAX_ITER * OPT_MAX_REORDERING / OPT_EPOCH));

  init_gradient(x,w_gradient);

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
	  sprintf(iter_str,"Iteration %d > ",iter+1);

	  // init the optimization specific parameters
	  std::copy(class_order.begin(),class_order.end(), prev_class_order.begin());
		    
	  cout << "\n Class_Order: ";
		    
	  for (k = 0; k < noClasses; k++)
	    {
	      cout << class_order[k] << "  ";
	    }
	  cout << "\n";fflush(stdout);

	  cout << "w norm: " << w.norm()<< endl;
		    
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
	      setZero(w_gradient);
	      l_gradient.setZero();
	      u_gradient.setZero();
			    
	      for (unsigned int idx = 0; idx < batch_size; idx++)// batch_size will be equal to n for complete GD
		{
		  if(STOCHASTIC_BATCH_SIZE > 0)
		    {
		      i = ((int) rand()) % ((int)n);
		    }
		  else
		    {
		      i=idx;
		    }
				
		  c = (int) y[i] - 1; // Assuming all the class labels start from 1
				
		  double multiplier = 0;
				
		  if(PRINT_I)
		    {	
		      cout << i << ", c: " << c << endl;
		    }
				
		  tmp = rowprod(x,i, w);
				
		  if (hinge_loss(tmp - l(c)) > 0)// I1 Condition
		    {
		      if(PRINT_I)
			{
			  cout << "I1 : " << idx << ", " << i<< endl;
			}
		      multiplier -= C1;
		      l_gradient(c) += C1;
		    } // end if
				
		  if (hinge_loss(-tmp + u(c)) > 0)//  I2 Condition
		    {
		      if(PRINT_I)
			{
			  cout << "I2 : " << idx << ", " << i << endl;
			}
		      multiplier +=C1;
		      u_gradient(c) -= C1;
		    } // end if
				
		  for (cp = 0; cp < noClasses; cp++)
		    {
		      if (cp != c)
			{
			  sj = (class_order[c] < class_order[cp]) ? 1 : 0;
			  if (sj == 1 && hinge_loss(-tmp + l(cp)) > 0) // I3 Condition
			    {
			      if(PRINT_I)
				{
				  cout << "I3 : " << idx << ", " << i << endl;
				}
			      multiplier +=C2;
			      l_gradient(cp) -= C2;
			    }
					
			  if (sj == 0 && hinge_loss(tmp - u(cp)) > 0) //  I4 Condition
			    {
			      if(PRINT_I)
				{
				  cout << "I4 : " << idx << ", " << i<< endl;
				}
			      multiplier -= C2;
			      u_gradient(cp) += C2;
			    }
			} // end if cp != c
		    } // end for cp
		  if (multiplier != 0)
		    {
		      assign_add(w_gradient,x,i,multiplier);
		    }
		} // end for idx (first)
			    
	      // after computing the gradients update the variables
			    
	      // update the gradient
	      prod(w_gradient, 1.0 / batch_size);// * n /n
	      // update for the reglarizer
	      w-=lambda*w*eta_t;
	      // update for the penalty
	      update_gradient(w, w_gradient, eta_t);
	      l = l - ( l_gradient * eta_t * 1.0 / batch_size );
	      u = u - ( u_gradient * eta_t * 1.0 / batch_size );
			    
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
}

// ---------------------------------------
int main()
{
  MatrixXd x(2, 3);
  float C1 = 1, C2 = 1;
  VectorXd y(10), s(10);
  for (int i = 0; i < y.size(); i++)
    {
      y(i) = 10 - i;
      if (i == 5)
	{
	  y(i) = 100;
	}
    }

  std::vector<int> cranks(y.size());
  IndexComparator cmp(y);
  sort_index(y, cranks);
  // for(int i=0;i<y.size();i++){cranks[i]=i;}
  //std::sort(cranks.begin(), cranks.end(), cmp);
  for (std::vector<int>::iterator it = cranks.begin(); it != cranks.end();
       ++it)
    std::cout << ' ' << *it << "[" << y(*it) << "]";
  std::cout << '\n';

  MatrixXd x1(4, 3);
  x1 << 0.54343, 0.14613, 0.55317, 0.62082, 0.41308, 0.59649, 0.64365, 0.58947, 0.94562, 0.99690, 0.91504, 0.92640;
  SparseM x2 = x1.sparseView();
  cout << "before init ... \n" << x1 << endl;
  // MatrixXd xx(x1);
  normalize(x1);	// should print out :
  cout << "\n---------------\n" << endl;
  normalize(x2);	// should print out :
  // -0.78203  -1.14666  -0.96574
  //  -0.39843  -0.31890  -0.75890
  // -0.28525   0.22802   0.90822
  //  1.46571   1.23754   0.81642

  cout << x2 << endl;

  // std::cout << getClasses(y) << std::endl;
  // std::vector<int> classes = getClasses(y);
  // for (std::vector<int>::iterator it = classes.begin(); it != classes.end(); ++it)
  //  std::cout << ' ' << *it;
  // std::cout << '\n';

  //  VectorXd w = solve_optimization(x,y,s,C1,C2);

  // std::cout << w << "---\n" << OPT_MAX_ITER << std::endl;

  // read_sparse("../data/IPC/ipc.train.svm");

  return 0;
}
