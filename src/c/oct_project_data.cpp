#include "filter.h"
#include "boolmatrix.h"
#include "EigenOctave.h"
#include "Eigen/Dense"
#include "Eigen/Sparse"
#include <octave/oct.h>
#include <octave/ov-struct.h>
#include <octave/Cell.h>
#include <iostream>
#include <typeinfo>

using namespace std;
using boost::dynamic_bitset;

void print_usage()
{
  cout << "oct_project_data x p w l u" << endl;
  cout << "     x - the data matrix (can be dense or sparse)" << endl;
  cout << "     p - the ova predictions (sparse for now)" << endl;
  cout << "     w - the projection vectors" << endl;
  cout << "     l - the lower bounds" << endl;
  cout << "     u - the upper bounds" << endl;
}

void update_active(boolmatrix& active, bool all_active, const VectorXd& projection, const VectorXd& l, const VectorXd& u)
{
  int noClasses = l.size();			
  if (all_active)
    {
      for (size_t i = 0; i < projection.size(); i++)
	{      
	  double proj = projection.coeff(i);
	  for (size_t cp=0; cp<noClasses; cp++)
	    {
	      bool val = (proj>l.coeff(cp))&&(proj<u.coeff(cp))?true:false;
	      if (val) 
		{
		  active.set(i,cp); 
		}	  
	    }
	}
    }
  else
    {
      size_t i=0,cp=0;
      active.findFirst(i,cp);
      while( i < projection.size())
	{
	  double proj = projection.coeff(i);
	  bool val = (proj>l.coeff(cp))&&(proj<u.coeff(cp))?true:false;
	  if (!val) 
	    {
	      active.set(i,cp,false); 
	    }	  
	  active.findNext(i,cp);
	}
    }
}

SparseBoolMatrix toSparseBoolMatrix(const boolmatrix& boolmat)
{  
  octave_idx_type r = boolmat.rows();
  octave_idx_type c = boolmat.cols();
  octave_idx_type nnz = boolmat.count();
  SparseBoolMatrix ret = SparseBoolMatrix(r,c,nnz);
  cout << "Sparsebool " << ret.nnz() << "   " << ret.nzmax() << endl;
  size_t i, j;
  boolmat.findFirst(i,j);
  while (i < boolmat.rows())
    {
      cout << i << "   " << j << endl;
      ret(i,j) = true;
      boolmat.findNext(i,j);
    }
  return ret;
}

void filterpreds(SparseM& out, const boolmatrix& active)
{
  size_t i;
  for (i=0; i<out.rows(); i++)
    {
      for (SparseM::InnerIterator it(out,i); it; ++it)
	{
	  //	  cout << i << "   " << it.index() << "   " << active.get(i,it.index()) << endl;
	  if (!active.get(i,it.index()))
	    {
	      it.valueRef() = 0;
	    }
	}
    }
  out.prune(0.0);
}

void filterpreds(SparseM& out, const vector<dynamic_bitset<>*>& active)
{
  size_t i;
  for (i=0; i<out.rows(); i++)
    {
      for (SparseM::InnerIterator it(out,i); it; ++it)
	{
	  //	  cout << i << "   " << it.index() << "   " << active.get(i,it.index()) << endl;
	  if (!active[i]->test(it.index()))
	    {
	      it.valueRef() = 0;
	    }
	}
    }
  out.prune(0.0);
}
     



DEFUN_DLD (oct_project_data, args, nargout,
		"Interface to project data on a set of vectors")
{

#ifdef _OPENMP
  Eigen::initParallel();
  cout << "initialized Eigen parallel"<<endl;
#endif  

  int nargin = args.length();
  if (nargin == 0)
    {
      print_usage();
      return octave_value_list(0);
    }
  
  DenseColM wmat = toEigenMat<DenseColM>(args(2).float_array_value());
  DenseColM lmat = toEigenMat<DenseColM>(args(3).float_array_value());
  DenseColM umat = toEigenMat<DenseColM>(args(4).float_array_value());
  DenseM projections;

  if(args(0).is_sparse_type())
    {
      // Sparse data
      SparseM x = toEigenMat(args(0).sparse_matrix_value());
      projections = x * wmat;
    }
  else
    {
      // Dense data
      DenseM x = toEigenMat<DenseM>(args(0).float_array_value());
      projections = x * wmat;
    }

  SparseM preds = toEigenMat(args(1).sparse_matrix_value());
  

  Cell ret = Cell(dim_vector(projections.cols(),1));
  
  vector<dynamic_bitset<>*> active(projections.rows());
  size_t n = projections.rows();
  size_t noClasses = lmat.rows();
  for (vector<dynamic_bitset<>*>::iterator it = active.begin(); it != active.end(); ++it)
    {
      *it = new dynamic_bitset<>(noClasses);
      (*it) -> set();
    }

  for (int i = 0; i < projections.cols(); i++)
    {
      cout << i << endl;
      VectorXd proj = projections.col(i);
      VectorXd l = lmat.col(i);
      VectorXd u = umat.col(i);
      cout << "init filter" << endl;
      Filter f(l,u);      
      cout << i << "  update filter" << endl; 
      size_t j;
      vector<dynamic_bitset<>*>::iterator it;
      size_t count = 0;
      for (it = active.begin(), j=0; it != active.end(); ++it, j++)
	{
	  (**it) &= *(f.filter(proj.coeff(j)));
	  count += (*it) -> count();
	}
      cout << i << " active  " << count << " out of " << n*noClasses << " (" << count*1.0/(n*noClasses) << ")" << endl; 
      //      ret(i) = toSparseBoolMatrix(filtered);
      filterpreds(preds,active);
      ret(i) = toMatrix(preds);
    }

  // boolmatrix active = boolmatrix(projections.rows(),lmat.rows());
  // for (int i = 0; i < projections.cols(); i++)
  //   {
  //     cout << i << endl;
  //     VectorXd proj = projections.col(i);
  //     VectorXd l = lmat.col(i);
  //     VectorXd u = umat.col(i);
  //     cout << i << "  update filter" << endl; 
  //     update_active(active, i==0, projections.col(i), lmat.col(i), umat.col(i));
  //     cout << i << " active  " << active.count() << " out of " << active.rows()*active.cols() << " (" << active.count()*1.0/(active.rows()*active.cols()) << ")" << endl; 
  //     //      ret(i) = toSparseBoolMatrix(filtered);
  //     filterpreds(preds,active);
  //     ret(i) = toMatrix(preds);
  //   }

  octave_value_list retval(1);// return value
  retval(0) = ret;

  // clean up
  for (vector<dynamic_bitset<>*>::iterator it = active.begin(); it != active.end(); ++it)
    {
      delete *it;
    }


  return retval;
}

