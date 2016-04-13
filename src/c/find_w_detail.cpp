#include "typedefs.h"
#include "constants.h"
#include "WeightVector.h"
#include "printing.h"
#include "utils.h"
#include "find_w.h"
#include "find_w_detail.h"
#include "Eigen/Dense"
#include "Eigen/Sparse"
//#include "boost/iterator/counting_iterator.hpp"
#include <boost/numeric/conversion/bounds.hpp>  // boost::numeric::bounds<T>
#include <iostream>
#include <iomanip>
#include <vector>
//#include <stdio.h>
//#include <typeinfo>
#include <math.h>
//#include <stdlib.h>

using Eigen::VectorXd;
using Eigen::VectorXi;

using namespace std;


#define __restricted /* __restricted seems to be an error */


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
/** Ranks the classes to build the switches.
 * \c sortKey   values to be sorted
 * \c indices   of the sorted permutation of \c sortkey
 * \c cranks    reverse permutation (cranks[indices[i]] == i)
 */
void rank_classes(std::vector<int>& indices, std::vector<int>& cranks, const VectorXd& sortkey)
{
    if( indices.size() != static_cast<size_t>(sortkey.size()) ||
        cranks.size() != static_cast<size_t>(sortkey.size()) )
        throw std::runtime_error("ERROR: rank_classes(indices,cranks,sortKey): indices and cranks must match size of sortKey");
    //cout<<" sort_index "<<endl; cout.flush();
    sort_index(sortkey, indices); // <---   un-init
    //cout<<" cranks... "<<endl; cout.flush();
    for (int i = 0; i < sortkey.size(); ++i) {
        assert( indices[i] >= 0 && indices[i] < static_cast<int>(cranks.size()) );
        cranks[indices[i]] = i;
    }
    //cout<<" rank_classes DONE "<<endl; cout.flush();
}

// **********************************************
// get l and u in the original class order

void get_lu (VectorXd& l, VectorXd& u, const VectorXd& sortedLU, const vector<int>& sorted_class)
{
    assert( (size_t)l.size() == sorted_class.size() );
    assert( (size_t)u.size() == sorted_class.size() );
    vector<int>::const_iterator scIter;
    const double* sortedLU_iter;
    for (scIter = sorted_class.begin(),sortedLU_iter=sortedLU.data(); scIter != sorted_class.end(); ++scIter) {
        int const cp = *scIter;
        l.coeffRef(cp) = *(sortedLU_iter++);
        u.coeffRef(cp) = *(sortedLU_iter++);
    }
}
void get_unsorted_lu_sum (VectorXd & lusum, VectorXd const& l, VectorXd const& u, const VectorXd& sortedLU, const vector<int>& sorted_class)
{
    assert( (size_t)l.size() == sorted_class.size() );
    assert( (size_t)u.size() == sorted_class.size() );
    assert( (size_t)lusum.size() == sorted_class.size() );
    vector<int>::const_iterator scIter=sorted_class.begin();
    const double* sortedLU_iter = sortedLU.data();
    for ( ; scIter != sorted_class.end(); ++sortedLU_iter, ++scIter)
    {
        double const sum = *sortedLU_iter;
        lusum.coeffRef(*scIter) = sum + *++sortedLU_iter;
    }
}

// **********************************
// sort l and u in the new class order

void get_sortedLU(VectorXd& sortedLU, const VectorXd& l, const VectorXd& u, const vector<int>& sorted_class)
{
  for (size_t i = 0; i < sorted_class.size(); i++)
    {
      sortedLU.coeffRef(2*i) = l.coeff(sorted_class[i]);
      sortedLU.coeffRef(2*i+1) = u.coeff(sorted_class[i]);
    }
}

// ************************************
// Get the number of examples in each class

void init_nc(VectorXi& nc, VectorXi& nclasses, const SparseMb& y)
{
    int noClasses = y.cols();
    int n = y.rows();
    nc.setZero(noClasses);
    nclasses.setZero(n);
    for (int i=0;i<n;i++) {
        for (SparseMb::InnerIterator it(y,i);it;++it) {
            if (it.value()) {
                ++nc.coeffRef(it.col());
                ++nclasses.coeffRef(i);
            }
        }
    }
    ostringstream err;
    // Until supported, check that all data is needed
    // E.g. many places may divide by nc[i] or ...
    int nclassesZero = 0;
    for (int i=0;i<n;i++) if( nclasses[i]==0 ) ++nclassesZero;
    if(nclassesZero) err<<"\nERROR: it seems "<<nclassesZero<<" examples have been assigned to no class at all";

    int ncZero = 0;
    int i0=0;
    for (int i=0;i<noClasses;i++) if( nc[i]==0 ) {++ncZero; if(i0==0) i0=i;}
    if(ncZero) err<<"\nERROR: it seems "<<ncZero<<" classes have been assigned to NO training examples (nc["<<i0<<"]==0)";

    if(err.str().size()){
        err<<"\n\tPlease check whether code should support this, since"
            <<"\n\tboth nc[class] and nclasses[example] may be used as divisors"<<endl;
        throw runtime_error(err.str());
    }
}

// ************************************
// Get the sum of the weight of all examples in each class

void init_wc(VectorXd& wc, const VectorXi& nclasses, const SparseMb& y, const param_struct& params)
{
    if( params.optimizeLU_epoch > 0 ){
        double ml_wt_class = 1.0;
        int noClasses = y.cols();
        wc.setZero(noClasses);
        int n = y.rows();
        if (nclasses.size() != n) {
            cerr << "init_wc has been called with vector nclasses of wrong size" << endl;
            exit(-1);
        }
        for (int i=0;i<n;i++) {
            if (params.ml_wt_class_by_nclasses) {
                ml_wt_class = 1.0/nclasses.coeff(i);
            }
            for (SparseMb::InnerIterator it(y,i);it;++it) {
                if (it.value()) {
                    wc.coeffRef(it.col()) += ml_wt_class;
                }
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
  for (int i = 0; i < projection.size(); i++)
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


// internal function that initializes some counts and the class vectors
// used by compute_gradiends, compute_single_w_gradient_size and update_single_sortedLU

void init_ordered_class_list(int& left_classes, double& left_update,
			     int& right_classes, double& right_update,
			     vector<int>& classes,
			     double class_weight, double other_weight,
			     size_t i, const SparseMb& y, const VectorXi& nclasses,
			     const vector<int>& class_order,
			     int sc_start, int sc_end)
{

  left_classes = 0; //number of classes to the left of the current one
  left_update = 0; // left_classes * other_weight
  right_classes = nclasses.coeff(i); //number of classes to the right of the current one
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
  classes.push_back(sc_end+1); // this will always be the last. This avoids an extra check if s < sc_end inside the while loop.
  std::sort(classes.begin(),classes.end());
}



// ***********************************************
// calculate the multipliers (for the w gradient update)
// and the gradients for l and u updates
// on a subset of classes and instances

void compute_gradients (VectorXd& multipliers , VectorXd& sortedLU_gradient,
			const size_t idx_start, const size_t idx_end,
			const int sc_start, const int sc_end,
			const VectorXd& proj, const VectorXsz& index,
			const SparseMb& y, const VectorXi& nclasses,
			const int maxclasses,
			const vector<int>& sorted_class,
			const vector<int>& class_order,
			const VectorXd& sortedLU,
			const boolmatrix& filtered,
			const double C1, const double C2,
			const param_struct& params )
{
  int sc, cp;
  //int noClasses = y.cols();
  size_t idx, i;
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

      class_weight = C1;
      if (params.ml_wt_by_nclasses) other_weight /= nclasses.coeff(i);
      other_weight = C2;
      if (params.ml_wt_class_by_nclasses) class_weight /= nclasses.coeff(i);


      init_ordered_class_list(left_classes, left_update,
			      right_classes, right_update,
			      classes,
			      class_weight, other_weight,
			      i, y, nclasses, class_order, sc_start, sc_end);


      sc=sc_start;
      class_iter = classes.begin();
      sortedLU_iter = sortedLU.data() + 2*sc_start;
      sortedLU_gradient_iter = sortedLU_gradient.data();
      while (1)
	{
	  while (sc == *class_iter) // classes.back = sc_end+1 so this must end before sc == sc_end
	    {
	      // example has class cp
	      cp = sorted_class[sc];
	      if (!params.remove_class_constraints || !(filtered.get(i,cp)))
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
	  if (sc == sc_end)
	    {
	      break; // we are done
	    }
	  // example is not of class cp
	  cp = sorted_class[sc];
	  if (!(filtered.get(i,cp)))
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
	} // while(1)
      multipliers_iter++;
    }  // end for idx (second)
}




// function to calculate the multiplier of the gradient for w for a single example.

double compute_single_w_gradient_size (const int sc_start, const int sc_end,
				       const double proj, const size_t i,
				       const SparseMb& y, const VectorXi& nclasses,
				       const int maxclasses,
				       const vector<int>& sorted_class,
				       const vector<int>& class_order,
				       const VectorXd& sortedLU,
				       const boolmatrix& filtered,
				       const double C1, const double C2,
				       const param_struct& params )
{
  double multiplier = 0.0;
  int sc, cp;
  //int noClasses = y.cols();
  double class_weight, other_weight, left_update, right_update;
  vector<int> classes;
  vector<int>::iterator class_iter;
  int left_classes, right_classes;
  const double *sortedLU_iter;


  double pp1 = proj + 1;
  double pm1 = proj - 1;

  class_weight = C1;
  other_weight = C2;
  if (params.ml_wt_by_nclasses)
    {
      other_weight /= nclasses.coeff(i);
    }
  if (params.ml_wt_class_by_nclasses)
    {
      class_weight /= nclasses.coeff(i);
    }

  init_ordered_class_list(left_classes, left_update,
			  right_classes, right_update,
			  classes,
			  class_weight, other_weight,
			  i, y, nclasses, class_order, sc_start, sc_end);

  sc=sc_start;
  class_iter = classes.begin();
  sortedLU_iter = sortedLU.data() + 2*sc_start;
  while (1)
    {
      while (sc == *class_iter)
	{
	  // example has class cp
	  cp = sorted_class[sc];
	  if (!params.remove_class_constraints || !(filtered.get(i,cp)))
	    {
	      //if ((1 - proj + *(sortedLU_iter++)) > 0)// I1 Condition  w*x < l(c)+1
	      if (*(sortedLU_iter++) > pm1)// I1 Condition  w*x < l(c)+1
		{
#ifdef PRINTI
		  {
		    cout << "I1 : " << i<< endl;
		  }
#endif
		  multiplier -= class_weight;
		} // end if

	      //if ((1 + proj - *(sortedLU_iter++)) > 0)//  I2 Condition  w*x > u(c)-1
	      if (pp1 > *(sortedLU_iter++))//  I2 Condition  w*x > u(c)-1
		{
#ifdef PRINTI
		  {
		    cout << "I2 : " << i << endl;
		  }
#endif
		  multiplier += class_weight;
		} // end if
	    }
	  else
	    {
	      sortedLU_iter +=2; //the iterator needs to be incremeted even if the class is filtered
	    }
	  //update the left and right classes;
	  ++left_classes;
	  left_update += other_weight;
	  --right_classes;
	  right_update -= other_weight;
	  ++class_iter;
	  ++sc;
	}
      if (sc == sc_end)
	{
	  break; // we are done
	}
      // example is not of class cp
      cp = sorted_class[sc];
      if (!(filtered.get(i,cp)))
	{
	  //if (left_classes && ((1 - *sortedLU_iter + proj) > 0)) // I3 Condition w*x > l(cp) - 1
	  if (left_classes && (pp1 > *sortedLU_iter)) // I3 Condition w*x > l(cp) - 1
	    {
#ifdef PRINTI
	      {
		cout << "I3 : " << i << endl;
	      }
#endif
	      multiplier += left_update;
	    }
	  ++sortedLU_iter;
	  //if (right_classes && ((1 - proj + *sortedLU_iter) > 0)) //  I4 Condition  w*x < u(cp) + 1
	  if (right_classes && (*sortedLU_iter > pm1)) //  I4 Condition  w*x < u(cp) + 1
	    {
#ifdef PRINTI
	      {
		cout << "I4 : " << i<< endl;
	      }
#endif
	      multiplier -= right_update;
	    }
	  ++sortedLU_iter;
	}
      else
	{
	  sortedLU_iter += 2; //the iterator needs to be incremeted even if the class is filtered
	}
      ++sc;
    } // while(1)
  return multiplier;
}



// function to update L and U for a single example, given w.

void update_single_sortedLU( VectorXd& sortedLU,
			     int sc_start, int sc_end,
			     const double proj, const size_t i,
			     const SparseMb& y, const VectorXi& nclasses,
			     int maxclasses,
			     const vector<int>& sorted_class,
			     const vector<int>& class_order,
			     const boolmatrix& filtered,
			     double C1, double C2, const double eta_t,
			     const param_struct& params)
{

  int sc, cp;
  //int noClasses = y.cols();
  double class_weight, other_weight, left_update, right_update;
  vector<int> classes;
  vector<int>::iterator class_iter;
  int left_classes, right_classes;
  double *sortedLU_iter;

  double pp1 = proj + 1;
  double pm1 = proj - 1;

  // absorbe the learning rate in C1 and C2
  class_weight = C1*eta_t;
  other_weight = C2*eta_t;
  if (params.ml_wt_by_nclasses)
    {
      other_weight /= nclasses.coeff(i);
    }
  if (params.ml_wt_class_by_nclasses)
    {
      class_weight /= nclasses.coeff(i);
    }
  init_ordered_class_list(left_classes, left_update,
			  right_classes, right_update,
			  classes,
			  class_weight, other_weight,
			  i, y, nclasses, class_order, sc_start, sc_end);
  sc=sc_start;
  class_iter = classes.begin();
  sortedLU_iter = sortedLU.data() + 2*sc_start;
  while (1)
    {
      while (sc == *class_iter)
	{
	  // example has class cp
	  cp = sorted_class[sc];
	  if (!params.remove_class_constraints || !(filtered.get(i,cp)))
	    {
	      if (*sortedLU_iter > pm1)// I1 Condition  w*x < l(c)+1
		{
#ifdef PRINTI
		  {
		    cout << "I1 : " << idx << ", " << i<< endl;
		  }
#endif
		  *sortedLU_iter = max(*sortedLU_iter - class_weight, pm1);
		} // end if

	      ++sortedLU_iter;

	      if (pp1 > *sortedLU_iter)//  I2 Condition  w*x > u(c)-1
		{
#ifdef PRINTI
		  {
		    cout << "I2 : " << idx << ", " << i << endl;
		  }
#endif
		  *sortedLU_iter = min(*sortedLU_iter + class_weight, pp1);
		} // end if
	      ++sortedLU_iter;
	    }
	  else
	    {
	      sortedLU_iter +=2; //the iterator needs to be incremeted even if the class is filtered
	    }
	  //update the left and right classes;
	  ++left_classes;
	  left_update += other_weight;
	  --right_classes;
	  right_update -= other_weight;
	  ++class_iter;
	  ++sc;
	}
      if (sc == sc_end)
	{
	  break;
	}
      // example is not of class cp
      cp = sorted_class[sc];
      if (!(filtered.get(i,cp)))
	{
	  if (left_classes && (pp1 > *sortedLU_iter))// I3 Condition w*x > l(cp) - 1
	    {
#ifdef PRINTI
	      {
		cout << "I3 : " << idx << ", " << i << endl;
	      }
#endif
	      *sortedLU_iter = min(*sortedLU_iter + left_update, pp1);
	    }
	  ++sortedLU_iter;
	  if (right_classes && (*sortedLU_iter > pm1)) //  I4 Condition  w*x < u(cp) + 1
	    {
#ifdef PRINTI
	      {
		cout << "I4 : " << idx << ", " << i<< endl;
	      }
#endif
	      *sortedLU_iter = max(*sortedLU_iter - right_update, pm1);
	    }
	  ++sortedLU_iter;
	}
      else
	{
	  sortedLU_iter += 2; //the iterator needs to be incremeted even if the class is filtered
	}
      ++sc;
    } // while(1)
}

/** sample with replacement .. i.e. may contain duplicates */
void get_ordered_sample(vector<int>& sample, int max, int num_samples)
{
  sample.resize(num_samples);
  for (int i=0;i<num_samples;i++)
    {
      sample[i] = ((int)rand()) % max;
    }
  std::sort(sample.begin(),sample.end());
}

// internal function that initializes some counts and the class vectors
// used by compute_single_w_gradient_size_sample and update_single_sortedLU_sample
void init_ordered_class_list_sample(int& left_classes, double& left_update,
				    int& right_classes, double& right_update,
				    vector<int>& classes,
				    double class_weight, double other_weight,
				    size_t i, const SparseMb& y, const VectorXi& nclasses,
				    const vector<int>& class_order,
				    int& sc_start, int& sc_end, const vector<int>& sc_sample)
{
#ifndef NDEBUG
  assert(static_cast<size_t>(sc_end) < sc_sample.size());
  assert(sc_sample.back() == y.cols());
#endif

  left_classes = 0; //number of classes to the left of the current one
  left_update = 0; // left_classes * other_weight
  right_classes = nclasses.coeff(i); //number of classes to the right of the current one
  right_update = other_weight * right_classes;

  // calling y.coeff is expensive so get the classes here
  classes.resize(0);
  for (SparseMb::InnerIterator it(y,i); it; ++it)
    {
      if (it.value())
	{
	  int c = class_order[it.col()];
	  if ( sc_start > 0 && c < sc_sample[sc_start] )
	    {
	      left_classes++;
	      left_update += other_weight;
	      right_classes--;
	      right_update -= other_weight;
	    }
	  else if (c < sc_sample[sc_end]) //noClasses will always be the last element of sc_sample and sc_end will always be < sc_sample.size(), so sc_sample[sc_end] is well defined.
	    {
	      classes.push_back(c);
	    }
	  else if (c == sc_sample[sc_end])
	    {
	      // this is neaded to avoid having multiple samples of the same class appearing
	      // at the end of the segment. These samples need to be discarded from the current segment.
	      // the class will be processed in the next segment
	      while (sc_end >= sc_start && c == sc_sample[sc_end])
		{
		  sc_end--;
		}
	      sc_end++;
	    }
	}
    }
  classes.push_back(sc_sample[sc_end]+1); // this will always be the last. Done to make sure that classes has at least one element. the +1 is to handle ties if sampling is done with replacement
  std::sort(classes.begin(),classes.end());
}



// function to calculate the multiplier of the gradient for w for a single example.
// subsampling the negative class constraints

double compute_single_w_gradient_size_sample ( int sc_start, int sc_end,
					       const vector<int>& sc_sample,
					       const double proj, const size_t i,
					       const SparseMb& y, const VectorXi& nclasses,
					       int maxclasses,
					       const vector<int>& sorted_class,
					       const vector<int>& class_order,
					       const VectorXd& sortedLU,
					       const boolmatrix& filtered,
					       double C1, double C2,
					       const param_struct& params )
{
  double multiplier = 0.0;
  int s, sc, cp;
  int noClasses = y.cols();
  double class_weight, other_weight, left_update, right_update;
  vector<int> classes;
  vector<int>::iterator class_iter;
  int left_classes, right_classes;

  double pp1 = proj + 1;
  double pm1 = proj - 1;

  class_weight = C1;
  other_weight = C2*noClasses*1.0/(sc_sample.size()-1); //-1 because we added noClasses at the end;
  if (params.ml_wt_by_nclasses)
    {
      other_weight /= nclasses.coeff(i);
    }
  if (params.ml_wt_class_by_nclasses)
    {
      class_weight /= nclasses.coeff(i);
    }

  init_ordered_class_list_sample(left_classes, left_update,
				 right_classes, right_update,
				 classes,
				 class_weight, other_weight,
				 i, y, nclasses, class_order,
				 sc_start, sc_end, sc_sample);

  s=sc_start;
  class_iter = classes.begin();
  int cls = *class_iter;
  sc = sc_sample[s];
  while (1)
    {
      while (cls <= sc)
	{
	  // example has class cp
	  cp = sorted_class[cls];
	  if (!params.remove_class_constraints || !(filtered.get(i,cp)))
	    {
	      //if ((1 - proj + *(sortedLU_iter++)) > 0)// I1 Condition  w*x < l(c)+1
	      if (sortedLU[2*cls] > pm1)// I1 Condition  w*x < l(c)+1
		{
#ifdef PRINTI
		  {
		    cout << "I1 : " << i<< endl;
		  }
#endif
		  multiplier -= class_weight;
		} // end if

	      //if ((1 + proj - *(sortedLU_iter++)) > 0)//  I2 Condition  w*x > u(c)-1
	      if (pp1 > sortedLU[2*cls+1])//  I2 Condition  w*x > u(c)-1
		{
#ifdef PRINTI
		  {
		    cout << "I2 : " << i << endl;
		  }
#endif
		  multiplier += class_weight;
		} // end if
	    }
	  //update the left and right classes;
	  ++left_classes;
	  left_update += other_weight;
	  --right_classes;
	  right_update -= other_weight;
	  while ( sc == cls)
	    {
	      // if a true class was sampled, advance
	      sc = sc_sample[++s];  // because classes.back() = sc_samples[sc_end]+1, it has to be that the while loop terminates before s > sc_end
	    }
	  cls = *(++class_iter);
	}
      if (s == sc_end) // if we are done
	{
	  break;
	}
      // example is not of class cp
      cp = sorted_class[sc];
      if (!(filtered.get(i,cp)))
	{
	  //if (left_classes && ((1 - *sortedLU_iter + proj) > 0)) // I3 Condition w*x > l(cp) - 1
	  if (left_classes && (pp1 > sortedLU[2*sc])) // I3 Condition w*x > l(cp) - 1
	    {
#ifdef PRINTI
	      {
		cout << "I3 : " << i << endl;
	      }
#endif
	      multiplier += left_update;
	    }
	  //if (right_classes && ((1 - proj + *sortedLU_iter) > 0)) //  I4 Condition  w*x < u(cp) + 1
	  if (right_classes && (sortedLU[2*sc+1] > pm1)) //  I4 Condition  w*x < u(cp) + 1
	    {
#ifdef PRINTI
	      {
		cout << "I4 : " << i<< endl;
	      }
#endif
	      multiplier -= right_update;
	    }
	}
      sc = sc_sample[++s];
    } // while(1)
  return multiplier;
}

// function to update L and U for a single example, given w.
// subsampling the negative classes

void update_single_sortedLU_sample ( VectorXd& sortedLU,
				     int sc_start, int sc_end,
				     const vector<int>& sc_sample,
				     const double proj, const size_t i,
				     const SparseMb& y, const VectorXi& nclasses,
				     int maxclasses,
				     const vector<int>& sorted_class,
				     const vector<int>& class_order,
				     const boolmatrix& filtered,
				     double C1, double C2, const double eta_t,
				     const param_struct& params)
{

  int s, sc, cp;
  int noClasses = y.cols();
  double class_weight, other_weight, left_update, right_update;
  vector<int> classes;
  vector<int>::iterator class_iter;
  int left_classes, right_classes;
  double *sl;
  double *su;

  double pp1 = proj + 1;
  double pm1 = proj - 1;

  // absorbe the learning rate in C1 and C2
  class_weight = C1*eta_t;
  other_weight = C2*eta_t*noClasses*1.0/sc_sample.size();
  if (params.ml_wt_by_nclasses)
    {
      other_weight /= nclasses.coeff(i);
    }
  if (params.ml_wt_class_by_nclasses)
    {
      class_weight /= nclasses.coeff(i);
    }

  init_ordered_class_list_sample(left_classes, left_update,
				 right_classes, right_update,
				 classes,
				 class_weight, other_weight,
				 i, y, nclasses, class_order,
				 sc_start, sc_end, sc_sample);

  s=sc_start;
  sc = sc_sample[s];
  class_iter = classes.begin();
  int cls = *class_iter;
  while (1)
    {
      while (cls <= sc)
	{
	  // example has class cp
	  cp = sorted_class[cls];
	  sl = sortedLU.data() + 2*cls;
	  su = sl + 1;
	  if (!params.remove_class_constraints || !(filtered.get(i,cp)))
	    {
	      if (*sl > pm1)// I1 Condition  w*x < l(c)+1
		{
#ifdef PRINTI
		  {
		    cout << "I1 : " << idx << ", " << i<< endl;
		  }
#endif
		  *sl = max(*sl - class_weight, pm1);
		} // end if
	      if (pp1 > *su)//  I2 Condition  w*x > u(c)-1
		{
#ifdef PRINTI
		  {
		    cout << "I2 : " << idx << ", " << i << endl;
		  }
#endif
		  *su = min(*su + class_weight, pp1);
		} // end if
	    }
	  //update the left and right classes;
	  ++left_classes;
	  left_update += other_weight;
	  --right_classes;
	  right_update -= other_weight;
	  while (sc == cls)
	    {
	      // if a true class was sampled, advance
	      sc = sc_sample[++s]; // because classes.back() = sc_samples[sc_end]+1, it has to be that the while loop terminates before s > sc_end
	    }
	  cls = *(++class_iter);
	}
      if ( s == sc_end)
	{
	  break;
	}
      // example is not of class cp
      cp = sorted_class[sc];
      sl = sortedLU.data() + 2*sc;
      su = sl + 1 ;
      if (!(filtered.get(i,cp)))
	{
	  if (left_classes && (pp1 > *sl))// I3 Condition w*x > l(cp) - 1
	    {
#ifdef PRINTI
	      {
		cout << "I3 : " << idx << ", " << i << endl;
	      }
#endif
	      *sl = min(*sl + left_update, pp1);
	    }
	  if (right_classes && (*su > pm1)) //  I4 Condition  w*x < u(cp) + 1
	    {
#ifdef PRINTI
	      {
		cout << "I4 : " << idx << ", " << i<< endl;
	      }
#endif
	      *su = max(*su - right_update, pm1);
	    }
	}
      sc = sc_sample[++s];
    } // while(1)
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
  double class_weight, other_weight;
  std::vector<int> classes;
  std::vector<int>::iterator class_iter;
  classes.reserve(nclasses.coeff(i)+1);
  int left_classes, right_classes;
  double left_weight, right_weight;
  int sc,cp;
  const double* sortedLU_iter;

  class_weight = C1;
  other_weight = C2;
  if (params.ml_wt_by_nclasses)
    {
      other_weight /= nclasses.coeff(i);
    }
  if (params.ml_wt_class_by_nclasses)
    {
      class_weight /= nclasses.coeff(i);
    }

  left_classes = 0; //number of classes to the left of the current one
  left_weight = 0; // left_classes * other_weight
  right_classes = nclasses.coeff(i); //number of classes to the right of the current one
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
	      obj_val += (left_classes?(left_weight * hinge_loss(*sortedLU_iter - proj)):0)
		+ (right_classes?(right_weight * hinge_loss(proj - *(sortedLU_iter+1))):0);
	    }
	  sc++;
	  sortedLU_iter+=2;
	}
      if (sc < noClasses) // test if we are done
	{
	  // example has class cp
	  cp = sorted_class[sc];
	  // compute the loss incured by the example no being withing the bounds
	  //    of class cp
	  // could also test if params.remove_class_constraints is set. If not, this constarint can not be filtered
	  //    essentially add || !params.remove_class_constraint to the conditions below
	  //    would speed things up by a tiny bit .. not significant
	  if (none_filtered || !(filtered.get(i,cp)))
	    {
	      obj_val += (class_weight
			  * (hinge_loss(proj - *sortedLU_iter)
			     + hinge_loss(*(sortedLU_iter+1) - proj)));
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
  const double* sortedLU_iter;
  for (size_t i = i_start; i < i_end; i++)
    {
      tmp = projection.coeff(i);

      class_weight = C1;
      other_weight = C2;
      if (params.ml_wt_by_nclasses)
	{
	  other_weight /= nclasses.coeff(i);
	}
      if (params.ml_wt_class_by_nclasses)
	{
	  class_weight /= nclasses.coeff(i);
	}

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
		  obj_val += (left_classes?(left_weight * hinge_loss(*sortedLU_iter - tmp)):0)
		    + (right_classes?(right_weight * hinge_loss(tmp - *(sortedLU_iter+1))):0);

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
  //const int noClasses = y.cols();
  double obj_val = 0;
  //  size_t no_filtered = filtered.count();
  bool none_filtered = filtered.count()==0;     // OUCH ! slow operation
  //int maxclasses = nclasses.maxCoeff();
#if MCTHREADS
#pragma omp parallel for default(shared) reduction(+:obj_val)
#endif
  for (int i = 0; i < projection.size(); i++)
    {
      obj_val += calculate_ex_objective_hinge(i, projection.coeff(i),  y,
					      nclasses,
					      sorted_class,class_order,
					      sortedLU, filtered,
					      none_filtered,
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
#if MCTHREADS
# pragma omp parallel for  default(shared) collapse(2) reduction(+:obj_val)
#endif
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

double set_eta(param_struct const& params, size_t const t, double const lambda)
{
  double eta_t;
  switch (params.eta_type)
    {
    case ETA_CONST:
      eta_t = params.eta;
      break;
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
      cerr << "Eta option "<<params.eta_type<<" unknown" << endl;
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
void proj_means(VectorXd& means, VectorXi const& nc,
		VectorXd const& projection, SparseMb const& y)
{
  size_t noClasses = y.cols();
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

void init_lu( VectorXd& l, VectorXd& u, VectorXd& means,
              enum Reorder_Type const reorder_type, VectorXd const& projection,
              SparseMb const& y, VectorXi const& nc )
{
    size_t const noClasses = y.cols();
    size_t const n = y.rows();
    l.conservativeResize(noClasses);
    u.conservativeResize(noClasses);
    means.conservativeResize(noClasses);
    for (size_t k = 0; k < noClasses; k++)
    {
        // need /10 because octave gives an error when reading the saved file otherwise.
        // this should not be a problem. If this is a problem then we have bigger issues
        l(k)=boost::numeric::bounds<double>::highest()/10;
        u(k)=boost::numeric::bounds<double>::lowest()/10;
    }
    if( reorder_type == REORDER_AVG_PROJ_MEANS // have no avg'ing yet!
        || reorder_type == REORDER_PROJ_MEANS ){
        means.setZero();
        // XXX if we iterate over CLASSES, then loop can be parallelized
        for (size_t i=0; i<n; ++i) {
            for (SparseMb::InnerIterator it(y,i); it; ++it) {
                if (it.value()) {
                    size_t const c = it.col();
                    double const pr = projection.coeff(i);
                    means.coeffRef(c) += pr;
                    l.coeffRef(c) = std::min(pr,l.coeff(c));
                    u.coeffRef(c) = std::max(pr,u.coeff(c));
                }
            }
        }
        for (size_t k = 0; k < noClasses; k++){
            means.coeffRef(k) /= nc.coeff(k);
        }
    }else if( reorder_type == REORDER_RANGE_MIDPOINTS ){
        // means = l+u, no need to scale since only order important
        for (size_t i=0; i<n; ++i) {
            for (SparseMb::InnerIterator it(y,i); it; ++it) {
                if (it.value()) {
                    size_t const c = it.col();
                    double const pr = projection.coeff(i);
                    l.coeffRef(c)=std::min(pr,l.coeff(c));
                    u.coeffRef(c)=std::max(pr,u.coeff(c));
                }
            }
        }
        means = l + u;
    }else{
        throw std::runtime_error("Error - unsupported reorder_type in init_lu");
    }
}


// *************************************
// need to reimplement this funciton to work with (inside) the WeightVector class
// this might be a costly operation that might be not needed
// we'll implement this when we get there

/** Projection to a new vector that is orthogonal to the rest.
 * - sets w to be ortho to weights[0..projection_dim-1]
 * - It is basically Gram-Schmidt Orthogonalization.
 * - sequentially remove projections of \c w onto each of the
 *   \c weights[i] directions.
 * - original implementation <B>only works of weights.cols(0..projection_dim-1)
 *   are already orthogonal</B>
 * \throw runtime_error in debug compile if \c w not orthogonal to each \c weights.col(0..projection_dim-1)
 */
void project_orthogonal( VectorXd& w, const DenseM& weights,
                         const int& projection_dim)
{
    int const verbose=0;
    if (projection_dim == 0)
        return;
    if(verbose) cout<<" project_orthogonal : w["<<w.rows()<<"x"<<w.cols()<<"] |w_0|="<<w.norm()
        <<" weight["<<weights.rows()<<"x"<<weights.cols()<<"]"
        <<" projection_dim = "<<projection_dim<<endl;
#define ASSUME_WEIGHTS_ALREADY_ORTHOGONAL 0
#if ASSUME_WEIGHTS_ALREADY_ORTHOGONAL
#if 0 // original code
    //
    // [ejk] original method only works if weights cols 0..projection_dim-1 are already ortho
    //
    // Assuming the first to the current projection_dim are the ones we want to be orthogonal to
    VectorXd proj_sum(w.rows());
    //DenseM wt = w.transpose();
    proj_sum.setZero();
    for (int i = 0; i < projection_dim; ++i) {
        double const norm = weights.col(i).norm();
        proj_sum = proj_sum
            + weights.col(i) * ((w.transpose() * weights.col(i)) / (norm*norm));
        if(verbose) cout<<" w vs w["<<i<<"] |w[i]|="<<norm<<" dot(wi,w-proj_sum)) ~ "
            <<(w-proj_sum).transpose() * weights.col(i)<<endl;
    }

    w = (w - proj_sum); // NO
    // Suppose two identical cols of weights.
    // Then that component would get removed TWICE.
    // So must immediately apply projections to w ...
#else 0 // just update w sequentially...
    for (int i = 0; i < projection_dim; ++i) {
        double const norm = weights.col(i).norm();
        if( norm > 1.e-6 ){
            w.array() -= (weights.col(i) * ((w.transpose() * weights.col(i)) / (norm*norm))).array();
        }
        if(verbose) cout<<" remove wi=weights.col("<<i<<") --> |w'_"<<i<<"|="<<w.norm()
            <<" dot(w',wi)"<<w.transpose() * weights.col(i)<<endl;
    }
#endif
#else // ! ASSUME_WEIGHTS_ALREADY_ORTHOGONAL
    if(verbose) cout<<" Orthogonalizing first "<<projection_dim<<" cols of weights..."<<endl;
    // we orthogonalize weights, and then apply to w
    DenseM o = weights.topLeftCorner( weights.rows(), projection_dim );  // orthonormalized weights
    assert( o.cols() == projection_dim );
    VectorXd onorm(projection_dim);
    for(int i = 0; i < projection_dim; ++i){
        onorm[i] = o.col(i).norm();                     // valgrind issues. why?
    }
    for(int i = 0; i < projection_dim; ++i){
        for(int j = 0; j < i; ++j) {
            double const nj = o.col(j).norm(); //double const nj = onorm[j];
            if( nj > 1.e-6 ){
                o.col(i).array() -= (o.col(j) * ((o.col(i).transpose() * o.col(j)) / (nj*nj))).array();
            }
            if(verbose) cout<<" o"<<i<<" remove o"<<j<<" --> |o"<<i<<"|="<<o.col(i).norm()
                <<" dot(o"<<i<<",o"<<j<<") = "<<o.col(i).transpose() * o.col(j)<<endl;
        }
        onorm[i] = o.col(i).norm();
    }
    for(int i = 0; i < projection_dim; ++i){
        double const norm = onorm[i]; //o.col(i).norm();        // XXX valgrind
        if( norm > 1.e-6 ){
            w.array() -= (o.col(i) * ((w.transpose() * o.col(i)) / (norm*norm))).array();
        }
        if(verbose) cout<<" remove wi=o.col("<<i<<") --> |w'_"<<i<<"|="<<w.norm()
            <<" dot(w',wi)"<<w.transpose() * o.col(i)<<endl;
    }
#endif
#ifndef NDEBUG
    // check orthogonality
    std::ostringstream err;
    for( int i=0; i<projection_dim; ++i ){
        double z = w.transpose() * weights.col(i);
        if( fabs(z) > 1.e-8 ){
            err<<" ERROR: w wrt "<<weights.cols()<<" vectors,"
                " orthogonality violated for w vs weights.col("<<i<<")"
                "\n\t|w'| = "<<w.norm() // <<" |proj_sum| = "<<proj_sum.norm()
                <<"\n\tdot product = "<<z<<" too large";
        }
    }
    if(err.str().size()){
        throw std::runtime_error(err.str()); // perhaps it is not that serious?
    }
#endif //NDEBUG
}


// *********************************
// Solve the optimization using the gradient decent on hinge loss

/********* template functions are implemented in the header
template<typename EigenType>
void solve_optimization(DenseM& weights, DenseM& lower_bounds,
			DenseM& upper_bounds, VectorXd& objective_val,
			EigenType& x, const VectorXd& y,
			const double C1_, const double C2_, bool resumed);
***********/

