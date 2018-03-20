#include "mcupdate.hh"
#include "boolmatrix.h"

namespace mcsolver_detail{


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
			  const VectorXd& inside_weight, const VectorXd& outside_weight,		    
			  const vector<int>& sorted_class,
			  const vector<int>& class_order,
			  const VectorXd& sortedLU,
			  const boolmatrix& filtered,
			  const double C1, const double C2,
			  const param_struct& params )
  {
    int sc, cp;
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

	class_weight = C1 * inside_weight.coeff(i);
	other_weight = C2 * outside_weight.coeff(i);

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
			*multipliers_iter -= class_weight;
			*sortedLU_gradient_iter -= class_weight;
		      } // end if

		    sortedLU_gradient_iter++;

		    if ((1 + tmp - *(sortedLU_iter++)) > 0)//  I2 Condition  w*x > u(c)-1
		      {
			*multipliers_iter += class_weight;
			*sortedLU_gradient_iter += class_weight;
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
		    *multipliers_iter += left_update; 
		    *sortedLU_gradient_iter += left_update;
		  }
		sortedLU_iter++;
		sortedLU_gradient_iter++;
		if (right_classes && ((1 - tmp + *sortedLU_iter) > 0)) //  I4 Condition  w*x < u(cp) + 1
		  {
		    *multipliers_iter -= right_update;
		    *sortedLU_gradient_iter -= right_update;
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
				       const VectorXd& inside_weight, const VectorXd& outside_weight,		    
				       const vector<int>& sorted_class,
				       const vector<int>& class_order,
				       const VectorXd& sortedLU,
				       const boolmatrix& filtered,
				       const double C1, const double C2,
				       const param_struct& params )
{
  double multiplier = 0.0;
  int sc, cp;
  double class_weight, other_weight, left_update, right_update;
  vector<int> classes;
  vector<int>::iterator class_iter;
  int left_classes, right_classes;
  double const *sortedLU_iter;


  double pp1 = proj + 1;
  double pm1 = proj - 1;

  class_weight = C1 * inside_weight.coeff(i);
  other_weight = C2 * outside_weight.coeff(i);

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
	      // I1 Condition  w*x < l(c)+1 and I2 Condition  w*x > u(c)-1

	      multiplier -= class_weight*((*(sortedLU_iter) > pm1) - (pp1 > *(sortedLU_iter + 1)));
	    }
	  sortedLU_iter +=2; //the iterator needs to be incremeted even if the class is filtered
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
	  // I3 Condition w*x > l(cp) - 1 and I4 Condition  w*x < u(cp) + 1
	  multiplier += (pp1 > *(sortedLU_iter))*left_update - (*(sortedLU_iter + 1) > pm1)*right_update; 
	}
      sortedLU_iter += 2; //the iterator needs to be incremeted even if the class is filtered
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
			     const VectorXd& inside_weight, const VectorXd& outside_weight,		    
			     const vector<int>& sorted_class,
			     const vector<int>& class_order,
			     const boolmatrix& filtered,
			     double C1, double C2, const double eta_t,
			     const param_struct& params)
{

  int sc, cp;
  double class_weight, other_weight, left_update, right_update;
  vector<int> classes;
  vector<int>::iterator class_iter;
  int left_classes, right_classes;
  double *sortedLU_iter;

  double pp1 = proj + 1;
  double pm1 = proj - 1;

  // absorbe the learning rate in C1 and C2
  class_weight = C1 * inside_weight.coeff(i) * eta_t;
  other_weight = C2 * outside_weight.coeff(i) *eta_t;

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
		  *sortedLU_iter = max(*sortedLU_iter - class_weight, pm1);
		} // end if

	      ++sortedLU_iter;

	      if (pp1 > *sortedLU_iter)//  I2 Condition  w*x > u(c)-1
		{
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
	      *sortedLU_iter = min(*sortedLU_iter + left_update, pp1);
	    }
	  ++sortedLU_iter;
	  if (right_classes && (*sortedLU_iter > pm1)) //  I4 Condition  w*x < u(cp) + 1
	    {
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

// function to calculate the multiplier of the gradient for w for a single example.
// subsampling the negative class constraints

double compute_single_w_gradient_size_sample ( int sc_start, int sc_end,
					       const vector<int>& sc_sample,
					       const double proj, const size_t i,
					       const SparseMb& y, const VectorXi& nclasses,
					       int maxclasses,
					       const VectorXd& inside_weight, const VectorXd& outside_weight,    
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

  class_weight = C1*inside_weight.coeff(i);
  other_weight = C2*outside_weight.coeff(i)*noClasses*1.0/(sc_sample.size()-1); //-1 because we added noClasses at the end;

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
 	      // I1 Condition  w*x < l(c)+1 and  I2 Condition  w*x > u(c)-1

	      multiplier -= class_weight*((sortedLU[2*cls] > pm1) - (pp1 > sortedLU[2*cls+1]));
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
 	  // I3 Condition w*x > l(cp) - 1 and I4 Condition  w*x < u(cp) + 1
	  multiplier += (pp1 > sortedLU[2*sc])*left_update - (sortedLU[2*sc+1] > pm1)*right_update;
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
				     const VectorXd& inside_weight, const VectorXd& outside_weight,		    
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
  class_weight = C1*inside_weight.coeff(i)*eta_t;
  other_weight = C2*outside_weight.coeff(i)*eta_t*noClasses*1.0/sc_sample.size();
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
		  *sl = max(*sl - class_weight, pm1);
		} // end if
	      if (pp1 > *su)//  I2 Condition  w*x > u(c)-1
		{
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
	      *sl = min(*sl + left_update, pp1);
	    }
	  if (right_classes && (*su > pm1)) //  I4 Condition  w*x < u(cp) + 1
	    {
	      *su = max(*su - right_update, pm1);
	    }
	}
      sc = sc_sample[++s];
    } // while(1)
}



}
