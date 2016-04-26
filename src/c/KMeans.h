#ifndef __KMEANS_H
#define __KMEANS_H

#include "typedefs.h"
#include "utils.h"
#include <iostream>
#include <omp.h>

#ifdef PROFILE
#include <gperftools/profiler.h>
#endif


//using Eigen::VectorXi;
//using Eigen::RowVectorXd;
//using namespace std;


template<typename EigenType>
void init_centers_random(DenseColM& centers, const EigenType& x)
{
  int k = centers.cols();
  size_t n = x.rows();
  int i;
  for (i=0;i<k;i++)
    {
      centers.col(i) = x.row(((size_t) rand()) % n);
    }
}

template<typename EigenType>
double cluster_test_kmeans(Eigen::VectorXi& assignments, const DenseColM& centers,
			   const EigenType& x, bool spherical, bool verbose=false)
{
  using namespace std;
  size_t i;
  int k = centers.cols();
  size_t n = x.rows();
  double obj=0.0;
  Eigen::RowVectorXd sims(k);
  
  // Calculate the assignments given the centers
  if (verbose)
    {
      cout << "Assign points to clusters" << endl;
    }
  obj=0.0;
# pragma omp parallel for firstprivate(sims) default(shared) reduction(+:obj)
  for (i=0; i<n;i++)
    {      
      //DotProductInnerVector(sims,centers,x,i);
      sims = x.row(i)*centers;
      obj+=sims.maxCoeff(&assignments[i]);
    }
  
  if (verbose)
    {
      cout << "Objective: " << obj << endl;
    }
  
  return obj;
}

template<typename EigenType>
double run_kmeans(DenseColM& centers, Eigen::VectorXi& assignments,
		  const EigenType& x, int iterations, bool spherical, bool verbose=false)
{
  using namespace std;
  #ifdef PROFILE
  ProfilerStart("kmeans.profile");
  #endif

  size_t i;
  int k = centers.cols();
  size_t n = x.rows();
  double obj,old_obj=0.0;
  Eigen::RowVectorXd sizes(k);
  Eigen::RowVectorXd sims(k);

  init_centers_random(centers,x);

  for (int it=0 ; it<iterations;it++)
    {
      // Calculate the assignments given the centers
      if (verbose)
	{
	  cout << "E step: Assign points to clusters" << endl;
	}
      obj=0.0;
# pragma omp parallel for firstprivate(sims) default(shared) reduction(+:obj)
      for (i=0; i<n;i++)
	{      
	  //DotProductInnerVector(sims,centers,x,i);
	  sims = x.row(i)*centers;
	  obj+=sims.maxCoeff(&assignments[i]);
	}

      if (verbose)
	{
	  cout << "Iteration " << it << ". Objective: " << obj << endl;
	}
      
      if ((obj - old_obj)/obj < 1e-4)
	{
	  return obj;
	}
      old_obj = obj;
      
      // Recompute the centers given the assignments  
      if (verbose)
	{
	  cout << "M step: Centroid recomputation" << endl;
	}
      centers.setZero();
      sizes.setZero();
      for (i=0; i<n;i++)
	{
	  addInnerVector(centers.col(assignments(i)),x,i);
	  sizes[assignments[i]]++;    
	}
      for (i=0;i<k;i++)
	{
	  // if a cluster is empty reinitialize its center to a random data point.
	  if (sizes[i] == 0) 
	    {
	      centers.col(i) = x.row(((size_t) rand()) % n);
	      sizes[i] = 1;
	    }
	}
      
      if (spherical)
	{
	  centers.colwise().normalize();
	}
      else
	{
	  centers.array().rowwise()/=(sizes.array());
	}
    }
  
  // Calculate the final assignments given the centers
  if (verbose)
    {
      cout << "Final assignment of points to clusters" << endl;
    }
  obj=0.0;  
# pragma omp parallel for firstprivate(sims) default(shared) reduction(+:obj)
  for (i=0; i<n;i++)
    {
      //DotProductInnerVector(sims,centers,x,i);
      sims = x.row(i)*centers;
      obj+=sims.maxCoeff(&(assignments[i]));
    }
  #ifdef PROFILE
  ProfilerStop();
  #endif
  return obj;
}

#endif 
