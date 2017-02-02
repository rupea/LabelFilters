#include "constants.h" // MCTHREADS
#include "predict.h"
#include "filter.h"

ActiveDataSet* projectionsToActiveSet( VectorXsz& no_active, DenseM const& projections,
                                       const DenseColM& lmat, const DenseColM& umat,
                                       bool verbose)
{
    ActiveDataSet* active;
    size_t noClasses = lmat.rows();
    assert( umat.rows() == (int)noClasses );

    no_active.resize(projections.cols());
    no_active.setZero();

    // active = new ActiveDataSet(n);
    // for (ActiveDataSet::iterator it = active->begin(); it != active->end(); ++it) {
    //     *it = new boost::dynamic_bitset<>();
    //     (*it) -> resize(noClasses,true);        // if no projections, every class must be possible
    // }

    for (int i = 0; i < projections.cols(); i++)
    {
        Eigen::VectorXd proj = projections.col(i);
        Eigen::VectorXd l = lmat.col(i);
        Eigen::VectorXd u = umat.col(i);
        if (0 && verbose) // debug only
        {
            cout<<"Init filter col i="<<i<<endl;
            cout<<"lmat: "<<lmat.transpose()<<endl;
            cout<<"umat: "<<umat.transpose()<<endl;
            cout<<"proj: "<<proj.transpose()<<endl;
        }
        Filter f(l,u);
        if (verbose)
        {
            cout << "Update filter, projection " << i << endl;
        }
 
	no_active[i] = update_active(&active, f, proj);
    }
    return active;
}

size_t update_active(ActiveDataSet** active, Filter const& f, Eigen::VectorXd const&  proj)
{ 
  size_t count = 0;
  size_t n = proj.size();  // # of examples
  if (!(*active))
    {
      //this is the first filter applied. Initialize active with the first filter values. 
      *active = new ActiveDataSet(n);
#if MCTHREADS
#pragma omp parallel for default(shared) reduction(+:count)
#endif
      for (size_t j=0; j < n; ++j)
	{
	  (**active)[j] = new boost::dynamic_bitset<>(*(f.filter(proj.coeff(j))));
	  count += (**active)[j]->count();
	}
    }
  else
    {      
#if MCTHREADS
#pragma omp parallel for default(shared) reduction(+:count)
#endif
      for(size_t j=0; j < n; ++j)
	{
	  boost::dynamic_bitset<>* act = (**active)[j];
	  (*act) &= *(f.filter(proj.coeff(j)));
	  count += act -> count();
	}
    }
  return count;
}
