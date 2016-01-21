
#include "predict.h"
#include "filter.h"

ActiveDataSet* projectionsToActiveSet( VectorXsz& no_active, DenseM const& projections,
                                       const DenseColM& lmat, const DenseColM& umat,
                                       bool verbose)
{
    ActiveDataSet* active;
    size_t n = projections.rows();  // # of features
    size_t noClasses = lmat.rows();
    assert( umat.rows() == (int)noClasses );
    active = new ActiveDataSet(n);

    no_active.resize(projections.cols());
    no_active.setZero();

    for (ActiveDataSet::iterator it = active->begin(); it != active->end(); ++it) {
        *it = new boost::dynamic_bitset<>();
        (*it) -> resize(noClasses,true);        // if no projections, every class must be possible
    }

    for (int i = 0; i < projections.cols(); i++)
    {
        VectorXd proj = projections.col(i);
        VectorXd l = lmat.col(i);
        VectorXd u = umat.col(i);
        if (verbose) 
        {
            cout << "Init filter" << endl;
        }
        Filter f(l,u);      
        if (verbose)
        {
            cout << "Update filter, projection " << i << endl; 
        }
        ActiveDataSet::iterator it;
        size_t count = 0;
#pragma omp parallel for default(shared) reduction(+:count)
        for(size_t j=0; j < n; ++j)
        {
            boost::dynamic_bitset<>* act = (*active)[j];
            (*act) &= *(f.filter(proj.coeff(j)));
            count += act -> count();
        }
        no_active[i]=count;
    }
    return active;
}

