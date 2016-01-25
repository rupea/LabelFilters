#ifndef FIND_W_DETAIL_HH
#define FIND_W_DETAIL_HH
/** \file
 * template definitions for templates of \ref find_w_detail.h
 */

#include "find_w_detail.h"                      // internal functions
//#include <boost/numeric/conversion/bounds.hpp>  // boost::numeric::bounds<T>
//#include <boost/limits.hpp>     // boost::numeric::bounds<T>
//#include <iosfwd>                             // std::cerr
#include <iostream>                             // std::cerr

/** initialize projection vector \c w prior to calculating \c weights[*,projection_dim].
 * \p w         WeightedVector for SGD iterations
 * \p x         nExamples x d training data (each row is one input vector)
 * \p y         bool matrix of class info
 * \p nc        number of examples in each class
 * \p weights   d x projections matrix of col-wise projection vectors
 * \p projection_dim the column of weights for which we are initializing \c w
 */
    template<typename EigenType>
void init_w( WeightVector& w,
             EigenType const& x, SparseMb const& y, VectorXi const& nc,
             DenseM const& weights, int const projection_dim)
{
    using namespace std;
    cout<<" init_w : weights.cols()="<<weights.cols()<<" projection_dim="<<projection_dim<<" random"; cout.flush();
    int const d = x.cols();
    int const noClasses = y.cols();
    assert( projection_dim < weights.cols() );

    VectorXd init(d);
    //init.setZero();
    init.setRandom(); init.normalize();

    if(1){
        cout<<" + vector-between-2-classes"; cout.flush();
        // initialize w as vector between the means of two random classes.
        // should find cleverer initialization schemes
        int c1, c2, tries=0;
        // Actually better to select to far classes, probability prop. to dist between classes?
        // BUT also want to avoid previously chosen directions.
        // XXX fix this !!!
        do{
            c1 = ((int) rand()) % noClasses;
            ++tries;
        }while( nc(c1) == 0 && tries < 10 );
        do{
            c2 = ((int) rand()) % noClasses;
            ++tries;
        }while( (nc(c2) == 0 || c2 == c1) && tries < 50 );
        if( nc(c1) > 0 && nc(c2) > 0 && c2 != c1 ){
            VectorXd difference;
            difference_means(difference,x,y,nc,c1,c2);
            //cout<<" init["<<init.rows()<<"x"<<init.cols()<<"]"
            //    <<" difference["<<difference.rows()<<"x"<<difference.cols()<<"]"; cout.flush();
            assert( difference.rows() == init.rows() );
            assert( difference.cols() == init.cols() );
            init.array() += difference.array()*(10.0/difference.norm()); // 10 ? perhaps match margins?
        }
    }
    cout<<endl;

    if(1){ // I think starting w should be ~ orthogonal to previous projections.
        //cout<<" + orthogal to prev"; cout.flush();
        try{
            // orthogonalize to current projection dirns w[*, col<projection_dim]
            project_orthogonal( init, weights, projection_dim );
        }catch(std::runtime_error const& e){
            cout<<e.what();
            cout<<" Continuing anyway (just initializing a fresh \"random\" projection vector)"<<endl;
        }
        double inorm = init.norm();
        if( init.norm() < 1.e-6 ) {init.setRandom(); init.normalize();}
        else                      {init *= (1.0 / inorm);}
    }
    //cout<<endl;

    w = WeightVector(init);
    // w.setRandom(); // initialize to a random value
}

    template<typename EigenType>
void difference_means(VectorXd& difference, const EigenType& x, const SparseMb& y, const VectorXi& nc, const int c1, const int c2)
{
    size_t const d = x.cols();
    size_t const n = x.rows();
    difference.resize(d);
    difference.setZero();
    double wt1 = 1.0 / nc(c1);
    double wt2 = 1.0 / nc(c2);
    for (size_t row=0;row<n; ++row) {
        if (y.coeff(row,c1)) {
            typename EigenType::InnerIterator it(x,row);
            for (; it; ++it)
                difference.coeffRef(it.col())+=it.value()*wt1;
        }
        if (y.coeff(row,c2)) {
            typename EigenType::InnerIterator it(x,row);
            for (; it; ++it)
                difference.coeffRef(it.col())-=it.value()*wt2;
        }
    }
}

    template<typename EigenType>
void finite_diff_test(const WeightVector& w, const EigenType& x, size_t idx,
                      const SparseMb& y, const VectorXi& nclasses, int maxclasses,
                      const std::vector<int>& sorted_class, const std::vector<int>& class_order,
                      const VectorXd& sortedLU,
                      const boolmatrix& filtered,
                      double C1, double C2, const param_struct& params)
{
    using namespace std;
    double delta = params.finite_diff_test_delta;
    VectorXd proj(1);
    proj.coeffRef(0) = w.project_row(x,idx);
    bool none_filtered = filtered.count()==0;
    double obj = calculate_ex_objective_hinge(idx, proj.coeff(0), y, nclasses, sorted_class, class_order, sortedLU, filtered, none_filtered, C1, C2, params);

    VectorXsz index(1);
    index.coeffRef(0) = idx;
    size_t idx_start = 0;
    size_t idx_end = 1;
    int sc_start = 0;
    int sc_end = y.cols();
    VectorXd multipliers;
    VectorXd sortedLU_gradient;

    compute_gradients(multipliers, sortedLU_gradient,
                      idx_start, idx_end, sc_start, sc_end,
                      proj, index, y, nclasses, maxclasses,
                      sorted_class, class_order, sortedLU,
                      filtered, C1, C2, params);

    WeightVector w_new(w);
    double xnorm = x.row(idx).norm();
    double multsign;
    if (multipliers.coeff(0) > 0)
        multsign = 1.0;
    if (multipliers.coeff(0) < 0)
        multsign = -1.0;

    w_new.gradient_update(x, idx, multsign*delta/xnorm);// divide delta by multipliers.coeff(0)*xnorm . the true gradient is multpliers.coeff(0)*x.

    double obj_w_grad = calculate_ex_objective_hinge(idx, w_new.project_row(x,idx), y, nclasses, sorted_class, class_order, sortedLU, filtered, none_filtered, C1, C2, params);
    double w_grad_error = fabs(obj_w_grad - obj + multsign*delta*multipliers.coeff(0)*xnorm);

    VectorXd sortedLU_new(sortedLU);
    sortedLU_new += sortedLU_gradient * delta / sortedLU_gradient.norm();  // have some delta that is inversely proportional to the norm of the gradient

    double obj_LU_grad = calculate_ex_objective_hinge(idx, proj.coeff(0), y, nclasses, sorted_class, class_order, sortedLU_new, filtered, none_filtered, C1, C2, params);
    double LU_grad_error = fabs(obj_LU_grad - obj + delta*sortedLU_gradient.norm());

    cerr << "w_grad_error:  " << w_grad_error << "   " << obj_w_grad - obj << "  " << obj_w_grad << "  " << obj << "  " << multsign*delta*multipliers.coeff(0)*xnorm << "   " << xnorm << "  " << idx << "   " << proj.coeff(0) << "  " << w_new.project_row(x,idx)  << "  ";

    for (SparseMb::InnerIterator it(y,idx); it; ++it)
    {
        int order = class_order[it.col()];
        cerr << it.col() << ":" << it.value() << ":" << order << ":" <<sortedLU.coeff(2*order) + 1 << ":" << sortedLU.coeff(2*order+1) - 1  << "  ";
    }
    cerr << endl;
    /* if (idx == 9022) */
    /*   {						 */
    /*     cerr << sortedLU.transpose() - VectorXd::Ones(sortedLU.size()).transpose() << endl; */
    /*     cerr << sortedLU.transpose() +  VectorXd::Ones(sortedLU.size()).transpose() << endl; */
    /*   } */
    cerr << "LU_grad_error: " << LU_grad_error << "  " << obj_LU_grad - obj << "  " << "  " << obj_LU_grad << "  " << obj << "  " << delta*sortedLU_gradient.norm() << "  " << "  " << idx << "  " << proj.coeff(0) << "  ";
    for (SparseMb::InnerIterator it(y,idx); it; ++it)
    {
        int order = class_order[it.col()];
        cerr << it.col() << ":" << it.value() << ":" << order << ":" << sortedLU.coeff(2*order) + 1 << ":" << sortedLU.coeff(2*order+1) - 1  << " - " << sortedLU_new.coeff(2*order) + 1 << ":" << sortedLU_new.coeff(2*order+1) - 1  << "  ";
    }
    cerr << endl;
}

    template<typename EigenType>
void update_safe_SGD (WeightVector& w, VectorXd& sortedLU, VectorXd& sortedLU_avg,
                      const EigenType& x, const SparseMb& y,
                      const double C1, const double C2, const double lambda,
                      const unsigned long t, const double eta_t,
                      const size_t n, const VectorXi& nclasses, const int maxclasses,
                      const std::vector<int>& sorted_class, const std::vector<int>& class_order,
                      const boolmatrix& filtered,
                      const int sc_chunks, const int sc_chunk_size, const int sc_remaining,
                      const param_struct& params)
{
    using namespace std;

    double multiplier = 0;

#ifndef NDEBUG
    // batch size should be 1
    assert(params.batch_size == 1);
#endif

    size_t i = ((size_t) rand()) % n;
    // WARNING: for now it assumes that norm(x) = 1!!!!!
    assert(x.row(i).norm() == 1);
    double proj = w.project_row(x,i);

    vector<int> sample;
    if (params.class_samples)
    {
        get_ordered_sample(sample, y.cols(), params.class_samples);
        sample.push_back(y.cols()); // need the last entry of the sample to be the number of classes
    }


#pragma omp parallel for default(shared) reduction(+:multiplier)
    for (int sc_chunk = 0; sc_chunk < sc_chunks; sc_chunk++)
    {
        // the first chunks will have an extra iteration
        int sc_start = sc_chunk*sc_chunk_size + (sc_chunk<sc_remaining?sc_chunk:sc_remaining);
        int sc_incr = sc_chunk_size + (sc_chunk<sc_remaining);
        if (params.class_samples)
        {
            multiplier +=
                compute_single_w_gradient_size_sample(sc_start, sc_start+sc_incr,
                                                      sample,
                                                      proj, i,
                                                      y, nclasses, maxclasses,
                                                      sorted_class, class_order,
                                                      sortedLU, filtered, C1, C2, params);
        }
        else
        {
            multiplier += compute_single_w_gradient_size(sc_start, sc_start+sc_incr,
                                                         proj, i,
                                                         y, nclasses, maxclasses,
                                                         sorted_class, class_order,
                                                         sortedLU, filtered, C1, C2, params);
        }
    }

    // make sure we do not overshoot with the update
    // this is expensive, so we might want an option to turn it off
    double new_multiplier, new_proj;
    double eta = eta_t;
    do
    {
        // WARNING: for now it assumes that norm(x) = 1!!!!!
        new_proj = proj - eta*lambda*proj - eta*multiplier;
        new_multiplier=0;
#pragma omp parallel for  default(shared) reduction(+:new_multiplier)
        for (int sc_chunk = 0; sc_chunk < sc_chunks; sc_chunk++)
        {
            // the first chunks will have an extra iteration
            int sc_start = sc_chunk*sc_chunk_size + (sc_chunk<sc_remaining?sc_chunk:sc_remaining);
            int sc_incr = sc_chunk_size + (sc_chunk<sc_remaining);
            if (params.class_samples)
            {
                new_multiplier +=
                    compute_single_w_gradient_size_sample(sc_start, sc_start+sc_incr,
                                                          sample,
                                                          new_proj, i,
                                                          y, nclasses, maxclasses,
                                                          sorted_class, class_order,
                                                          sortedLU, filtered, C1, C2, params);
            }
            else
            {
                new_multiplier +=
                    compute_single_w_gradient_size(sc_start, sc_start+sc_incr,
                                                   new_proj, i,
                                                   y, nclasses, maxclasses,
                                                   sorted_class, class_order,
                                                   sortedLU, filtered, C1, C2, params);
            }
        }
        eta = eta/2;
    } while (multiplier*new_multiplier < -1e-5);

    // last eta did not overshooot so restore it
    eta = eta*2;
    //update w
    if (params.avg_epoch && t >= params.avg_epoch)
    {
        // updates both the curent w and the average w
        w.batch_gradient_update_avg(x,i,multiplier,lambda,eta);
    }
    else
    {
        // update only the current w
        w.batch_gradient_update(x, i, multiplier, lambda, eta);
    }

    // update L and U with w fixed.
    // use new_proj since it is exactly the projection obtained with the new w
    bool const accumulate_sortedLU = (params.optimizeLU_epoch <= 0 && params.avg_epoch > 0 && t >= params.avg_epoch);
#pragma omp parallel for  default(shared)
    for (int sc_chunk = 0; sc_chunk < sc_chunks; sc_chunk++)
    {
        int sc_start = sc_chunk*sc_chunk_size + (sc_chunk<sc_remaining?sc_chunk:sc_remaining);
        int sc_incr = sc_chunk_size + (sc_chunk<sc_remaining);
        if (params.class_samples)
        {
            update_single_sortedLU_sample(sortedLU, sc_start, sc_start+sc_incr,
                                          sample, new_proj, i,
                                          y, nclasses, maxclasses,
                                          sorted_class, class_order,
                                          filtered, C1, C2, eta_t, params);
        }
        else
        {
            update_single_sortedLU(sortedLU, sc_start, sc_start+sc_incr, new_proj, i,
                                   y, nclasses, maxclasses, sorted_class, class_order,
                                   filtered, C1, C2, eta_t, params);
        }
        // update the average LU
        // need to do something special when samplin classes to avoid the O(noClasses) complexity.
        // for now we leave it like this since we almost always we optimize LU at the end
        if (accumulate_sortedLU)
        {
            // if we optimize the LU, we do not need to
            // keep track of the averaged lower and upper bounds
            // We optimize the bounds at the end based on the
            // average w

            // do not divide by t-params.avg_epoch + 1 here
            // do it when using sortedLU_avg
            // it might become too big!, but through division it
            //might become too small
            sortedLU_avg.segment(2*sc_start, 2*sc_incr) += sortedLU.segment(2*sc_start, 2*sc_incr);
        }
    }
#if MCPRM>0
    if (accumulate_sortedLU) {
        if(t==params.avg_epoch) std::cout<<" ACC "; std::cout.flush();
        //++luPerm.nAccSortlu_avg;
    }
#endif
}

    template<typename EigenType>
void update_minibatch_SGD(WeightVector& w, VectorXd& sortedLU, VectorXd& sortedLU_avg,
                          const EigenType& x, const SparseMb& y,
                          const double C1, const double C2, const double lambda,
                          const unsigned long t, const double eta_t,
                          const size_t n, const size_t batch_size,
                          const VectorXi& nclasses, const int maxclasses,
                          const std::vector<int>& sorted_class, const std::vector<int>& class_order,
                          const boolmatrix& filtered,
                          const size_t sc_chunks, const size_t sc_chunk_size, const size_t sc_remaining,
                          const int idx_chunks, const int idx_chunk_size, const int idx_remaining,
                          MutexType* idx_locks, MutexType* sc_locks,
                          const param_struct& params)
{
    // use statics to avoid the cost of alocation at each iteration?
    static VectorXd proj(batch_size);
    static VectorXsz index(batch_size);
    static VectorXd multipliers(batch_size);
    VectorXd multipliers_chunk;
    //  VectorXd sortedLU_gradient(2*noClasses); // used to improve cache performance
    VectorXd sortedLU_gradient_chunk;
    size_t i,idx;

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
    // now we can update w and L,U directly

    multipliers.setZero();
    //  sortedLU_gradient.setZero();

    //#pragma omp parallel for  default(shared) shared(idx_locks,sc_locks) private(multipliers_chunk,sortedLU_gradient_chunk) collapse(2)
    //#pragma omp parallel for  default(shared) shared(idx_locks,sc_locks) private(multipliers_chunk,sortedLU_gradient_chunk) collapse(1)
    //#pragma omp parallel for  default(shared) shared(idx_locks,sc_locks) collapse(1) if(idx_chunks > 1)
    //#pragma omp parallel for  default(shared) shared(idx_locks,sc_locks) collapse(2) if(idx_chunks > 1)
    for (int idx_chunk = 0; idx_chunk < idx_chunks; idx_chunk++)
    {
        //VectorXd sortedLU_gradient_chunk;
        //VectorXd multipliers_chunk;
        //multipliers.setZero();

        for (size_t sc_chunk = 0; sc_chunk < sc_chunks; sc_chunk++)
        {
            // the first chunks will have an extra iteration
            size_t idx_start = idx_chunk*idx_chunk_size + (idx_chunk<idx_remaining?idx_chunk:idx_remaining);
            size_t idx_incr = idx_chunk_size + (idx_chunk<idx_remaining);
            // the first chunks will have an extra iteration
            size_t sc_start = sc_chunk*sc_chunk_size + (sc_chunk<sc_remaining?sc_chunk:sc_remaining);
            int sc_incr = sc_chunk_size + (sc_chunk<sc_remaining);
            compute_gradients(multipliers_chunk, sortedLU_gradient_chunk,
                              idx_start, idx_start+idx_incr,
                              sc_start, sc_start+sc_incr,
                              proj, index, y, nclasses, maxclasses,
                              sorted_class, class_order,
                              sortedLU, filtered,
                              C1, C2, params);

            //#pragma omp task default(none) shared(sc_chunk, idx_chunk, multipliers, sc_start, idx_start, sc_incr, idx_incr, sortedLU, sortedLU_gradient_chunk, multipliers_chunk, sc_locks,  idx_locks)
            {
                //#pragma omp task default(none) shared(idx_chunk, multipliers, multipliers_chunk, idx_start, idx_incr, idx_locks)
                {
                    //idx_locks[idx_chunk].YieldLock();
                    multipliers.segment(idx_start, idx_incr) += multipliers_chunk;
                    //idx_locks[idx_chunk].Unlock();
                }
                //sc_locks[sc_chunk].YieldLock();
                // update the lower and upper bounds
                // divide by batch_size here because the gradients have
                // not been averaged
                sortedLU.segment(2*sc_start, 2*sc_incr) += sortedLU_gradient_chunk * (eta_t / batch_size);
                //		  sortedLU_gradient.segment(2*sc_start, 2*sc_incr) += sortedLU_gradient_chunk;
                //sc_locks[sc_chunk].Unlock();
                //#pragma omp taskwait
            }
            //#pragma omp taskwait
        }
    }

    //update w
    if (params.avg_epoch && t >= params.avg_epoch)
    {
        // updates both the curent w and the average w
        w.batch_gradient_update_avg(x, index, multipliers, lambda, eta_t);
    }
    else
    {
        // update only the current w
        w.batch_gradient_update(x, index, multipliers, lambda, eta_t);
    }

    ///// did this above in parallel
    // update the lower and upper bounds
    // divide by batch_size here because the gradients have
    // not been averaged
    // should be done above
    // sortedLU += sortedLU_gradient * (eta_t / batch_size);


    // update the average version
    // should do in parallel (maybe Eigen already does it?)
    // especially for small batch sizes.
    if (params.optimizeLU_epoch <= 0 && params.avg_epoch > 0 && t >= params.avg_epoch)
    {
        // if we optimize the LU, we do not need to
        // keep track of the averaged lower and upper bounds
        // We optimize the bounds at the end based on the
        // average w

        // do not divide by t-params.avg_epoch + 1 here
        // do it when using sortedLU_avg
        // it might become too big!, but through division it
        //might become too small
        sortedLU_avg += sortedLU;
        if(t==params.avg_epoch) std::cout<<" ACC "; std::cout.flush();
#if MCPRM>0
        //++luPerm.nAccSortlu_avg;
#endif
    }
}

#endif // FIND_W_DETAIL_HH
