#ifndef FIND_W_HH
#define FIND_W_HH
/** \file
 * inline template definitions if your \c x is not DenseM or SparseM.
 *
 * ACTUALLY client code should not include this, so this file could
 * be find_w_detail0.hh or something.
 *
 * The library instantiates for EIGENTYPE SparseM and DenseM,
 * so client SHOULD NOT need to ever include this (if linking with library)
 */

#include "find_w.h"             // public api declarations
#include "find_w_detail.hh"     // declarations and template definitions

#include "constants.h"          // PRINT_O setting
#include "printing.hh"          // print_report, print_progress

#ifdef PROFILE
#include <gperftools/profiler.h>
#define OPT_PROFILE_START( FIRSTFILE ) do{ ProfilerStart( FIRSTFILE ); }while(0)
#define OPT_PROFILE_NEXT( NEXTFILE )   do{ ProfilerStop(); ProfilerStart( NEXTFILE ); }while(0)
#define OPT_PROFILE_STOP()             do{ ProfilerStop(); }while(0)
#else
#define OPT_PROFILE_START( FIRSTFILE ) do{}while(0)
#define OPT_PROFILE_NEXT( NEXTFILE )   do{}while(0)
#define OPT_PROFILE_STOP()             do{}while(0)
#endif

#include <iostream>
#include <iomanip>

#if GRADIENT_TEST // compile-time option, disabled by default, macro to unclutter iteration loop
#define OPT_GRADIENT_TEST do \
{   /* switched to thread-safe rand, just in case */ \
    static unsigned thread_local seed = static_cast<unsigned>(this_thread.get_id()); \
    bool const bFiniteDiff= params.finite_diff_test_epoch > 0 && t%params.finite_diff_test_epoch == 0; \
    if(bFiniteDiff) \
        for (uint32_t fdtest=0U; fdtest<params.no_finite_diff_tests; ++fdtest){ \
            size_t idx = ((size_t) rand_r(&seed)) % n; \
                finite_diff_test(w, x, idx, y, nclasses, maxclasses, sorted_class, \
                                 class_order, sortedLU, filtered, C1, C2, params); \
        } \
}while(0)
#else
#define OPT_GRADIENT_TEST do{}while(0)
#endif

// --------------- inline template definition to solve optimization problem ----------------
/** \c params is const \b mostly -- if we had to stop before params.no_projections,
 * then params.no_projections is reduce in value. */
    template<typename EIGENTYPE> inline
void solve_optimization(DenseM& weights, DenseM& lower_bounds,
                        DenseM& upper_bounds, VectorXd& objective_val,
                        DenseM& weights_avg, DenseM& lower_bounds_avg,
                        DenseM& upper_bounds_avg, VectorXd& objective_val_avg,
                        EIGENTYPE const& x,
                        SparseMb const& y,
                        param_struct const& params)
{
    using namespace std;
    OPT_PROFILE_START("init.profile");
    size_t const d = x.cols();          // training examples are d-dim row vectors
    size_t const n = x.rows();          // # of training examples
    size_t const noClasses = y.cols();
    int const no_projections = params.no_projections;
    cout << " find_w.hh, solve_optimization: no_projections: " << no_projections << endl;
    if( (size_t)no_projections >= d ){
        cout<<"WARNING: no_projections > example dimensionality"<<endl;
    }
    //if( params.tot_projections >= d || params.tot_projections < params.no_projections ){
    //    params.tot_projections == params.no_projections;
    //    cout<<"WARNING: will add no random projections";
    //}

    size_t const batch_size = (params.batch_size < 1 || params.batch_size > n) ? (size_t) n : params.batch_size;
    VectorXd xSqNorms;
    if (params.update_type == SAFE_SGD)
    {
        assert(batch_size == 1); // safe_sgd update only works with batch size 1
        calc_sqNorms( x, xSqNorms );
    }

    //std::vector<int> classes = get_classes(y);
    cout << "size x: " << x.rows() << " rows and " << x.cols() << " columns.\n";
    cout << "size y: " << y.rows() << " rows and " << y.cols() << " columns.\n";

    WeightVector w;
    VectorXd projection, projection_avg;
    VectorXd l(noClasses), u(noClasses);
    //l.setZero(); u.setZero();           // <-- valgrind proves these are uninitialized sometimes.
    VectorXd sortedLU(2*noClasses); // holds l and u interleaved in the curent class sorting order (i.e. l,u,l,u,l,u)
    //  VectorXd sortedLU_gradient(2*noClasses); // used to improve cache performance
    //  VectorXd sortedLU_gradient_chunk;
    VectorXd l_avg(noClasses),u_avg(noClasses); // the lower and upper bounds for the averaged gradient
    VectorXd sortedLU_avg(2*noClasses); // holds l_avg and u_avg interleaved in the curent class sorting order (i.e. l_avg,u_avg,l_avg,u_avg,l_avg,u_avg)
    VectorXi nc; // the number of examples in each class
    VectorXd wc; // the number of examples in each class
    VectorXi nclasses; // the number of examples in each class
    int maxclasses; // the maximum number of classes an example might have
    double eta_t;
    size_t obj_idx = 0, obj_idx_avg = 0;
    //  bool order_changed = 1;
    //  VectorXd proj(batch_size);
    //  VectorXsz index(batch_size);
    //  VectorXd multipliers(batch_size);
    //  VectorXd multipliers_chunk;
    // in the multilabel case each example will have an impact proportinal
    // to the number of classes it belongs to. ml_wt and ml_wt_class
    // allows weighting that impact when updating params for the other classes
    // respectively its own class.
    //  size_t  i=0, idx=0;
    unsigned long t = 1;
    VectorXd means(noClasses); // used for initialization of the class order vector;
    std::vector<int> sorted_class(noClasses), class_order(noClasses);//, prev_class_order(noClasses);// used as the switch
    //means.setZero();
    //for( int i=0; i<static_cast<int>(noClasses); ++i ){ sorted_class[i] = i; class_order[i] = i; } // valgrind debug
    char iter_str[30];

    // how to split the work for gradient update iterations
#ifdef _OPENMP
    int nthreads;
    {
        if (params.num_threads < 0)
            nthreads = omp_get_num_procs();   // use # of CPUs
        else if (params.num_threads == 0)
            nthreads = omp_get_max_threads(); // use OMP_NUM_THREADS
        else
            nthreads = params.num_threads;
    }
    omp_set_num_threads( nthreads );
    std::cout<<" solve_ with _OPENMP and params.num_threads set to "<<params.num_threads
        <<", nthreads is "<<nthreads<<", and omp_max_threads is now "<<omp_get_max_threads()<<endl;
    int total_chunks = nthreads; // NOTE: omp_get_num_threads==1 because we are not in an omp section
    int sc_chunks = total_chunks;  // floor(sqrt(total_chunks));
    int idx_chunks = total_chunks/sc_chunks; //total_chunks;
#else
    std::cout<<" no _OPENMP";
    int idx_chunks = 1;
    int sc_chunks = 1;
#endif
    std::cout<<" idx_chunks="<<idx_chunks<<std::endl;
    std::cout<<" sc_chunks="<<sc_chunks<<std::endl;
    MutexType* sc_locks = new MutexType [sc_chunks];
    MutexType* idx_locks = new MutexType [idx_chunks];
    int sc_chunk_size = (params.class_samples?params.class_samples:noClasses)/sc_chunks;
    int sc_remaining = (params.class_samples?params.class_samples:noClasses) % sc_chunks;
    int idx_chunk_size = batch_size/idx_chunks;
    int idx_remaining = batch_size % idx_chunks;

    init_nc(nc, nclasses, y);
    if (params.optimizeLU_epoch > 0||params.reoptimize_LU)
    {
        init_wc(wc, nclasses, y, params);
    }

    maxclasses = nclasses.maxCoeff();
    //keep track of which classes have been elimninated for a particular example
    boolmatrix filtered(n,noClasses);
    unsigned long total_constraints = n*noClasses - (1-params.remove_class_constraints)*nc.sum();
    size_t no_filtered=0;
    int projection_dim = 0;
    VectorXd vect;

    // this should not be here. Params should be specified before this function is called 
    // if (params.C1<0.0 || params.C2 < 0.0){
    //    // a reasonable "auto" mode
    //    params.C2 = 1.0;
    //    params.C2 = 2.0*params.C2*noClasses;
    // }
    assert(params.C1 >= 0.0);
    assert(params.C2 >= 0.0);

    //have to do this better. Maybe divide by exp(log(params.C1)/2)??
    double lambda, C1, C2;
    lambda = 1.0/params.C2;
    C1 = params.C1/params.C2;
    C2 = 1.0;
    cout<<" begin solve_optimization, lambda="<<lambda<<" C1="<<C1<<" C2="<<C2<<endl;

    if(1) { // throw if input dims inconsistent or conflicting with params
        ostringstream err;
#define MCSOLVER_CHK( COND ) do{ if(!(COND)){err<<"\nMCsolver::solve ERROR: bad input dimensions, " #COND;}}while(0)
        MCSOLVER_CHK( x.rows() == y.rows() );   // x and y must agree on number of training examples (rows)
        MCSOLVER_CHK( y.cols() > 1 );           // noClasses >= 2
        MCSOLVER_CHK( x.cols() > 0 );           // x dim > 0
        MCSOLVER_CHK( weights.cols() == lower_bounds.cols() );   // = projections (or 0)
        MCSOLVER_CHK( weights.cols() == upper_bounds.cols() );   // = projections (or 0)
        MCSOLVER_CHK( lower_bounds.rows() == upper_bounds.rows() );   // = noClasses (or 0)
        MCSOLVER_CHK( weights_avg.cols() == lower_bounds_avg.cols() );   // = projections (or 0)
        MCSOLVER_CHK( weights_avg.cols() == upper_bounds_avg.cols() );   // = projections (or 0)
        MCSOLVER_CHK( lower_bounds_avg.rows() == upper_bounds_avg.rows() );   // = noClasses (or 0)
        MCSOLVER_CHK( weights.rows() == 0 || weights.rows() == x.cols() );
        MCSOLVER_CHK( weights_avg.rows() == 0 || weights_avg.rows() == x.cols() );
        if( weights.rows() == 0 )
            MCSOLVER_CHK( weights.rows() == 0 && lower_bounds.rows() == 0 && upper_bounds.rows() == 0 );
        if( weights_avg.rows() == 0 )
            MCSOLVER_CHK( weights_avg.rows() == 0 && lower_bounds_avg.rows() == 0 && upper_bounds_avg.rows() == 0 );
        if( err.str().size() > 0U ){
            err<<endl;
            throw runtime_error(err.str());
        }
#undef MCSOLVER_CHK
        if( weights.rows() == 0 && weights_avg.rows() == 0
            && (params.resume || params.reoptimize_LU ))
                throw("MCsolver::solve ERROR: missing all weights -- cannot possible resume or reoptimize_LU");
    }
    // missing one of FOO or FOO_avg ? copy from one zero-rows to the other
    // XXX TODO

    if (weights.cols() > no_projections)
    {
        cerr << "Warning: the number of requested filters is lower than the number of filters already learned. Dropping the extra filters" << endl;
        cerr<<"           XXX actually should only drop if tot_projections says to do so?"<<endl;
        weights.conservativeResize(d, no_projections);
        weights_avg.conservativeResize(d, no_projections);
        lower_bounds.conservativeResize(noClasses, no_projections);
        upper_bounds.conservativeResize(noClasses, no_projections);
        lower_bounds_avg.conservativeResize(noClasses, no_projections);
        upper_bounds_avg.conservativeResize(noClasses, no_projections);
    }

    if (params.reoptimize_LU)
    {
        lower_bounds.setZero(noClasses, no_projections);
        upper_bounds.setZero(noClasses, no_projections);
        lower_bounds_avg.setZero(noClasses, no_projections);
        upper_bounds_avg.setZero(noClasses, no_projections);
    }

    if (params.resume || params.reoptimize_LU)
    {
        cout<<" Continuing a run ..."<<endl;
        if(params.reoptimize_LU || params.remove_constraints)
        {
            for (projection_dim = 0; projection_dim < weights.cols(); projection_dim++)
            {
                // use weights_avg since they will hold the correct weights regardless if
                // averaging was performed on a prior run or not
                w = WeightVector(weights_avg.col(projection_dim));

                if (params.reoptimize_LU || (params.remove_constraints && projection_dim < no_projections-1)){
                    w.project(projection,x);
                }

                if (params.reoptimize_LU) {
                    switch (params.reorder_type) {
                      case REORDER_AVG_PROJ_MEANS:
                          // use the current w since averaging has not started yet
                      case REORDER_PROJ_MEANS:
                          proj_means(means, nc, projection, y);
                          break;
                      case REORDER_RANGE_MIDPOINTS:
                          // this should not work with optimizeLU since it depends on LU and LU on the reordering
                          //		      means = l+u; //no need to divide by 2 since it is only used for ordering
                          cerr << "Error, reordering " << params.reorder_type << " should not be used when reoptimizing the LU parameters" << endl;
                          exit(-1);
                          break;
                      default:
                          cerr << "Error, reordering " << params.reorder_type << " not implemented" << endl;
                          exit(-1);
                    }
                    rank_classes(sorted_class, class_order, means);

                    optimizeLU(l,u,projection,y,class_order, sorted_class, wc, nclasses, filtered, C1, C2, params);
                    lower_bounds_avg.col(projection_dim) = l;
                    upper_bounds_avg.col(projection_dim) = u;
                    // copy w, lower_bounds, upper_bounds from the coresponding averaged terms.
                    // this way we do not spend time reoptimizing LU for non-averaged terms we probably won't use.
                    // The right way to handle this would be to know whether we want to return only the averaged values or we also need the non-averaged ones.

                    w.toVectorXd(vect);
                    weights.col(projection_dim) = vect;
                    lower_bounds.col(projection_dim) = l;
                    upper_bounds.col(projection_dim) = u;
                }
                else
                {
                    l = lower_bounds_avg.col(projection_dim);
                    u = upper_bounds_avg.col(projection_dim);
                }
                // should we do this in parallel?
                // the main problem is that the bitset is not thread safe (changes to one bit can affect changes to other bits)
                // should update to use the filter class
                // things will not work correctly with remove_class_constrains on. We need to update wc, nclass
                //       and maybe nc
                // check if nclass and nc are used for anything else than weighting examples belonging
                //       to multiple classes
                if (params.remove_constraints && projection_dim < no_projections-1)
                {
                    update_filtered(filtered, projection, l, u, y, params.remove_class_constraints);
                    no_filtered = filtered.count();
                    cout << "Filtered " << no_filtered << " out of " << total_constraints << endl;
                }

                // work on this. This is just a crude approximation.
                // now every example - class pair introduces nclass(example) constraints
                // if weighting is done, the number is different
                // eliminating one example -class pair removes nclass(example) potential
                // if the class not among the classes of the example
                if (params.reweight_lambda != REWEIGHT_NONE)
                {
                    long int no_remaining = total_constraints - no_filtered;
                    lambda = no_remaining*1.0/(total_constraints*params.C2);
                    if (params.reweight_lambda == REWEIGHT_ALL)
                    {
                        C1 = params.C1*lambda;
                    }
                }
            }
        }
        projection_dim = weights.cols();
        obj_idx = objective_val.size();
        obj_idx_avg = objective_val_avg.size();
        if(1){
            cout<<" Continuing a run ... starting with weights"<<prettyDims(weights)<<":\n"<<weights<<endl;
            cout<<" Continuing a run ... starting with weights_avg"<<prettyDims(weights_avg)<<":\n"<<weights_avg<<endl;
            cout<<" Continuing a run ... beginning at projection_dim="<<projection_dim<<endl;
        }
    }
    // XXX make more robust to continued runs?
    weights.conservativeResize(d, no_projections);
    weights_avg.conservativeResize(d, no_projections);
    lower_bounds.conservativeResize(noClasses, no_projections);
    upper_bounds.conservativeResize(noClasses, no_projections);
    lower_bounds_avg.conservativeResize(noClasses, no_projections);
    upper_bounds_avg.conservativeResize(noClasses, no_projections);

    {// more space for objective_val history, per new projection...
        size_t const newProjs = (no_projections < projection_dim? 0U
                                 : no_projections - projection_dim);
        if(params.report_epoch > 0){
            size_t const nReports = params.max_iter / params.report_epoch + 1U;
            size_t const more = newProjs * (nReports+1U) + 1000U;
            objective_val    .conservativeResize(obj_idx     + more );
        }
        if(params.report_avg_epoch > 0){
            size_t const nReports = params.max_iter / params.report_avg_epoch + 1U;
            size_t const more = newProjs * (nReports+1U) + 1000U;
            objective_val_avg.conservativeResize(obj_idx_avg + more );
        }
    }

    cout << "start projection " << projection_dim << endl;
    cout.flush(); //fflush(stdout);
    for(; projection_dim < no_projections; projection_dim++)
    {
        // XXX ohoh, if reoptimize_LU, then are some things already known?
        init_w      ( w, x,y,nc, weights_avg,projection_dim);
        w.project   ( projection,x);        // project each example onto current projection dirn, w
        init_lu     ( l,u,projection, y,nc); // init l, u 

	// get the initial ordering of classes
	switch (params.reorder_type)
	  {
	  case REORDER_AVG_PROJ_MEANS:
	    // use the current w since averaging has not started yet 
	  case REORDER_PROJ_MEANS:	       
	    proj_means(means, nc, projection, y);
	    break;
	  case REORDER_RANGE_MIDPOINTS:
	    // what to do if u < l? 
	    means = l+u; //no need to divide by 2 since it is only used for ordering
	    break;
	  default:
	    cerr << "Error, reordering " << params.reorder_type << " not implemented" << endl;
	    exit(-1);	      
	  }	  
        rank_classes( sorted_class, class_order, means);
        cout << "start optimize LU" << endl; cout.flush();
        OPT_PROFILE_NEXT("optimizeLU.profile");
        if (params.optimizeLU_epoch > 0) {
            optimizeLU(l,u,projection,y,class_order, sorted_class, wc, nclasses, filtered, C1, C2, params);
        }
        cout << "end optimize LU" << endl; cout.flush();
        get_sortedLU(sortedLU, l, u, sorted_class);
        if (params.optimizeLU_epoch <= 0) {
            // we do not need the average sortedLU since we will
            // optimize the bounds at the end based on the
            // average w
            sortedLU_avg.setZero();
        }
        print_report(projection_dim,batch_size, noClasses,C1,C2,lambda,w.size(),print_report(x));
        OPT_PROFILE_NEXT("learning.profile");
        t = 0;
        if(PRINT_O){
            cout<<"objective_val[  t   ]: value    w.norm (initially "<<w.norm()<<")\n"
                <<"--------------------- -------  -------"<<endl;
        }

        while (t < params.max_iter)
        {
            ++t;
            bool const bReorder   = params.reorder_epoch    > 0 && t%params.reorder_epoch    == 0;
            bool const bReport    = params.report_epoch     > 0 && t%params.report_epoch     == 0;
            bool const bReportAvg = params.report_avg_epoch > 0 && t%params.report_avg_epoch == 0;
            bool const bOptLU     = params.optimizeLU_epoch > 0 && t%params.optimizeLU_epoch == 0;
            // print some progress
            if (!params.report_epoch && t % 1000 == 0)
            {
                snprintf(iter_str,30, "Projection %d > ", projection_dim+1);
                print_progress(iter_str, t, params.max_iter);
                cout.flush();
            }
            OPT_GRADIENT_TEST;
            eta_t = set_eta(params, t, lambda); // set eta for this iteration
            if (params.update_type == SAFE_SGD) { // compute the gradient and update
                update_safe_SGD(w, sortedLU, sortedLU_avg,
                                x, y, xSqNorms, C1, C2, lambda, t, eta_t, n,
                                nclasses, maxclasses, sorted_class, class_order, filtered,
                                sc_chunks, sc_chunk_size, sc_remaining,
                                params);
            } else if (params.update_type == MINIBATCH_SGD) {
                update_minibatch_SGD(w, sortedLU, sortedLU_avg,
                                     x, y, C1, C2, lambda, t, eta_t, n, batch_size/*<--*/,
                                     nclasses, maxclasses, sorted_class, class_order, filtered,
                                     sc_chunks, sc_chunk_size, sc_remaining,
                                     idx_chunks, idx_chunk_size, idx_remaining, //<-- new
                                     idx_locks, sc_locks,                       //<-- new
                                     params);
            }
            if( bReportAvg || (bReorder && params.reorder_type == REORDER_AVG_PROJ_MEANS))
                w.project_avg( projection_avg,x);
            if (bReport || bOptLU || (bReorder && params.reorder_type == REORDER_PROJ_MEANS))
                w.project    ( projection    ,x);
            if(bReorder){                       // reorder the classes
                // do this in a function?
                // get the current l and u in the original class order
                get_lu(l,u,sortedLU,sorted_class);
                if ( params.optimizeLU_epoch <= 0 && params.avg_epoch &&  t >= params.avg_epoch)
                    get_lu(l_avg,u_avg,sortedLU_avg,sorted_class);
                switch (params.reorder_type){
                  case REORDER_AVG_PROJ_MEANS:    // if averaging has not started yet, this defaults
                      proj_means(means, nc, projection_avg, y); // to projecting using the current w
                      break;
                  case REORDER_PROJ_MEANS:
                      proj_means(means, nc, projection, y);
                      break;
                  case REORDER_RANGE_MIDPOINTS:         // no need to divide by 2 since
                      means = l+u;                      // only used for ordering
                      break;
                  default:
                      cerr << "Error, reordering " << params.reorder_type << " not implemented" << endl;
                      exit(-1);
                }
                assert( u.size() == l.size() );
                assert( means.size() == l.size() );
                // calculate the new class order
                rank_classes(sorted_class, class_order, means); // valgrind!!
                assert( means.size() == l.size() );
                // sort the l and u in the order of the classes
                assert( static_cast<size_t>(sorted_class.size()) == static_cast<size_t>(l.size()) );
                assert( sortedLU.size() == l.size()*2U );
                get_sortedLU(sortedLU, l, u, sorted_class);
                if ( params.optimizeLU_epoch <= 0 && params.avg_epoch &&  t >= params.avg_epoch)
                {
                    // if we optimize the LU, we do not need to keep track of
                    // the averaged lower and upper bounds We optimize the
                    // bounds at the end based on the average w
                    get_sortedLU(sortedLU_avg, l_avg, u_avg, sorted_class);
                }
            }
            // optimize the lower and upper bounds (done after class ranking)
            // since it depends on the ranks. If ranking type is
            // REORDER_RANGE_MIDPOINTS, then class ranking depends on this but
            // should still be done before since it is less expensive (could
            // also be done after the LU optimization
            if(bOptLU){
                optimizeLU(l,u,projection,y,class_order, sorted_class, wc, nclasses, filtered, C1, C2, params);
                get_sortedLU(sortedLU, l, u, sorted_class);
            }
            // calculate the objective functions with respect to the current w and bounds
            if(bReport){
                // use the current w to calculate objective
                objective_val[obj_idx++] =
                    calculate_objective_hinge( projection, y, nclasses,
                                               sorted_class, class_order,
                                               w.norm(), sortedLU, filtered,
                                               lambda, C1, C2, params); // save the objective value
                if(PRINT_O) {
                    cout << "objective_val[" <<setw(6)<<t << "]: " << objective_val[obj_idx-1] << " "<< w.norm() << endl;
                }
            }
            // Calculate the objective for the averaged w.  If optimizing LU
            // then this is expensive since it runs the optimizaion
            if(bReportAvg){
                if ( params.avg_epoch && t >= params.avg_epoch) { // use avg to calculate objective
                    VectorXd sortedLU_test( l_avg.size()+u_avg.size() );
                    if (params.optimizeLU_epoch > 0) {
                        optimizeLU(l_avg, u_avg, projection_avg, y, class_order, sorted_class, wc, nclasses, filtered, C1, C2, params);
                        get_sortedLU(sortedLU_test, l_avg, u_avg, sorted_class);
                    } else {
                        sortedLU_test = sortedLU_avg/(t - params.avg_epoch + 1);
                    }
                    objective_val_avg[obj_idx_avg++] =
                        calculate_objective_hinge( projection_avg, y, nclasses,
                                                   sorted_class, class_order,
                                                   w.norm_avg(), sortedLU_test,
                                                   filtered,
                                                   lambda, C1, C2, params); // save the objective value
                } else if(bReport){
                    // the objective has just been computed for the current w, use it.
                    objective_val_avg[obj_idx_avg++] = objective_val[obj_idx - 1];
                } else {
                    // Since averaging has not started yet, compute the
                    // objective for the current w.  We can use projection_avg
                    // because if averaging has not started this is just the
                    // projection using the current w
                    objective_val_avg[obj_idx_avg++] =
                        calculate_objective_hinge( projection_avg, y, nclasses,
                                                   sorted_class, class_order,
                                                   w.norm(), sortedLU, filtered,
                                                   lambda, C1, C2, params); // save the objective value
                }
                if(PRINT_O) {
                    cout << "objective_val_avg[" << t << "]: " << objective_val_avg[obj_idx_avg-1] << " "<< w.norm_avg() << endl;
                }
            }
        } // ********* end while t ********
        OPT_PROFILE_NEXT("filtering.profile");
        // define these here just in case I got some of the conditons wrong
        VectorXd projection, projection_avg;

        if ( params.optimizeLU_epoch <= 0 || params.reorder_type == REORDER_RANGE_MIDPOINTS ){
            // get l and u back to original class ordering
            get_lu(l,u,sortedLU,sorted_class);
        }
        // Optimize LU and compute objective for averaging if it is turned on
        // If t = params.avg_epoch, everything is exactly the same as
        // just using the current w
        if ( params.avg_epoch && t > params.avg_epoch ){
            if ( params.optimizeLU_epoch <= 0){
                // get l_avg and u_avg to reflect original class ordering
                get_lu(l_avg,u_avg,sortedLU_avg/(t - params.avg_epoch + 1),sorted_class);
            }
            if (params.report_avg_epoch > 0 || params.optimizeLU_epoch > 0){
                // project all the data on the average w if needed
                w.project_avg(projection_avg,x);
            }
            if (params.reorder_epoch > 0 && (params.optimizeLU_epoch > 0 || params.report_avg_epoch > 0)){
                // Only need to reorder the classes if optimizing LU
                // or if we are interested in the last obj value.
                // Do the reordering based on the averaged w
                proj_means(means, nc, projection_avg, y);
                rank_classes(sorted_class, class_order, means); // calculate the new class order
            }
            if (params.optimizeLU_epoch > 0){
                // Optimize the lower and upper bounds (done after class
                // ranking) since it depends on the ranks
                optimizeLU(l_avg,u_avg,projection_avg,y,class_order, sorted_class, wc, nclasses, filtered, C1, C2, params);
            }
            if( params.report_avg_epoch>0 ) {
                // get the current sortedLU in case bounds or order changed
                // could test for changes!
                get_sortedLU(sortedLU_avg, l_avg, u_avg, sorted_class);
                // calculate the objective for the averaged w
                objective_val_avg[obj_idx_avg++] =
                    calculate_objective_hinge( projection_avg, y, nclasses,
                                               sorted_class, class_order,
                                               w.norm_avg(), sortedLU_avg,
                                               filtered,
                                               lambda, C1, C2, params); // save the objective value
                if(PRINT_O)
                {
                    cout << "objective_val_avg[" << t << "]: " << objective_val_avg[obj_idx_avg-1] << " "<< w.norm_avg() << endl;
                }
            }
        }
        // do everything for the current w .
        // it might be wasteful if we are not interested in the current w
        if (params.report_epoch > 0 || params.optimizeLU_epoch > 0){
            w.project(projection,x);
        }
        // only need to reorder the classes if optimizing LU
        // or if we are interested in the last obj value
        // do the reordering based on the averaged w
        if (params.reorder_epoch > 0 && (params.optimizeLU_epoch > 0 || params.report_epoch > 0))
        {
            switch (params.reorder_type){
              case REORDER_AVG_PROJ_MEANS:
                  // even if reordering is based on the averaged w
                  // do it here based on the w to get the optimal LU and
                  // the best objective with respect to w
              case REORDER_PROJ_MEANS:
                  proj_means(means, nc, projection, y);
                  break;
              case REORDER_RANGE_MIDPOINTS:
                  means = l+u; //no need to divide by 2 since it is only used for ordering
                  break;
              default:
                  cerr << "Error, reordering " << params.reorder_type << " not implemented" << endl;
                  exit(-1);
            }
            rank_classes(sorted_class, class_order, means); // calculate the new class order
        }
        // optimize the lower and upper bounds (done after class ranking)
        // since it depends on the ranks
        // if ranking type is REORDER_RANGE_MIDPOINTS, then class ranking depends on this
        // but shoul still be done before since it is less expensive
        // (could also be done after the LU optimization
        // do this for the average class
        if (params.optimizeLU_epoch > 0){
            optimizeLU(l,u,projection,y,class_order, sorted_class, wc, nclasses, filtered, C1, C2, params);
        }

        if( params.report_epoch>0 ){                    // calculate objective for current w
            // get the current sortedLU in case bounds or order changed
            get_sortedLU(sortedLU, l, u, sorted_class);
            objective_val[obj_idx++] =
                calculate_objective_hinge( projection, y, nclasses,
                                           sorted_class, class_order,
                                           w.norm(), sortedLU,
                                           filtered,
                                           lambda, C1, C2, params); // save objective value
            if(PRINT_O) cout<<"objective_val["<<setw(6)<<t <<"]: "<<objective_val[obj_idx-1]<<" "<<w.norm()<<endl;
        }

        w.toVectorXd(vect);
        weights.col(projection_dim) = vect;
        lower_bounds.col(projection_dim) = l;
        upper_bounds.col(projection_dim) = u;
        if ( params.avg_epoch && t > params.avg_epoch ){
            w.toVectorXd_avg(vect);
            weights_avg.col(projection_dim) = vect;
            lower_bounds_avg.col(projection_dim) = l_avg;
            upper_bounds_avg.col(projection_dim) = u_avg;
        }else{
            weights_avg.col(projection_dim) = vect;
            lower_bounds_avg.col(projection_dim) = l;
            upper_bounds_avg.col(projection_dim) = u;
        }

        // should we do this in parallel?
        // the main problem is that the bitset is not thread safe (changes to one bit can affect changes to other bits)
        // should update to use the filter class
        // things will not work correctly with remove_class_constrains on. We need to update wc, nclass
        //       and maybe nc
        // check if nclass and nc are used for anything else than weighting examples belonging
        //       to multiple classes
        if (params.remove_constraints && projection_dim < no_projections-1){
            // are the project[_avg]() calls here un-needed? Here w is the weight vector for just prjax
            if (params.avg_epoch && t > params.avg_epoch ){
                w.project_avg(projection_avg,x);
                update_filtered(filtered, projection_avg, l_avg, u_avg, y, params.remove_class_constraints);
            }else{
                w.project(projection,x);
                update_filtered(filtered, projection, l, u, y, params.remove_class_constraints);
            }

            no_filtered = filtered.count();
            cout << "Filtered " << no_filtered << " out of " << total_constraints << endl;
            // work on this. This is just a crude approximation.
            // now every example - class pair introduces nclass(example) constraints
            // if weighting is done, the number is different
            // eliminating one example -class pair removes nclass(exmple) potential
            // if the class not among the classes of the example
            if (params.reweight_lambda != REWEIGHT_NONE){
                long int no_remaining = total_constraints - no_filtered;
                lambda = no_remaining*1.0/(total_constraints*params.C2);
                if (params.reweight_lambda == REWEIGHT_ALL){
                    C1 = params.C1*no_remaining*1.0/(total_constraints*params.C2);
                }
                // New: test for early exit ...
                cout<<setw(20)<<tostring(params.reweight_lambda)<<": total_constraints="
                    <<total_constraints<<" minus no_filtered="<<no_filtered<<"\n"<<setw(20)
                    <<""<<"  leaving no_remaining="<<no_remaining<<" lambda="<<lambda<<" C1="<<C1<<endl;
                if( no_filtered > total_constraints )
                    throw std::runtime_error(" programmer error: removed more constraints than exist?");
                if( no_remaining == 0 ){
                    cout<<setw(20)<<""<<"  CANNOT CONTINUE, no more constraints left to satisfy"<<endl;
                    const_cast<param_struct&>(params).no_projections = projection_dim+1U;
                    OPT_PROFILE_STOP();
                    break;
                }
            }
        }
        OPT_PROFILE_STOP();
    } // end for projection_dim
    delete[]  sc_locks;  sc_locks = nullptr;
    delete[] idx_locks; idx_locks = nullptr;
}

#endif // FIND_W_HH
