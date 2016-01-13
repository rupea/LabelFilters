#ifndef MCSOLVER_HH
#define MCSOLVER_HH
/** \file
 * MCsolver::solve impl
 */
#include "find_w.h"
#include "find_w_detail.hh"
#include "constants.h"          // PRINT_O
#include "printing.h"
#include <iomanip>

    template< typename EIGENTYPE >
void MCsolver::solve( EIGENTYPE const& x, SparseMb const& y,
                     param_struct const* const params_arg /*= nullptr*/ )
{
    using namespace std;
    if( params_arg ){
        // XXX compatibility checks?
        this->parms = *params_arg;      // if present, parms OVERWRITE any old ones
    }
    param_struct const& params = this->parms;
    // Possibly read in from save file:
    if( ! params.resume ){      // NEW --- needs checking
        this->t = 0U;
        /*double*/this-> lambda = 1.0/params.C2;
        /*double*/this-> C1 = params.C1/params.C2;
        /*double*/this-> C2 = 1.0;
        //this->eta_t = params.eta; // actually, via seteta(params,t,lambda), below
    }


#ifdef PROFILE
    ProfilerStart("init.profile");
#endif

    this->nProj = params.no_projections;
    this->d = x.cols();
    cout << "nProj: " << nProj << endl;
    const size_t n = x.rows();
    const size_t batch_size = (params.batch_size < 1 || params.batch_size > n) ? (size_t) n : params.batch_size;
    if (params.update_type == SAFE_SGD)
    {
        // save_sgd update only works with batch size 1
        assert(batch_size == 1);
    }

    //std::vector<int> classes = get_classes(y);
    cout << "size x: " << x.rows() << " rows and " << x.cols() << " columns.\n";
    cout << "size y: " << y.rows() << " rows and " << y.cols() << " columns.\n";

    /*const size_t*/this-> nClass = y.cols();
    WeightVector w;
    VectorXd projection, projection_avg;
    VectorXd l(nClass),u(nClass);
    VectorXd sortedLU(2*nClass); // holds l and u interleaved in the curent class sorting order (i.e. l,u,l,u,l,u)
    //  VectorXd sortedLU_gradient(2*nClass); // used to improve cache performance
    //  VectorXd sortedLU_gradient_chunk;
    VectorXd l_avg(nClass),u_avg(nClass); // the lower and upper bounds for the averaged gradient
    VectorXd sortedLU_avg(2*nClass); // holds l_avg and u_avg interleaved in the curent class sorting order (i.e. l_avg,u_avg,l_avg,u_avg,l_avg,u_avg)
    VectorXd means(nClass); // used for initialization of the class order vector;
    int maxclasses; // the maximum number of classes an example might have
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
    std::vector<int> sorted_class(nClass), class_order(nClass);//, prev_class_order(nClass);// used as the switch
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
    int idx_chunks = total_chunks; //total_chunks/sc_chunks;
#else
    std::cout<<" no _OPENMP";
    int idx_chunks = 1;
    int sc_chunks = 1;
#endif
    std::cout<<" idx_chunks="<<idx_chunks<<std::endl;
    MutexType* sc_locks = new MutexType [sc_chunks];
    MutexType* idx_locks = new MutexType [idx_chunks];
    int sc_chunk_size = (params.class_samples?params.class_samples:nClass)/sc_chunks;
    int sc_remaining = (params.class_samples?params.class_samples:nClass) % sc_chunks;
    int idx_chunk_size = batch_size/idx_chunks;
    int idx_remaining = batch_size % idx_chunks;

    VectorXi nc;       // nc[class]         = number of training examples of each class
    VectorXi nclasses; // nclasses[example] = number of classes assigned to each training example
    init_nc(nc, nclasses, y);
    // Suppose example y[i] --> weight of 1.0, or if params.ml_wt_class_by_nclasses, 1.0/nclasses[i]
    // Then what is total weight of each class?
    VectorXd wc; // wc[class] = weight of each class (= nc[class] if params.ml_wt_class_by_nclasses==false)
    init_wc(wc, nclasses, y, params);   // wc is used if optimizeLU_epoch>0

    maxclasses = nclasses.maxCoeff();
    //keep track of which classes have been elimninated for a particular example
    boolmatrix filtered(n,nClass);
    VectorXd difference(d);
    unsigned long total_constraints = n*nClass - (1-params.remove_class_constraints)*nc.sum();
    size_t no_filtered=0;
    int projection_dim = 0;
    VectorXd vect;

    if (weights.cols() > nProj)
    {
        cerr<<"Warning: the number of requested filters is lower than the number of filters already learned."
           "\n\tDropping the extra filters" << endl;
        weights.conservativeResize(d, nProj);
        weights_avg.conservativeResize(d, nProj);
        lower_bounds.conservativeResize(nClass, nProj);
        upper_bounds.conservativeResize(nClass, nProj);
        lower_bounds_avg.conservativeResize(nClass, nProj);
        upper_bounds_avg.conservativeResize(nClass, nProj);
    }

    if (params.reoptimize_LU)
    {
        lower_bounds.setZero(nClass, nProj);
        upper_bounds.setZero(nClass, nProj);
        lower_bounds_avg.setZero(nClass, nProj);
        upper_bounds_avg.setZero(nClass, nProj);
    }

    if (params.resume || params.reoptimize_LU)
    {
        if(params.reoptimize_LU || params.remove_constraints)
        {
            for (projection_dim = 0; projection_dim < weights.cols(); projection_dim++)
            {
                // use weights_avg since they will hold the correct weights regardless if
                // averaging was performed on a prior run or not
                w = WeightVector(weights_avg.col(projection_dim));

                if (params.reoptimize_LU || (params.remove_constraints && projection_dim < nProj-1))
                {
                    w.project(projection,x);
                }

                if (params.reoptimize_LU)
                {
                    switch (params.reorder_type)
                    {
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
                    // coppy w, lower_bound, upper_bound from the coresponding averaged terms.
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
                if (params.remove_constraints && projection_dim < nProj-1)
                {
                    update_filtered(filtered, projection, l, u, y, params.remove_class_constraints);
                    no_filtered = filtered.count();
                    cout << "Filtered " << no_filtered << " out of " << total_constraints << endl;
                }

                // work on this. This is just a crude approximation.
                // now every example - class pair introduces nclass(example) constraints
                // if weighting is done, the number is different
                // eliminating one example -class pair removes nclass(exmple) potential
                // if the class not among the classes of the example
                if (params.reweight_lambda != REWEIGHT_NONE)
                {
                    long int no_remaining = total_constraints - no_filtered;
                    lambda = no_remaining*1.0/(total_constraints*params.C2);
                    if (params.reweight_lambda == REWEIGHT_ALL)
                    {
                        C1 = params.C1*no_remaining*1.0/(total_constraints*params.C2);
                    }
                }
            }
        }
        projection_dim = weights.cols();
        obj_idx = objective_val.size();
        obj_idx_avg = objective_val_avg.size();
    }
    // XXX make more robust to continued runs?
    weights.conservativeResize(d, nProj);
    weights_avg.conservativeResize(d, nProj);
    lower_bounds.conservativeResize(nClass, nProj);
    upper_bounds.conservativeResize(nClass, nProj);
    lower_bounds_avg.conservativeResize(nClass, nProj);
    upper_bounds_avg.conservativeResize(nClass, nProj);
    {// more space for objective_val history, per new projection...
        size_t const newProjs = (nProj < projection_dim? 0U
                                 : nProj - projection_dim);
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

    cout << "start projection " << projection_dim << endl; cout.flush();
    for(; projection_dim < nProj; projection_dim++)
    {
        init_w      (w, x,y,nc, weights_avg,projection_dim);
        w.project   (projection,x);        // project each example onto current projection dirn, w
        init_lu     (l,u,means, params.reorder_type,projection,y,nc); // init l, u and means
        rank_classes(sorted_class, class_order, means);
        cout << "start optimize LU" << endl; cout.flush();
#ifdef PROFILE
        ProfilerStop();
        ProfilerStart("optimizeLU.profile");
#endif
        if (params.optimizeLU_epoch > 0) {
            optimizeLU( l,u,projection,y,class_order, sorted_class, wc,
                        nclasses, filtered, C1, C2, params);
        }else{
            // we do not need the average sortedLU since we will
            // optimize the bounds at the end based on the
            // average w
            sortedLU_avg.setZero();
        }
        get_sortedLU(sortedLU, l, u, sorted_class);
        cout << "end optimize LU" << endl; cout.flush();
#ifdef PROFILE
        ProfilerStop();
#endif
        print_report(projection_dim,batch_size, nClass,C1,C2,lambda,w.size(),print_report(x));
#ifdef PROFILE
        ProfilerStart("learning.profile");
#endif
        t = 0;        // part of MCsoln data -- TODO -- continue iterating
        if(PRINT_O){
            cout<<"objective_val[  t   ]: value    w.norm\n"
                <<"--------------------- -------  -------"<<endl;
        }
        // -------- main iteration loop --------
        while (t < params.max_iter) {
            ++t;
            // 't' is "Time For" ... various things
            // t --> bool t4_XXX = (params.XXX_epoch && t%params.XXX_epoch==0)
#define MCSOLVER_TIME_FOR(XXX) bool const t4_##XXX = params.XXX##_epoch && t % params.XXX##_epoch == 0
            MCSOLVER_TIME_FOR( reorder );               // defines bool t4_reorder
            MCSOLVER_TIME_FOR( report );
            MCSOLVER_TIME_FOR( report_avg );
            MCSOLVER_TIME_FOR( optimizeLU );
            MCSOLVER_TIME_FOR( finite_diff_test );
#undef MCSOLVER_TIME_FOR
            // print some progress
            if (!params.report_epoch && t % 1000 == 0) {
                snprintf(iter_str,30, "Projection %d > ", projection_dim+1);
                print_progress(iter_str, t, params.max_iter);
                cout.flush();
            }
            // perform finite differences test
            if(t4_finite_diff_test){
                for (size_t fdtest=0; fdtest<params.no_finite_diff_tests; ++fdtest) {
                    size_t idx = ((size_t) rand()) % n;
                    finite_diff_test( w, x, idx, y, nclasses, maxclasses, sorted_class,
                                      class_order, sortedLU, filtered, C1, C2, params);
                }
            }
            // set eta for this iteration
            eta_t = set_eta(params, t, lambda);
            // compute the gradient and update
            if (params.update_type == SAFE_SGD) {
                update_safe_SGD(w, sortedLU, sortedLU_avg,
                                x, y, C1, C2, lambda, t, eta_t, n,
                                nclasses, maxclasses, sorted_class, class_order, filtered,
                                sc_chunks, sc_chunk_size, sc_remaining,
                                params);
            } else if (params.update_type == MINIBATCH_SGD) {
                update_minibatch_SGD(w, sortedLU, sortedLU_avg,
                                     x, y, C1, C2, lambda, t, eta_t, n, batch_size,
                                     nclasses, maxclasses, sorted_class, class_order, filtered,
                                     sc_chunks, sc_chunk_size, sc_remaining,
                                     idx_chunks, idx_chunk_size, idx_remaining,
                                     idx_locks, sc_locks,
                                     params);
            }
            if (t4_report_avg || (t4_reorder && params.reorder_type == REORDER_AVG_PROJ_MEANS)){
                w.project_avg(projection_avg, x);
            }
            if (t4_report || t4_optimizeLU || (t4_reorder && params.reorder_type == REORDER_PROJ_MEANS)){
                w.project    (projection    , x);
            }
            // reorder the classes
            if(t4_reorder){
                // do this in a function?
                // get the current l and u in the original class order
                get_lu(l,u,sortedLU,sorted_class);
                if ( params.optimizeLU_epoch <= 0 && params.avg_epoch &&  t >= params.avg_epoch){
                    get_lu(l_avg,u_avg,sortedLU_avg,sorted_class);
                }
                switch (params.reorder_type){
                  case REORDER_AVG_PROJ_MEANS:
                      // if averaging has not started yet, this defaults projecting
                      // using the current w
                      proj_means(means, nc, projection_avg, y);
                      break;
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
                // calculate the new class order
                rank_classes(sorted_class, class_order, means);
                // sort the l and u in the order of the classes
                get_sortedLU(sortedLU, l, u, sorted_class);
                if ( params.optimizeLU_epoch <= 0 && params.avg_epoch &&  t >= params.avg_epoch)
                {
                    // if we optimize the LU, we do not need to
                    // keep track of the averaged lower and upper bounds
                    // We optimize the bounds at the end based on the
                    // average w
                    get_sortedLU(sortedLU_avg, l_avg, u_avg, sorted_class);
                }
            }

            // optimize the lower and upper bounds (done after class ranking)
            // since it depends on the ranks
            // if ranking type is REORDER_RANGE_MIDPOINTS, then class ranking depends on this
            // but shoul still be done before since it is less expensive
            // (could also be done after the LU optimization
            if (t4_optimizeLU){
                optimizeLU( l,u,projection,y,class_order, sorted_class, wc, nclasses,
                            filtered, C1, C2, params);
                get_sortedLU(sortedLU, l, u, sorted_class);
            }
            // calculate the objective functions with respect to the current w and bounds
            if(t4_report){ // params.report_epoch && (t % params.report_epoch == 0) )
                // use the current w to calculate objective
                objective_val[obj_idx++] =
                    calculate_objective_hinge( projection, y, nclasses,
                                               sorted_class, class_order,
                                               w.norm(), sortedLU, filtered,
                                               lambda, C1, C2, params); // save the objective value
                if(PRINT_O) {
                    cout<<"objective_val["<<setw(6)<<t<<"]: "<<setw(15)<<objective_val[obj_idx-1]
                        << " "<< w.norm() << endl;
                }
            }
            // calculate the objective for the averaged w
            // if optimizing LU then this is expensive since
            // it runs the optimizaion
            if(t4_report_avg){
                if ( params.avg_epoch && t >= params.avg_epoch) {
                    // use the average to calculate objective
                    VectorXd sortedLU_test;
                    if (params.optimizeLU_epoch > 0) {
                        optimizeLU( l_avg, u_avg, projection_avg, y, class_order, sorted_class,
                                    wc, nclasses, filtered, C1, C2, params);
                        get_sortedLU( sortedLU_test, l_avg, u_avg, sorted_class);
                    } else {
                        sortedLU_test = sortedLU_avg/(t - params.avg_epoch + 1);
                    }
                    objective_val_avg[obj_idx_avg++] =
                        calculate_objective_hinge( projection_avg, y, nclasses,
                                                   sorted_class, class_order,
                                                   w.norm_avg(), sortedLU_test,
                                                   filtered,
                                                   lambda, C1, C2, params); // save the objective value
                } else if(t4_report){
                        // the objective has just been computed for the current w, use it.
                        objective_val_avg[obj_idx_avg++] = objective_val[obj_idx - 1];
                } else {
                    // since averaging has not started yet, compute the objective for
                    // the current w.
                    // we can use projection_avg because if averaging has not started
                    // this is just the projection using the current w
                    objective_val_avg[obj_idx_avg++] =
                        calculate_objective_hinge( projection_avg, y, nclasses,
                                                   sorted_class, class_order,
                                                   w.norm(), sortedLU, filtered,
                                                   lambda, C1, C2, params); // save the objective value
                }
                if(PRINT_O) {
                    cout<<"objective_val_avg[" << t << "]: " << objective_val_avg[obj_idx_avg-1] << " "<< w.norm_avg() << endl;
                }
            }


        } // end while t
#ifdef PROFILE
        ProfilerStop();
        ProfilerStart("filtering.profile");
#endif

        // define these here just in case I got some of the conditons wrong
        VectorXd projection, projection_avg;

        // get l and u if needed
        // have to do this here because class order might change
        if ( params.optimizeLU_epoch <= 0 || params.reorder_type == REORDER_RANGE_MIDPOINTS ) {
            get_lu(l,u,sortedLU,sorted_class);
        }

        // optimize LU and compute objective for averaging if it is turned on
        // if t = params.avg_epoch, everything is exactly the same as
        // just using the current w
        if ( params.avg_epoch && t > params.avg_epoch ) {
            // get the current l_avg and u_avg if needed
            if ( params.optimizeLU_epoch <= 0) {
                get_lu(l_avg,u_avg,sortedLU_avg/(t - params.avg_epoch + 1),sorted_class);
            }

            // project all the data on the average w if needed
            if (params.report_avg_epoch > 0 || params.optimizeLU_epoch > 0) {
                w.project_avg(projection_avg,x);
            }
            // only need to reorder the classes if optimizing LU
            // or if we are interested in the last obj value
            // do the reordering based on the averaged w
            if (params.reorder_epoch > 0
                && (params.optimizeLU_epoch > 0 || params.report_avg_epoch > 0))
            {
                proj_means(means, nc, projection_avg, y);
                // calculate the new class order
                rank_classes(sorted_class, class_order, means);
            }

            // optimize the lower and upper bounds (done after class ranking)
            // since it depends on the ranks
            if (params.optimizeLU_epoch > 0)
            {
                optimizeLU( l_avg,u_avg,projection_avg,y,class_order, sorted_class, wc,
                            nclasses, filtered, C1, C2, params);
            }

            // calculate the objective for the averaged w
            if( params.report_avg_epoch>0 )
            {
                // get the current sortedLU in case bounds or order changed
                // could test for changes!
                get_sortedLU(sortedLU_avg, l_avg, u_avg, sorted_class);
                objective_val_avg[obj_idx_avg++] =
                    calculate_objective_hinge( projection_avg, y, nclasses,
                                               sorted_class, class_order,
                                               w.norm_avg(), sortedLU_avg,
                                               filtered,
                                               lambda, C1, C2, params); // save the objective value
                if(PRINT_O) {
                    cout << "objective_val_avg[" << t << "]: " << objective_val_avg[obj_idx_avg-1] << " "<< w.norm_avg() << endl;
                }
            }
        }



        // do everything for the current w .
        // it might be wasteful if we are not interested in the current w
        if (params.report_epoch > 0 || params.optimizeLU_epoch > 0) {
            w.project(projection,x);
        }
        // only need to reorder the classes if optimizing LU
        // or if we are interested in the last obj value
        // do the reordering based on the averaged w
        if (params.reorder_epoch > 0 && (params.optimizeLU_epoch > 0 || params.report_epoch > 0))
        {
            switch (params.reorder_type)
            {
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
            // calculate the new class order
            rank_classes(sorted_class, class_order, means);
        }

        // optimize the lower and upper bounds (done after class ranking)
        // since it depends on the ranks
        // if ranking type is REORDER_RANGE_MIDPOINTS, then class ranking depends on this
        // but shoul still be done before since it is less expensive
        // (could also be done after the LU optimization
        // do this for the average class
        if (params.optimizeLU_epoch > 0) {
            optimizeLU(l,u,projection,y,class_order, sorted_class, wc, nclasses, filtered, C1, C2, params);
        }

        // calculate the objective for the current w
        if( params.report_epoch>0 )
        {
            // get the current sortedLU in case bounds or order changed
            get_sortedLU(sortedLU, l, u, sorted_class);
            objective_val[obj_idx++] =
                calculate_objective_hinge( projection, y, nclasses,
                                           sorted_class, class_order,
                                           w.norm(), sortedLU,
                                           filtered,
                                           lambda, C1, C2, params); // save the objective value
            if(PRINT_O)
            {
                cout << "objective_val[" <<setw(6)<<t << "]: " << objective_val[obj_idx-1] << " "<< w.norm() << endl;
            }
        }

        w.toVectorXd(vect);
        weights.col(projection_dim) = vect;
        lower_bounds.col(projection_dim) = l;
        upper_bounds.col(projection_dim) = u;
        if ( params.avg_epoch && t > params.avg_epoch )
        {
            w.toVectorXd_avg(vect);
            weights_avg.col(projection_dim) = vect;
            lower_bounds_avg.col(projection_dim) = l_avg;
            upper_bounds_avg.col(projection_dim) = u_avg;
        }
        else
        {
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
        if (params.remove_constraints && projection_dim < nProj-1)
        {
            if (params.avg_epoch && t > params.avg_epoch )
            {
                w.project_avg(projection_avg,x); // could eliminate this since it has most likely been calculated above, but we keep it here for now for clarity
                update_filtered(filtered, projection_avg, l_avg, u_avg, y, params.remove_class_constraints);
            }
            else
            {
                w.project(projection,x); // could eliminate this since it has most likely been calculated above, but we keep it here for now for clarity
                update_filtered(filtered, projection, l, u, y, params.remove_class_constraints);
            }

            no_filtered = filtered.count();
            cout << "Filtered " << no_filtered << " out of " << total_constraints << endl;
            // work on this. This is just a crude approximation.
            // now every example - class pair introduces nclass(example) constraints
            // if weighting is done, the number is different
            // eliminating one example -class pair removes nclass(exmple) potential
            // if the class not among the classes of the example
            if (params.reweight_lambda != REWEIGHT_NONE)
            {
                long int no_remaining = total_constraints - no_filtered;
                lambda = no_remaining*1.0/(total_constraints*params.C2);
                if (params.reweight_lambda == REWEIGHT_ALL)
                {
                    C1 = params.C1*no_remaining*1.0/(total_constraints*params.C2);
                }
            }
        }

        //      C2*=((n-1)*nClass)*1.0/no_remaining;
        //C1*=((n-1)*nClass)*1.0/no_remaining;

#ifdef PROFILE
        ProfilerStop();
#endif

    } // end for projection_dim
}
#endif //MCSOLVER_HH
