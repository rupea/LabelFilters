#ifndef MCSOLVER_HH
#define MCSOLVER_HH
/** \file
 * MCsolver::solve impl
 */
//#include "find_w.h"
#include "mcsolver.h"
#include "find_w_detail.hh"
#include "mcupdate.h"
#include "constants.h"          // PRINT_O
#include "printing.h"
#include <iomanip>
#include <vector>

// first let's define mcsolver.h utilities...

#define MCITER_PERIODIC(XXX) XXX( params.XXX##_epoch && t % params.XXX##_epoch == 0 )
    inline MCiterBools::MCiterBools( uint64_t const t, param_struct const& params )
    : MCITER_PERIODIC( reorder )
    , MCITER_PERIODIC( report )
    , MCITER_PERIODIC( report_avg )
    , MCITER_PERIODIC( optimizeLU )
    , MCITER_PERIODIC( finite_diff_test )
      , doing_avg_epoch( params.avg_epoch && t >= params.avg_epoch )    // <-- true after a certain 't'
{
}
#undef MCITER_PERIODIC

inline void Perm::rank( VectorXd const& sortkey ){
    assert( (size_t)sortkey.size() == perm.size() );
    std::iota( perm.begin(), perm.end(), size_t{0U} );
    std::sort( perm.begin(), perm.end(), [&sortkey](size_t const i, size_t const j){return sortkey[i]<sortkey[j];} );
    for(size_t i=0U; i<perm.size(); ++i)
        rev[perm[i]] = i;
}
inline void MCpermState::toLu( VectorXd & ll, VectorXd & uu, VectorXd const& sorted ){
    double const* plu = sorted.data();
    auto pPerm = perm.begin(); // perm ~ old perm
    for(; pPerm != perm.end(); ++pPerm){
        size_t const cp = *pPerm;
        ll.coeffRef(cp) = *(plu++);
        uu.coeffRef(cp) = *(plu++);
    }
}
inline void MCpermState::toSorted( VectorXd & sorted, VectorXd const& ll, VectorXd const& uu ){
    for(size_t i=0; i<perm.size(); ++i){
        sorted.coeffRef(2U*i)    = l.coeff(perm[i]);
        sorted.coeffRef(2U*i+1U) = u.coeff(perm[i]);
    }
}
inline void MCpermState::toSorted( VectorXd & sorted, VectorXd const& ll, VectorXd const& uu ) const{
    for(size_t i=0; i<perm.size(); ++i){
        sorted.coeffRef(2U*i)    = l.coeff(perm[i]);
        sorted.coeffRef(2U*i+1U) = u.coeff(perm[i]);
    }
}
inline void MCpermState::chg_sortlu(){
    //assert( ok_sortlu ); // oh it is ok to call this multiple times
    ok_lu = false;
}
inline void MCpermState::chg_sortlu_avg(){      // multiple calls OK
    ok_lu_avg = false;
}
inline void MCpermState::chg_lu(){              // multiple calls OK
    ok_sortlu = false;
}
inline void MCpermState::chg_lu_avg(){ // only called from optimizeLU_avg
    ok_lu_avg = true;
    // ok_sortlu_avg = false; // The accumulator value is NOT affected.
}

inline void MCpermState::mkok_lu(){
    if(!ok_lu && ok_sortlu){
        toLu( l, u, sortlu );
        ok_lu = true;
    }
    assert( ok_lu == true );
}
inline void MCpermState::mkok_lu_avg(){
    if(!ok_lu_avg && ok_sortlu_avg){
        toLu( l_avg, u_avg, sortlu_avg );
        ok_lu_avg = true;
    }
    // ok_lu_avg MAY be false -- should assert if it's required
}
inline VectorXd& MCpermState::mkok_sortlu(){
    if(!ok_sortlu){
        assert(ok_lu);
        toSorted( l, u, sortlu );
        ok_sortlu = true;
    }
    return sortlu;
}
inline VectorXd& MCpermState::mkok_sortlu_avg(){
    if(!ok_sortlu_avg && ok_lu_avg){
        toSorted( l_avg, u_avg, sortlu_avg );
        ok_sortlu_avg = true;
    }
    return sortlu_avg;
}
inline void MCpermState::rank( VectorXd& sortkey ){
    mkok_lu();
    //mkok_lu_avg();
    Perm::rank( sortkey );
    ok_sortlu = false;
    if( nAccSortlu_avg ) ok_sortlu_avg = false;
}
inline void MCpermState::getSortlu_avg( VectorXd& sortlu_test ) const
{
    if( ok_sortlu_avg ){
        sortlu_test = sortlu_avg;
    }else{
        assert( ok_lu_avg );
        toSorted( sortlu_test, l_avg, u_avg );
    }
}

// upgrade path via macros for oft-called functions...
#define ITER_TRACE 0
#define W_HAS_CHANGED do { \
    ok_projection = false; \
    ok_projection_avg = false; \
}while(0)
#define MKOK_PROJECTION do { \
    if( ! ok_projection     ){ w.project    (projection    , x); ok_projection     = true; } \
}while(0)
#define MKOK_PROJECTION_AVG do { \
    if( ! ok_projection_avg ){ w.project_avg(projection_avg, x); ok_projection_avg = true; } \
}while(0)

#if MCPRM>=3
#define OPTIMIZE_LU( MSG ) do { \
    if(ITER_TRACE){ std::cout<<MSG; std::cout.flush(); } \
    /** -- no -- luPerm l/u/sortlu prerequisites : outputs l, u */ \
    MKOK_PROJECTION; \
    assert( ok_projection ); \
    luPerm.optimizeLU(projection, y, wc, nclasses, filtered, C1, C2, params); \
    assert( luPerm.ok_lu ); assert( ! luPerm.ok_sortlu ); \
}while(0)
#define OPTIMIZE_LU_AVG( MSG ) do { \
    if(ITER_TRACE){ std::cout<<MSG; std::cout.flush(); } \
    MKOK_PROJECTION_AVG; \
    /* -- no -- luPerm prerequisites, but sets {l,u}_avg instead of {l,u} */ \
    /* sortlu_avg is still some slightly stale accumulated version, but we can't change it */ \
    assert( ok_projection_avg ); \
    luPerm.optimizeLU_avg(projection_avg, y, wc, nclasses, filtered, C1, C2, params); \
    assert( luPerm.ok_lu_avg ); \
    if( luPerm.nAccSortlu_avg ) luPerm.ok_sortlu_avg=false; \
}while(0)
#endif

#if MCPRM>=3
#define GET_LU do { \
    luPerm.mkok_sortlu(); \
    assert( luPerm.ok_sortlu ); \
    luPerm.mkok_lu(); \
    assert( luPerm.ok_lu ); \
}while(0)
#define GET_LU_AVG do { \
    luPerm.mkok_lu_avg(); \
    assert( luPerm.ok_lu_avg ); \
    /* in principle, might get (0,0)(0,0)... for {l,u}_avg if update never accumulated avgs */ \
    get_lu( l_avg,u_avg, sortedLU,sorted_class ); \
    assert( luPerm.ok_lu ); \
}while(0)
#endif

#if MCPRM>=3
#define GRADIENT_UPDATE do { \
            /* cout<<" t="<<t<<" before 'update', luPerm.ok_sortlu="<<luPerm.ok_sortlu<<endl; */ \
            luPerm.mkok_sortlu(); \
            assert( luPerm.ok_sortlu ); \
            MCupdate::update(w, luPerm,                                 /*sortedLU, sortedLU_avg,*/ \
                             x, y, C1, C2, lambda, t, eta_t, nTrain,    /*batch_size,*/ \
                             nclasses, maxclasses, /*sorted_class, class_order,*/ filtered, \
                             /* sc_chunks, sc_chunk_size, sc_remaining,*/ \
                             /* idx_chunks, idx_chunk_size, idx_remaining,*/ \
                             /* idx_locks, sc_locks,*/ \
                             updateSettings, \
                             params); \
}while(0)
#endif

#if MCPRM>=3
#define GET_SORTEDLU do { \
    assert( luPerm.ok_lu ); \
    get_sortedLU(sortedLU, l, u, sorted_class); /* l, u --> sortedLU */ \
    luPerm.ok_sortlu = true; \
}while(0)
/** \post BOTH ok_lu_avg and ok_sortlu_avg true. */
#define GET_SORTEDLU_AVG do { \
    luPerm.mkok_lu_avg(); \
    assert( luPerm.ok_lu_avg ); \
    get_sortedLU(sortedLU_avg, l_avg, u_avg, sorted_class); /* l, u --> sortedLU */ \
    luPerm.ok_sortlu_avg = true; \
}while(0)
#endif
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
    cout << " mcsolver.hh, solve_optimization: nProj: "<<nProj << endl;
    if( (size_t)nProj >= d ){
        cout<<"WARNING: no_projections > example dimensionality"<<endl;
    }
    const size_t nTrain = x.rows();

    //std::vector<int> classes = get_classes(y);
    cout << "size x: " << x.rows() << " rows and " << x.cols() << " columns.\n";
    cout << "size y: " << y.rows() << " rows and " << y.cols() << " columns.\n";

    /*const size_t*/this-> nClass = y.cols();
    WeightVector w;
    VectorXd projection, projection_avg;
    bool ok_projection = false;     // whenever w changes (init or gradient update)
    bool ok_projection_avg = false; // ok_projection* invalidated.
#if MCPRM==3
    MCpermState luPerm( nClass );
    VectorXd & l                = luPerm.l;
    VectorXd & u                = luPerm.u;
    VectorXd & sortedLU         = luPerm.sortlu;
    VectorXd & l_avg            = luPerm.l_avg;
    VectorXd & u_avg            = luPerm.u_avg;
    VectorXd & sortedLU_avg     = luPerm.sortlu_avg;
    std::vector<int> & sorted_class = luPerm.perm;      // original class numbering --> sortedLU index
    std::vector<int> & class_order  = luPerm.rev;       // reverse permutation
#else // begin eliminating variables
    MCpermState luPerm( nClass );
    VectorXd & l                = luPerm.l;
    VectorXd & u                = luPerm.u;
    VectorXd & sortedLU         = luPerm.sortlu;
    VectorXd & l_avg            = luPerm.l_avg;
    VectorXd & u_avg            = luPerm.u_avg;
    VectorXd & sortedLU_avg     = luPerm.sortlu_avg;
    std::vector<int> & sorted_class = luPerm.perm;
    std::vector<int> & class_order  = luPerm.rev;
#endif
    VectorXd means(nClass); // used for initialization of the class order vector;
    int maxclasses; // the maximum number of classes an example might have
    size_t obj_idx = 0, obj_idx_avg = 0;
    // in the multilabel case each example will have an impact proportinal
    // to the number of classes it belongs to. ml_wt and ml_wt_class
    // allows weighting that impact when updating params for the other classes
    // respectively its own class.
    //  size_t  i=0, idx=0;
    unsigned long t = 1;
    char iter_str[30];

    // how to split the work for gradient update iterations
    int const nThreads = this->getNthreads( params );
    MCupdateChunking updateSettings( nTrain/*x.rows()*/, nClass, nThreads, params );
#ifndef NDEBUG
    {// in debug mode check no change from original settings
        const size_t batch_size = (params.batch_size < 1 || params.batch_size > nTrain) ? (size_t) nTrain : params.batch_size;
        if (params.update_type == SAFE_SGD)
        {
            // save_sgd update only works with batch size 1
            assert(batch_size == 1);
        }
        cout<<" batch_size = "<<batch_size<<endl;
        int const idx_chunks = nThreads;
        int const idx_chunk_size = batch_size/idx_chunks;
        int const idx_remaining = batch_size % idx_chunks;
        int const sc_chunks = nThreads;
        std::cout<<" idx_chunks="<<idx_chunks<<std::endl;
        int const sc_chunk_size = (params.class_samples?params.class_samples:nClass)/sc_chunks;
        int const sc_remaining = (params.class_samples?params.class_samples:nClass) % sc_chunks;
        assert( idx_chunks == updateSettings.idx_chunks );
        assert( idx_chunk_size == updateSettings.idx_chunk_size );
        assert( idx_remaining == updateSettings.idx_remaining );
        assert( sc_chunks == updateSettings.sc_chunks );
        assert( sc_chunk_size == updateSettings.sc_chunk_size );
        assert( sc_remaining == updateSettings.sc_remaining );
        //MutexType* sc_locks = new MutexType [sc_chunks];
        //MutexType* idx_locks = new MutexType [idx_chunks];
        assert( updateSettings.idx_locks != nullptr );
        assert( updateSettings.sc_locks != nullptr );
    }
#endif
    size_t const& batch_size    = updateSettings.batch_size; // really only for print_report ?

    VectorXi nc;       // nc[class]         = number of training examples of each class
    VectorXi nclasses; // nclasses[example] = number of classes assigned to each training example
    init_nc(nc, nclasses, y);
    // Suppose example y[i] --> weight of 1.0, or if params.ml_wt_class_by_nclasses, 1.0/nclasses[i]
    // Then what is total weight of each class?
    VectorXd wc; // wc[class] = weight of each class (= nc[class] if params.ml_wt_class_by_nclasses==false) ONLY FOR optimizeLU
    init_wc(wc, nclasses, y, params);   // wc is used if optimizeLU_epoch>0

    maxclasses = nclasses.maxCoeff();
    //keep track of which classes have been elimninated for a particular example
    boolmatrix filtered(nTrain,nClass);
    VectorXd difference(d);
    unsigned long total_constraints = nTrain*nClass - (1-params.remove_class_constraints)*nc.sum();
    size_t no_filtered=0;
    int projection_dim = 0;
    VectorXd vect;

    if (weights.cols() > nProj) {
        cerr<<"Warning: the number of requested filters is lower than the number of filters already learned."
           "\n\tDropping the extra filters" << endl;
        weights.conservativeResize(d, nProj);
        weights_avg.conservativeResize(d, nProj);
        lower_bounds.conservativeResize(nClass, nProj);
        upper_bounds.conservativeResize(nClass, nProj);
        lower_bounds_avg.conservativeResize(nClass, nProj);
        upper_bounds_avg.conservativeResize(nClass, nProj);
    }
    if (params.reoptimize_LU) {
        lower_bounds.setZero(nClass, nProj);
        upper_bounds.setZero(nClass, nProj);
        lower_bounds_avg.setZero(nClass, nProj);
        upper_bounds_avg.setZero(nClass, nProj);
    }

    if (params.resume || params.reoptimize_LU) {
        if(params.reoptimize_LU || params.remove_constraints) {
            for (projection_dim = 0; projection_dim < weights.cols(); projection_dim++) {
                // use weights_avg since they will hold the correct weights regardless if
                // averaging was performed on a prior run or not
                w = WeightVector(weights_avg.col(projection_dim));
                W_HAS_CHANGED;  // projections of 'x' onto 'w' no longer valid

                if (params.reoptimize_LU) {
                    switch (params.reorder_type) {
                      case REORDER_AVG_PROJ_MEANS:
                          // use the current w since averaging has not started yet
                          std::cout<<" ***CHECKME-FALLTHROUGH***"<<endl;
                          assert( luPerm.nAccSortlu_avg == 0 );
                      case REORDER_PROJ_MEANS:
                          MKOK_PROJECTION;
                          proj_means(means, nc, projection, y);
                          break;
                      case REORDER_RANGE_MIDPOINTS:
                          // this should not work with optimizeLU since it depends on LU and LU on the reordering
                          // means = l+u; //no need to divide by 2 since it is only used for ordering
                          cerr << "Error, reordering " << params.reorder_type << " should not be used when reoptimizing the LU parameters" << endl;
                          exit(-1);
                          break;
                      default:
                          cerr << "Error, reordering " << params.reorder_type << " not implemented" << endl;
                          exit(-1);
                    }
                    rank_classes(sorted_class, class_order, means);
                    //luPerm.ok_sortlu = false; // ok_sortlu_avg accum still ok because it begins all-zero
                    luPerm.ok_sortlu = false; if( luPerm.nAccSortlu_avg ) luPerm.ok_sortlu_avg=false;
                    OPTIMIZE_LU(" OPTLU-Init");         // OPTIMIZE_LU(w,projection,sort_order) ----> luPerm.{l,u}
                    // copy w, lower_bound, upper_bound from the coresponding averaged terms.
                    // this way we do not spend time reoptimizing LU for non-averaged terms we probably won't use.
                    // The right way to handle this would be to know whether we want to return
                    // only the averaged values or we also need the non-averaged ones.
                    w.toVectorXd(vect);
                    weights.col(projection_dim) = vect;
                    lower_bounds.col(projection_dim) = luPerm.l;
                    upper_bounds.col(projection_dim) = luPerm.u;
                    //
                    lower_bounds_avg.col(projection_dim) = luPerm.l;
                    upper_bounds_avg.col(projection_dim) = luPerm.u;
                }else{
                    luPerm.set_lu( lower_bounds_avg.col(projection_dim), upper_bounds_avg.col(projection_dim) );
                }
                // should we do this in parallel?
                // the main problem is that the bitset is not thread safe (changes to one bit can affect changes to other bits)
                // should update to use the filter class
                // things will not work correctly with remove_class_constrains on. We need to update wc, nclass
                //       and maybe nc
                // check if nclass and nc are used for anything else than weighting examples belonging
                //       to multiple classes
                if (params.remove_constraints && projection_dim < (int)nProj-1) {
                    update_filtered(filtered, projection, l, u, y, params.remove_class_constraints);
                    no_filtered = filtered.count();
                    cout << "Filtered " << no_filtered << " out of " << total_constraints << endl;
                }

                // work on this. This is just a crude approximation.
                // now every example - class pair introduces nclass(example) constraints
                // if weighting is done, the number is different
                // eliminating one example -class pair removes nclass(exmple) potential
                // if the class not among the classes of the example
                if (params.reweight_lambda != REWEIGHT_NONE) {
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
        size_t const newProjs = (nProj < (size_t)projection_dim? 0U
                                 : nProj - (size_t)projection_dim);
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

    for(; projection_dim < (int)nProj; projection_dim++)
    {
        init_w      (w, x,y,nc, weights_avg,projection_dim);
        cout << "start projection "<<projection_dim<<" w.norm="<<w.norm()<<endl;
        W_HAS_CHANGED;  // ok_projection* --> false

        MKOK_PROJECTION;                  // guarantee valid 'projection' of 'x' [every example] onto 'w'
        luPerm.init( projection, y, nc );
        assert( luPerm.ok_lu );
        assert( luPerm.ok_sortlu_avg );   // this is an accumulator, all-zero is the actual valid initial state
        assert( y.cols() == luPerm.l.size() );
        if( params.reorder_type == REORDER_RANGE_MIDPOINTS ) means = luPerm.l + luPerm.u;
        else /* we have no *_avg yet, so use l,u          */ proj_means( means,  nc,projection,y );  
        cout<<" ranking..."; cout.flush();
        rank_classes(sorted_class, class_order, means);
        luPerm.ok_sortlu = false;
        assert(!luPerm.ok_lu_avg);
        assert(luPerm.ok_sortlu_avg); // ok_sortlu_avg remains true (all zeroes, iteration has not begun yet)
        assert(luPerm.nAccSortlu_avg==0U);
        //if( luPerm.nAccSortlu_avg ) luPerm.ok_sortlu_avg=false;
        cout<<"start optimize LU" << endl; cout.flush();
#ifdef PROFILE
        ProfilerStop();
        ProfilerStart("optimizeLU.profile");
#endif
        if (params.optimizeLU_epoch > 0) { // questionable.. l,u at this point are random, so why do it this early?
            OPTIMIZE_LU(" PROJ_INIT");
        }
        GET_SORTEDLU;
        cout << "end optimize LU" << endl; cout.flush();
#ifdef PROFILE
        ProfilerStop();
#endif
        print_report(projection_dim,batch_size, nClass,C1,C2,lambda,w.size(),print_report(x));
        t = 0;        // part of MCsoln data -- TODO -- continue iterating
        if(PRINT_O){
            cout<<" t=0 luPerm.{ok_lu="<<luPerm.ok_lu<<",ok_sortlu="<<luPerm.ok_sortlu
                <<",  ok_lu_avg="<<luPerm.ok_lu_avg<<",ok_sortlu_avg="<<luPerm.ok_sortlu_avg<<"}"<<endl;
            cout<<"objective_val[  t   ]: value    w.norm (initially "<<w.norm()<<")\n"
                <<"--------------------- -------  -------"<<endl;

        }
#ifdef PROFILE
        ProfilerStart("learning.profile");
#endif
        cout<<" start iterations:"
            <<" * ok_projection="<<ok_projection<<" ok_projection_avg="<<ok_projection_avg
            <<" * luPerm ok_lu="<<luPerm.ok_lu<<" ok_sortlu="<<luPerm.ok_sortlu
            <<" * luPerm ok_lu_avg="<<luPerm.ok_lu_avg<<" ok_sortlu_avg="<<luPerm.ok_sortlu_avg<<" nAccSortlu_avg="<<luPerm.nAccSortlu_avg
            <<endl;
        // -------- main iteration loop --------
        while (t < params.max_iter) {
            ++t;
            MCiterBools t4(t, params);          // "time for" ... various const bool
            eta_t = set_eta(params, t, lambda); // set eta for this iteration
            if (!params.report_epoch && t % 1000 == 0) { // print some progress
                snprintf(iter_str,30, "Projection %d > ", projection_dim+1);
                print_progress(iter_str, t, params.max_iter);
                cout.flush();
            }
            if(t4.finite_diff_test){            // perform finite differences test
                for (size_t fdtest=0; fdtest<params.no_finite_diff_tests; ++fdtest) {
                    size_t idx = ((size_t) rand()) % nTrain;
                    finite_diff_test( w, x, idx, y, nclasses, maxclasses, sorted_class,
                                      class_order, sortedLU, filtered, C1, C2, params);
                }
            }
            if(0) cout<<" t="<<setw(7)<<t
                <<" * ok_projection="<<ok_projection<<" ok_projection_avg="<<ok_projection_avg
                    <<" * luPerm ok_lu="<<luPerm.ok_lu<<" ok_sortlu="<<luPerm.ok_sortlu
                    <<" * luPerm ok_lu_avg="<<luPerm.ok_lu_avg<<" ok_sortlu_avg="<<luPerm.ok_sortlu_avg<<" nAccSortlu_avg="<<luPerm.nAccSortlu_avg
                    <<endl;

            // compute the gradient and update
            //      modifies w, sortedLU, (sortedLU_avg) ONLY  (l,u,l_avg,u_avg values may be stale)
            // --> update( w, MCpermState, /* R/O: */x, y, MCsoln, MCiterBools, MCiterState, MCbatchState, params )
            // --> the ONLY place where 'w' is modified
            // update --> invalidates projection_avg and projection
            GRADIENT_UPDATE;
            W_HAS_CHANGED; // Projections no good. They map raw data in 'x' ---> the updated line in 'w'

#ifndef NDEBUG
            // Can this invariant (at this point) simplify [ or debug ] anything?
            if( params.optimizeLU_epoch <= 0 && t4.doing_avg_epoch ){
                if(!( luPerm.nAccSortlu_avg > 0U )) cout<<"OH? t="<<t
                    <<" luPerm.nAccSortlu_avg = "<<luPerm.nAccSortlu_avg<<" not > 0"<<endl;
            }else{
                if(!( luPerm.nAccSortlu_avg == 0U )) cout<<"OH? t="<<t
                    <<" luPerm.nAccSortlu_avg = "<<luPerm.nAccSortlu_avg<<" not == 0"<<endl;
            }
#endif

            if(t4.reorder) {
                if(ITER_TRACE){ std::cout<<" REORDER"<<t<<" "; std::cout.flush(); }
                switch (params.reorder_type){
                  case REORDER_AVG_PROJ_MEANS:
                      MKOK_PROJECTION_AVG;
                      proj_means(means, nc, projection_avg, y);
                      break;
                  case REORDER_PROJ_MEANS:
                      MKOK_PROJECTION;
                      proj_means(means, nc, projection, y);
                      break;
                  case REORDER_RANGE_MIDPOINTS:
                      luPerm.mkok_lu(); assert(luPerm.ok_lu);
                      means = l+u; //no need to divide by 2 since it is only used for ordering
                      break;
                }
                // calculate the new class order
                luPerm.mkok_lu();
                //if ( params.optimizeLU_epoch <= 0 && t4.doing_avg_epoch ){ GET_LU_AVG; }
                if( luPerm.nAccSortlu_avg ) GET_LU_AVG;
                rank_classes(sorted_class, class_order, means); // valgrind error above -- goto VALGRIND_ERROR
                luPerm.ok_sortlu = false; if( luPerm.nAccSortlu_avg ) luPerm.ok_sortlu_avg=false;
                GET_SORTEDLU; // sort the l and u in the order of the classes
                //if ( params.optimizeLU_epoch <= 0 && t4.doing_avg_epoch ) GET_SORTEDLU_AVG;
                if( luPerm.nAccSortlu_avg ) GET_SORTEDLU_AVG;
                // else .. NO change in validity of zeroed accumulator
            }

            // optimize the lower and upper bounds (done after class ranking)
            // since it depends on the ranks
            // if ranking type is REORDER_RANGE_MIDPOINTS, then class ranking depends on this
            // but shoul still be done before since it is less expensive
            // (could also be done after the LU optimization
            if (t4.optimizeLU){  // w, luPerm constant, optimize {l,u}. Needs valid 'projection'
                OPTIMIZE_LU(" OPTLU"<<t<<" ");  assert( luPerm.ok_lu ); // sort order unchanged, so sortlu_avg ...
                GET_SORTEDLU;           assert( luPerm.ok_sortlu );     // ... accumulator validity, NOT affected :)
            }
            // calculate the objective functions with respect to the current w and bounds
            if(t4.report){ // params.report_epoch && (t % params.report_epoch == 0) )
                if(ITER_TRACE){ std::cout<<" REPORT"<<t<<" "; std::cout.flush(); }
                MKOK_PROJECTION;
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
            if(t4.report_avg){
                if ( t4.doing_avg_epoch ){
                    if(ITER_TRACE){ std::cout<<" REPORT_AVG_a"<<t; std::cout.flush(); }
                    // use the average to calculate objective
                    VectorXd sortedLU_test( l_avg.size()+u_avg.size() );
                    if (params.optimizeLU_epoch > 0) {  // <-- {l,u}_avg used as temporaries here
                        OPTIMIZE_LU_AVG("_opt");        // NO effect on sortlu_avg (still a sortlu accumulator)
                        get_sortedLU( sortedLU_test, l_avg, u_avg, sorted_class);
                    } else {
                        assert( luPerm.nAccSortlu_avg > 0U );
                        if(ITER_TRACE){ std::cout<<"_avg"; std::cout.flush(); }
                        sortedLU_test = sortedLU_avg/(t - params.avg_epoch + 1);
                    }
                    MKOK_PROJECTION_AVG;
                    objective_val_avg[obj_idx_avg++] =
                        calculate_objective_hinge( projection_avg, y, nclasses, sorted_class, class_order,
                                                   w.norm_avg(), sortedLU_test, filtered,
                                                   lambda, C1, C2, params); // save the objective value
                } else if(t4.report){
                    if(ITER_TRACE){ std::cout<<" REPORT_AVG_b"<<t<<" "; std::cout.flush(); }
                    // the objective has just been computed for the current w, use it.
                    objective_val_avg[obj_idx_avg++] = objective_val[obj_idx - 1];
                } else {
                    if(ITER_TRACE){ std::cout<<" REPORT_noAVGyet"<<t<<" "; std::cout.flush(); }
                    MKOK_PROJECTION;
                    objective_val_avg[obj_idx_avg++] =
                        calculate_objective_hinge( projection, y, nclasses, sorted_class, class_order,
                                                   w.norm(), sortedLU, filtered,
                                                   lambda, C1, C2, params); // save the objective value
                }
                if(PRINT_O) {
                    cout<<"objective_val_avg[" << t << "]: " << objective_val_avg[obj_idx_avg-1] << " "<< w.norm_avg() << endl;
                }
            }
        } // **** **** end while t **** ****
#ifdef PROFILE
        ProfilerStop();
        ProfilerStart("filtering.profile");
#endif
        cout<<" * end iterations:"
            <<" * ok_projection="<<ok_projection<<" ok_projection_avg="<<ok_projection_avg
            <<" * luPerm ok_lu="<<luPerm.ok_lu<<" ok_sortlu="<<luPerm.ok_sortlu
            <<" * luPerm ok_lu_avg="<<luPerm.ok_lu_avg<<" ok_sortlu_avg="<<luPerm.ok_sortlu_avg<<" nAccSortlu_avg="<<luPerm.nAccSortlu_avg
            <<endl;

        // define these here just in case I got some of the conditons wrong
        //VectorXd projection, projection_avg;

#if 0
        // get l and u if needed
        // have to do this here because class order might change
        if ( params.optimizeLU_epoch <= 0 || params.reorder_type == REORDER_RANGE_MIDPOINTS ) {
            GET_LU;
        }
#endif
        // optimize LU and compute objective for averaging if it is turned on
        // if t = params.avg_epoch, everything is exactly the same as
        // just using the current w
        if ( params.avg_epoch && t > params.avg_epoch ) {
            cout<<" mp02 bugpoint: luPerm ok_lu_avg="<<luPerm.ok_lu_avg<<" ok_sortlu_avg="<<luPerm.ok_sortlu_avg
                <<" nAccSortlu_avg="<<luPerm.nAccSortlu_avg<<endl;
#if 0
            if ( params.optimizeLU_epoch <= 0) {
                GET_LU_AVG; // get the current l_avg and u_avg if needed
            }
#endif
#if 0
            if (params.report_avg_epoch > 0 || params.optimizeLU_epoch > 0) {
                // project all the data on the average w if needed
                w.project_avg(projection_avg,x);
            }
#endif
            // only need to reorder the classes if optimizing LU
            // or if we are interested in the last obj value
            // do the reordering based on the averaged w
            if (params.reorder_epoch > 0
                && (params.optimizeLU_epoch > 0 || params.report_avg_epoch > 0)) {
                if(ITER_TRACE){ std::cout<<" proj_RANK_AVG "; std::cout.flush(); }
                MKOK_PROJECTION_AVG;
                proj_means(means, nc, projection_avg, y);
                // calculate the new class order
                        //GET_LU_AVG;     // mp02 ok_lu_avg failed
                rank_classes(sorted_class, class_order, means);
                luPerm.ok_sortlu = false; if( luPerm.nAccSortlu_avg ) luPerm.ok_sortlu_avg=false;
                        //GET_SORTEDLU_AVG;
                // XXX if reorder AND report_avg AND !optLU, then could have sortlu_avg false forever after
            }
            // optimize the lower and upper bounds (done after class ranking)
            // since it depends on the ranks
            if (params.optimizeLU_epoch > 0) {
                OPTIMIZE_LU_AVG(" proj_OPTLU_avg");
            }
            if( params.report_avg_epoch>0 ) { // calculate the objective for the averaged w
                if(ITER_TRACE){ std::cout<<" proj_OBJ_avg "; std::cout.flush(); }
                GET_SORTEDLU_AVG;               // if !ok from OPTIMIZE_LU_AVG, then set it from sortlu_avg
                MKOK_PROJECTION_AVG;
                objective_val_avg[obj_idx_avg++] =
                    calculate_objective_hinge( projection_avg, y, nclasses,
                                               sorted_class, class_order,
                                               w.norm_avg(), sortedLU_avg, filtered,
                                               lambda, C1, C2, params); // save the objective value
                if(PRINT_O) {
                    cout << "objective_val_avg[" << t << "]: " << objective_val_avg[obj_idx_avg-1] << " "<< w.norm_avg() << endl;
                }
            }
        }
        // only need to reorder the classes if optimizing LU
        // or if we are interested in the last obj value
        // do the reordering based on the averaged w
        if (params.reorder_epoch > 0 && (params.optimizeLU_epoch > 0 || params.report_epoch > 0)) {
            if(ITER_TRACE){ std::cout<<" proj_REORDER_nonavg "; std::cout.flush(); }
            switch (params.reorder_type) {
              case REORDER_AVG_PROJ_MEANS:
                  // even if reordering is based on the averaged w
                  // do it here based on the w to get the optimal LU and
                  // the best objective with respect to w
              case REORDER_PROJ_MEANS:
                  MKOK_PROJECTION;
                  proj_means(means, nc, projection, y);
                  break;
              case REORDER_RANGE_MIDPOINTS:
                  luPerm.mkok_lu();
                  means = l+u; //no need to divide by 2 since it is only used for ordering
                  break;
              default:
                  cerr << "Error, reordering " << params.reorder_type << " not implemented" << endl;
                  exit(-1);
            }
            // calculate the new class order
            rank_classes(sorted_class, class_order, means);
            luPerm.ok_sortlu = false; if( luPerm.nAccSortlu_avg ) luPerm.ok_sortlu_avg=false;
        }

        // optimize the lower and upper bounds (done after class ranking)
        // since it depends on the ranks
        // if ranking type is REORDER_RANGE_MIDPOINTS, then class ranking depends on this
        // but shoul still be done before since it is less expensive
        // (could also be done after the LU optimization
        // do this for the average class
        if (params.optimizeLU_epoch > 0) {
            OPTIMIZE_LU(" proj_OPTLU");
        }
        // calculate the objective for the current w
        if( params.report_epoch>0 ) {
            // get the current sortedLU in case bounds or order changed
            GET_SORTEDLU;
            MKOK_PROJECTION;
            objective_val[obj_idx++] =
                calculate_objective_hinge( projection, y, nclasses,
                                           sorted_class, class_order,
                                           w.norm(), sortedLU,
                                           filtered,
                                           lambda, C1, C2, params); // save the objective value
            if(PRINT_O) {
                cout << "objective_val[" <<setw(6)<<t << "]: " << objective_val[obj_idx-1] << " "<< w.norm();
            }
        }

        w.toVectorXd(vect);
        weights.col(projection_dim) = vect;
        lower_bounds.col(projection_dim) = l;
        upper_bounds.col(projection_dim) = u;
        if ( params.avg_epoch && t > params.avg_epoch ) {
            luPerm.mkok_lu_avg();
            if( ! luPerm.ok_lu_avg ) throw std::runtime_error("BAD final luPerm.ok_lu_avg");
            w.toVectorXd_avg(vect);
            weights_avg.col(projection_dim) = vect;
            lower_bounds_avg.col(projection_dim) = l_avg;
            upper_bounds_avg.col(projection_dim) = u_avg;
        }else{
            luPerm.mkok_lu();
            if( ! luPerm.ok_lu ) throw std::runtime_error("BAD final luPerm.ok_lu");
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
        if (params.remove_constraints && projection_dim < (int)nProj-1) {
            if(ITER_TRACE){ std::cout<<" proj_UPDATE_FILTERED "; std::cout.flush(); }
            if (params.avg_epoch && t > params.avg_epoch ) {
                MKOK_PROJECTION_AVG;
                update_filtered(filtered, projection_avg, l_avg, u_avg, y, params.remove_class_constraints);
            }else{
                MKOK_PROJECTION;
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
            }
        }
        //      C2*=((nTrain-1)*nClass)*1.0/no_remaining;
        //C1*=((nTrain-1)*nClass)*1.0/no_remaining;

#ifdef PROFILE
        ProfilerStop();
#endif

        if(ITER_TRACE || PRINT_O)
            cout<<endl;
    } // end for projection_dim
}
#endif //MCSOLVER_HH
