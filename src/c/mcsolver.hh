#ifndef MCSOLVER_HH
#define MCSOLVER_HH
/** \file
 * MCsolver::solve impl
 */
//#include "find_w.h"
#include "mcsolver.h"
#include "find_w_detail.hh"
#if MCUC
#include "mcupdate.h"
#endif
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
inline void MCpermState::chg_sortlu_avg(){
    assert( ok_sortlu_avg );
    ok_lu_avg = false;
}
inline void MCpermState::chg_lu(){
    //assert( ok_lu );
    ok_sortlu = false;
}
inline void MCpermState::chg_lu_avg(){
    throw std::runtime_error("I do not think anything should directly modifies {l,u}_avg.  sortlu_avg is an ACCUMULATOR.");
    //assert( ok_lu_avg );
    ok_sortlu_avg = false;
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
    // ok_lu_avg MAY be false (until avg'ing is started)
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
    mkok_lu_avg();
    Perm::rank( sortkey );
    ok_sortlu = false;
    ok_sortlu_avg = false;
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
#if MCPRM==0
#define OPTIMIZE_LU do { \
    optimizeLU(l,u,projection,y,class_order, sorted_class, wc, nclasses, filtered, C1, C2, params); \
}while(0)
#elif MCPRM==1
#define OPTIMIZE_LU do{ \
    optimizeLU(l,u,projection,y,class_order, sorted_class, wc, nclasses, filtered, C1, C2, params); \
    luPerm.ok_lu = true; luPerm.ok_sortlu = false; \
    assert( luPerm.ok_lu ); \
}while(0)
#else
#define OPTIMIZE_LU do { \
    luPerm.optimizeLU(projection, y, wc, nclasses, filtered, C1, C2, params); \
    assert( luPerm.ok_lu ); \
}while(0)
#endif

#if MCPRM==0
#define OPTIMIZE_LU_AVG do { \
    optimizeLU(l_avg,u_avg,projection_avg,y,class_order, sorted_class, wc, nclasses, filtered, C1, C2, params); \
}while(0)
#elif MCPRM==1
#define OPTIMIZE_LU_AVG do{ \
                assert( luPerm.ok_sortlu_avg ); \
    optimizeLU(l_avg,u_avg,projection_avg,y,class_order, sorted_class, wc, nclasses, filtered, C1, C2, params); \
    luPerm.ok_lu_avg = true; luPerm.ok_sortlu_avg = false; \
    assert( luPerm.ok_lu_avg ); \
}while(0)
#else
#define OPTIMIZE_LU_AVG do { \
                assert( luPerm.ok_sortlu_avg ); \
    luPerm.optimizeLU_avg(projection, y, wc, nclasses, filtered, C1, C2, params); \
    assert( luPerm.ok_lu_avg ); \
}while(0)
#endif

#if MCPRM==0
#define GET_LU do { \
    get_lu( l,u, sortedLU,sorted_class ); \
}while(0)
#elif MCPRM==1
#define GET_LU do { \
    assert( luPerm.ok_sortlu ); \
    get_lu( l,u, sortedLU,sorted_class ); \
    luPerm.ok_lu = true; \
}while(0)
#else
#define GET_LU do { \
    luPerm.mkok_lu(); \
    assert( luPerm.ok_lu ); \
}while(0)
#endif

#if MCPRM==0
#define GET_LU_AVG do { \
    get_lu( l_avg,u_avg, sortedLU,sorted_class ); \
}while(0)
#elif MCPRM==1
#define GET_LU_AVG do { \
    assert( luPerm.ok_sortlu ); \
    get_lu( l_avg,u_avg, sortedLU,sorted_class ); \
    luPerm.ok_lu_avg = true; \
}while(0)
#else
#define GET_LU_AVG do { \
    luPerm.mkok_lu_avg(); \
    assert( luPerm.ok_lu ); \
}while(0)
#endif

#if MCUC==0 || MCUC==1
#define GRADIENT_UPDATE do { \
            if (params.update_type == SAFE_SGD) { \
                update_safe_SGD(w, sortedLU, sortedLU_avg, \
                                x, y, C1, C2, lambda, t, eta_t, nTrain, /* nTrain is just x.rows()*/ \
                                nclasses, maxclasses, sorted_class, class_order, filtered, \
                                sc_chunks, sc_chunk_size, sc_remaining, \
                                params); \
            } else if (params.update_type == MINIBATCH_SGD) { \
                assert( idx_locks != nullptr ); \
                assert( sc_locks != nullptr ); \
                update_minibatch_SGD(w, sortedLU, sortedLU_avg, \
                                     x, y, C1, C2, lambda, t, eta_t, nTrain, batch_size, \
                                     nclasses, maxclasses, sorted_class, class_order, filtered, \
                                     sc_chunks, sc_chunk_size, sc_remaining, \
                                     idx_chunks, idx_chunk_size, idx_remaining, \ \
                                     idx_locks, sc_locks, \
                                     params); \
            } \
}while(0)
#elif MCUC>1 && MCPRM==0
#define GRADIENT_UPDATE do { \
            MCupdate::update(w, sortedLU, sortedLU_avg, \
                             x, y, C1, C2, lambda, t, eta_t, nTrain, /*batch_size,*/ \
                             nclasses, maxclasses, sorted_class, class_order, filtered, \
                             /*sc_chunks, sc_chunk_size, sc_remaining,*/ \
                             /*idx_chunks, idx_chunk_size, idx_remaining,*/ \
                             /*idx_locks, sc_locks,*/ \
                             updateSettings, \
                             params); \
}while(0)
#elif MCUC>1 && MCPRM>0
#define GRADIENT_UPDATE do { \
            /* cout<<" t="<<t<<" before 'update', luPerm.ok_sortlu="<<luPerm.ok_sortlu<<endl; */ \
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
#else
#error "OHOH need to define GRADIENT_UPDATE call"
#endif

#if MCPRM==0
#define GET_SORTEDLU do { \
    get_sortedLU(sortedLU, l, u, sorted_class); /* l, u --> sortedLU */ \
}while(0)
#define GET_SORTEDLU_AVG do { \
    get_sortedLU(sortedLU_avg, l_avg, u_avg, sorted_class); /* l, u --> sortedLU */ \
}while(0)
#else
#define GET_SORTEDLU do { \
    assert( luPerm.ok_lu ); \
    get_sortedLU(sortedLU, l, u, sorted_class); /* l, u --> sortedLU */ \
    luPerm.ok_sortlu = true; \
}while(0)
#define GET_SORTEDLU_AVG do { \
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
#if MCPRM==0
    VectorXd l(nClass),u(nClass);
    VectorXd sortedLU(2*nClass); // holds l and u interleaved in the curent class sorting order (i.e. l,u,l,u,l,u)
    //  VectorXd sortedLU_gradient(2*nClass); // used to improve cache performance
    //  VectorXd sortedLU_gradient_chunk;
    VectorXd l_avg(nClass),u_avg(nClass); // the lower and upper bounds for the averaged gradient
    VectorXd sortedLU_avg(2*nClass); // holds l_avg and u_avg interleaved in the curent class sorting order (i.e. l_avg,u_avg,l_avg,u_avg,l_avg,u_avg)
    std::vector<int> sorted_class(nClass), class_order(nClass);//, prev_class_order(nClass);// used as the switch
#elif MCPRM>=1
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
#if 1
    int const nThreads = getNthreads( params );
#else
#ifdef _OPENMP
    int nThreads;
    {
        if (params.num_threads < 0)
            nThreads = omp_get_num_procs();   // use # of CPUs
        else if (params.num_threads == 0)
            nThreads = omp_get_max_threads(); // use OMP_NUM_THREADS
        else
            nThreads = params.num_threads;
    }
    omp_set_num_threads( nThreads );
    std::cout<<" solve_ with _OPENMP and params.num_threads set to "<<params.num_threads
        <<", nThreads is "<<nThreads<<", and omp_max_threads is now "<<omp_get_max_threads()<<endl;
#else
    nThreads = 1;
    std::cout<<" no _OPENMP";
#endif
#endif

#if MCUC==0
    const size_t batch_size = (params.batch_size < 1 || params.batch_size > nTrain) ? (size_t) nTrain : params.batch_size;
    if (params.update_type == SAFE_SGD)
    {
        // save_sgd update only works with batch size 1
        assert(batch_size == 1);
    }
    int const idx_chunks = nThreads;
    int const sc_chunks = nThreads;
    std::cout<<" idx_chunks="<<idx_chunks<<std::endl;
    MutexType* sc_locks = new MutexType [sc_chunks];
    MutexType* idx_locks = new MutexType [idx_chunks];
    int const sc_chunk_size = (params.class_samples?params.class_samples:nClass)/sc_chunks;
    int const sc_remaining = (params.class_samples?params.class_samples:nClass) % sc_chunks;
    int const idx_chunk_size = batch_size/idx_chunks;
    int const idx_remaining = batch_size % idx_chunks;
#else // MCUC
    MCupdateChunking updateSettings( nTrain/*x.rows()*/, nClass, nThreads, params );
    //        size_t const& batch_size    = updateSettings.batch_size;
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
    size_t const& batch_size    = updateSettings.batch_size;
#if MCUC==1 // keep all old-style variables
    int const& sc_chunks        = updateSettings.sc_chunks;
    int const& sc_chunk_size    = updateSettings.sc_chunk_size;
    int const& sc_remaining     = updateSettings.sc_remaining;

    int const& idx_chunks       = updateSettings.idx_chunks;
    int const& idx_chunk_size   = updateSettings.idx_chunk_size;
    int const& idx_remaining    = updateSettings.idx_remaining;
    MutexType* idx_locks        = updateSettings.idx_locks;
    MutexType* sc_locks         = updateSettings.sc_locks;
#else // MCUC > 1 ---> hide most old-style variables
    // force to **only** use updateSettings
#endif
#endif

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
                if (params.reoptimize_LU || (params.remove_constraints && projection_dim < int(nProj)-1))
                    w.project(projection,x);
                if (params.reoptimize_LU) {
                    switch (params.reorder_type) {
                      case REORDER_AVG_PROJ_MEANS:
                          // use the current w since averaging has not started yet
                      case REORDER_PROJ_MEANS:
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
                    OPTIMIZE_LU;
                    // copy w, lower_bound, upper_bound from the coresponding averaged terms.
                    // this way we do not spend time reoptimizing LU for non-averaged terms we probably won't use.
                    // The right way to handle this would be to know whether we want to return
                    // only the averaged values or we also need the non-averaged ones.
                    w.toVectorXd(vect);
                    weights.col(projection_dim) = vect;
#if MCPRM==0
                    lower_bounds.col(projection_dim) = l;
                    upper_bounds.col(projection_dim) = u;
                    //
                    lower_bounds_avg.col(projection_dim) = l;
                    upper_bounds_avg.col(projection_dim) = u;
#else
                    lower_bounds.col(projection_dim) = luPerm.l;
                    upper_bounds.col(projection_dim) = luPerm.u;
                    //
                    lower_bounds_avg.col(projection_dim) = luPerm.l;
                    upper_bounds_avg.col(projection_dim) = luPerm.u;
#endif
                }else{
#if MCPRM==0
                    l = lower_bounds_avg.col(projection_dim);
                    u = upper_bounds_avg.col(projection_dim);
#else
                    luPerm.set_lu( lower_bounds_avg.col(projection_dim), upper_bounds_avg.col(projection_dim) );
#endif
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

    cout << "start projection " << projection_dim << endl; cout.flush();
    for(; projection_dim < (int)nProj; projection_dim++)
    {
#if 0 && MCPRM>=1
        luPerm.reset(); // including sortedLU_avg.setZero()
        assert( y.cols() == luPerm.l.size() );
        assert( luPerm.ok_sortlu_avg ); // this is an accumulator, all-zero is the actual valid initial state (sorta')
#endif
        init_w      (w, x,y,nc, weights_avg,projection_dim);
        w.project   (projection,x);        // project each example onto current projection dirn, w
#if MCPRM==0
        init_lu     (l,u,means, params.reorder_type,projection,y,nc); // init l, u and means
        if( params.optimizeLU_epoch <= 0 )    sortedLU_avg.setZero(); // not needed
#else
        luPerm.init( projection, y, nc );
        if( params.reorder_type == REORDER_RANGE_MIDPOINTS ) means = luPerm.l + luPerm.u;
        else /* we have no *_avg yet, so use l,u          */ proj_means( means,  nc,projection,y );  
        assert( y.cols() == luPerm.l.size() );
        assert( luPerm.ok_sortlu_avg ); // this is an accumulator, all-zero is the actual valid initial state (sorta')
#endif
        rank_classes(sorted_class, class_order, means);
        cout << "start optimize LU" << endl; cout.flush();
#ifdef PROFILE
        ProfilerStop();
        ProfilerStart("optimizeLU.profile");
#endif
        if (params.optimizeLU_epoch > 0) { // questionable.. l,u at this point are random, so why do it this early?
            OPTIMIZE_LU;
        }
        GET_SORTEDLU;
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
#if MCPRM>=1
            cout<<" t=0 luPerm.{ok_lu="<<luPerm.ok_lu<<",ok_sortlu="<<luPerm.ok_sortlu
                <<",  ok_lu_avg="<<luPerm.ok_lu_avg<<",ok_sortlu_avg="<<luPerm.ok_sortlu_avg<<"}"<<endl;
#endif
            cout<<"objective_val[  t   ]: value    w.norm\n"
                <<"--------------------- -------  -------"<<endl;
        }
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

            // compute the gradient and update
            //      modifies w, sortedLU, (sortedLU_avg) ONLY  (l,u,l_avg,u_avg values may be stale)
            // --> update( w, MCpermState, /* R/O: */x, y, MCsoln, MCiterBools, MCiterState, MCbatchState, params )
            // update --> invalidates projection_avg and projection
            GRADIENT_UPDATE;

            if (t4.report_avg || (t4.reorder && params.reorder_type == REORDER_AVG_PROJ_MEANS)){
                w.project_avg(projection_avg, x);
            }
            if (t4.report || t4.optimizeLU || (t4.reorder && params.reorder_type == REORDER_PROJ_MEANS)){
                w.project    (projection    , x);
            }
            // reorder the classes
            if(t4.reorder){
                GET_LU;
                if ( params.optimizeLU_epoch <= 0 && t4.doing_avg_epoch ){
                    GET_LU_AVG;
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
                rank_classes(sorted_class, class_order, means); // valgrind error above -- goto VALGRIND_ERROR

                // sort the l and u in the order of the classes
                GET_SORTEDLU;

                if ( params.optimizeLU_epoch <= 0 && t4.doing_avg_epoch ){
                    GET_SORTEDLU_AVG;
                }
            }

            // optimize the lower and upper bounds (done after class ranking)
            // since it depends on the ranks
            // if ranking type is REORDER_RANGE_MIDPOINTS, then class ranking depends on this
            // but shoul still be done before since it is less expensive
            // (could also be done after the LU optimization
            if (t4.optimizeLU){
                OPTIMIZE_LU;
                GET_SORTEDLU;
            }
            // calculate the objective functions with respect to the current w and bounds
            if(t4.report){ // params.report_epoch && (t % params.report_epoch == 0) )
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
            if(t4.report_avg){
                if ( t4.doing_avg_epoch ){
                    // use the average to calculate objective
                    VectorXd sortedLU_test;
                    if (params.optimizeLU_epoch > 0) {
                        OPTIMIZE_LU_AVG;
#if MCPRM==0
                        get_sortedLU( sortedLU_test, l_avg, u_avg, sorted_class);
#else
                        assert( luPerm.ok_lu_avg );
                        luPerm.getSortlu_avg( sortedLU_test );  // XXX double-check equivalence XXX
#endif
                    } else {
                        sortedLU_test = sortedLU_avg/(t - params.avg_epoch + 1);
                    }
                    objective_val_avg[obj_idx_avg++] =
                        calculate_objective_hinge( projection_avg, y, nclasses,
                                                   sorted_class, class_order,
                                                   w.norm_avg(), sortedLU_test,
                                                   filtered,
                                                   lambda, C1, C2, params); // save the objective value
                } else if(t4.report){
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
        } // **** **** end while t **** ****
#ifdef PROFILE
        ProfilerStop();
        ProfilerStart("filtering.profile");
#endif

        // define these here just in case I got some of the conditons wrong
        VectorXd projection, projection_avg;

        // get l and u if needed
        // have to do this here because class order might change
        if ( params.optimizeLU_epoch <= 0 || params.reorder_type == REORDER_RANGE_MIDPOINTS ) {
            GET_LU;
        }
        // optimize LU and compute objective for averaging if it is turned on
        // if t = params.avg_epoch, everything is exactly the same as
        // just using the current w
        if ( params.avg_epoch && t > params.avg_epoch ) {
            if ( params.optimizeLU_epoch <= 0) {
                GET_LU_AVG; // get the current l_avg and u_avg if needed
            }
            if (params.report_avg_epoch > 0 || params.optimizeLU_epoch > 0) {
                // project all the data on the average w if needed
                w.project_avg(projection_avg,x);
            }
            // only need to reorder the classes if optimizing LU
            // or if we are interested in the last obj value
            // do the reordering based on the averaged w
            if (params.reorder_epoch > 0
                && (params.optimizeLU_epoch > 0 || params.report_avg_epoch > 0)) {
                // calculate the new class order
                proj_means(means, nc, projection_avg, y);
                rank_classes(sorted_class, class_order, means);
            }
            // optimize the lower and upper bounds (done after class ranking)
            // since it depends on the ranks
            if (params.optimizeLU_epoch > 0) {
                OPTIMIZE_LU_AVG;
            }
            if( params.report_avg_epoch>0 ) {
                // get the current sortedLU in case bounds or order changed // could test for changes!
                GET_SORTEDLU_AVG;
                // calculate the objective for the averaged w
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
        if (params.reorder_epoch > 0 && (params.optimizeLU_epoch > 0 || params.report_epoch > 0)) {
            switch (params.reorder_type) {
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
            OPTIMIZE_LU;
        }
        // calculate the objective for the current w
        if( params.report_epoch>0 ) {
            // get the current sortedLU in case bounds or order changed
            GET_SORTEDLU;
            objective_val[obj_idx++] =
                calculate_objective_hinge( projection, y, nclasses,
                                           sorted_class, class_order,
                                           w.norm(), sortedLU,
                                           filtered,
                                           lambda, C1, C2, params); // save the objective value
            if(PRINT_O) {
                cout << "objective_val[" <<setw(6)<<t << "]: " << objective_val[obj_idx-1] << " "<< w.norm() << endl;
            }
        }

        w.toVectorXd(vect);
        weights.col(projection_dim) = vect;
        lower_bounds.col(projection_dim) = l;
        upper_bounds.col(projection_dim) = u;
        if ( params.avg_epoch && t > params.avg_epoch ) {
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
        if (params.remove_constraints && projection_dim < (int)nProj-1) {
            if (params.avg_epoch && t > params.avg_epoch ) {
                w.project_avg(projection_avg,x); // most likely been calculated above (?)
                update_filtered(filtered, projection_avg, l_avg, u_avg, y, params.remove_class_constraints);
            }else{
                w.project(projection,x); // most likely calculated above (?)
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

    } // end for projection_dim
}
#endif //MCSOLVER_HH
