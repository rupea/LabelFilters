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
#include "printing.hh"          // print_report
#include <iomanip>
#include <vector>

#ifdef PROFILE
#define PROFILER_START( CSTR )      do{ ProfilerStart(CSTR); }while(0)
#define PROFILER_STOP_START( CSTR ) do{ ProfilerStop(); ProfilerStart(CSTR); }while(0)
#define PROFILER_STOP               ProfilerStop()
#else
#define PROFILER_START( CSTR )      do{}while(0)
#define PROFILER_STOP_START( CSTR ) do{}while(0)
#define PROFILER_STOP               do{}while(0)
#endif

#if GRADIENT_TEST // compile-time option, disabled by default, macro to unclutter iteration loop
#define OPT_GRADIENT_TEST do \
{   /* now thread-safe, just in case */ \
    static unsigned thread_local seed = static_cast<unsigned>(this_thread.get_id()); \
    if(t4.finite_diff_test){            /* perform finite differences test */ \
        for (size_t fdtest=0; fdtest<params.no_finite_diff_tests; ++fdtest) { \
            size_t idx = ((size_t) rand_r(&seed)) % n; \
            finite_diff_test( w, x, idx, y, nclasses, maxclasses, luPerm.perm/*sorted_class*/, \
                              luPerm.rev, luPerm.sortlu, filtered, C1, C2, params); \
        } \
    } \
}while(0)
#else
#define OPT_GRADIENT_TEST do{}while(0)
#endif

// first let's define mcsolver.h utilities...

#define MCITER_PERIODIC(XXX) XXX( params.XXX##_epoch && t % params.XXX##_epoch == 0 )
    inline MCiterBools::MCiterBools( uint64_t const t, param_struct const& params )
    : MCITER_PERIODIC( reorder )
    , MCITER_PERIODIC( report )
    , MCITER_PERIODIC( report_avg )
    , MCITER_PERIODIC( optimizeLU )
#if GRADIENT_TEST
    , MCITER_PERIODIC( finite_diff_test )
#endif
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
        //assert( ll.coeff(cp) <= uu.coeff(cp) );
    }
}
#if 0
inline void MCpermState::toSorted( VectorXd & sorted, VectorXd const& ll, VectorXd const& uu ){
    for(size_t i=0; i<perm.size(); ++i){
#ifndef NDEBUG
        if( !(ll.coeff(perm[i]) <= uu.coeff(perm[i])) ){
            std::cerr<<" OHOH. perm["<<i<<"] = class "<<perm[i]<<" with {l,u} = {"
                <<l.coeff(perm[i])<<", "<<u.coeff(perm[i])<<"}"<<std::endl;
        }
#endif
        sorted.coeffRef(2U*i)    = l.coeff(perm[i]);
        sorted.coeffRef(2U*i+1U) = u.coeff(perm[i]);
    }
}
#endif
inline void MCpermState::toSorted( VectorXd & sorted, VectorXd const& ll, VectorXd const& uu ) const{
//#ifndef NDEBUG
//    int nErr=0U;
//#endif
    for(size_t i=0; i<perm.size(); ++i){
//#ifndef NDEBUG
//        if( !(ll.coeff(perm[i]) <= uu.coeff(perm[i])) ){
//            std::cerr<<" OHOH. perm["<<i<<"] = class "<<perm[i]<<" with {ll,uu} = {"
//                <<ll.coeff(perm[i])<<", "<<uu.coeff(perm[i])<<"}"<<std::endl;
//            ++nErr;
//        }
//#endif
        sorted.coeffRef(2U*i)    = ll.coeff(perm[i]);
        sorted.coeffRef(2U*i+1U) = uu.coeff(perm[i]);
    }
//#ifndef NDEBUG
//    assert( nErr == 0 );
//#endif
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
        // wasteful if called with nAccSortlu_avg == 0, because sortlu_avg is all-zeroes ???
        assert( nAccSortlu_avg > 0U );
        toLu( l_avg, u_avg, sortlu_avg );
        ok_lu_avg = true;
    }
    // ok_lu_avg MAY be false -- should assert if it's required
    assert( ok_lu_avg );        // Maybe
}
inline VectorXd& MCpermState::mkok_sortlu(){
    if(!ok_sortlu){
//#ifndef NDEBUG
//        assert(ok_lu);
//        assert( l.size() == u.size() );
//        for(size_t c=0U; c<l.size(); ++c){
//            assert( l.coeff(c) <= u.coeff(c) );
//        }
//#endif
        toSorted( sortlu, l, u );
        ok_sortlu = true;
    }
    assert( ok_sortlu );
    return sortlu;
}
inline VectorXd& MCpermState::mkok_sortlu_avg(){
    if(!ok_sortlu_avg && ok_lu_avg){
//#ifndef NDEBUG
//        assert( ok_lu_avg );
//        for(size_t c=0U; c<l_avg.size(); ++c){
//            assert( l_avg.coeff(c) <= u_avg.coeff(c) );
//        }
//#endif
        toSorted( sortlu_avg, l_avg, u_avg);
        ok_sortlu_avg = true;
    }
    return sortlu_avg;
}
inline void MCpermState::rank( VectorXd const& sortkey ){
    mkok_lu();
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

#define OPTIMIZE_LU( MSG ) do { \
    if(ITER_TRACE){ std::cout<<MSG; std::cout.flush(); } \
    /** -- no -- luPerm l/u/sortlu prerequisites : outputs l, u */ \
    /* MKOK_PROJECTION; assert( ok_projection ); */\
    luPerm.optimizeLU( xwProj.std(), y, wc, nclasses, filtered, C1, C2, params); \
    assert( luPerm.ok_lu ); assert( ! luPerm.ok_sortlu ); \
}while(0)
#define OPTIMIZE_LU_AVG( MSG ) do { \
    if(ITER_TRACE){ std::cout<<MSG; std::cout.flush(); } \
    /*MKOK_PROJECTION_AVG; assert( ok_projection_avg );*/ \
    /* -- no -- luPerm prerequisites, but sets {l,u}_avg instead of {l,u} */ \
    /* sortlu_avg is still some slightly stale accumulated version, but we can't change it */ \
    luPerm.optimizeLU_avg( xwProj.avg(), y, wc, nclasses, filtered, C1, C2, params); \
    assert( luPerm.ok_lu_avg ); \
    if( luPerm.nAccSortlu_avg ) luPerm.ok_sortlu_avg=false; \
}while(0)

#if MCPRM>=3
#define GRADIENT_UPDATE do { \
            /* cout<<" t="<<t<<" before 'update', luPerm.ok_sortlu="<<luPerm.ok_sortlu<<endl; */ \
            luPerm.mkok_sortlu(); \
            assert( luPerm.ok_sortlu ); \
            MCupdate::update(w, luPerm, eta_t,      x, y, C1, C2, lambda, t, nTrain, \
                             nclasses, maxclasses, filtered, updateSettings, params); \
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
    PROFILER_START("init.profile");
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

    /** projector of row-wise data examples \c x onto evolving projection vector \c w.
     * - \c x can be projected on \c w in two ways, \c STD and \c AVG.
     * - if \c w has not yet begun maintaining an \c AVG form of the projection vector,
     *   then we silently demote and use \c STD (simpler) form.
     * - destructor prints stats -- hopefully you usually need just one form of the projection
     *   and do not toggle between the two types a lot between calls to \c w_changed().
     * - Initial tests showed \c nSwitch==0 for a wide variety of \c param_struct settings [GOOD].*/
    class Proj {
    public:
        enum Type { STD, AVG } type;
        Proj( EIGENTYPE const& x, WeightVector const& w ) : x(x), w(w), v(x.rows()), ngetSTD(0U), ngetAVG(0U), nReuse(0U), nSwitch(0U), nDemote(0U), valid(false)
        {}
        ~Proj(){
            std::cout<<" ~Proj{ngetSTD="<<ngetSTD<<",ngetAVG="<<ngetAVG<<",nReuse="<<nReuse
                <<",nSwitch="<<nSwitch<<",nDemote="<<nDemote<<"}"<<std::endl;
        }
        /** Every time client changes \c w, \c Proj \b must be informed that the world is now different. */
        void w_changed() {
            //cout<<" - "; cout.flush();
            valid = false;
        }
        /** get w's projection, silently demote from AVG to STD if w.getAvg_t() is still zero */
        VectorXd const& operator()( enum Type t ){
            // can we do what was asked?
            if ( t == STD ){
                ++ngetSTD;
            }else if( t == AVG ){
                ++ngetAVG;
                if( w.getAvg_t() > 0 ){
                    ++nDemote;
                    t = STD;        // silently demote AVG --> STD if w has not yet begun averaging
                }
            }
            // can we reuse last projection?
            if( valid ){
                if( type==t ){
                    ++nReuse;
                    return v;
                }else{
                    ++nSwitch;
                }
            }
            // recalculate projection 'v' of [possibly demoted] type 't'
            if( t == STD ){
                //cout<<" S "; cout.flush();
                w.project( v, x );
            }else{                      assert( t== STD );
                //cout<<" A "; cout.flush();
                w.project_avg( v, x );
            }
            type = t;
            valid = true;
            return v;
        }
        VectorXd const& std(){ return operator()(STD); }
        VectorXd const& avg(){ return operator()(AVG); }
        /// for assertions
        bool isValid() const {return valid;}
        //VectorXd const& get() const {return v;}       ///< even if it's invalid
    private:
        EIGENTYPE const& x;
        WeightVector const& w;
        VectorXd v;
        // stats: if some of these large [nSwitch], maybe keep 2 vectors handy?
        size_t ngetSTD;
        size_t ngetAVG;
        size_t nReuse;  ///< count how many get() could be elided because v was already OK.
        size_t nSwitch; ///< count "valid but of wrong format" during get()
        size_t nDemote; ///< count "ask for AVG but give STD" during get()
        bool valid;
    };

    MCpermState luPerm( nClass );
    VectorXd means(nClass); // used for initialization of the class order vector;

    

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
#ifndef NDEBUG
    MCupdateChunking updateSettings( nTrain/*x.rows()*/, nClass, nThreads, params );
    {// in debug mode check no change from original settings
        const size_t batch_size = (params.batch_size < 1 || params.batch_size > nTrain) ? (size_t) nTrain : params.batch_size;
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
#else
    MCupdateChunking updateSettings( nTrain/*x.rows()*/, nClass, nThreads, params );
    //MCupdateChunking updateSettings( nTrain/*x.rows()*/, nClass, 2, params );  // 2 might be fastest?
#endif
    size_t const& batch_size    = updateSettings.batch_size; // really only for print_report ?

    VectorXi nc;       // nc[class]         = number of training examples of each class
    VectorXi nclasses; // nclasses[example] = number of classes assigned to each training example
    init_nc(nc, nclasses, y);
    int maxclasses = nclasses.maxCoeff(); // the maximum number of classes an example might have
    // Suppose example y[i] --> weight of 1.0, or if params.ml_wt_class_by_nclasses, 1.0/nclasses[i]
    // Then what is total weight of each class? (used for optimizeLU)
    VectorXd wc; // wc[class] = weight of each class (= nc[class] if params.ml_wt_class_by_nclasses==false)
    init_wc(wc, nclasses, y, params);   // wc is used if optimizeLU_epoch>0
    VectorXd xSqNorms;
    if (params.update_type == SAFE_SGD) calc_sqNorms( x, xSqNorms ); 

    //keep track of which classes have been elimninated for a particular example
    boolmatrix filtered(nTrain,nClass);
    //VectorXd difference(d);
    VectorXd tmp(d);
    unsigned long const total_constraints = nTrain*nClass
        - (1-params.remove_class_constraints) * nc.sum();
    size_t no_filtered=0;
    // Possibly read in from save file:
    if( ! params.resume ){      // NEW --- needs checking
        if(  params.C1 >= 0.0 && params.C2 >= 0.0 ){
            this->t = 0U;
#if 1 // original
            /*double*/this-> lambda = 1.0/params.C2;
            /*double*/this-> C1 = params.C1/params.C2;
            /*double*/this-> C2 = 1.0;
            //this->eta_t = params.eta; // actually, via seteta(params,t,lambda), below
#else // new -- For un-row-normalized data, want to scale C1,C2 proportionally ??
            C1 = params.C1;
            C2 = params.C2;
            //if( C2 > 1.e-4 ) lambda = 1.0/C2; else lambda = 100.0;
            lambda = C2;
#endif
        }else{
            //this-> C2 = 1.0; // / sqrt(nClass);
            //long int no_remaining = total_constraints - no_filtered;
            //this->lambda = no_remaining*1.0/(total_constraints*this->C2);
            this->lambda = sqrt(nc.sum());
            //this-> lambda = nTrain*nClass * 1.0 / (total_constraints*params.C2);
            //this-> lambda = nTrain*nClass * 1.0 / (total_constraints*params.C2);
            //this-> C1 = nClass;
            this-> C1 = sqrt(nc.sum());
            this-> C2 = 1.0/this->C1;
        }
    }
    cout<<" initial lambda="<<lambda<<" C1="<<C1<<" C2="<<C2<<endl;

    Proj xwProj( x, w );

    // define a lambda to prepackage this recurring calculation
    // It silently demotes REORDER_AVG_PROJ_MEANS if 'w' has not begun averaging.
    auto GetMeans = [&]( enum Reorder_Type reorder ) -> VectorXd const&
    {
        switch (reorder){
          case REORDER_AVG_PROJ_MEANS:
              if( w.getAvg_t() > 0 ){
                  proj_means(means, nc, xwProj.avg(), y);
                  break;
              }/* else continue */
          case REORDER_PROJ_MEANS:
              proj_means(means, nc, xwProj.std(), y);
              break;
          case REORDER_RANGE_MIDPOINTS:
              luPerm.mkok_lu();
              means = luPerm.l+luPerm.u; /*no need to divide by 2 since it is only used for ordering*/
              break;
        }
        return means;
    };
    // forcibly ignore REORDER_AVG_PROJ_MEANS
    auto GetMeansNoAvg = [&means,&GetMeans]( enum Reorder_Type reorder ) -> VectorXd const&
    {
        if( reorder == REORDER_AVG_PROJ_MEANS )
            reorder = REORDER_PROJ_MEANS;
        GetMeans(reorder);
        return means;
    };
    auto ObjectiveHinge = [&] ( char const* msg, size_t const t ) -> double
    {
        if(ITER_TRACE && msg && msg[0]!='\0'){ std::cout<<msg<<t; std::cout.flush(); }
        return calculate_objective_hinge( xwProj.std(), y, nclasses, luPerm.perm, luPerm.rev,
                                          w.norm(), luPerm.sortlu, filtered,
                                          lambda, C1, C2, params); 
    };
    auto ObjectiveHingeAvg = [&] ( char const* msg, size_t const t ) -> double
    {
        // ??? if w.getAvg_t() ?
        if(ITER_TRACE && msg && msg[0]!='\0'){ std::cout<<msg<<t; std::cout.flush(); }
        return calculate_objective_hinge( xwProj.avg(), y, nclasses, luPerm.perm, luPerm.rev,
                                          w.norm_avg(), luPerm.sortlu, filtered,
                                          lambda, C1, C2, params); 
    };

    // does resume mean "add new dims" or "recalc old"?
    bool const resume_newdims = (params.resume && nProj > weights_avg.cols()
                                 ? true : false );
    chopProjections(nProj);
    int prjax = 0;
    int reuse_dim = weights_avg.cols();
    cout<<"  ... begin with weights"<<prettyDims(weights)<<" weights_avg"<<prettyDims(weights_avg)
        <<" lower_bounds_avg"<<prettyDims(lower_bounds_avg)<<" upper_bounds_avg"<<prettyDims(upper_bounds_avg)
        <<endl;
    if (params.reoptimize_LU && params.reorder_type == REORDER_RANGE_MIDPOINTS )
        throw std::runtime_error("Error, reordering REORDER_RANGE_MIDPOINTS should "
                                 "not be used when reoptimizing the LU parameters");
    if (resume_newdims || params.reoptimize_LU) {
        if (params.reoptimize_LU) {
            lower_bounds.setZero(nClass, reuse_dim);
            upper_bounds.setZero(nClass, reuse_dim);
            lower_bounds_avg.setZero(nClass, reuse_dim);
            upper_bounds_avg.setZero(nClass, reuse_dim);
        }
        cout<<" Continuing a run ... reuse_dim=weights_avg.cols()="<<reuse_dim<<endl;
        assert( lower_bounds_avg.cols() >= reuse_dim ); // if not, resize them
        assert( upper_bounds_avg.cols() >= reuse_dim );
        if(params.reoptimize_LU || params.remove_constraints) {
            // set {l,u,w}_avg.  If possible copy into {l,u,w}
            cout<<"\tInitial tweaks to {l,u,w}_avg for projections 0.."<<reuse_dim-1<<endl;
            for (prjax = 0; prjax < reuse_dim; ++prjax) {
                cout<<"\tp:"<<prjax;
                w.init(weights_avg.col(prjax));
                xwProj.w_changed();  // projections of 'x' onto 'w' no longer valid
                if (params.reoptimize_LU) {
                    luPerm.init( xwProj.std(), y, nc );     // try to appease valgrind?
                    luPerm.rank( GetMeans(params.reorder_type) );
                    PROFILER_STOP_START("optimizeLU.profile");
                    OPTIMIZE_LU(" OPTLU-Init"); // w,projection,sort_order ----> luPerm.l,u
                    PROFILER_STOP;
                    weights_avg.col(prjax) = w.getVec();
                    lower_bounds_avg.col(prjax) = luPerm.l;
                    upper_bounds_avg.col(prjax) = luPerm.u;
                    //
                    if( prjax < weights.cols() ) weights.col(prjax) = weights_avg.col(prjax);
                    if( prjax < lower_bounds.cols() ) lower_bounds.col(prjax) = luPerm.l;
                    if( prjax < upper_bounds.cols() ) upper_bounds.col(prjax) = luPerm.u;
                }else{
                    luPerm.set_lu( lower_bounds_avg.col(prjax), upper_bounds_avg.col(prjax) );
                }
                // should update to use the filter class
                // things will not work correctly with remove_class_constrains on.
                // We need to update wc, nclass and maybe nc
                // Check if nclass and nc are used for anything other than weighting
                // examples belonging to multiple classes
                if (params.remove_constraints && prjax < (int)nProj-1) {
                    update_filtered(filtered, xwProj.std(), luPerm.l, luPerm.u, y, params.remove_class_constraints);
                    no_filtered = filtered.count();
                    cout<<"Filter["<<filtered.rows()<<"x"<<filtered.cols()<<"] removed "<<no_filtered
                        <<" of "<<total_constraints<<" constraints"<<endl;
                }

                // work on this. This is just a crude approximation.
                // now every example - class pair introduces nclass(example) constraints
                // if weighting is done, the number is different
                // eliminating one example -class pair removes nclass(exmple) potential
                // if the class not among the classes of the example
                if( params.reweight_lambda != REWEIGHT_NONE ){ // XXX code duplication here
                    long int no_remaining = (long)total_constraints - no_filtered;
                    lambda = no_remaining*1.0/(total_constraints*params.C2);
                    if( params.reweight_lambda == REWEIGHT_ALL ){
                        C1 = params.C1*no_remaining*1.0/(total_constraints*params.C2);
                    }
                    // New: test for early exit ...
                    cout<<setw(20)<<tostring(params.reweight_lambda)<<": total_constraints="
                        <<total_constraints<<" minus no_filtered="<<no_filtered<<"\n"<<setw(20)
                        <<""<<"  leaving no_remaining="<<no_remaining<<" lambda="<<lambda<<" C1="<<C1<<endl;
                    if( no_filtered > total_constraints )
                        throw std::runtime_error(" programmer error: removed more constraints than exist?");
                    if( no_remaining == 0 ){
                        cout<<setw(20)<<""<<"  CANNOT CONTINUE, no more constraints left\n"
                            "  nProj "<<nProj<<" becomes "<<prjax+1U<<endl;
                        nProj = prjax+1U;
                        reuse_dim = nProj;
                        const_cast<param_struct*>(params_arg)->no_projections = nProj; // <-- NB
                        const_cast<param_struct&>(params).no_projections = nProj; // <-- NB
                        chopProjections( nProj );
                        break;
                    }
                }
                cout<<endl;
            }
        }
        obj_idx = objective_val.size();
        obj_idx_avg = objective_val_avg.size();
        // Note: if soln NOT stored in LONG format, we will redo all projections
        //       if soln IS  stored in LONG format, we will never re-iterate over previous projections
        //prjax = weights.cols();
    }
    if(1){
        cout<<"  ... starting with     weights"<<prettyDims(weights)<<":\n"<<weights<<endl;
        cout<<"  ... starting with weights_avg"<<prettyDims(weights_avg)<<":\n"<<weights_avg<<endl;
        cout<<"  ... beginning at prjax="<<prjax<<" reuse_dim="<<reuse_dim<<endl;
        cout<<"  ... sc_chunks="<<updateSettings.sc_chunks<<" MCTHREADS="<<MCTHREADS<<endl;
    }
    if( params.reoptimize_LU ){
        cout<<" ... optlu : reoptimized projections 0.."<<reuse_dim<<endl;
        if ( !params.resume ){
            cout<<"  ... --reoptlu with no --resume --> early solver exit"<<endl;
            return;
        }
    }
    if( params.resume ){
        if( !resume_newdims && reuse_dim > 0U){ // recalc beginning with axis 0
            prjax = 0U;
            filtered.reset();
            // XXX should really agree with above initial settings
            lambda = params.C1;
            C1 = params.C1/params.C2;
            C2 = 1.0;
        }
        weights = weights_avg;
        lower_bounds = lower_bounds_avg;
        upper_bounds = upper_bounds_avg;
    }
    // XXX make more robust to continued runs (zero-initalize new projections) ?
    weights             .conservativeResize(d, nProj);
    weights_avg         .conservativeResize(d, nProj);
    lower_bounds        .conservativeResize(nClass, nProj);
    upper_bounds        .conservativeResize(nClass, nProj);
    lower_bounds_avg    .conservativeResize(nClass, nProj);
    upper_bounds_avg    .conservativeResize(nClass, nProj);
    if(0){ // conservativeResize may make many entries undefined...
        cout<<"  ... resized     weights"<<prettyDims(weights)<<":\n"<<weights<<endl;
        cout<<"  ... resized weights_avg"<<prettyDims(weights_avg)<<":\n"<<weights_avg<<endl;
        cout<<"  ... beginning at prjax="<<prjax<<" reuse_dim="<<reuse_dim<<endl;
    }
    {// more space for objective_val history, per new projection...
        size_t const newProjs = (nProj < (size_t)prjax? 0U
                                 : nProj - (size_t)prjax);
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
    assert(w.getAvg_t() == 0);
    // if(0) ---> valgrind only 616 errors, 17 contexts, all assoc'd with Eigen::operator<< (likely ignorable?)
    if(1) for(; prjax < (int)nProj; ++prjax)
    {
        init_w( w, x,y,nc, weights_avg,prjax, (prjax<reuse_dim) );
        cout<<" start projection "<<prjax<<" w.norm="<<w.norm();
        if( w.size()<50U ){ w.toVectorXd(tmp); cout<<" w: "<<tmp.transpose(); } cout<<endl;
        xwProj.w_changed();                     // invalidate w-dependent stuff (projections)
        luPerm.init( xwProj.std(), y, nc );     // std because w can't have started averaging yet
        luPerm.rank( GetMeans(params.reorder_type) );
        PROFILER_STOP_START("optimizeLU.profile");
        if (params.optimizeLU_epoch > 0) { // questionable.. l,u at this point are random, so why do it this early?
            OPTIMIZE_LU(" PROJ_INIT");
            // above call can give mis-ordered {lower,upper}, with lower > upper !!!
        }
        luPerm.mkok_sortlu();
        PROFILER_STOP;
        print_report(prjax,batch_size, nClass,C1,C2,lambda,w.size(),print_report(x));
        t = 0;        // part of MCsoln data -- TODO -- continue iterating
        if(PRINT_O){
            if(0)cout<<" t=0 luPerm.{ok_lu="<<luPerm.ok_lu<<",ok_sortlu="<<luPerm.ok_sortlu
                <<",  ok_lu_avg="<<luPerm.ok_lu_avg<<",ok_sortlu_avg="<<luPerm.ok_sortlu_avg
                    <<" nAccSortlu_avg="<<luPerm.nAccSortlu_avg<<"}"<<endl;
            cout<<"objective_val[  t   ]:       value    w.norm (initially "<<w.norm()<<")\n"
                <<"--------------------- --------------- ------"<<endl;

        }
        PROFILER_START("learning.profile");
        // -------- main iteration loop --------
        while (t < params.max_iter) {
            ++t;
            MCiterBools t4(t, params);          // "time for" ... various const bool
            if (!params.report_epoch && t % 1000 == 0) { // print some progress
                snprintf(iter_str,30, "Projection %d > ", prjax+1);
                print_progress(iter_str, t, params.max_iter);
                cout.flush();
            }
            OPT_GRADIENT_TEST;
            eta_t = set_eta(params, t, lambda); // set eta for this iteration
            // compute the gradient and update
            //      modifies w, sortedLU, (sortedLU_avg) ONLY  (l,u,l_avg,u_avg values may be stale)
            // --> the ONLY place where 'w' is modified
            MCupdate::update(w, luPerm, eta_t,   /*R/O:*/x, y, xSqNorms, C1, C2, lambda, t,
                             nTrain, nclasses, maxclasses, filtered, updateSettings, params);
            // update 'w' ==> projections of raw 'x' data (projection_avg & projection) invalid.
            xwProj.w_changed();
            if(t4.reorder) {
                if(ITER_TRACE){ std::cout<<" REORDER"<<t<<" "; std::cout.flush(); }
                if( luPerm.nAccSortlu_avg ) luPerm.mkok_lu_avg();       // sortlu_avg --> {l,u}_avg
                luPerm.mkok_lu();                                       // sortlu     --> {l,u}
                luPerm.rank( GetMeans(params.reorder_type) );   // <-- new sort order (sortlu* no good)
                luPerm.mkok_sortlu();                                   // {l,u}     --> sortlu
                if( luPerm.nAccSortlu_avg ) luPerm.mkok_sortlu_avg();   // {l,u}_avg --> sortlu_avg
            }
            if (t4.optimizeLU){  // w, luPerm-ranking constant, optimize {l,u}. Needs valid 'projection'
                // if ranking type is REORDER_RANGE_MIDPOINTS, perhaps optimizeLU before luPerm.rank()?
                OPTIMIZE_LU(" OPTLU"<<t<<" ");  assert( luPerm.ok_lu );
                luPerm.mkok_sortlu(); // XXX is this ever nec.?
            }
            bool haveObjectiveHinge = false;
            if(t4.report){ // params.report_epoch && (t % params.report_epoch == 0) )
                // calculate the objective functions with respect to the current w and bounds
                objective_val[obj_idx++] = ObjectiveHinge("REPORT",t);
                haveObjectiveHinge = true; // maybe avoid recalc
                if(PRINT_O) {
                    cout<<"objective_val["<<setw(6)<<t<<"]: "<<setw(15)<<objective_val[obj_idx-1]<<" ";
                    cout<<w.norm()<<endl;
                }
            }
#ifndef NDEBUG
            if( t4.doing_avg_epoch ) assert( w.getAvg_t() > 0U );
            else                     assert( w.getAvg_t() == 0U );
#endif
            if(t4.report_avg){
                double objective_avg;       // calculate the objective for the averaged w [if avail]
                // if optimizing LU then this is expensive since it runs the optimization
                if ( w.getAvg_t() ) { // equiv. t4.doing_avg_epoch, so use *_avg for objective
                    VectorXd sortedLU_test( luPerm.l_avg.size()*2U );
                    if (params.optimizeLU_epoch > 0) { // <-- Note: {l,u}_avg as SCRATCH storage ...
                        OPTIMIZE_LU_AVG("REPORT_AVG_opt"); // sortlu_avg UNCHANGED, sortlu ~ accumulator
                        //get_sortedLU( sortedLU_test, l_avg, u_avg, sorted_class);
                        luPerm.toSorted( sortedLU_test, luPerm.l_avg, luPerm.u_avg );
                    }else{ assert( luPerm.nAccSortlu_avg > 0U );
                        if(ITER_TRACE){ std::cout<<"_avg"; std::cout.flush(); }
                        sortedLU_test = luPerm.sortlu_avg/(t - params.avg_epoch + 1);
                    }
                    objective_avg = ObjectiveHingeAvg("REPORT_AVG_a",t);
                }else if(haveObjectiveHinge){           // no avg avail: prev calc avail
                    if(ITER_TRACE){ std::cout<<" REPORT_AVG_b"<<t<<" "; std::cout.flush(); }
                    objective_avg = objective_val[obj_idx - 1];
                } else {                                // no avg avail: do full calc
                    objective_avg = ObjectiveHinge("REPORT_c",t);
                }
                objective_val_avg[obj_idx_avg++] = objective_avg;
                if(PRINT_O) {
                    cout<<"objective_avg["<<t<<"]:"<<objective_avg<<" ";
                    if( w.getAvg_t() ) cout<<w.norm_avg()<<" (avg)"<<endl; // 'if' for valgrind? XXX
                    else               cout<<w.norm()<<" (std)"<<endl;
                }
            }
        } // **** **** end while t **** ****
        PROFILER_STOP_START("filtering.profile");
        if(PRINT_O) cout<<" * end iterations:"
            //<<" * ok_projection="<<ok_projection<<" ok_projection_avg="<<ok_projection_avg
            <<" * luPerm ok_lu="<<luPerm.ok_lu<<" ok_sortlu="<<luPerm.ok_sortlu
                <<" * luPerm ok_lu_avg="<<luPerm.ok_lu_avg<<" ok_sortlu_avg="<<luPerm.ok_sortlu_avg<<" nAccSortlu_avg="<<luPerm.nAccSortlu_avg
                <<endl;

        // optimize LU and compute objective for averaging if it is turned on
        // if t = params.avg_epoch, everything is exactly the same as
        // just using the current w
        if ( params.avg_epoch && t > params.avg_epoch ) {
            // only need to reorder the classes if optimizing LU
            // or if we are interested in the last obj value
            // do the reordering based on the averaged w
            if (params.reorder_epoch > 0 && (params.optimizeLU_epoch > 0 || params.report_avg_epoch > 0)) {
                if(ITER_TRACE){ std::cout<<" proj_RANK_AVG "; std::cout.flush(); }
                proj_means(means, nc, xwProj.avg(), y);
                luPerm.rank(means);
            }
            // optimize the lower and upper bounds (done after class ranking)
            // since it depends on the ranks
            if (params.optimizeLU_epoch > 0) {
                OPTIMIZE_LU_AVG(" proj_OPTLU_avg");
            }
            if( params.report_avg_epoch>0 ) { // calculate the objective for the averaged w
                assert( luPerm.ok_lu );
                if(ITER_TRACE){ std::cout<<" proj_OBJ_avg "; std::cout.flush(); }
                luPerm.mkok_sortlu_avg();
                objective_val_avg[obj_idx_avg++] = ObjectiveHingeAvg("proj_OBJ_avg",t);
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
            // even if reordering is based on the averaged w
            // do it here based on the w to get the optimal LU and
            // the best objective with respect to w
            luPerm.rank( GetMeansNoAvg(params.reorder_type) );  // ok_sortlu_avg is no longer needed.
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
            luPerm.mkok_sortlu();
            objective_val[obj_idx++] = ObjectiveHinge("",t);
            if(PRINT_O) {
                cout << "objective_val[" <<setw(6)<<t << "]: " << objective_val[obj_idx-1] << " "<< w.norm();
            }
        }

        if(PRINT_O) cout<<" * END iterations: * luPerm ok_lu="<<luPerm.ok_lu<<" ok_sortlu="<<luPerm.ok_sortlu
                <<" * luPerm ok_lu_avg="<<luPerm.ok_lu_avg<<" ok_sortlu_avg="<<luPerm.ok_sortlu_avg
                <<" nAccSortlu_avg="<<luPerm.nAccSortlu_avg<<endl;
        if(ITER_TRACE){ std::cout<<" UPD-W-AVG.col("<<prjax<<")"; }
        weights.col(prjax) = w.getVec();
        lower_bounds.col(prjax) = luPerm.l;
        upper_bounds.col(prjax) = luPerm.u;
        if( w.getAvg_t() ) { // equiv. ( params.avg_epoch && t > params.avg_epoch )
            weights_avg.col(prjax) = w.getVecAvg();
            luPerm.mkok_lu_avg(); if( ! luPerm.ok_lu_avg ) throw std::runtime_error("BAD final luPerm.ok_lu_avg");
            lower_bounds_avg.col(prjax) = luPerm.l_avg;
            upper_bounds_avg.col(prjax) = luPerm.u_avg;
        }else{
            weights_avg.col(prjax) = weights.col(prjax);
            luPerm.mkok_lu(); if( ! luPerm.ok_lu ) throw std::runtime_error("BAD final luPerm.ok_lu");
            lower_bounds_avg.col(prjax) = luPerm.l;
            upper_bounds_avg.col(prjax) = luPerm.u;
        }

        // should we do this in parallel?
        // the main problem is that the bitset is not thread safe (changes to one bit can affect changes to other bits)
        // should update to use the filter class
        // things will not work correctly with remove_class_constrains on. We need to update wc, nclass
        //       and maybe nc
        // check if nclass and nc are used for anything else than weighting examples belonging
        //       to multiple classes
        if (params.remove_constraints && prjax < (int)nProj-1) {
            if( w.getAvg_t() ){ // equiv. (params.avg_epoch && t > params.avg_epoch )
                if(ITER_TRACE){ std::cout<<" proj_FILTER_avg "; std::cout.flush(); }
                luPerm.mkok_lu_avg();  //-- *just* finished doing this
                update_filtered(filtered,  /*inputs:*/ xwProj.avg(), luPerm.l_avg, luPerm.u_avg
                                , y, params.remove_class_constraints);
            }else{
                if(ITER_TRACE){ std::cout<<" proj_FILTER_std "; std::cout.flush(); }
                luPerm.mkok_lu();
                update_filtered(filtered, xwProj.std(), luPerm.l, luPerm.u, y, params.remove_class_constraints);
            }

            no_filtered = filtered.count();
            cout<<"Filter["<<filtered.rows()<<"x"<<filtered.cols()<<"] removed "<<no_filtered
                <<" of "<<total_constraints<<" constraints"<<endl;
            // work on this. This is just a crude approximation.
            // now every example - class pair introduces nclass(example) constraints
            // if weighting is done, the number is different
            // eliminating one example -class pair removes nclass(exmple) potential
            // if the class not among the classes of the example
            if (params.reweight_lambda != REWEIGHT_NONE){
                long const no_remaining = (int)total_constraints - no_filtered;
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
                    const_cast<param_struct*>(params_arg)->no_projections = prjax+1U; // <-- NB
                    const_cast<param_struct&>(params).no_projections = prjax+1U; // <-- NB
                    chopProjections( prjax+1U );
                    break;
                }
            }
        }
        //C2*=((nTrain-1)*nClass)*1.0/no_remaining;
        //C1*=((nTrain-1)*nClass)*1.0/no_remaining;
        PROFILER_STOP;
        if(ITER_TRACE || PRINT_O)
            cout<<endl;
    } // end for prjax
    PROFILER_STOP;
}
#endif //MCSOLVER_HH
