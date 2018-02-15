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
#include <time.h>
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
//-- make this a lambda function?
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
  , MCITER_PERIODIC( optimizeLU )
#if GRADIENT_TEST
  , MCITER_PERIODIC( finite_diff_test )
#endif
  , doing_avg_epoch( params.avg_epoch && t >= params.avg_epoch )    // <-- true after a certain 't'
  , progress( params.verbose >= 1 && !params.report_epoch && t % 1000 == 0)  // print some progress
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
inline void MCpermState::toSorted( VectorXd & sorted, VectorXd const& ll, VectorXd const& uu ) const{
  for(size_t i=0; i<perm.size(); ++i){
    sorted.coeffRef(2U*i)    = ll.coeff(perm[i]);
    sorted.coeffRef(2U*i+1U) = uu.coeff(perm[i]);
  }
}

// inline void MCpermState::accumulate_sortlu(){
//   mkok_sortlu();
//   sortlu_acc = sortlu_acc + sortlu;
//   ++nAccSortlu;
//   ok_lu_avg = false; // we've accumulated, which invalidates lu_avg.
// }

inline void MCpermState::chg_sortlu(){
    ok_lu = false;
    ok_lu_avg = false;
    ok_sortlu_avg = false;
}

// inline void MCpermState::chg_lu(){              // multiple calls OK
//   // this should only be called from optimize_LU, so move the code there
//   ok_sortlu = false;
//   ok_sortlu_avg = false;
// }

inline void MCpermState::mkok_lu(){
  if(!ok_lu)
    {
      assert (ok_sortlu);
      toLu( l, u, sortlu );
      ok_lu = true;
    }
}
inline void MCpermState::mkok_lu_avg(){
  if(!ok_lu_avg)
    {
      if(nAccSortlu > 0) // averaging has started
	{
	  mkok_sortlu_avg();	  
	  toLu( l_avg, u_avg, sortlu_avg );
	  ok_lu_avg = true;
	}
      else // averaging has not started, Just coppy l and u. 
	{
	  mkok_lu();
	  l_avg = l;
	  u_avg = u;
	}	
    }
}

inline VectorXd& MCpermState::mkok_sortlu(){
  if(!ok_sortlu){
    assert(ok_lu && nAccSortlu == 0); //the only time sortlu is not ok is when optmize_LU has been called
                                    // which resets the averaging. 
    toSorted( sortlu, l, u );
    ok_sortlu = true;
  }
  return sortlu;
}

inline VectorXd& MCpermState::mkok_sortlu_avg(){
  if(!ok_sortlu_avg) {
    if (ok_lu_avg){ // if we have good lu_avg then use them
      toSorted( sortlu_avg, l_avg, u_avg);
    }else if (nAccSortlu > 0){ // if averaging has started the sortlu must be ok. 
      assert(ok_sortlu);
      sortlu_avg = sortlu_acc/nAccSortlu;
    } else { // averaging has not started. Copy sortlu into sortlu_avg 
      mkok_sortlu();
      sortlu_avg = sortlu;
    }
    ok_sortlu_avg = true;
  }      
  return sortlu_avg;
}

inline void MCpermState::rank( VectorXd const& sortkey ){
    mkok_lu();
    if (nAccSortlu > 0U)
      {
	mkok_lu_avg();  // coppy sortlu_avg to lu_avg
      }
    Perm::rank( sortkey );
    ok_sortlu = false;
    mkok_sortlu();
    if (nAccSortlu > 0U)
      {
	ok_sortlu_avg = false;
	// this is a bit wastefull, but avoids new code, and it is not called often. 
	mkok_sortlu_avg(); // because ok_lu_avg = true, this will coppy lu_avg to sortedlu_avg
	sortlu_acc = sortlu_avg * nAccSortlu; //restore sortlu_acc
      }
}



    template< typename EIGENTYPE >
void MCsolver::solve( EIGENTYPE const& x, SparseMb const& y,
                     param_struct const& params_arg /*= nullptr*/ )
{
    using namespace std;

    this->d = x.cols();
    const size_t nTrain = x.rows();
    this-> nClass = y.cols();


    param_struct params(params_arg);
    finalize_default_params(params); // just in case it was not done until now
    // set the default avg_epoch if not already set
    if (params.default_avg_epoch)
      {
	params.avg_epoch = nTrain>d?nTrain:d;
	params.default_avg_epoch = false;
      }
    // multiply C1 by number of classes
    params.C1 = params.C1 * nClass;
    
    this->nProj = params.no_projections;
    if( (size_t)nProj >= d ){
      cerr<<"WARNING: no_projections > example dimensionality"<<endl;
    }

    if ( params.verbose >= 1)
      {
	cout << "size x: " << x.rows() << " rows and " << x.cols() << " columns.\n";
	cout << "size y: " << y.rows() << " rows and " << y.cols() << " columns.\n";
      }

    // initialize the random seed generator
    if (params.seed > 1)
      {
	srand(params.seed);
      }
    else if (params.seed == 0)
      {
	srand (time(NULL));
      }

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
	  //	  std::cout<<" ~Proj{ngetSTD="<<ngetSTD<<",ngetAVG="<<ngetAVG<<",nReuse="<<nReuse
	  //               <<",nSwitch="<<nSwitch<<",nDemote="<<nDemote<<"}"<<std::endl;
        }
        /** Every time client changes \c w, \c Proj \b must be informed that the world is now different. */
        void w_changed() {
            valid = false;
        }
        /** get w's projection, silently demote from AVG to STD if w.getAvg_t() is still zero */
        VectorXd const& operator()( enum Type t ){
            // can we do what was asked?
            if ( t == STD ){
                ++ngetSTD;
            }else if( t == AVG ){
                ++ngetAVG;
                if( w.getAvg_t() == 0 ){
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
	      w.project( v, x );
            }else{
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

    

    size_t obj_idx = 0;
    // in the multilabel case each example will have an impact proportinal
    // to the number of classes it belongs to. ml_wt and ml_wt_class
    // allows weighting that impact when updating params for the other classes
    // respectively its own class.
    //  size_t  i=0, idx=0;
    char iter_str[30];

    // how to split the work for gradient update iterations
    int const nThreads = this->getNthreads( params );
    MCupdateChunking updateSettings( nTrain/*x.rows()*/, nClass, nThreads, params );
    //MCupdateChunking updateSettings( nTrain/*x.rows()*/, nClass, 2, params );  // 2 might be fastest?
#ifndef NDEBUG
    {// in debug mode check no change from original settings
        const size_t batch_size = (params.batch_size < 1 || params.batch_size > nTrain) ? (size_t) nTrain : params.batch_size;
        if (params.verbose >= 1) cout<<" batch_size = "<<batch_size<<endl;
        int const sc_chunks = nThreads;
        int const sc_chunk_size = (params.class_samples?params.class_samples:nClass)/sc_chunks;
        int const sc_remaining = (params.class_samples?params.class_samples:nClass) % sc_chunks;
        int const idx_chunks = nThreads/sc_chunks;
        if (params.verbose >= 1) std::cout<<" idx_chunks="<<idx_chunks<<std::endl;
        int const idx_chunk_size = batch_size/idx_chunks;
        int const idx_remaining = batch_size % idx_chunks;
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
    
    double lambda = 1.0/params.C2;
    double C1 = params.C1/params.C2;
    double C2 = 1.0;
        
    Proj xwProj( x, w );
    int prjax = 0;
    
    // --------------- define lambdas to prepackage recurring calculation ------------------------

    // It silently demotes REORDER_AVG_PROJ_MEANS if 'w' has not begun averaging.
    auto GetMeans = [&]( enum Reorder_Type reorder ) -> VectorXd const&
      {
	switch (reorder){
	case REORDER_AVG_PROJ_MEANS:
	  proj_means(means, nc, xwProj.avg(), y); // if avg has not started ,xwProj silently uses non-averaged w
	  break;
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

    auto ObjectiveHinge = [&] ( ) -> double
      {
	luPerm.mkok_sortlu();
	return calculate_objective_hinge( xwProj.std(), y, nclasses, luPerm.perm, luPerm.rev,
					  w.norm(), luPerm.sortlu, filtered,
					  lambda, C1, C2, params); 
      };

    auto ObjectiveHingeAvg = [&] ( ) -> double
      {
	luPerm.mkok_sortlu_avg();
	return calculate_objective_hinge( xwProj.avg(), y, nclasses, luPerm.perm, luPerm.rev,
					  w.norm_avg(), luPerm.sortlu_avg, filtered,
					  lambda, C1, C2, params); 
      };
    
    auto OptimizeLU = [&] () -> void
      {					  
	/* changes l, u, invalidates sortlu */	
	luPerm.optimizeLU( xwProj.std(), y, wc, nclasses, filtered, C1, C2, params); 
      };
    auto OptimizeLU_avg =[&] () -> void
      {
	/* changes {l,u}_avg instead of {l,u}. Invalidates sortlu_avg */ 
	luPerm.optimizeLU_avg( xwProj.avg(), y, wc, nclasses, filtered, C1, C2, params);
      };

    auto RemoveConstraints = [&] () -> void
      {	
	// should we do this in parallel?
	// the main problem is that the bitset is not thread safe (changes to one bit can affect changes to other bits)
	// should update to use the filter class
	// things will not work correctly with remove_class_constrains on. We need to update wc, nclass
	//       and maybe nc
	// check if nclass and nc are used for anything else than weighting examples belonging
	//       to multiple classes

	// if averagin has not started this will be using the non-averaged w,l and u
	luPerm.mkok_lu_avg(); // should be OK, but just in case
	update_filtered(filtered,  /*inputs:*/ xwProj.avg(), luPerm.l_avg, luPerm.u_avg
			, y, params.remove_class_constraints);
	
	no_filtered = filtered.count(); 
	if (params.verbose >= 1)
	  {
	    cout<<"Filter["<<filtered.rows()<<"x"<<filtered.cols()<<"] removed "<<no_filtered
		<<" of "<<total_constraints<<" constraints"<<endl;
	  }
	if( no_filtered > total_constraints )
	  throw std::runtime_error(" programmer error: removed more constraints than exist?");
	if( no_filtered == total_constraints )
	  {
	    cerr<<setw(20)<<""<<"  CANNOT CONTINUE, no more constraints left\n"
	      "  Stopping at "<<prjax+1U<< " filters." << endl;
	    setNProj(prjax+1U, true, true);
	    return;
	  }
	
	if (params.reweight_lambda != REWEIGHT_NONE)
	  {
	    long const no_remaining = (int)total_constraints - no_filtered;
	    lambda = no_remaining*1.0/(total_constraints*params.C2);
	    if (params.reweight_lambda == REWEIGHT_ALL)
	      {
		C1 = params.C1*no_remaining*1.0/(total_constraints*params.C2);
	      }
	  }
      };

    // ------------- end defining lambda functions --------------------------

    if (params.reoptimize_LU && !params.resume && params.no_projections > nProj)
      throw std::runtime_error("Error, --reoptlu specified with more projections than current solution requested and --resume.\n");
    
    bool const keep_weights = params.resume || params.reoptimize_LU;
    bool const keep_LU = keep_weights && !params.reoptimize_LU;
    int reuse_dim = keep_weights?min(nProj,params.no_projections):0;
    setNProj(params.no_projections, keep_weights, keep_LU);
    
    if (params.verbose >= 1)
      {
	cout<<"  ... begin with weights"<<prettyDims(weights)
	    <<" lower_bounds"<<prettyDims(lower_bounds)<<" upper_bounds"<<prettyDims(upper_bounds)
	    <<endl;
      }
    
    //-- move this to parameters check
    if (params.reoptimize_LU && params.reorder_type == REORDER_RANGE_MIDPOINTS )
      {
        throw std::runtime_error("Error, reordering REORDER_RANGE_MIDPOINTS should "
				 "not be used when reoptimizing the LU parameters");
      }
    
    for ( ;prjax < reuse_dim; ++prjax)
      { 
	w.init(weights.col(prjax));
	xwProj.w_changed();  // projections of 'x' onto 'w' no longer valid
	if (params.reoptimize_LU) {
	  luPerm.init( xwProj.std(), y, nc );     // try to appease valgrind?
	  luPerm.rank( GetMeans(params.reorder_type) );
	  OptimizeLU(); // w,projection,sort_order ----> luPerm.l,u
	  lower_bounds.col(prjax) = luPerm.l;
	  upper_bounds.col(prjax) = luPerm.u;
	}else{
	  luPerm.set_lu( lower_bounds.col(prjax), upper_bounds.col(prjax) );
	}
	
	if (params.remove_constraints && prjax < (int)nProj-1) 
	  {
	    RemoveConstraints();
	  }	
      }

    obj_idx = objective_val.size();
    
    if(params.verbose >= 1)
      {
      cout<<"  ... starting with     weights"<<prettyDims(weights)<<endl;
      cout<<"  ... beginning at prjax="<<prjax<<" reuse_dim="<<reuse_dim<<endl;
      cout<<"  ... sc_chunks="<<updateSettings.sc_chunks<<" MCTHREADS="<<MCTHREADS<<endl;
      }
    {// more space for objective_val history, per new projection...
      int const newProjs = nProj - prjax;
      if(params.report_epoch > 0){
	size_t const nReports = params.max_iter / params.report_epoch + 1U;
	size_t const more = newProjs * (nReports+1U);
	objective_val.conservativeResize(obj_idx + more );
      }
    }
    assert(w.getAvg_t() == 0);
    for(; prjax < nProj; ++prjax)
      {
	//        init_w( w, x,y,nc, weights_avg,prjax, (prjax<reuse_dim) );
	init_w( w, x, y, nc, weights, prjax, params);
	
	if (params.verbose >= 1)
	  cout<<" start projection "<<prjax<<" w.norm="<<w.norm() << endl;
	xwProj.w_changed();                     // invalidate w-dependent stuff (projections)
	luPerm.init( xwProj.std(), y, nc );     // std because w can't have started averaging yet
	luPerm.rank( GetMeans(params.reorder_type) );
	if (params.optimizeLU_epoch > 0) { 
	  OptimizeLU();
	}
	luPerm.mkok_sortlu();
	if (params.verbose >= 1)
	  {
	    print_report(prjax, updateSettings.batch_size, nClass,C1,C2,lambda,w.size(),print_report(x));
	  }
	if(params.verbose >= 1){
	  cout<<"Iteration   "<<setw(10)<<"Objective   "<<"w.norm"<<endl;
	  cout<<"----------  ----------  -------"<<endl;	
	} 
	uint64_t t = 0;   	// -------- main iteration loop --------
	while (t < params.max_iter) {
	  ++t;
	  MCiterBools t4(t, params);          // "time for" ... various const bool
	  if (t4.progress) { // print some progress
	    snprintf(iter_str,30, "Projection %d > ", prjax+1);
	    print_progress(iter_str, t, params.max_iter);
	    cout.flush();
	  }
	  OPT_GRADIENT_TEST;
	  double eta_t = set_eta(params, t, lambda); // set eta for this iteration
	  // compute the gradient and update
	  //      modifies w, sortedLU, (sortedLU_avg) ONLY  (l,u,l_avg,u_avg values may be stale)
	  // --> the ONLY place where 'w' is modified
	  MCupdate::update(w, luPerm, eta_t,   /*R/O:*/x, y, xSqNorms, C1, C2, lambda, t,
			   nTrain, nclasses, maxclasses, filtered, updateSettings, params);
	  // update 'w' ==> projections of raw 'x' data (projection_avg & projection) invalid.
	  xwProj.w_changed();
	  if(t4.reorder) {
	    if(params.verbose >= 2){ std::cout<<" REORDER"<<t<<" "; std::cout.flush(); }
	    luPerm.rank( GetMeans(params.reorder_type) );   // <-- new sort order (sortlu* no good)
	  }
	  if (t4.optimizeLU){  // w, luPerm-ranking constant, optimize {l,u}. Needs valid 'projection'
	    // optimize the lower and upper bounds (done after class ranking)
	    // since it depends on the ranks
	    OptimizeLU();  
	  }
	  if(t4.report){ // params.report_epoch && (t % params.report_epoch == 0) )
	    // calculate the objective functions with respect to the current w and bounds
	    objective_val[obj_idx++] = ObjectiveHinge();
	    if(params.verbose >= 1) {
	      cout<<setw(10)<<t<<"  "<<setw(10)<<objective_val[obj_idx-1]<<"  "<<w.norm()<<endl;
	    }
	  }
	} // **** **** end while t **** ****
	if( params.report_epoch>0 && t % params.report_epoch != 0 ) {
	  objective_val[obj_idx++] = ObjectiveHinge();
	  if(params.verbose >= 1 ) {
	    cout<<setw(10)<<t<<"  "<<setw(10)<<objective_val[obj_idx-1]<<"  "<<w.norm()<<endl;
	  }
	}

	if(params.verbose >= 1)
	  {
	    cout<<"----------  ----------  -------"<<endl;	
	    cout<<" * end iterations" <<endl;
	  }
	// optimize LU and compute objective for averaging if it is turned on
	// if t = params.avg_epoch, everything is exactly the same as
	// just using the current w

	// only need to reorder the classes if optimizing LU
	// or if we are interested in the last obj value
	if (params.reorder_epoch > 0 && (params.optimizeLU_epoch > 0 || params.report_epoch > 0)) {
	  luPerm.rank(GetMeans(params.reorder_type));
	}
	// optimize the lower and upper bounds (done after class ranking)
	// since it depends on the ranks
	if (params.optimizeLU_epoch > 0) {
	  OptimizeLU_avg();
	}
	if( params.report_epoch>0 ) { // calculate the objective for the averaged w
	  objective_val[obj_idx++] = ObjectiveHingeAvg();
	  if(params.verbose>=1) {
	    cout << "Final objective_val[" << prjax << "]: " << objective_val[obj_idx-1] << endl;
	    }
	  }
			
	weights.col(prjax) = w.getVecAvg();
	luPerm.mkok_lu_avg();
	lower_bounds.col(prjax) = luPerm.l_avg;
	upper_bounds.col(prjax) = luPerm.u_avg;
	
	if (params.remove_constraints && prjax < (int)nProj-1) 
	  {
	    RemoveConstraints();
	  }   
	if(params.verbose >= 1) 
	  cout<<" * END projection "<<prjax << endl << endl;
	
      } // end for prjax
}
#endif //MCSOLVER_HH
