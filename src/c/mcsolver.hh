#ifndef MCSOLVER_HH
#define MCSOLVER_HH
/** \file
 * MCsolver::solve impl
 */

#include "mcsolver.h"
#include "mcsolver_detail.hh"
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
#include "gradient_test.h"
//-- make this a lambda function?
#define OPT_GRADIENT_TEST do \
{   /* now thread-safe, just in case */ \
    static unsigned thread_local seed = static_cast<unsigned>(this_thread.get_id()); \
    if(t4.finite_diff_test){            /* perform finite differences test */ \
        for (size_t fdtest=0; fdtest<params.no_finite_diff_tests; ++fdtest) { \
            size_t idx = ((size_t) rand_r(&seed)) % n; \
            finite_diff_test( w, x, idx, y, nclasses, maxclasses, inside_weight, outside_weight, \
			      luPerm.perm()/*sorted_class*/,		\
                              luPerm.rev(), luPerm.sortlu(), filtered, C1, C2, params); \
        } \
    } \
}while(0)
#else
#define OPT_GRADIENT_TEST do{}while(0)
#endif

// first let's define mcsolver.h utilities...

/** iteration state that does not need saving -- important stuff is in MCsoln */
struct MCiterBools
{
  /** constructor includes "print some progress" block to cout */
  MCiterBools( uint64_t const t, param_struct const& params );
  bool const reorder;              ///< true if param != 0 && t%param==0
  bool const report;               ///< true if param != 0 && t%param==0
  bool const optimizeLU;           ///< true if param != 0 && t%param==0
  //  bool const doing_avg_epoch;      ///< avg_epoch && t >= avg_epoch
  bool const progress;             ///< params.verbose >= 1 && !params.report_epoch && t % 1000 == 0
#if GRADIENT_TEST
  bool const finite_diff_test;     ///< true if param != 0 && t%param==0
#endif
};


#define MCITER_PERIODIC(XXX) XXX( params.XXX##_epoch && t % params.XXX##_epoch == 0 )
inline MCiterBools::MCiterBools( uint64_t const t, param_struct const& params )
  : MCITER_PERIODIC( reorder )
  , MCITER_PERIODIC( report )
  , MCITER_PERIODIC( optimizeLU )
#if GRADIENT_TEST
  , MCITER_PERIODIC( finite_diff_test )
#endif
  , progress( params.verbose >= 1 && !params.report_epoch && t % 1000 == 0)  // print some progress
{
}
#undef MCITER_PERIODIC

inline void Perm::rank( VectorXd const& sortkey ){
    assert( (size_t)sortkey.size() == m_perm.size() );
    std::iota( m_perm.begin(), m_perm.end(), size_t{0U} );
    std::sort( m_perm.begin(), m_perm.end(), [&sortkey](size_t const i, size_t const j){return sortkey[i]<sortkey[j];} );
    for(size_t i=0U; i<m_perm.size(); ++i)
        m_rev[m_perm[i]] = i;
}
inline void MCpermState::toLu( VectorXd & ll, VectorXd & uu, VectorXd const& sorted ) const{
  double const* plu = sorted.data();
  auto pPerm = m_perm.begin(); // perm ~ old perm
  for(; pPerm != m_perm.end(); ++pPerm){
    size_t const cp = *pPerm;
    ll.coeffRef(cp) = *(plu++);
    uu.coeffRef(cp) = *(plu++);
  }
}
inline void MCpermState::toSorted( VectorXd & sorted, VectorXd const& ll, VectorXd const& uu ) const{
  for(size_t i=0; i<m_perm.size(); ++i){
    sorted.coeffRef(2U*i)    = ll.coeff(m_perm[i]);
    sorted.coeffRef(2U*i+1U) = uu.coeff(m_perm[i]);
  }
}

inline void MCpermState::reset_acc(){
  // reset accumulation since changes to lu do not come from a gradient step
  if (nAccSortlu > 0U)
    {
      nAccSortlu = 0U;
      m_sortlu_acc.setZero();
    }    
}  

inline void MCpermState::chg_sortlu(){
    ok_lu = false;
    ok_lu_avg = false;
    ok_sortlu_avg = false;
}

inline void MCpermState::mkok_lu(){
  if(!ok_lu)
    {
      assert (ok_sortlu);
      toLu( m_l, m_u, m_sortlu );
      if (nAccSortlu > 0 ) // averaging has started. Need to update l_acc and u_acc from  sortlu_acc
	{
	  toLu(m_l_acc, m_u_acc, m_sortlu_acc);
	}
      ok_lu = true;
    }
}

inline void MCpermState::mkok_lu_avg(){
  if(!ok_lu_avg)
    {
      if (ok_sortlu)
	{
	  mkok_sortlu_avg();	  
	  toLu( m_l_avg, m_u_avg, m_sortlu_avg );
	}
      else
	{
	  mkok_lu();
	  if(nAccSortlu > 0) // averaging has started
	    {
	      m_l_avg = m_l_acc/nAccSortlu;
	      m_u_avg = m_u_acc/nAccSortlu;
	    }
	  else // averaging has not started 
	    {
	      m_l_avg = m_l;
	      m_u_avg = m_u;
	    }	      
	}
      ok_lu_avg = true;
    }
}

inline void MCpermState::mkok_sortlu(){
  if(!ok_sortlu){
    assert(ok_lu);
    toSorted( m_sortlu, m_l, m_u );
    if (nAccSortlu > 0) //averaging has started
      {
	toSorted(m_sortlu_acc, m_l_acc, m_u_acc);
      }
    ok_sortlu = true;
  }
}

inline void MCpermState::mkok_sortlu_avg(){
  if(!ok_sortlu_avg) {    
    if (ok_lu_avg){ // if we have good lu_avg then use them
      toSorted( m_sortlu_avg, m_l_avg, m_u_avg);
    } else {
      mkok_sortlu();
      if (nAccSortlu > 0){ // averaging has started
	m_sortlu_avg = m_sortlu_acc/nAccSortlu;
      } else { // averaging has not started. Copy sortlu into sortlu_avg 
	m_sortlu_avg = m_sortlu;
      }
    }
    ok_sortlu_avg = true;
  }      
}

inline VectorXd const& MCpermState::l() 
{
  mkok_lu();
  return m_l;
}
inline VectorXd const& MCpermState::u()
{
  mkok_lu();
  return m_u;
}
inline VectorXd const& MCpermState::sortlu()
{
  mkok_sortlu();
  return m_sortlu;
}
inline VectorXd const& MCpermState::l_avg()
{
  mkok_lu_avg();
  return m_l_avg;
}
inline VectorXd const& MCpermState::u_avg()
{
  mkok_lu_avg();
  return m_u_avg;
}
inline VectorXd const& MCpermState::sortlu_avg()
{
  mkok_sortlu_avg();
  return m_sortlu_avg;
}

std::vector<int> const& MCpermState::perm() const
{
  return Perm::m_perm;
}
std::vector<int> const& MCpermState::rev() const
{
  return Perm::m_rev;
}



inline void MCpermState::rank( VectorXd const& sortkey ){
  mkok_lu();
  Perm::rank( sortkey );
  ok_sortlu = false;
  mkok_sortlu();
}



template< typename EIGENTYPE >
void MCsolver::solve( EIGENTYPE const& x, SparseMb const& y,
		      param_struct const& params_arg)
{
  using namespace std;
  
  this->d = x.cols();
  const size_t nTrain = x.rows();
  this-> nClass = y.cols();
  
  
  param_struct params(params_arg);

  // set the default avg_epoch if not already set
  if (params.averaged_gradient && params.avg_epoch == 0)
    {
      params.avg_epoch = nTrain>d?nTrain:d;
    }
  // multiply C1 by number of classes
  params.C1 = params.C1 * nClass;
    
  this->nProj = params.nfilters;
  if( (size_t)nProj >= d ){
    cerr<<"WARNING: nfilters > example dimensionality"<<endl;
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

  //keep track of which classes have been elimninated for a particular example
  boolmatrix filtered(nTrain,nClass);    
  VectorXi nclasses; // nclasses[example] = number of classes assigned to each training example
  VectorXd inside_weight; // inside_weight[example] = weight of this example in the inside interval constraints 
  VectorXd outside_weight; // inside_weight[example] = weight of this example in the outside interval constraints 
  mcsolver_detail::init_nclasses(nclasses, inside_weight, outside_weight, y, params);
  int maxclasses = nclasses.maxCoeff(); // the maximum number of classes an example might have
  // Suppose example y[i] --> weight of 1.0, or if params.ml_wt_class_by_nclasses, 1.0/nclasses[i]
  // Then what is total weight of each class? (used for optimizeLU)
  VectorXi nc;       // nc[class]         = number of training examples of each class
  VectorXd wc; // wc[class] = weight of each class (= nc[class] if params.ml_wt_class_by_nclasses==false)
  mcsolver_detail::init_nc(nc, wc, inside_weight, y, params, filtered);   // wc is used if optimizeLU_epoch>0
  VectorXd xSqNorms;
  if (params.update_type == SAFE_SGD) mcsolver_detail::calc_sqNorms( x, xSqNorms ); 
    
  size_t total_constraints = nTrain * nClass;
  size_t total_inside_constraints = nc.sum();
  size_t total_outside_constraints = total_constraints - total_inside_constraints;
    
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
      case REORDER_PROJ_MEANS:
      {
	mcsolver_detail::proj_means(means, nc, xwProj.avg(), y, params, filtered); // if avg has not started ,xwProj silently uses non-averaged w
	break;
      }
      case REORDER_RANGE_MIDPOINTS:
      {
	means = luPerm.l()+luPerm.u(); /*no need to divide by 2 since it is only used for ordering*/
	break;
      }
      }
      return means;
    };

  auto ObjectiveHinge = [&] ( ) -> double
    {
      return mcsolver_detail::calculate_objective_hinge( xwProj.std(), y, 
							 nclasses, inside_weight, outside_weight, 
							 luPerm.perm(), luPerm.rev(),
							 w.norm(), luPerm.sortlu(), filtered,
							 lambda, C1, C2, params); 
    };

  auto ObjectiveHingeAvg = [&] ( ) -> double
    {
      return mcsolver_detail::calculate_objective_hinge( xwProj.avg(), y,
							 nclasses, inside_weight, outside_weight,
							 luPerm.perm(), luPerm.rev(),
							 w.norm_avg(), luPerm.sortlu_avg(), filtered,
							 lambda, C1, C2, params); 
    };
    
  auto OptimizeLU = [&] () -> void
    {					  
      /* changes l, u, invalidates sortlu */	
      luPerm.optimizeLU( xwProj.std(), y, wc, nclasses, inside_weight, outside_weight, filtered, C1, C2, params); 
    };
  auto OptimizeLU_avg =[&] () -> void
    {
      /* changes {l,u}_avg instead of {l,u}. Invalidates sortlu_avg */ 
      luPerm.optimizeLU_avg( xwProj.avg(), y, wc, nclasses, inside_weight, outside_weight, filtered, C1, C2, params);
    };

  auto RemoveConstraints = [&] () -> void
    {	
      // should we do this in parallel?
      // the main problem is that the bitset is not thread safe (changes to one bit can affect changes to other bits)
      // should update to use the filter class

      // if averagin has not started this will be using the non-averaged w,l and u
      mcsolver_detail::update_filtered(filtered, /*inputs:*/ xwProj.avg(), luPerm.l_avg(), luPerm.u_avg(),
				       y, params.remove_class_constraints);
      
      // recalculate nc, wc if class constraints have been removed
      if (params.remove_class_constraints)
	{
	  mcsolver_detail::init_nc(nc, wc, inside_weight, y, params, filtered);
	}

      size_t no_filtered = filtered.count(); 
      size_t filtered_inside = params.remove_class_constraints?total_inside_constraints - nc.sum():0;
      size_t filtered_outside = no_filtered - filtered_inside;

      if (params.verbose >= 1)
	{
	  cout<<"Filter["<<filtered.rows()<<"x"<<filtered.cols()<<"] removed "<<no_filtered
	      <<" of "<<total_constraints<<" constraints"<<endl;
	}
      if( filtered_inside > total_inside_constraints || filtered_outside > total_outside_constraints )
	throw std::runtime_error(" programmer error: removed more constraints than exist?");
      if( filtered_outside == total_outside_constraints || filtered_inside == total_inside_constraints )
	{
	  cerr<<setw(20)<<""<<"  CANNOT CONTINUE, no more constraints left\n"
	    "  Stopping at "<<prjax+1U<< " filters." << endl;
	  if (filtered_inside == total_inside_constraints) {
	    cerr << "All classes have been eliinated for all examples. Increase C1!" << endl;
	  }
	  setNProj(prjax+1U, true, true);
	  return;
	}

      if (params.adjust_C)
	{
	  size_t no_remaining= total_outside_constraints - filtered_outside;
	  C2 = total_outside_constraints*params.C2*1.0/no_remaining;
	  no_remaining = total_inside_constraints - filtered_inside;
	  C1 = total_inside_constraints * params.C1 * 1.0 / no_remaining; 
	  
	  lambda = 1.0/C2;
	  C1 /= C2;
	  C2 = 1.0;
	}
    };

  // ------------- end defining lambda functions --------------------------

  if (params.reoptimize_LU && !params.resume && params.nfilters > nProj)
    throw std::runtime_error("Error, --reoptlu specified with more projections than current solution requested and --resume.\n");
    
  bool const keep_weights = params.resume || params.reoptimize_LU;
  bool const keep_LU = keep_weights && !params.reoptimize_LU;
  int reuse_dim = keep_weights?min(nProj,params.nfilters):0;
  setNProj(params.nfilters, keep_weights, keep_LU);
    
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
	luPerm.init( xwProj.std(), y, params, filtered );     // try to appease valgrind?
	luPerm.rank( GetMeans(params.reorder_type) );
	OptimizeLU(); // w,projection,sort_order ----> luPerm.l,u
	lower_bounds.col(prjax) = luPerm.l();
	upper_bounds.col(prjax) = luPerm.u();
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
      mcsolver_detail::init_w( w, weights, x, y, nc,  prjax, params, filtered);
	
      if (params.verbose >= 1)
	cout<<" start projection "<<prjax<<" w.norm="<<w.norm() << endl;
      xwProj.w_changed();                     // invalidate w-dependent stuff (projections)
      luPerm.init( xwProj.std(), y, params, filtered);     // std because w can't have started averaging yet
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
	double eta_t = mcsolver_detail::set_eta(params, t, lambda); // set eta for this iteration
	// compute the gradient and update
	//      modifies w, sortedLU, (sortedLU_avg) ONLY  (l,u,l_avg,u_avg values may be stale)
	// --> the ONLY place where 'w' is modified

	mcsolver_detail::MCupdate::update(w, luPerm, eta_t,   /*R/O:*/x, y, xSqNorms, C1, C2, lambda, t,
					  nTrain, nclasses, maxclasses, inside_weight, outside_weight,
					  filtered, updateSettings, params);
	// update 'w' ==> projections of raw 'x' data (projection_avg & projection) invalid.
	xwProj.w_changed();
	if(t4.reorder) {
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
      lower_bounds.col(prjax) = luPerm.l_avg();
      upper_bounds.col(prjax) = luPerm.u_avg();
	
      if (params.remove_constraints && prjax < (int)nProj-1) 
	{
	  RemoveConstraints();
	}   
      if(params.verbose >= 1) 
	cout<<" * END projection "<<prjax << endl << endl;
	
    } // end for prjax
}


#endif //MCSOLVER_HH
