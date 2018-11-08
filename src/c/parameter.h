/*  Copyright (C) 2017 NEC Laboratories America, Inc. ("NECLA"). All rights reserved.
 *
 * This source code is licensed under the license found in the LICENSE file in
 * the root directory of this source tree. An additional grant of patent rights
 * can be found in the PATENTS file in the same directory.
 */
#ifndef __PARAMETER_H
#define __PARAMETER_H

#include <assert.h>
#include <stdexcept>  //runtime_error
#include <cstdlib>      // size_t
#include <string>
#include <iosfwd>
#include <iostream>
#include <cstdint>

/** \file
 * Reorganized option groups.
 *\verbatim
 * Options you want to play with:
 * C1, C2  -- penalty parameters. C1 is multiplied by the number of classes internally
 * nfilters -- how many projections to have
 * max_iter  —- default should be 10^8/batch_size
 * report_epoch —- default max_iter/10
 *              -- increasing it or turning it off (report_epoch=0) would
 *              speed things up if you do not need to see the convergence. 
 * eta —- initial learning rate. default 0.1
 * seed —- random seed. 0= using time, 1=no random number initilization, 
 *      --             >1=initizile using seeed.  default 0
 * num_threads —- default 0  (i.e. all threads)
 * resume -- use projections from a previous run. Train more projectoins if requested
 * reoptimize_LU -- reoptimize lower, upper bounds on projections from previous run
 * class_samples -- subsample xxx negative classes for each gradient computation. 
 *               -- default 0 (no subsample)
 * 
 * verbose -- default 1. Set to 0 to have no output
 *
 * Development options:
 *   update_type —- MITNIBATCH_SGD (standard minibatch)
 *               -- or SAFE_SGD (protects against overshooting). default SAFE_SGD 
 *   batch_size —- default: min(1000,nExamples) for MINIBATCH_SGD update. 0 = full gradient
 *              -- Must be 1 for SAFE_SGD update (if not 1 it is set to 1 with warning)
 *   avg_epoch —- When to start averaging the gradient.  Default max(#examples,dimension).
 *   reorder_epoch — Reorder classes every reorder_epoch iterations. default 10^3
 *   reorder_type — How to reorder classes:
 *                    - REORDER_PROJ_MEANS - reorder by the mean of the projections of examples in the class. Examples are projected using the current, filter direction (averaged w). Default
 *                    _ REORDER_RANGE_MIDPOINTS - reorder by the midpoint of the current active interval (L+U/2)
 *   optimizeLU_epoch — Optimize L and U every optimizeLU_epoch iterations. Expensive. default max_iter.
 *   min_eta — minimum leanrning rate. default 0
 *   eta_type — learning rate decay: 
 *              - ETA_CONST  - eta_t = eta. no decay
 *              - ETA_SQRT  - eta_t = eta/sqrt(t)
 *              - ETA_LIN - eta_t = eta/(1+eta*lambda*t) (default if avg_epoch == 0)
 *              - ETA_3_4 - eta_t = eta/(1+eta*lambda*t)^(3/4) (default if avg_epoch>0)
 *   remove_constraints — remove constraints for labels eliminated by previous filters
 *                          does not remove constraints involving the example lables
 *                          default 1 (true)
 *   remove_class_constraints — default 0  -- to be removed. Code brakes if it is turned on
 *   adjust_C — increase parameters C1 and C2 to compensate for constraints removed 
 *                        by a previous filter (true).
 *
 *   ml_wt_by_nclasses — weight example by 1/number_of_labels_it_has for constraints 
 *                         on intervals on other labels. default 0(false)
 *                     - never tried with this option turned on so the code might break
 *   ml_wt_class_by_nclasses — weight example by 1/number_of_labels_it_has for
 *                               constraints on intervals of its labels. default 0
 *                           - never tried with this option turned on so the code might break 
 * 
 * These should be removed, since they are only used for testing that the
 * gradient computation is accurate. The user should not need to do this.
 *   finite_diff_test_epoch — default 1
 *   no_finite_diff_tests — default 1000
 *   finite_diff_test_delta — default 10e-4
 *\endverbatim
 */
#ifndef GRADIENT_TEST
/** At compile-time, do we remove \c finite_diff_test_epoch,
 * \c no_finite_diff_tests, and \c finite_diff_test_delta? */
#define GRADIENT_TEST 0
#endif

#if GRADIENT_TEST
/** compile-time short-circuit for code or preprocessor snippets */
#define IF_GRADIENT_TEST( ... ) __VA_ARGS
#else
#define IF_GRADIENT_TEST( ... )
#endif
// -------- enum constants

enum Eta_Type
{
    ETA_CONST, ///< eta
    ETA_SQRT,  ///< eta/sqrt(t)
    ETA_LIN,   ///< eta/(1+eta*lambda*t) [*]
    ETA_3_4,    ///< eta*(1+eta*lambda*t)^(-3/4)
    DEFAULT   /// flag for default value
};
enum Update_Type
{
    MINIBATCH_SGD, ///< update w,L and U at the same time using minibatch SGD [*]
    SAFE_SGD       ///< update w first, then L and U using projected gradietn using a minibatch of 1
};

enum Reorder_Type
{
    REORDER_PROJ_MEANS, ///< reorder by the mean of the projection based on averaged w [*]
    REORDER_RANGE_MIDPOINTS ///< reorder by the mean of the range of the class (i.e. (u+l)/2 )
};


enum Init_W_Type
  {
    INIT_ZERO,    // initiazlie with zero
    INIT_PREV,    // initialize with previous value
    INIT_RANDOM,  // initialize with random vector
    INIT_DIFF     // initialize with vector between two class centers
  };

// ------- enum conversions & I/O

/** enum --> string */
std::string tostring( enum Eta_Type const e );
std::string tostring( enum Update_Type const e );
std::string tostring( enum Reorder_Type const e );
std::string tostring( enum Init_W_Type const e );
/** throw runtime_error if invalid string. Matches on some substring. */
void fromstring( std::string s, enum Eta_Type &e );
void fromstring( std::string s, enum Update_Type &e );
void fromstring( std::string s, enum Reorder_Type &e );
void fromstring( std::string s, enum Init_W_Type &e );
/** ostreams write enums as strings. XXX Why do these need templating? */
template<typename OSTREAM> inline
OSTREAM& operator<<(OSTREAM& os, enum Eta_Type const e) { return os<<tostring(e); }
template<typename OSTREAM> inline
OSTREAM& operator<<(OSTREAM& os, enum Update_Type const e) { return os<<tostring(e); }
template<typename OSTREAM> inline
OSTREAM& operator<<(OSTREAM& os, enum Reorder_Type const e) { return os<<tostring(e); }
template<typename OSTREAM> inline
OSTREAM& operator<<(OSTREAM& os, enum Init_W_Type const e) { return os<<tostring(e); }

// --------- parameter structure

typedef struct
{
    /// \group Main options
    //@{
  uint32_t nfilters; ///< number of projections to be solved for
  //uint32_t tot_projections; ///< >= nfilters, random unit vectors orthogonal to solved nfilters [default = nfilters]
  double C1;               ///< the penalty for an example being outside it's class bounary
  double C2;               ///< the penalty for an example being inside other class' boundary
  uint32_t max_iter;       ///< maximum number of iterations
  double eta;              ///< the initial learning rate, decay type via \c Eta_Type \c eta_type
  uint32_t seed;           ///< the random seed (if 0 then ititialized from time)
  uint32_t num_threads;    ///< number of threads to run on (0 for max threads)
  bool resume;             ///< whether to train more projections (Old projections should be passed to the program)
  bool reoptimize_LU;      ///< whether to reoptimize the bounds of the class intervals.
  uint32_t class_samples;  ///< the number of negative classes to use at each gradient iteration. 0 to use all the classes
  //@}
  /// \group Development options
  //@{
  Update_Type update_type; ///< how to update w, L and U (need a better name)
  uint32_t batch_size;     ///< size of the minibatch
  Eta_Type eta_type;       ///< how does the learning rate decay.
  double min_eta;          ///<the minimum value of the lerarning rate. E.g. lr ~ \c max(min_eta, eta/sqrt(t), min_eta)
  bool averaged_gradient;  ///< use averaged gradient   
  uint32_t avg_epoch;      ///< the iteration at which averaging starts (0=none)
  uint32_t reorder_epoch;  ///<number of iterations between class reorderings (0=none)
  uint32_t report_epoch;   ///<number of iterations between computation and objective value report (expensive: calculated on whole training set) (0=none)
  uint32_t optimizeLU_epoch; ///< number of iterations between full optimizations of the lower and upper bounds
  bool remove_constraints; ///< whether to remove the constraints for instances that fall outside the class boundaries in previous projections.
  bool remove_class_constraints; ///< whether to remove the constraints for examples that fell outside their own class boundaries in previous projections.
  bool adjust_C; ///< whether to increase C1 and C2 as constraints are eliminated;
  Reorder_Type reorder_type; ///< whether to rank the classes by the mean of the projected examples or by the midpoint \c (u+l)/2 of its [l,u] interval
  bool ml_wt_by_nclasses;  ///< UNTESTED - whether to weight an example by the number of classes it belongs to when considering other class contraints
  bool ml_wt_class_by_nclasses; ///< UNTESTED - whether to weight an example by the number of classes it belongs to when considering its class contraints
  Init_W_Type init_type; // how to initialize w
  bool init_orthogonal; // initialize w to be orthogonal on previous projections?
  double init_norm;  // initialize w to this norm. If negative, no renormalization is performed. 
  int verbose; // verbosity level 
  //@}
  /// \group Compile-time options
  //@{
#if GRADIENT_TEST // Off, by default, at compile time
  uint32_t finite_diff_test_epoch; ///< number of iterations between testing the gradient with finite differences. 0 for no testing.
  uint32_t no_finite_diff_tests; ///< number of instances to perform the finite differences test at each testing round. The instances are randomly picked from the training set.
  double finite_diff_test_delta; ///< the size of the finite differene
#endif
  //@}
} param_struct;

/// comes with a pretty-printer
std::ostream& operator<<( std::ostream& os, param_struct const& p );

// parameter initialization is done in two stages. First default are set here. 
// some parameters defaults depend on other parameters of the data
// It is recommended to use SolverParams class to set parameters as it sets the correct defaults
inline param_struct set_default_params()
{
  param_struct def;
  // Main options
  def.nfilters = 5;
  def.C1=-1.0; // this is a required parameter (error if not set > 0)
  def.C2=-1.0; // this is a required parameter (error if not set > 0)
  def.max_iter = 1e8;
  def.eta=0.1;
  def.seed = 0;
  def.num_threads = 0;          // use OMP_NUM_THREADS
  def.resume = false;
  def.reoptimize_LU = false;
  def.class_samples = 0;
  // Development options
  def.update_type = SAFE_SGD;
  def.batch_size=1U; //default value depends on update_type.
  def.eta_type = ETA_3_4; //default value depends on the value of avg_epoch.
  def.min_eta = 0;
  def.optimizeLU_epoch = 1e8+1; // this is very expensive so it should not be done often
  def.reorder_epoch=1000;
  def.reorder_type = REORDER_PROJ_MEANS; // defaults to REORDER_PROJ_MEANS if averaging is off
  def.report_epoch=1e7;
  def.averaged_gradient=true;
  def.avg_epoch=0; //default depends on the data. initialized in mcsolver::solve
  def.adjust_C = true; 
  def.remove_constraints = true;
  def.remove_class_constraints = false;
  def.ml_wt_by_nclasses = false;        // UNTESTED
  def.ml_wt_class_by_nclasses = false;  // UNTESTED
  def.init_type = INIT_DIFF;
  def.init_norm = 10; // no good reason for using 10. 
  def.init_orthogonal = false;
  def.verbose = 1; // print some output
  // Compile-time options
#if GRADIENT_TEST
  def.finite_diff_test_epoch=1;
  def.no_finite_diff_tests=1000;
  def.finite_diff_test_delta=1e-4;
#endif

  return def;
}

/** \ref erik/parameter-args.h has a boost::program_options description
 * that can also generate useful help. */
void print_parameter_usage();

/// \name I/O (non-milde -- independent of os/util_io.hpp)
/// return non-zero on error (stream.fail())
//@{
int write_ascii( std::ostream& os, param_struct const& p );
int read_ascii( std::istream& is, param_struct& p );
int write_binary( std::ostream& is, param_struct const& p );
int read_binary( std::istream& is, param_struct& p );
//@}


// class that encapsulates the param_struct to make sure the various parameters are compatible.
// also manages the default settings as the defaults change depending on the parameter settigns 
class SolverParams  
{
 public:
  SolverParams();
  ~SolverParams(){}
  
  inline param_struct const& params() const
  {
    return m_params;
  }

  /// \group Main options
    //@{
  inline uint32_t nfilters() const {return m_params.nfilters;} ///< number of projections to be solved for
  inline void nfilters(uint32_t nf) {m_params.nfilters = nf;} ///< number of projections to be solved for
  
  inline double C1() const {return m_params.C1;}               ///< the penalty for an example being outside it's class bounary
  inline void C1(double c1) {if (c1 < 0.0) throw std::runtime_error("C1 must be positive"); m_params.C1 = c1;}               ///< the penalty for an example being outside it's class bounary
  inline double C2() const {return m_params.C2;}               ///< the penalty for an example being inside other class' boundary
  inline void C2( double c2 ) {if (c2 < 0.0) throw std::runtime_error("C2 must be positive"); m_params.C2 = c2;}               ///< the penalty for an example being inside other class' boundary
  inline uint32_t max_iter() const {return m_params.max_iter;}       ///< maximum number of iterations
  inline void max_iter(uint32_t mi, bool default_val = false)        ///< maximum number of iterations
  {
    m_params.max_iter = mi;
    default_max_iter = default_val;
    if (default_report_epoch) report_epoch(mi/10, true);
    if (default_optimizeLU_epoch) optimizeLU_epoch(mi+1, true); // only optimize at the beginning and at the end
  }
  inline double eta() const {return m_params.eta;}              ///< the initial learning rate, decay type via \c Eta_Type \c eta_type
  inline void eta(double e) {if (e < 0.0) throw std::runtime_error("eta must be positive");m_params.eta = e;}              ///< the initial learning rate, decay type via \c Eta_Type \c eta_type
  inline uint32_t seed() const{return m_params.seed;}           ///< the random seed (if 0 then ititialized from time)
  inline void seed(uint32_t s) {m_params.seed =  s;}           ///< the random seed (if 0 then ititialized from time)
  inline uint32_t num_threads() const {return m_params.num_threads;}    ///< number of threads to run on (0 for max threads)
  inline void  num_threads( uint32_t nt) {m_params.num_threads = nt;}    ///< number of threads to run on (0 for max threads)
  inline bool resume() const {return m_params.resume;}             ///< whether to train more projections (Old projections should be passed to the program)
  inline void resume(bool r) {m_params.resume = r;}             ///< whether to train more projections (Old projections should be passed to the program)
  inline bool reoptimize_LU() const {return m_params.reoptimize_LU;}      ///< whether to reoptimize the bounds of the class intervals.
  inline void reoptimize_LU( bool r) {m_params.reoptimize_LU = r;}      ///< whether to reoptimize the bounds of the class intervals.
  inline uint32_t class_samples() const {return m_params.class_samples;}  ///< the number of negative classes to use at each gradient iteration. 0 to use all the classes
  inline void class_samples(uint32_t cs) {m_params.class_samples = cs;}  ///< the number of negative classes to use at each gradient iteration. 0 to use all the classes
  //@}
  /// \group Development options
  //@{
  inline Update_Type update_type() const {return m_params.update_type;} ///< how to update w, L and U (need a better name)
  inline void update_type(Update_Type ut)  ///< how to update w, L and U (need a better name). Must be set before batch_size
  {
    switch (ut)
      {
      case SAFE_SGD:
	if (default_batch_size) 
	  {
	    batch_size(1U, true);
	  }
	if (m_params.batch_size !=1U) 
	  {
	    std::cerr << "Warning: batch size must be 1 for SAFE_SGD update method. Setting batch size to 1" << std::endl;
	    batch_size(1U, true);
	  }
	break;
      case MINIBATCH_SGD:
	if (default_batch_size)
	  {
	    batch_size(1000U, true);
	  }
	break;
      default:
	throw std::runtime_error("Unrecognized update method");
      }
    m_params.update_type = ut;
  } 
  inline uint32_t batch_size() const {return m_params.batch_size;}     ///< size of the minibatch
  inline void batch_size(uint32_t bs, bool default_val = false)    ///< size of the minibatch
  {
    if (bs != 1U && m_params.update_type == SAFE_SGD)
      {
	std::cerr << "WARNING: SAVE_SGD update method only works with batch size of 1. Ignoring batch size request." << std::endl;
      }
    else
      {
	m_params.batch_size=bs;
	default_batch_size = default_val;
      }
    if (default_max_iter) max_iter(1e8/m_params.batch_size,true); 
  }   
  inline Eta_Type eta_type() const {return m_params.eta_type;}       ///< how does the learning rate decay.
  inline void eta_type(Eta_Type et, bool default_val = false)   ///< how does the learning rate decay.
  {
    m_params.eta_type=et;
    default_eta_type = default_val;
  }
  inline double min_eta() const {return m_params.min_eta;}          ///<the minimum value of the lerarning rate. E.g. lr ~ \c max(min_eta, eta/sqrt(t), min_eta)
  inline void min_eta(double me) {m_params.min_eta=me;}          ///<the minimum value of the lerarning rate. E.g. lr ~ \c max(min_eta, eta/sqrt(t), min_eta)
  inline bool averaged_gradient() const {return m_params.averaged_gradient;} ///<use averaged gradient 
  inline void averaged_gradient(bool ag)
  {
    m_params.averaged_gradient = ag;
    if (default_eta_type)
      {
	if (ag)
	  eta_type(ETA_3_4, true);
	else
	  eta_type(ETA_LIN, true);
      }
  } ///<use averaged gradient 
  inline uint32_t avg_epoch() const {return m_params.avg_epoch;}      ///< the iteration at which averaging starts (0=none)
  inline void avg_epoch(uint32_t ae)   ///< the iteration at which averaging starts (0=none)
  {
    m_params.avg_epoch=ae;
    averaged_gradient(true);  ///< setting avg_epoch turns on average gradient. 
  }
  
  inline uint32_t reorder_epoch() const {return m_params.reorder_epoch;}  ///<number of iterations between class reorderings (0=none)
  inline void reorder_epoch(uint32_t re) {m_params.reorder_epoch=re;}  ///<number of iterations between class reorderings (0=none)
  inline uint32_t report_epoch() const {return m_params.report_epoch;}   ///<number of iterations between computation and objective value report (expensive: calculated on whole training set) (0=none)
  inline void report_epoch(uint32_t re, bool default_val = false)   ///<number of iterations between computation and objective value report (expensive: calculated on whole training set) (0=none)
  {
    m_params.report_epoch=re;
    default_report_epoch = default_val;
  } 
  inline uint32_t optimizeLU_epoch() const {return m_params.optimizeLU_epoch;} ///< number of iterations between full optimizations of the lower and upper bounds
  inline void optimizeLU_epoch(uint32_t oe, bool default_val = false) ///< number of iterations between full optimizations of the lower and upper bounds
  {
    m_params.optimizeLU_epoch=oe;
    default_optimizeLU_epoch = default_val;
  }
  inline bool remove_constraints() const {return m_params.remove_constraints;} ///< whether to remove the constraints for instances that fall outside the class boundaries in previous projections.
  inline void remove_constraints(bool rc) {m_params.remove_constraints=rc;} ///< whether to remove the constraints for instances that fall outside the class boundaries in previous projections.
  inline bool remove_class_constraints() const {return m_params.remove_class_constraints;} ///< whether to remove the constraints for examples that fell outside their own class boundaries in previous projections.
  inline void remove_class_constraints(bool rcc) {m_params.remove_class_constraints=rcc;} ///< whether to remove the constraints for examples that fell outside their own class boundaries in previous projections.
  inline bool adjust_C() const {return m_params.adjust_C;} ///< whether to increase C1 and C2 as constraints are eliminated;
  inline void adjust_C(bool adjust) {m_params.adjust_C=adjust;} ///< whether to increase C1 and C2 as constraints are eliminated;
  inline Reorder_Type reorder_type() const {return m_params.reorder_type;} ///< whether to rank the classes by the mean of the projected examples or by the midpoint \c (u+l)/2 of its [l,u] interval
  inline void reorder_type(Reorder_Type rtype) {m_params.reorder_type=rtype;} ///< whether to rank the classes by the mean of the projected examples or by the midpoint \c (u+l)/2 of its [l,u] interval
  inline bool ml_wt_by_nclasses() const {return m_params.ml_wt_by_nclasses;}  ///< UNTESTED - whether to weight an example by the inverse of the number of classes it belongs to when considering other class contraints
  inline void ml_wt_by_nclasses(bool wc) {m_params.ml_wt_by_nclasses=wc;}  ///< UNTESTED - whether to weight an example by the inverse of the number of classes it belongs to when considering other class contraints
  inline bool ml_wt_class_by_nclasses() const {return m_params.ml_wt_class_by_nclasses;} ///< UNTESTED - whether to weight an example by the inverse of the  number of classes it belongs to when considering its class contraints
  inline void ml_wt_class_by_nclasses(bool wcc) {m_params.ml_wt_class_by_nclasses=wcc;} ///< UNTESTED - whether to weight an example by the inverse of the number of classes it belongs to when considering its class contraints
  inline Init_W_Type init_type() const {return m_params.init_type;} // how to initialize w
  inline void init_type(Init_W_Type it)  // how to initialize w. Should be caled before init_orthogonal, and init_norm. 
  {
    m_params.init_type=it;
    switch (it)
      {
      case INIT_DIFF:
	{
	  if (default_init_norm)
	    init_norm(10, true);
	  if (init_orthogonal())
	    {
	      std::cerr << "WARNING: DIFF initialization is not orthogonal. Setting init_orthogonal to false." << std::endl; 	      
	      init_orthogonal(false);
	    }
	}
	break;
      case INIT_RANDOM:
	{
	  if (default_init_norm)
	    init_norm(10, true);
	}
	break;
      case INIT_PREV:
	{
	  if (!default_init_norm)
	    {
	      std::cerr << "WARNING: Initializaing with given vector. Inti_norm is not used" << std::endl;
	    }
	  init_norm(0, true);
	  if (init_orthogonal())
	    {
	      std::cerr << "WARNING: PREV initialization is not orthogonal. Setting init_orthogonal to false." << std::endl; 	      
	      init_orthogonal(false);
	    }
	}
	break;
      case INIT_ZERO:
	{
	  if (!default_init_norm)
	    {
	      std::cerr << "WARNING: Initializaing with given vector. Inti_norm is not used" << std::endl;
	    }
	  init_norm(0,true);
	  if (init_orthogonal())
	    {
	      std::cerr << "WARNING: PREV initialization is not orthogonal. Setting init_orthogonal to false." << std::endl; 	      
	      init_orthogonal(false);
	    }
	}
	break;
      default:
	throw std::runtime_error("Unknown initialization type");
      }    
  }
  inline bool init_orthogonal() const {return m_params.init_orthogonal;} // initialize w to be orthogonal on previous projections?
  inline void init_orthogonal(bool orth)  // initialize w to be orthogonal on previous projections?
  {
    if (orth && (init_type() == INIT_ZERO || init_type() == INIT_PREV || init_type() == INIT_DIFF))
      {
	std::cerr << "WARNING: Initialziation type " << tostring(init_type()) << " can not be orthogonal. Ignoring init_orthogonal" << std::endl;
      }
    else
      {
	m_params.init_orthogonal=orth;
      }
  }
  inline double init_norm() const {return m_params.init_norm;}  // initialize w to this norm. If negative, no renormalization is performed. 
  inline void init_norm(double norm, bool default_val = false)  // initialize w to this norm. If negative, no renormalization is performed. 
  {
    if (!default_val && (init_type() == INIT_ZERO || init_type() == INIT_PREV))
      {
	std::cerr << "WARNING: init_norm is not used for this initialziation type" << std::endl;
      }
    else
      {
	m_params.init_norm = norm;
	default_init_norm = default_val;
      }
  }
  inline int verbose() const {return m_params.verbose;} // verbosity level 
  inline void verbose(int vb)
  {
    m_params.verbose=vb;
  }
  //@}
  /// \group Compile-time options
  //@{
#if GRADIENT_TEST // Off, by default, at compile time
  inline uint32_t finite_diff_test_epoch() const {return m_params.finite_diff_test_epoch;} ///< number of iterations between testing the gradient with finite differences. 0 for no testing.
  inline void finite_diff_test_epoch(uint32_t epoch) {m_params.finite_diff_test_epoch=epoch;} ///< number of iterations between testing the gradient with finite differences. 0 for no testing.
  inline uint32_t no_finite_diff_tests() const {return m_params.no_finite_diff_tests;} ///< number of instances to perform the finite differences test at each testing round. The instances are randomly picked from the training set.
  inline void no_finite_diff_tests(uint32_t ntests) {m_params.no_finite_diff_tests=ntests;} ///< number of instances to perform the finite differences test at each testing round. The instances are randomly picked from the training set.
  inline double finite_diff_test_delta() const {return m_params.finite_diff_test_delta;} ///< the size of the finite differene
  inline void finite_diff_test_delta(double delta) {m_params.finite_diff_test_delta=delta;} ///< the size of the finite differene
#endif
  //@}

 private:  
  param_struct m_params;

  //bools to keep track if some fields are set to the default value
  bool default_batch_size; ///< is batch_size set to the default value?
  bool default_report_epoch; ///< is report epoch set to the default value?
  bool default_optimizeLU_epoch; ///< is optimizeLU_epoch set to the default value?
  bool default_init_norm; ///< is init_norm set to the default value?
  bool default_eta_type; ///< is eta_type set the the default value?
  bool default_max_iter; ///< is max_iter set to the default value?
};

/// comes with a pretty-printer
std::ostream& operator<<( std::ostream& os, SolverParams const& p );


#endif
