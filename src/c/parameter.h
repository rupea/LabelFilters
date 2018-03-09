#ifndef __PARAMETER_H
#define __PARAMETER_H

#include <assert.h>
#include <cstdlib>      // size_t
#include <string>
#include <iosfwd>
#include <cstdint>

/** \file
 * Reorganized option groups.
 *\verbatim
 * Options you want to play with:
 * C1, C2  -- penalty parameters. C1 is multiplied by the number of classes internally
 * no_projections -- how many projections to have
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
 *   reweight_lambda — increase parameter C2 to compensate for constraints eliminated
 *                        by a previous filter.
 * 			     - REWEIGHT_NONE: do not increase C2
 *                           - REWEIGHT_LAMBDA: increase both C1 and C2
 *                           - REWEIGHT_ALL: increase only C2 (default)
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
 *   finite_diff_test_epoch — default 0
 *   no_finite_diff_tests — default 1
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

enum Reweight_Type
{
    REWEIGHT_NONE,      ///< do not diminish any
    REWEIGHT_LAMBDA,    ///< diminish lambda only [*]
    REWEIGHT_ALL        ///< diminish lambda and C1 ", increase C2
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
std::string tostring( enum Reweight_Type const e );
std::string tostring( enum Init_W_Type const e );
/** throw runtime_error if invalid string. Matches on some substring. */
void fromstring( std::string s, enum Eta_Type &e );
void fromstring( std::string s, enum Update_Type &e );
void fromstring( std::string s, enum Reorder_Type &e );
void fromstring( std::string s, enum Reweight_Type &e );
void fromstring( std::string s, enum Init_W_Type &e );
/** ostreams write enums as strings. XXX Why do these need templating? */
template<typename OSTREAM> inline
OSTREAM& operator<<(OSTREAM& os, enum Eta_Type const e) { return os<<tostring(e); }
template<typename OSTREAM> inline
OSTREAM& operator<<(OSTREAM& os, enum Update_Type const e) { return os<<tostring(e); }
template<typename OSTREAM> inline
OSTREAM& operator<<(OSTREAM& os, enum Reorder_Type const e) { return os<<tostring(e); }
template<typename OSTREAM> inline
OSTREAM& operator<<(OSTREAM& os, enum Reweight_Type const e) { return os<<tostring(e); }
template<typename OSTREAM> inline
OSTREAM& operator<<(OSTREAM& os, enum Init_W_Type const e) { return os<<tostring(e); }

// --------- parameter structure

typedef struct
{
    /// \group Main options
    //@{
  uint32_t no_projections; ///< number of projections to be solved for
  //uint32_t tot_projections; ///< >= no_projections, random unit vectors orthogonal to solved no_projections [default = no_projections]
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
  bool default_batch_size; ///< is batch_size uninitialized?
  uint32_t batch_size;     ///< size of the minibatch
  Eta_Type eta_type;       ///< how does the learning rate decay.
  double min_eta;          ///<the minimum value of the lerarning rate. E.g. lr ~ \c max(min_eta, eta/sqrt(t), min_eta)
  bool default_avg_epoch;  ///< is avg_epoch uninitialized?
  uint32_t avg_epoch;      ///< the iteration at which averaging starts (0=none)
  uint32_t reorder_epoch;  ///<number of iterations between class reorderings (0=none)
  bool default_report_epoch; ///< is report epoch uninitialized?
  uint32_t report_epoch;   ///<number of iterations between computation and objective value report (expensive: calculated on whole training set) (0=none)
  bool default_optimizeLU_epoch; ///< is optimizeLU_epoch uninitialized?
  uint32_t optimizeLU_epoch; ///< number of iterations between full optimizations of the lower and upper bounds
  bool remove_constraints; ///< whether to remove the constraints for instances that fall outside the class boundaries in previous projections.
  bool remove_class_constraints; ///< whether to remove the constraints for examples that fell outside their own class boundaries in previous projections.
  Reweight_Type reweight_lambda; ///< whether to diminish lambda (increase C1 and C2) as constraints are eliminated;
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

/* some defaults depend on other parameters. This function should be after parmeters 
 * have been set (e.g. through command line arguments).
 * after calling this function the avg_epoch parameter may remain not set as its 
 * default value depends on the data (i.e. avg_epoch = nExamples). It is set in 
 * MCsolver::solve if it has not been initialized by then */

void finalize_default_params( param_struct &p );

// parameter initialization is done in two stages. First default are set here. 
// some parameters defaults depend on other parameters of the data
// so after parameters aer parsed, finalize_default_params() must be called. 
// finally, default value of avg_epoch depends on the data, so it is initialized 
// by mcsolver::solve if not done already. 
inline param_struct set_default_params()
{
  param_struct def;
  // Main options
  def.no_projections = 5;
  def.C1=-1.0; // this is a required parameter (error if not set > 0)
  def.C2=-1.0; // this is a required parameter (error if not set > 0)
  def.max_iter=0U; //1e8;
  def.eta=0.1;
  def.seed = 0;
  def.num_threads = 0;          // use OMP_NUM_THREADS
  def.resume = false;
  def.reoptimize_LU = false;
  def.class_samples = 0;
  // Development options
  def.update_type = SAFE_SGD;
  def.default_batch_size = true;
  def.batch_size=0U; //default value depends on update_type. to be initialized in finalize default params
  def.eta_type = DEFAULT; //default value depends on the value of avg_epoch. Will be initialized by finalize_default_params.
  def.min_eta = 0;
  def.default_optimizeLU_epoch = true;//flag that optimizeLU_epoch has not been initialized. Default value depends on max_iter. Will be initialized by finalize_default_params
  def.optimizeLU_epoch = 0; // this is very expensive so it should not be done often
  def.reorder_epoch=1000;
  def.reorder_type = REORDER_PROJ_MEANS; // defaults to REORDER_PROJ_MEANS if averaging is off
  def.default_report_epoch = true;
  def.report_epoch=0;//1000000;
  def.default_avg_epoch = true; //flag that avg_epoch has not been initialized. Default depends on the data. Will be initialized by mcsolver::solve
  def.avg_epoch=0;
  def.reweight_lambda = REWEIGHT_ALL; 
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
  def.finite_diff_test_epoch=0;
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

#endif
