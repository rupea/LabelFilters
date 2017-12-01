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
 * C1, C2  - should tweak them a bit such that C1 is actually
 *           C1*number_of_classes no_projections
 * max_iter  — default should be 10^8/batch_size
 * report_epoch — default 10^6/batch_size
 *              -- increasing it or turning it off (report_epoch=0) would
 *              speed things up if you do not need to see the convergence. 
 * eta — default 0.1
 * seed — default 0 (not sure this should be here but some people
 *                   want to have reproducible runs)
 * num_threads — default 0  (i.e. all threads)
 * resume 
 * reoptimize_LU
 * class_samples -- default 0 (I am not sure this should be here, but the user
 *                  has the option to set it to 100 or 1000 if learning is too slow) 
 * 
 * verbose -- default 1. Set to 0 to have no output
 *
 * Development options:
 *   update_type — default safe 
 *   batch_size — default 1 (max(1000,nExamples) for minibatch update)
 *   avg_epoch — default #training examples.
 *   reorder_epoch — default 10^6/batch_size
 *   reorder_type — default avg_proj_means  
 *   optimizeLU_epoch — default max_iter 
 *   report_avg_epoch — default 0
 *   min_eta — default 0
 *   eta_type — default 3_4  ( default sqrt if avg_epoch == 0) 
 *   remove_constraints — default 1
 *   remove_class_constraints — default 0
 *   reweight_lambda — default 2
 *   ml_wt_by_nclasses — default 0(false)
 *                     - never tried with this option turned on so the code might break 
 *   ml_wt_class_by_nclasses — default 0(false)
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
    ETA_3_4    ///< eta*(1+eta*lambda*t)^(-3/4)
};
enum Update_Type
{
    MINIBATCH_SGD, ///< update w,L and U at the same time using minibatch SGD [*]
    SAFE_SGD       ///< update w first, then L and U using projected gradietn using a minibatch of 1
};

enum Reorder_Type
{
    REORDER_AVG_PROJ_MEANS, ///< reorder by the mean of the projection based on averaged w [*]
    REORDER_PROJ_MEANS,     ///< reorder by the means of the projection based on current w
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
  uint32_t batch_size;     ///< size of the minibatch
  double eps;              ///< not used
  Eta_Type eta_type;       ///< how does the learning rate decay.
  double min_eta;          ///<the minimum value of the lerarning rate. E.g. lr ~ \c max(min_eta, eta/sqrt(t), min_eta)
  uint32_t avg_epoch;      ///< the iteration at which averaging starts (0=none)
  uint32_t reorder_epoch;  ///<number of iterations between class reorderings (0=none)
  uint32_t report_epoch;   ///<number of iterations between computation and objective value report (expensive: calculated on whole training set) (0=none)
  uint32_t report_avg_epoch; ///<number of iterations between computation and objective value report for averaged w (expensive: calculated on whole training set). Even more expensive if optimizeLU_epoch > 0. (0=none)
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

/** adapt parameters with flag values for a specific problem. */
void param_finalize( uint32_t const nTrain, uint32_t nClass, param_struct &p );

inline param_struct set_default_params()
{
  param_struct def;
  // Main options
  def.no_projections = 5;
  //def.tot_projections = 5;
  def.C1=10.0;
  def.C2=1.0;
  def.max_iter=1e8;
  def.eta=0.1;
  def.seed = 0;
  def.num_threads = 0;          // use OMP_NUM_THREADS
  def.resume = false;
  def.reoptimize_LU = false;
  def.class_samples = 0;
  // Development options
  def.update_type = SAFE_SGD;
  def.batch_size=1U;
  def.eps=1e-4;
  def.eta_type = ETA_LIN;
  def.min_eta= 0;
  def.optimizeLU_epoch=10000; // this is very expensive so it should not be done often
  def.reorder_epoch=1000;
  def.reorder_type = REORDER_AVG_PROJ_MEANS; // defaults to REORDER_PROJ_MEANS if averaging is off
  def.report_epoch=1000000;
  def.avg_epoch=0;
  def.report_avg_epoch=0; // this is expensive so the default is 0
  def.reweight_lambda = REWEIGHT_ALL;   // <-- changed - was marked NOT YET DONE?
  def.remove_constraints = true;
  def.remove_class_constraints = false;
  def.ml_wt_by_nclasses = false;        // UNTESTED
  def.ml_wt_class_by_nclasses = false;  // UNTESTED
  def.init_type = INIT_DIFF;
  def.init_norm = 10; // no good reason for using 10. 
  def.init_orthogonal = false;
  def.verbose = 1; // print output
  // Compile-time options
#if GRADIENT_TEST
  def.finite_diff_test_epoch=0;
  def.no_finite_diff_tests=1000;
  def.finite_diff_test_delta=1e-4;
#endif
#if 0 //1-1
  // For reference, here are suggested paramater updates/defaults:
  if( def.update_type == MINIBATCH_SGD ){
      def.batch_size = def.nExamples;   // # of training examples, x.rows()?
      if( def.nExamples > 1000U ) def.batch_size = 1000U;
  }
  if( def.update_type == MINIBATCH_SGD && def.batch_size > 1U ){
      def.max_iter      /= def.batch_size;
      def.report_epoch  /= def.batch_size;
      if( def.max_iter < 1U ) def.max_iter = 1U;
  }
  def.C1 *= nClasses;                   // # of classes?
  def.optimizeLU_epoch = def.max_iter;  // max iter?
  def.avg_epoch = nExamples;            // # of training examples, x.rows()?
  if( def.avg_epoch == 0 ) def.eta_type == ETA_SQRT;
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

/** \fn param_finalize
 *
 * After we have read training data, we can switch from default
 * \em flag values to final values.
 *
 * \p nTrain    number of training examples, x.row()
 * \p nClass    number of classes (used to scale a default C1)
 *
 * - Reacts to flag values:
 *   - p.batch_size == 0U         ---> nTrain  (implies MINIBATCH_SGD full gradient)
 *   - p.batch_size > nTrain      --> nTrain
 *   - p.batch_size > 1           --> if nonzero, max_iter = max(1,max_iter/batch_size)
 *     - Are there cases where p.max-iter==0 makes sense (resume, reoptlu) ?
 *   - p.C1 <= 0                  ---> C1 *= -nClasses  ( or default.C1*nClass )
 *   - p.optimizeLU_epoch > p.max_iter ---> p.max_iter
 *   - p.avg_epoch > p.max_ite   ---> p.max_iter
 *   - etc.
 *
 * \sa  extract( po::variables_map const& vm, param_struct & parms )
 */
#endif
