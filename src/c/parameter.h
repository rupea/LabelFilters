#ifndef __PARAMETER_H
#define __PARAMETER_H

#include <assert.h>
#include <cstdlib>      // size_t
#include <string>
#include <iosfwd>
#include <cstdint>

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
    REWEIGHT_ALL        ///< diminish lambda and C1 ", increase C2" <-- NOT DONE !!!"
};


// ------- enum conversions & I/O

/** enum --> string */
std::string tostring( enum Eta_Type const e );
std::string tostring( enum Update_Type const e );
std::string tostring( enum Reorder_Type const e );
std::string tostring( enum Reweight_Type const e );
/** throw runtime_error if invalid string. Matches on some substring. */
void fromstring( std::string s, enum Eta_Type &e );
void fromstring( std::string s, enum Update_Type &e );
void fromstring( std::string s, enum Reorder_Type &e );
void fromstring( std::string s, enum Reweight_Type &e );
/** ostreams write enums as strings. XXX Why do these need templating? */
template<typename OSTREAM> inline
OSTREAM& operator<<(OSTREAM& os, enum Eta_Type const e) { return os<<tostring(e); }
template<typename OSTREAM> inline
OSTREAM& operator<<(OSTREAM& os, enum Update_Type const e) { return os<<tostring(e); }
template<typename OSTREAM> inline
OSTREAM& operator<<(OSTREAM& os, enum Reorder_Type const e) { return os<<tostring(e); }
template<typename OSTREAM> inline
OSTREAM& operator<<(OSTREAM& os, enum Reweight_Type const e) { return os<<tostring(e); }

// --------- parameter structure

typedef struct
{
  uint32_t no_projections; ///< number of projections to be solved for
  //uint32_t tot_projections; ///< >= no_projections, random unit vectors orthogonal to solved no_projections [default = no_projections]
  double C1;          ///< the penalty for an example being outside it's class bounary
  double C2;          ///< the penalty for an example being inside other class' boundary
  uint32_t max_iter;    ///< maximum number of iterations
  uint32_t batch_size;  ///< size of the minibatch
  Update_Type update_type; ///< how to update w, L and U (need a better name)
  double eps;         ///< not used
  Eta_Type eta_type;  ///< how does the learning rate decay
  double eta;         ///< the initial learning rate. The leraning rate is eta/sqrt(t) where t is the number of iterations
  double min_eta;     ///<the minimum value of the lerarning rate (i.e. lr will be max (eta/sqrt(t), min_eta)
  uint32_t avg_epoch;   ///< the iteration at which averaging starts. 0 for no averaging.
  uint32_t reorder_epoch; ///<number of iterations between class reorderings. 0 for no reordering of classes
  uint32_t report_epoch; ///<number of iterations between computation and report the objective value (can be expensive because obj is calculated on the entire training set). 0 for no reporting
  uint32_t report_avg_epoch; ///<number of iterations between computation and report the objective value for the averaged w (can be expensive because obj is calculated on the entire training set). Even more expensive if optimizeLU_epoch > 0.  0 for no reporting
  uint32_t optimizeLU_epoch; ///< number of iterations between full optimizations of the lower and upper bounds
  bool remove_constraints; ///< whether to remove the constraints for instances that fall outside the class boundaries in previous projections.
  bool remove_class_constraints; ///< whether to remove the constraints for examples that fell outside their own class boundaries in previous projections.
  Reweight_Type reweight_lambda; ///< whether to diminish lambda (increase C1 and C2) as constraints are eliminated;
  Reorder_Type reorder_type; ///< whether to rank the classes by the mean of the projected examples or by the midpoint of its [l,u] interval (i.e. (u+l)/2).
  bool ml_wt_by_nclasses; ///< whether to weight an example by the number of classes it belongs to when conssidering other class contraints
  bool ml_wt_class_by_nclasses; ///< whether to weight an example by the number of classes it belongs to when conssidering its class contraints.
  uint32_t num_threads; ///< number of threads to run on (0 for max threads)
  uint32_t seed; ///< the random seed. if 0 then ititialized from time.
  uint32_t finite_diff_test_epoch; ///< number of iterations between testing the gradient with finite differences. 0 for no testing.
  uint32_t no_finite_diff_tests; ///< number of instances to perform the finite differences test at each testing round. The instances are randomly picked from the training set.
  double finite_diff_test_delta; ///< the size of the finite differene
  bool resume; ///< whether to train more projections. Old projections should be passed to the program.
  bool reoptimize_LU; ///< whether to reoptimize the bounds of the class intervals.
  uint32_t class_samples; ///< the number of negative classes to use at each gradient iteration. 0 to use all the classes
} param_struct;


inline param_struct set_default_params()
{
  param_struct def;
  def.no_projections = 5;
  //def.tot_projections = 5;
  def.C1=10.0;
  def.C2=1.0;
  def.max_iter=1e6;
  def.batch_size=100;
  def.update_type = MINIBATCH_SGD;
  def.eps=1e-4;
  def.eta=0.1;
  def.eta_type = ETA_LIN;
  def.min_eta= 0;
  def.optimizeLU_epoch=10000; // this is very expensive so it should not be done often
  def.reorder_epoch=1000;
  def.reorder_type = REORDER_AVG_PROJ_MEANS; // defaults to REORDER_PROJ_MEANS if averaging is off
  def.report_epoch=1000;
  def.avg_epoch=0;
  def.report_avg_epoch=0; // this is expensive so the default is 0
  def.reweight_lambda = REWEIGHT_LAMBDA;
  def.ml_wt_by_nclasses = false;
  def.ml_wt_class_by_nclasses = false;
  def.remove_constraints = false;
  def.remove_class_constraints = false;
  def.num_threads = 0;          // use OMP_NUM_THREADS
  def.seed = 0;
  def.finite_diff_test_epoch=0;
  def.no_finite_diff_tests=1000;
  def.finite_diff_test_delta=1e-4;
  def.resume = false;
  def.reoptimize_LU = false;
  def.class_samples = 0;
  return def;
}

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
