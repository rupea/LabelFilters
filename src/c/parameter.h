#ifndef __PARAMETER_H
#define __PARAMETER_H

#include <assert.h>
#include <cstdlib>      // size_t
#include <string>
using std::size_t;

// -------- enum constants

enum Eta_Type 
  {
    ETA_CONST, // eta
    ETA_SQRT, // eta/sqrt(t)
    ETA_LIN, // eta/(1+eta*lambda*t)
    ETA_3_4  // eta*(1+eta*lambda*t)^(-3/4)
  }; 
enum Update_Type
  {
    MINIBATCH_SGD, //update w,L and U at the same time using minibatch SGD
    SAFE_SGD  // update w first, then L and U using projected gradietn using a minibatch of 1
  };

enum Reorder_Type
  {
    REORDER_AVG_PROJ_MEANS, // reorder by the mean of the projection based on averaged w
    REORDER_PROJ_MEANS, // reorder by the means of the projection based on current w
    REORDER_RANGE_MIDPOINTS  // reorder by the mean of the range of the class (i.e. (u+l)/2 )
  };

// ------- enum conversions & I/O

/** enum --> string */
std::string tostring( enum Eta_Type const e );
std::string tostring( enum Update_Type const e );
std::string tostring( enum Reorder_Type const e );
/** throw runtime_error if invalid string. Matches on some substring. */
void fromstring( std::string s, enum Eta_Type &e );
void fromstring( std::string s, enum Update_Type &e );
void fromstring( std::string s, enum Reorder_Type &e );
/** ostreams write enums as strings */
template<typename OSTREAM> inline
OSTREAM& operator<<(OSTREAM& os, enum Eta_Type const e) { return os<<tostring(e); }
template<typename OSTREAM> inline
OSTREAM& operator<<(OSTREAM& os, enum Update_Type const e) { return os<<tostring(e); }
template<typename OSTREAM> inline
OSTREAM& operator<<(OSTREAM& os, enum Reorder_Type const e) { return os<<tostring(e); }

// --------- parameter structure

typedef struct
{
  int no_projections; ///< number of projections to be made
  double C1;          ///< the penalty for an example being outside it's class bounary
  double C2;          ///< the penalty for an example being inside other class' boundary
  size_t max_iter;    ///< maximum number of iterations
  size_t batch_size;  ///< size of the minibatch
  Update_Type update_type; ///< how to update w, L and U (need a better name) 
  double eps;         ///< not used
  Eta_Type eta_type;  ///< how does the learning rate decay
  double eta;         ///< the initial learning rate. The leraning rate is eta/sqrt(t) where t is the number of iterations
  double min_eta;     ///<the minimum value of the lerarning rate (i.e. lr will be max (eta/sqrt(t), min_eta)
  size_t avg_epoch;   ///< the iteration at which averaging starts. 0 for no averaging. 
  size_t reorder_epoch; ///<number of iterations between class reorderings. 0 for no reordering of classes
  size_t report_epoch; ///<number of iterations between computation and report the objective value (can be expensive because obj is calculated on the entire training set). 0 for no reporting
  size_t report_avg_epoch; ///<number of iterations between computation and report the objective value for the averaged w (can be expensive because obj is calculated on the entire training set). Even more expensive if optimizeLU_epoch > 0.  0 for no reporting
  size_t optimizeLU_epoch; ///< number of iterations between full optimizations of the lower and upper bounds
  bool remove_constraints; ///< whether to remove the constraints for instances that fall outside the class boundaries in previous projections. 
  bool remove_class_constraints; ///< whether to remove the constraints for examples that fell outside their own class boundaries in previous projections. 
  int reweight_lambda; ///< whether to diminish lambda (increase C1 and C2) as constraints are eliminated;
  Reorder_Type reorder_type; ///< whether to rank the classes by the mean of the projected examples or by the midpoint of its [l,u] interval (i.e. (u+l)/2).
  bool ml_wt_by_nclasses; ///< whether to weight an example by the number of classes it belongs to when conssidering other class contraints. 
  bool ml_wt_class_by_nclasses; ///< whether to weight an example by the number of classes it belongs to when conssidering its class contraints. 
  int num_threads; ///< number of threads to run on (negative number for max threads)
  int seed; ///< the random seed. if 0 then ititialized from time.
  size_t finite_diff_test_epoch; ///< number of iterations between testing the gradient with finite differences. 0 for no testing.
  size_t no_finite_diff_tests; ///< number of instances to perform the finite differences test at each testing round. The instances are randomly picked from the training set. 
  double finite_diff_test_delta; ///< the size of the finite differene
  bool resume; // whether to train more projections. Old projections should be passed to the program.
  bool reoptimize_LU; ///< whether to reoptimize the baounds of the class intervals. 
  int class_samples; ///< the number of negative classes to use at each gradient iteration. 0 to use all the classes
} param_struct;


inline param_struct set_default_params()
{
  param_struct def;
  def.no_projections = 5;
  def.C1=10.0;
  def.C2=1.0;
  def.max_iter=1e6;
  def.eps=1e-4;
  def.update_type = MINIBATCH_SGD;
  def.eta_type = ETA_LIN;
  def.eta=0.1;
  def.min_eta= 0;
  def.batch_size=1000;
  def.avg_epoch=0;
  def.report_epoch=1000;
  def.report_avg_epoch=0; // this is expensive so the default is 0
  def.reorder_epoch=1000;
  def.reorder_type = REORDER_AVG_PROJ_MEANS; // defaults to REORDER_PROJ_MEANS if averaging is off
  def.optimizeLU_epoch=10000; // this is very expensive so it should not be done often
  def.remove_constraints = false;
  def.remove_class_constraints = false;
  def.reweight_lambda = 1;
  def.ml_wt_by_nclasses = false;
  def.ml_wt_class_by_nclasses = false;
  def.num_threads = 0;          // use OPENMP_NUM_THREADS
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
  
#endif
