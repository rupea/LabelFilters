#ifndef __PARAMETER_H
#define __PARAMETER_H

enum Eta_Type 
  {
    ETA_CONST, // eta
    ETA_SQRT, // eta/sqrt(t)
    ETA_LIN, // eta/(1+eta*lambda*t)
    ETA_3_4  // eta*(1+eta*lambda*t)^(-3/4)
  }; 

typedef struct
{
  int no_projections; // number of projections to be made
  double C1;   // the penalty for an example being outside it's class bounary
  double C2;   // the penalty for an example being inside other class' boundary
  size_t max_iter;  // maximum number of iterations
  int batch_size; // size of the minibatch
  double eps; // not used
  Eta_Type eta_type; // how does the learning rate decay
  double eta; // the initial learning rate. The leraning rate is eta/sqrt(t) where t is the number of iterations
  double min_eta; //the minimum value of the lerarning rate (i.e. lr will be max (eta/sqrt(t), min_eta)
  size_t avg_epoch; // the iteration at which averaging starts. 0 for no averaging. 
  size_t reorder_epoch; //number of iterations between class reorderings. 0 for no reordering of classes
  long int report_epoch; //number of iterations between computation and report the objective value (can be expensive because obj is calculated on the entire training set). 0 for no reporting
  bool remove_constraints; // whether to remove the constraints for instances that fall outside the class boundaries in previous projections. 
  bool remove_class_constraints; // whether to remove the constraints for examples that fell outside their own class boundaries in previous projections. 
  bool rank_by_mean; // whether to rank the classes by the mean of the projected examples or by the midpoint of its [l,u] interval (i.e. (u-l)/2).
  bool ml_wt_by_nclasses; // whether to weight an example by the number of classes it belongs to when conssidering other class contraints. 
  bool ml_wt_class_by_nclasses; // whether to weight an example by the number of classes it belongs to when conssidering its class contraints. 
  int num_threads; // number of threads to run on (negative number for max threads)
  int seed; // the random seed. if 0 then ititialized from time.
  size_t finite_diff_test_epoch; // number of iterations between testing the gradient with finite differences. 0 for no testing.
  size_t no_finite_diff_tests; // number of instances to perform the finite differences test at each testing round. The instances are randomly picked from the training set. 
  double finite_diff_test_delta; // the size of the finite differene
} param_struct;


inline param_struct set_default_params()
{
  param_struct def;
  def.no_projections = 5;
  def.C1=10.0;
  def.C2=1.0;
  def.max_iter=1e6;
  def.eps=1e-4;
  def.eta_type = ETA_LIN;
  def.eta=1;
  def.min_eta=1e-4;
  def.batch_size=1000;
  def.avg_epoch=0;
  def.report_epoch=1000;
  def.reorder_epoch=1000;
  def.remove_constraints = false;
  def.remove_class_constraints=false;
  def.rank_by_mean = true;
  def.ml_wt_by_nclasses = false;
  def.ml_wt_class_by_nclasses = false;
  def.num_threads = -1; 
  def.seed=0; 
  def.finite_diff_test_epoch=0;
  def.no_finite_diff_tests=1;
  def.finite_diff_test_delta=1e-2;
  return def;
}
  
#endif
