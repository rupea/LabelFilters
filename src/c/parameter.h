#ifndef __PARAMETER_H
#define __PARAMETER_H

typedef struct
{
  double C1;   // the penalty for an example being outside it's class bounary
  double C2;   // the penalty for an example being inside other class' boundary
  unsigned long max_iter;  // maximum number of iterations
  int batch_size; // size of the minibatch
  double eps; // not used
  double eta; // the initial learning rate. The leraning rate is eta/sqrt(t) where t is the number of iterations
  double min_eta; //the minimum value of the lerarning rate (i.e. lr will be max (eta/sqrt(t), min_eta)
  unsigned long reorder_epoch; //number of iterations between class reorderings. 0 for no reordering of classes
  int max_reorder; // maxumum number of class reorderings. Each reordering runs until convergence or for Max_Iter iterations and after each reordering the learning rate is reset
  long int report_epoch; //number of iterations between computation and report the objective value (can be expensive because obj is calculated on the entire training set). 0 for no reporting
  bool remove_constraints; // whether to remove the constraints for instances that fall outside the class boundaries in previous projections. 
  bool remove_class_constraints; // whether to remove the constraints for examples that fell outside their own class boundaries in previous projections. 
  bool rank_by_mean; // whether to rank the classes by the mean of the projected examples or by the midpoint of its [l,u] interval (i.e. (u-l)/2).
  bool ml_wt_by_nclasses; // whether to weight an example by the number of classes it belongs to when conssidering other class contraints. 
  bool ml_wt_class_by_nclasses; // whether to weight an example by the number of classes it belongs to when conssidering its class contraints. 
  int seed; // the random seed. if 0 then ititialized from time.
} param_struct;


inline param_struct set_default_params()
{
  param_struct def;
  def.C1=10.0;
  def.C2=1.0;
  def.max_iter=1e6;
  def.max_reorder=1;
  def.eps=1e-4;
  def.eta=1;
  def.min_eta=1e-4;
  def.batch_size=1000;
  def.report_epoch=1000;
  def.reorder_epoch=1000;
  def.remove_constraints = false;
  def.remove_class_constraints=false;
  def.rank_by_mean = true;
  def.ml_wt_by_nclasses = false;
  def.ml_wt_class_by_nclasses = false;
  def.seed=0;
  return def;
}
  
#endif
