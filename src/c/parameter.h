#ifndef __PARAMETER_H
#define __PARAMETER_H

typedef struct
{
  double C1;   // the penalty for an example being outside it's class bounary
  double C2;   // the penalty for an example being inside other class' boundary
  unsigned int max_iter;  // maximum number of iterations
  int batch_size; // size of the minibatch
  double eps; // not used
  double eta; // the initial learning rate. The leraning rate is eta/sqrt(t) where t is the number of iterations
  double min_eta; //the minimum value of the lerarning rate (i.e. lr will be max (eta/sqrt(t), min_eta)
  int reorder_epoch; //number of iterations between class reorderings. 0 for no reordering of classes
  int max_reorder; // maxumum number of class reorderings. Each reordering runs until convergence or for Max_Iter iterations and after each reordering the learning rate is reset
  int report_epoch; //number of iterations between computation and report the objective value (can be expensive because obj is calculated on the entire training set). 0 for no reporting
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
  return def;
}
  
#endif
