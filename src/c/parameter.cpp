
#include "parameter.h"
#include <iostream>

using namespace std;

void print_parameter_usage()
{
  cout << "     parameters - a structure with the optimization parameters. If a parmeter is not present the default is used" << endl;
  cout << "         Parameters (structure field names) are:" << endl;
  cout << "           no_projections - nubmer of projections to be learned [5]" << endl;
  cout << "           C1 - the penalty for an example being outside it's class bounary" << endl;
  cout << "           C2 - the penalty for an example being inside other class' boundary" << endl;
  cout << "           max_iter - maximum number of iterations [1e^6]" << endl;
  cout << "           batch_size - size of the minibatch [1000]" << endl;
  cout << "           update_type - how to update w, L and U [minibatch]" << endl;
  cout << "                           minibatch - update w, L and U together using minibatch SGD" <<endl;
  cout << "                           safe - update w first without overshooting, then update L and U using projected gradient. batch_size will be set to 1" << endl;
  cout << "           avg_epoch - iteration to start averaging at. 0 for no averaging [0]" << endl;
  cout << "           reorder_epoch - number of iterations between class reorderings. 0 for no reordering of classes [1000]" << endl;
  cout << "           reorder_type - how to order the classes [avg_proj_mean]: " << endl;
  cout << "                           avg_proj_means reorder by the mean of the projection on the averaged w (if averaging has not started is the ame as proj_mean" << endl;
  cout << "                           proj_means reorder by the mean of the projection on the current w" << endl;
  cout << "                           range_midpoints reorder by the midpoint of the [l,u] interval (i.e. (u-l)/2)" << endl;
  cout << "           optimizeLU_epoch - number of iterations between full optimizations of  the lower and upper class boundaries. Expensive. 0 for no optimization [10000]" << endl;
  cout << "           report_epoch - number of iterations between computation and report the objective value (can be expensive because obj is calculated on the entire training set). 0 for no reporting [1000]." << endl;
  cout << "           report_avg_epoch - number of iterations between computation and report the objective value for the averaged w (this can be quite expensive if full optimization of LU is turned on, since it first fully optimize LU and then calculates the obj on the entire training set). 0 for no reporting [0]." << endl;
  cout << "           eta - the initial learning rate. The leraning rate is eta/sqrt(t) where t is the number of iterations [1]" << endl;
  cout << "           eta_type - the type of learning rate decay:[lin]" << endl;
  cout << "                        const (eta)" << endl;
  cout << "                        sqrt (eta/sqrt(t))" << endl;
  cout << "                        lin (eta/(1+eta*lambda*t))" << endl;
  cout << "                        3_4 (eta*(1+eta*lambda*t)^(-3/4)" << endl;
  cout << "           min_eta - the minimum value of the lerarning rate (i.e. lr will be max (eta/sqrt(t), min_eta)  [1e-4]" << endl;
  cout << "           remove_constraints - whether to remove the constraints for instances that fall outside the class boundaries in previous projections. [false] " << endl;
  cout << "           remove_class_constraints - whether to remove the constraints for examples that fell outside their own class boundaries in previous projections. [false] " << endl;
  cout << "           reweight_lambda - whether to diminish lambda and/or C1 as constraints are eliminated. 0 - do not diminish any, 1 - diminish lambda only, 2 - diminish lambda and C1 (increase C2) [1]." << endl;
  cout << "           ml_wt_by_nclasses - whether to weight an example by the number of classes it belongs to when conssidering other class contraints. [false]" << endl;
  cout << "           ml_wt_class_by_nclasses - whether to weight an example by the number of classes it belongs to when conssidering its class contraints.[false]" << endl;
  cout << "           seed - random seed. 0 for time dependent seed. [0]" << endl;
  cout << "           num_threads - number of threads to run on. Negative value for architecture dependent maximum number of threads. [0]" << endl;
  cout << "           finite_diff_test_epoch - number of iterations between testign the gradient with finite differences. 0 for no testing [0]" << endl;
  cout << "           no_finite_diff_tests - number of instances to perform the finite differences test at each testing round. The instances are randomly picked from the training set. [1]" << endl;
  cout << "           finite_diff_test_delta - the size of the finite difference. [1e-2]" << endl;
  cout << "           resume - whether to continue with additional projections. Takes previous projections from w_prev l_prev and u_prev. [false]" << endl;
  cout << "           reoptimize_LU - optimize l and u for given projections w_prev. Implies resume is true (i.e. if no_projections > w_prev.cols() additional projections will be learned. [false]" << endl;
  cout << "           class_samples - the number of negative classes to sample for each example at each iteration. 0 to use all classes. [0]" << endl;
}
