/*  Copyright (C) 2017 NEC Laboratories America, Inc. ("NECLA"). All rights reserved.
 *
 * This source code is licensed under the license found in the LICENSE file in
 * the root directory of this source tree. An additional grant of patent rights
 * can be found in the PATENTS file in the same directory.
 */

#include "parameter.h"
#include "printing.hh"  // io_txt<T> io_bin<T>
#include <assert.h>
#include <iostream>
#include <stdexcept>
#include <cctype>       // std::toupper
#include <sstream>
#include <iomanip>

using namespace std;


void print_parameter_usage()
{
  cout<<
    " parameters - a structure with the optimization parameters."
    "\n              If a parmeter is not present the default is used"
    "\n   Main Parameters (structure field names) are:"
    "\n     nfilters - number of projections to be learned [5]"
    "\n     C1 - the penalty for an example being outside it's class bounary."
    "\n             Internally it is multiplied by the number of classes."
    "\n     C2 - the penalty for an example being inside other class' boundary"
    "\n     max_iter - maximum number of iterations [1e^8/batch_size]"
    "\n     eta - initial learning rate [0.1]"
    "\n     seed - random seed. 0 for time dependent seed. 1 for initializing RNG[0]"
    "\n     num_threads - number of threads to run on. [0 = default num threads]"
    "\n     resume - whether to continue with additional projections."
    "\n              Takes previous projections from w_prev l_prev and u_prev. [false]"
    "\n     reoptimize_LU - optimize l and u for given projections w_prev. Implies"
    "\n              resume is true (i.e. if nfilters > w_prev.cols() additional"
    "\n              projections will be learned. [false]"
    "\n     class_samples - the number of negative classes to sample for each example"
    "\n              at each iteration. 0 to use all classes. [0]"
    "\n   Development Parameters are:"
    "\n     update_type - how to update w, L and U [SAFE]"
    "\n              MINIBATCH - update w, L and U together using minibatch SGD" 
    "\n              SAFE - update w first without overshooting, then update L and U"
    "\n                     using projected gradient. batch_size will be set to 1"
    "\n     batch_size - size of the minibatch [1 for SAFE update, 1000 for MINIBATCH]"
    "\n     eta_type - type of learning rate decay:"
    "\n                  CONST (eta)"
    "\n                  SQRT (eta/sqrt(t))"
    "\n                  LIN (eta/(1+eta*lambda*t))[default if not averaging gradient]"
    "\n                  3_4 (eta*(1+eta*lambda*t)^(-3/4) [default if averaging gradient]"
    "\n     min_eta - minimum value of the learning rate [0]"
    "\n     avg_epoch - iteration to start averaging the gradient at."
    "\n              0 for no averaging [default max(nTrain,dim)]"
    "\n     reorder_epoch - number of iterations between class reorderings."
    "\n              0 for no reordering of classes [1000]"
    "\n     report_epoch - number of iterations between computation and report the objective value"
    "\n              (can be expensive because obj is calculated on the entire training set)."
    "\n              0 for no reporting [max_iter/10]."
    "\n     optimizeLU_epoch - number of iterations between full optimizations of  the"
    "\n              lower and upper class boundaries. Expensive."
    "\n              0 for no optimization [max_iter+1 (i.e. optimize at the beginning and the end)]"
    "\n     remove_constraints - whether to remove the constraints for instances that fall"
    "\n              outside the class boundaries in previous projections. [true] "
    "\n     remove_class_constraints - whether to remove the constraints for examples that fell"
    "\n              outside their own class boundaries in previous projections. [false] "
    "\n     adjust_C - whether to adjust C1 and C2 to account for removed constraints.[true]"
    "\n     reorder_type - how to order the classes [proj_means]: "
    "\n              PROJ_MEANS reorder by the mean of the projection on the"
    "\n                 AVERAGED w (if averaging has not started project on w)"
    "\n              RANGE_MIDPOINTS reorder by the midpoint of the [l,u] interval (i.e. (u-l)/2)"
    "\n     ml_wt_by_nclasses - UNTESTED whether to weight an example by the number of classes it belongs"
    "\n              to when conssidering other class contraints. [false]"
    "\n     ml_wt_class_by_nclasses - UNTESTED whether to weight an example by the number of classes it"
    "\n              belongs to when conssidering its class contraints.[false]"
    "\n     init_type - how to initialize w [3]"
    "\n              0 - initialize with the zero vector"
    "\n              1 - initialize with the previous projection from w_prev"
    "\n              2 - initialize with a random vector"
    "\n              3 - initialize with the vector between two random class centers"
    "\n     init_norm - renormalize the initial w to init_norm. Negative number means not renormalize. [10]"
    "\n     init_orthogonal - make the initial w orthogonal on previous projections [false]"
    "\n     verbose - output verbosity [1]"
    "\n            0 - no output"
    "\n            1 - output progress and objective values"
    
#if GRADIENT_TEST /* || others?*/
      "\n   Compile-time Parameters are:"
      "\n     finite_diff_test_epoch - number of iterations between testign the gradient with finite differences. 0 for no testing [1]"
      "\n     no_finite_diff_tests - number of instances to perform the finite differences test at each testing round. The instances are randomly picked from the training set. [1000]"
      "\n     finite_diff_test_delta - the size of the finite difference. [1e-4]"
#endif
      <<endl;
}

std::ostream& operator<<( std::ostream& os, SolverParams const& p )
{
  os << p.params();
  return os;
}

std::ostream& operator<<( std::ostream& os, param_struct const& p )
{
#define WIDE(OS,N,STUFF) do { \
    std::ostringstream oss; \
    oss<<STUFF; \
    os<<left<<setw(N)<<oss.str(); \
}while(0)
    os<<"MCfilter parameters:\n";
    uint32_t const c1=23U;
    uint32_t const c2=22U;
    uint32_t const c3=23U;
    WIDE(os,c1,right<<setw(14)<<"proj "<<left<<p.nfilters);
    WIDE(os,c2,right<<setw(11)<<"maxiter "<<left<<p.max_iter);
    WIDE(os,c3,right<<setw(15)<<"C1 "<<left<<p.C1);
    os<<endl;
    WIDE(os, c1, right<<setw(14)<<"updatetype "<<left<<tostring(p.update_type));
    WIDE(os,c1,right<<setw(14)<<"batchsize "<<left<<p.batch_size);
    WIDE(os,c3,right<<setw(15)<<"C2 "<<left<<p.C2);
    os<<endl;
    WIDE(os,c1,right<<setw(14)<<"etatype "<<left<<tostring(p.eta_type));
    WIDE(os,c2,"   "<<tostring(p.reorder_type));
    WIDE(os,c3,right<<setw(14)<<"adjustC"<<left<<p.adjust_C);
    os<<endl;
    WIDE(os,c1,right<<setw(14)<<"eta0 "<<left<<p.eta);
    WIDE(os,c2,right<<setw(11)<<"treport "<<left<<p.report_epoch);
    WIDE(os,c2,right<<setw(11)<<"treorder "<<left<<p.reorder_epoch);
    os<<endl;
    WIDE(os,c1,right<<setw(14)<<"etamin "<<left<<p.min_eta);
    WIDE(os,c2,right<<setw(11)<<"toptlu "<<left<<p.optimizeLU_epoch);
    WIDE(os,c3,right<<setw(15)<<"sample "<<left<<p.class_samples);
    os<<endl;
    WIDE(os,c1,right<<setw(14)<<"threads "<<left<<p.num_threads);
    WIDE(os,c1,right<<setw(14)<<"averaged_gradient "<<left<<p.averaged_gradient);     // bool    
    os<<endl;
    WIDE(os,c1,right<<setw(14)<<"remove_constraints "<<left<<p.remove_constraints);     // bool
    WIDE(os,c2,right<<setw(11)<<"avgstart "<<left<<p.avg_epoch);
    os<<endl;
    WIDE(os,c1,right<<setw(14)<<"remove_class "<<left<<p.remove_class_constraints);       // bool
    WIDE(os,c2,right<<setw(11)<<"resume "<<left<<p.resume);  // bool
    WIDE(os,c3,right<<setw(15)<<"wt_nclass_by_nclasses "<<left<<p.ml_wt_class_by_nclasses);        // bool
    os<<endl;
    WIDE(os,c1,right<<setw(14)<<"reoptlu "<<left<<p.reoptimize_LU);  // bool
    WIDE(os,c2,right<<setw(11)<<"seed "<<left<<p.seed);
    WIDE(os,c3,right<<setw(15)<<"wt_by_nclasses "<<left<<p.ml_wt_by_nclasses);      // bool
    os<<endl;
    WIDE(os, c1, right<<setw(14)<<"inittype "<<left<<tostring(p.init_type));
    WIDE(os, c2, right<<setw(11)<<"initnnorm "<<left<<p.init_norm);
    WIDE(os, c3, right<<setw(15)<<"initorthogonal "<<left<<p.init_orthogonal);   // bool
    os<<endl;
    WIDE(os, c1, right<<setw(14)<<"verbose "<<left<<p.verbose);
#if GRADIENT_TEST
    WIDE(os,c3,right<<setw(15)<<"tgrad "<<left<<p.finite_diff_test_epoch);
    WIDE(os,c3,right<<setw(15)<<"ngrad "<<left<<p.no_finite_diff_tests);
    WIDE(os,c3,right<<setw(15)<<"grad "<<left<<p.finite_diff_test_delta);
    os<<endl;
#endif
    return os;
}


#define ENUM_CASE(NAME) case(NAME): name= #NAME; break
std::string tostring( enum Eta_Type const e )
{
    char const *name=nullptr;
    switch(e){
        ENUM_CASE(ETA_CONST);
        ENUM_CASE(ETA_SQRT);
        ENUM_CASE(ETA_LIN);
        ENUM_CASE(ETA_3_4);
        ENUM_CASE(DEFAULT);
    }
    assert(name != nullptr);
    return name;
}
std::string tostring( enum Update_Type const e )
{
    char const *name=nullptr;
    switch(e){
        ENUM_CASE(MINIBATCH_SGD);
        ENUM_CASE(SAFE_SGD);
    }
    assert(name != nullptr);
    return name;
}
std::string tostring( enum Reorder_Type const e )
{
    char const *name=nullptr;
    switch(e){
        ENUM_CASE(REORDER_PROJ_MEANS);
        ENUM_CASE(REORDER_RANGE_MIDPOINTS);
    }
    assert(name != nullptr);
    return name;
}
std::string tostring( enum Init_W_Type const e )
{
    char const *name=nullptr;
    switch(e){
        ENUM_CASE(INIT_ZERO);
        ENUM_CASE(INIT_PREV);
        ENUM_CASE(INIT_RANDOM);
        ENUM_CASE(INIT_DIFF);
    }
    assert(name != nullptr);
    return name;
}
#undef ENUM_CASE

#define ENUM_FIND( NAME, VALUE ) do { \
    for(auto &c: s) c = std::toupper(c); \
    if( s.find(#NAME) != std::string::npos ) {e=VALUE; return;} \
}while(0)
void fromstring( std::string s, enum Eta_Type &e )
{
    ENUM_FIND(CONST, ETA_CONST);
    ENUM_FIND(SQRT,  ETA_SQRT);
    ENUM_FIND(LIN,   ETA_LIN);
    ENUM_FIND(3_4,   ETA_3_4);
    ENUM_FIND(DEFAULT,   DEFAULT);
    
    throw std::runtime_error("Unrecognized string for MCfilter enum Eta_Type");
}
void fromstring( std::string s, enum Update_Type &e )
{
    ENUM_FIND(MINIBATCH, MINIBATCH_SGD);
    ENUM_FIND(SAFE, SAFE_SGD);
    ENUM_FIND(PROJECTED, SAFE_SGD);
    throw runtime_error("Unrecognized string for MCfilter enum Update_Type");
}
void fromstring( std::string s, enum Reorder_Type &e )
{
    ENUM_FIND(PROJ, REORDER_PROJ_MEANS);
    ENUM_FIND(MID,  REORDER_RANGE_MIDPOINTS);
    throw runtime_error("Unrecognized string for MCfilter enum Reorder_Type");
}
void fromstring( std::string s, enum Init_W_Type &e )
{
    ENUM_FIND(ZERO,   INIT_ZERO);
    ENUM_FIND(PREV,  INIT_PREV);
    ENUM_FIND(RAND,  INIT_RANDOM);
    ENUM_FIND(DIFF,  INIT_DIFF);
    throw runtime_error("Unrecognized string for MCfilter enum Init_W_Type");
}
#undef ENUM_FIND


using namespace detail;

#define PARAM_STRUCT_IO \
        IO(nfilters); \
        IO(C1); \
        IO(C2); \
        IO(max_iter); \
        IO(seed); \
	IO(num_threads);	     \
        IO_AS(bool,uint32_t,resume); \
        IO_AS(bool,uint32_t,reoptimize_LU); \
        IO(class_samples); \
        IO_enum(update_type); \
        IO(batch_size); \
        IO_enum(eta_type); \
        IO(eta); \
        IO(min_eta); \
        IO_AS(bool,uint32_t,averaged_gradient); \
        IO(avg_epoch); \
        IO(reorder_epoch); \
        IO(report_epoch); \
        IO(optimizeLU_epoch); \
        IO_AS(bool,uint32_t,remove_constraints); \
        IO_AS(bool,uint32_t,remove_class_constraints); \
        IO_AS(bool,uint32_t,adjust_C); \
        IO_enum(reorder_type); \
        IO_AS(bool,uint32_t,ml_wt_by_nclasses); \
        IO_AS(bool,uint32_t,ml_wt_class_by_nclasses); \
	IO_enum(init_type); \
	IO(init_norm); \
	IO_AS(bool,uint32_t,init_orthogonal); \
	IO(verbose);\
        IF_GRADIENT_TEST( IO(finite_diff_test_epoch) ); \
        IF_GRADIENT_TEST( IO(no_finite_diff_tests) ); \
        IF_GRADIENT_TEST( IO(finite_diff_test_delta) ); \

int write_ascii( std::ostream& os, param_struct const& p )
{
    // types with standard sizes, or with io_txt|bin overrides
#define IO(PARM) do { io_txt(os, p.PARM); if(os.fail()) goto ERR; }while(0)
    // enums have to|from-string converters
#define IO_enum(PARM) do { string s = tostring(p.PARM); io_txt(os, s); if(os.fail()) goto ERR; }while(0)
    // for types like bool, which have no predefined size
#define IO_AS(TYPE,AS,PARM) do { AS t = (AS)(p.PARM); io_txt(os, t); if(os.fail()) goto ERR; }while(0)
    try {
        PARAM_STRUCT_IO;
    }catch(std::exception &err){
        cout<<err.what()<<endl;
        goto ERR;
    }
#undef IO_AS
#undef IO_enum
#undef IO
    return 0;   // no error;
ERR:
    return 1;   // failed read
}
int read_ascii( std::istream& is, param_struct& p )
{
#define IO(PARM) do { io_txt(is, p.PARM); if(is.fail()) goto ERR; }while(0)
#define IO_enum(PARM) do { string s; io_txt(is,s); fromstring(s,p.PARM); if(is.fail()) goto ERR; }while(0)
#define IO_AS(TYPE,AS,PARM) do { AS x; io_txt(is,x); p.PARM = (TYPE)x; if(is.fail()) goto ERR; }while(0)
    try{
        PARAM_STRUCT_IO;
    }catch(std::exception &err){
        cout<<err.what()<<endl;
        goto ERR;
    }
#undef IO_AS
#undef IO_enum
#undef IO
    return 0;   // no error;
ERR:
    return 1;   // failed read
}
int write_binary( std::ostream& os, param_struct const& p )
{
#define IO(PARM) do { io_bin(os, p.PARM); if(os.fail()) goto ERR; }while(0)
#define IO_enum(PARM) do { string s = tostring(p.PARM); io_bin(os, s); if(os.fail()) goto ERR; }while(0)
#define IO_AS(TYPE,AS,PARM) do { AS t = (AS)(p.PARM); io_bin(os, t); if(os.fail()) goto ERR; }while(0)
    try{
        PARAM_STRUCT_IO;
    }catch(std::exception &err){
        cout<<err.what()<<endl;
        goto ERR;
    }
#undef IO_AS
#undef IO_enum
#undef IO
    return 0;   // no error;
ERR:
    return 1;   // failed read
}
int read_binary( std::istream& is, param_struct& p )
{
#define IO(PARM) do { io_bin(is, p.PARM); if(is.fail()) goto ERR; }while(0)
#define IO_enum(PARM) do { string s; io_bin(is,s); fromstring(s,p.PARM); if(is.fail()) goto ERR; }while(0)
#define IO_AS(TYPE,AS,PARM) do { AS x; io_bin(is,x); p.PARM = (TYPE)x; if(is.fail()) goto ERR; }while(0)
    try{
        PARAM_STRUCT_IO;
    }catch(std::exception &err){
        cout<<err.what()<<endl;
        goto ERR;
    }
#undef IO_AS
#undef IO_enum
#undef IO
    return 0;   // no error;
ERR:
    return 1;   // failed read
}


SolverParams::SolverParams():    
  default_batch_size(true),
  default_report_epoch(true),
  default_optimizeLU_epoch(true),
  default_init_norm(true),
  default_eta_type(true),
  default_max_iter(true)
{
  m_params = set_default_params();
}

