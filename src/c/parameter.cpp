
#include "parameter.h"
#include "printing.hh"  // io_txt<T> io_bin<T>
#include <assert.h>
#include <iostream>
#include <stdexcept>
#include <cctype>       // std::toupper

using namespace std;

void print_parameter_usage()
{
  cout << "     parameters - a structure with the optimization parameters. If a parmeter is not present the default is used" << endl;
  cout << "         Parameters (structure field names) are:" << endl;
  cout << "           no_projections - number of projections to be learned [5]" << endl;
  //cout << "           tot_projections - number of projections total (rest will be random orthogonal unit vectors) [5]" << endl;
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
  cout << "           eta - initial learning rate [1]" << endl;
  cout << "           eta_type - type of learning rate decay:[lin]" << endl;
  cout << "                        const (eta)" << endl;
  cout << "                        sqrt (eta/sqrt(t))" << endl;
  cout << "                        lin (eta/(1+eta*lambda*t))" << endl;
  cout << "                        3_4 (eta*(1+eta*lambda*t)^(-3/4)" << endl;
  cout << "           min_eta - minimum value of the learning rate (i.e. lr will be max (eta/sqrt(t), min_eta)  [1e-4]" << endl;
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

#define ENUM_CASE(NAME) case(NAME): name= #NAME; break
std::string tostring( enum Eta_Type const e )
{
    char const *name=nullptr;
    switch(e){
        ENUM_CASE(ETA_CONST);
        ENUM_CASE(ETA_SQRT);
        ENUM_CASE(ETA_LIN);
        ENUM_CASE(ETA_3_4);
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
        ENUM_CASE(REORDER_AVG_PROJ_MEANS);
        ENUM_CASE(REORDER_PROJ_MEANS);
        ENUM_CASE(REORDER_RANGE_MIDPOINTS);
    }
    assert(name != nullptr);
    return name;
}
std::string tostring( enum Reweight_Type const e )
{
    char const *name=nullptr;
    switch(e){
        ENUM_CASE(REWEIGHT_NONE);
        ENUM_CASE(REWEIGHT_LAMBDA);
        ENUM_CASE(REWEIGHT_ALL);
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
    throw std::runtime_error("Unrecognized string for MCfilter enum Eta_Type");
}
void fromstring( std::string s, enum Update_Type &e )
{
    //cout<<"fromstring("<<s<<", Update_type&="<<(int)e<<")"<<endl;
    ENUM_FIND(BATCH, MINIBATCH_SGD);
    ENUM_FIND(SAFE, SAFE_SGD);
    throw runtime_error("Unrecognized string for MCfilter enum Update_Type");
}
void fromstring( std::string s, enum Reorder_Type &e )
{
    ENUM_FIND(AVG,  REORDER_AVG_PROJ_MEANS);
    ENUM_FIND(PROJ, REORDER_PROJ_MEANS);
    ENUM_FIND(MID,  REORDER_RANGE_MIDPOINTS);
    throw runtime_error("Unrecognized string for MCfilter enum Reorder_Type");
}
void fromstring( std::string s, enum Reweight_Type &e )
{
    ENUM_FIND(NO,   REWEIGHT_NONE);
    ENUM_FIND(LAM,  REWEIGHT_LAMBDA);
    ENUM_FIND(ALL,  REWEIGHT_ALL);
    throw runtime_error("Unrecognized string for MCfilter enum Reweight_Type");
}
#undef ENUM_FIND


using std::ostream;
using std::istream;
using std::string;

#if 0
namespace detail {
    template<typename T> inline ostream& io_txt( ostream& os, T const& x, char const* ws="\n" ){ return os << x << ws; }
    template<typename T> inline istream& io_txt( istream& is, T& x )                    { return is >> x; }
    template<typename T> inline ostream& io_bin( ostream& os, T const& x ) { return os.write(reinterpret_cast<char const*>(&x),sizeof(T)); }
    template<typename T> inline istream& io_bin( istream& is, T& x ) { return is.read (reinterpret_cast<char*>(&x),sizeof(T)); }

    // specializations
    //   strings as length + blob (no intervening space)
    template<> inline ostream& io_txt( std::ostream& os, std::string const& x, char const* /*ws="\n"*/ ){
        uint32_t len=(uint32_t)(x.size() * sizeof(string::traits_type::char_type));
        io_txt(os,len,"");      // no intervening whitespace
        if(os.fail()) throw std::overflow_error("failed string-len-->ostream");
        os<<x;
        if(os.fail()) throw std::overflow_error("failed string-data-->ostream");
        return os;
    }
    template<> inline istream& io_txt( istream& is, string& x ){
        uint32_t len;
        io_txt(is,len);
        if(is.fail()) throw std::underflow_error("failed istream-->string-len");
        x.resize(len,'\0');     // reserve string memory
        is.read(&x[0], len);    // read full string content
        if(is.fail()) throw std::underflow_error("failed istream-->string-data");
        return is;
    }
}
#endif
using namespace detail;

#define PARAM_STRUCT_IO \
        IO(no_projections); \
        /*IO(tot_projections);*/ \
        IO(C1); \
        IO(C2); \
        IO(max_iter); \
        IO(batch_size); \
        IO_enum(update_type); \
        IO(eps); \
        IO_enum(eta_type); \
        IO(eta); \
        IO(min_eta); \
        IO(avg_epoch); \
        IO(reorder_epoch); \
        IO(report_epoch); \
        IO(report_avg_epoch); \
        IO(optimizeLU_epoch); \
        IO_AS(bool,uint32_t,remove_constraints); \
        IO_AS(bool,uint32_t,remove_class_constraints); \
        IO_enum(reweight_lambda); \
        IO_enum(reorder_type); \
        IO_AS(bool,uint32_t,ml_wt_by_nclasses); \
        IO_AS(bool,uint32_t,ml_wt_class_by_nclasses); \
        IO(num_threads); \
        IO(seed); \
        IO(finite_diff_test_epoch); \
        IO(no_finite_diff_tests); \
        IO(finite_diff_test_epoch); \
        IO(no_finite_diff_tests); \
        IO(finite_diff_test_delta); \
        IO_AS(bool,uint32_t,resume); \
        IO_AS(bool,uint32_t,reoptimize_LU); \
        IO(class_samples);

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
