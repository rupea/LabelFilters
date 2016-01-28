
#include "parameter-args.h"

#include <iostream>
using namespace std;

void myUsage( std::ostream& os ){
    os<<"demo-parms does nothing but try to parse some MCfilter learning parameters\n"
        "  Usage:\n"
        "     demo-parms [options]\n";
}

int main(int argc, char**argv)
{
    param_struct parms;
    parms = set_default_params();
    cout<<" Going to parse args"<<endl;
#if 1
    auto other = opt::mcArgs(argc,argv, parms, myUsage);
    cout<<" Back from parsing args"<<endl;
    cout<<other.size()<<" other arguments"<<endl;
    for( auto s: other ){
        cout<<" other: "<<s<<endl;
    }
#else
    opt::mcArgs(argc,argv, parms, myUsage);
    cout<<" Back from parsing args"<<endl;
#endif
    cout<<" no_projections = "<<parms.no_projections<<endl;
    cout<<"\nGoodbye"<<endl;
}

