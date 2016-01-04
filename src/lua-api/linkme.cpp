#include "mclua.hpp"
#include <iostream>

using namespace std;
using namespace MILDE;
int main(int,char**)
{
    script_MCparm x = script_MCparm();
    scr_MCparm y = scr_MCparm();
    cout<<" GAS.d_si = "<<GAS.d_si<<endl;
    // the lua interpreter is not there, since GAS.d_si is still a nullptr.
    // ---> SIGSEGV as soon as f_type is called
    cout<<" y type is "<<script_MCparm::f_type()<<endl;
    cout<<" y get_no_projections "<<script_MCparm::f_get_no_projections()<<endl;
    cout<<"\nGoodbye"<<endl;
}
    

