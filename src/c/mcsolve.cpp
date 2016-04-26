/** \file
 * <B>M</B>ulti-<B>C</B>lass <B>solver</B> for discriminating projection lines.
 */

#include "mcsolveProg.hpp"
#include <iostream>

using namespace std;
using ::opt::MCsolveProgram;

int main(int argc, char**argv){
#ifndef NDEBUG
    int const verb = +1;        // verbosity modifier
#else
    int const verb = 0;         // verbosity modifier
#endif
    // run with -h for help
    MCsolveProgram prog(argc,argv,verb);
    cout<<"argc,argv -->\n"<<prog.args().parms<<endl;   // pretty print final parms
    prog.tryRead (verb);                 // read x,y training data
    prog.trySolve(verb);                 // determine best projection lines
    prog.trySave (verb);                 // write outfile, if asked to do so
    // tryDisplay NORMALIZES w for display, so run this AFTER trySave
    prog.tryDisplay(verb);                 // display soln
    cout<<"\nGoodbye"<<endl;
}
