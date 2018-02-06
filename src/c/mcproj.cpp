/** \file
 * For every example in a test set, use <B>M</B>ulti-<B>C</B>lass <B>proj</B>ections
 * to determine {possible class labels}.
 */

#include "mcprojProg.hpp"
#include "mcpredict.h"

#include <iostream>

using namespace std;
using opt::MCprojProgram;

int main(int argc, char**argv){
#ifndef NDEBUG
    int const verb = +1;        // verbosity modifier
#else
    int const verb = 0;         // verbosity modifier
#endif
    MCprojProgram prog(argc,argv,verb);
    prog.tryRead( verb );       // read soln, x [,y] data
    prog.tryProj( verb );       // project
    prog.trySave( verb );       // write outfile, if asked to do so
    cout<<"\nGoodbye"<<endl;
}
