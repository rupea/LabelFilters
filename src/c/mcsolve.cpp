/** \file
 * <B>M</B>ulti-<B>C</B>lass <B>solver</B> for discriminating projection lines.
 */

#include "mcsolveProg.hpp"
#include <iostream>

using namespace std;
using ::opt::MCsolveProgram;

int main(int argc, char**argv){
    // run with -h for help
    MCsolveProgram prog(argc,argv,/*verbose=*/1);
    cout<<"argc,argv -->\n"<<prog.args().parms<<endl;          // pretty print final parms
    prog.tryRead(/*verbose=*/1);                // read x,y training data
    prog.trySolve(/*verbose=*/1);               // determine best projection lines
    prog.trySave(/*verbose=*/1);                // write outFile, if asked to do so
    // tryDisplay NORMALIZES w for display, so run this AFTER trySave
    prog.tryDisplay(/*verbose=*/1);             // display soln
    cout<<"\nGoodbye"<<endl;
}
