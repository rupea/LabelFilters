
#include "standalone.h"

#include <stdexcept>
#include <iostream>
#include <omp.h>

namespace opt {
    using namespace std;
    MCsolveProgram::MCsolveProgram( MCsolveArgs const& a, int const verbose/*=0*/ )
        : a(a)
          , xDense()
          , denseOk(false)
          , xSparse()
          , sparseOk(false)
          , y() // SparseMb
    {
    }
    void MCsolveProgram::tryRead( int const verbose/*=0*/ ){
        if(verbose) cout<<"MCsolveProgram::tryRead()"<<endl;
        throw std::runtime_error("TBD");
    }
    void MCsolveProgram::trySolve( int const verbose/*=0*/ ){
        if(verbose) cout<<"MCsolveProgram::tryRead()"<<endl;
        throw std::runtime_error("TBD");
    }
    void MCsolveProgram::trySave( int const verbose/*=0*/ ){
        if(verbose) cout<<"MCsolveProgram::tryRead()"<<endl;
        throw std::runtime_error("TBD");
    }
    void MCsolveProgram::tryDisplay( int const verbose/*=0*/ ){
        if(verbose) cout<<"MCsolveProgram::tryRead()"<<endl;
        throw std::runtime_error("TBD");
    }
}//opt::
