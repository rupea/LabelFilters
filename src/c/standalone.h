#ifndef STANDALONE_H
#define STANDALONE_H
/** \file
 * Encapsulate original standalone mcsolve/mcproj into library.
 */
#include "find_w.h"
#include "parameter-args.h"

namespace opt {

    /** high level MCsolver api, as used in mcsolve executable */
    class MCsolveProgram : public ::MCsolver {
    public:
#ifdef NDBEUG
        static const int defaultVerbose=0;
#else
        static const int defaultVerbose=1;
#endif
        MCsolveProgram( MCsolveArgs const& a, int const verbose=0 );
        void tryRead( int const verbose=0 );        ///< read x,y data, or throw
        void trySolve( int const verbose=0 );       ///< solve for projections
        void trySave( int const verbose=0 );        ///< save projections to sonlFile
        void tryDisplay( int const verbose=0 );     ///< display normalized projections
    private:
        MCsolveArgs a;
        DenseM xDense;
        bool denseOk;
        SparseM xSparse;
        bool sparseOk;
        SparseMb y;
    };

}//opt::
#endif // STANDALONE_H
