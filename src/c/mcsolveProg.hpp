#ifndef MCSOLVEPROG_HPP
#define MCSOLVEPROG_HPP
/** \file
 * Encapsulate original standalone mcsolve/mcproj into library.
 */
#include "find_w.h"
#include "parameter-args.h"

namespace opt {

    /** high level MCsolver api, as used in mcsolve executable */
    class MCsolveProgram : private ::opt::MCsolveArgs, public ::MCsolver {
        typedef ::opt::MCsolveArgs A;
        typedef ::MCsolver S;
    public:
#ifdef NDBEUG
        static const int defaultVerbose=0;
#else
        static const int defaultVerbose=1;
#endif
        /** construct from cmdline args (ignoring argv[0]) */
        MCsolveProgram( int argc, char** argv
                        , int const verbose=defaultVerbose
                        , param_struct const* const defparms=nullptr );

        void tryRead( int const verbose=0 );        ///< read x,y data, or throw
        void trySolve( int const verbose=0 );       ///< solve for projections
        void trySave( int const verbose=0 );        ///< save projections to sonlFile
        /** display normalized projections.
         * \c tryDisplay \em normalizes projection directions for display,
         * so be sure to \c trySave \em before you \c tryDisplay. FIXME */
        void tryDisplay( int const verbose=0 );

        ::opt::MCsolveArgs const& args() const {return *this;}
        //::MCsolver const& solver() const {return *this;}
    private:
        DenseM xDense;
        bool denseOk;
        SparseM xSparse;
        bool sparseOk;
        SparseMb y;
    };

}//opt::
#endif // MCSOLVEPROG_HPP
