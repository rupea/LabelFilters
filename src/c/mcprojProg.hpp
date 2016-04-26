#ifndef MCPROJPROG_HPP
#define MCPROJPROG_HPP
/** \file
 * Encapsulate original standalone mcsolve/mcproj into library.
 */
#include "find_w.h"
#include "parameter-args.h"

#include <vector>
#include <boost/dynamic_bitset.hpp>

namespace opt {

    /** high level MCsolver api, as used in mcsolve executable */
    class MCprojProgram : private ::opt::MCprojArgs
                          //, public ::MCsolver
    {
        typedef ::opt::MCprojArgs A;
        //typedef ::MCfilter F; // or projector or predictor ???
    public:
#ifdef NDBEUG
        static const int defaultVerbose=0;
#else
        static const int defaultVerbose=1;
#endif
        /** construct from cmdline args (ignoring argv[0]).
         * \c verbose argument is only for the constructor.
         * tryFOO verbosity is from a default level of 0|1 (it is a debug
         * compilation) \b plus a --verbose[=int] commandline verbosity
         * of \c A::verbose.
         */
        MCprojProgram( int argc, char** argv
                        , int const verbose=defaultVerbose );

        void tryRead( int const verbose=defaultVerbose );            ///< read x,y data, or throw
        void tryProj( int const verbose=defaultVerbose );            ///< solve for projections
        void trySave( int const verbose=defaultVerbose );            ///< save projections to sonlFile (or cout)
        void tryValidate( int const verbose=defaultVerbose );        ///< validate (if y available) TBD

        ::opt::MCprojArgs const& args() const {return *this;}
        //::MCsolver const& solver() const {return *this;}
    private:
        // x examples, from A::xFile, is required, and can be sparse or dense
        MCsoln soln;            ///< required, from A::solnFile
        /// \name row-wise test data matrix
        //@{
        DenseM xDense;
        bool denseOk;
        SparseM xSparse;
        bool sparseOk;
        //@}
        SparseMb y;             ///< optional, from A::yFile (validiation TBD)
        /** projection data [output].
         * For each example row of \c x{Dense|Sparse},
         * raw output is a bitset of feasible classes. */
        std::vector<boost::dynamic_bitset<>> feasible;

#if 0 // pure binary outputs might not be very useful.
        /** Alt per-projection outputs (detail view of results).
         * If used, size of 1st dim will be \c soln.nProj. */
        std::vector< std::vector< boost::dynamic_bitset >> projFeasible;
#endif
    };

}//opt::
#endif // MCPROJPROG_HPP
