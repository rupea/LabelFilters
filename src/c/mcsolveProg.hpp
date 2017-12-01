#ifndef MCSOLVEPROG_HPP
#define MCSOLVEPROG_HPP
/** \file
 * Encapsulate original standalone mcsolve/mcproj into library.
 */
#include "mcsolver.h"
#include "parameter-args.h"
#include <memory>

class MCxyData; 

namespace opt {

  class MCprojProgram; // fwd decl, just in case.

    /** high level MCsolver api, as used in mcsolve executable */
    class MCsolveProgram : private ::opt::MCsolveArgs, public ::MCsolver {
        typedef ::opt::MCsolveArgs A;
        typedef ::MCsolver S;
        friend class MCprojProgram;
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
        ~MCsolveProgram();

        void tryRead( int const verbose=defaultVerbose );        ///< read x,y data, or throw
        void trySolve( int const verbose=defaultVerbose );       ///< solve for projections
        void trySave( int const verbose=defaultVerbose );        ///< save projections to solnFile
        /** display normalized projections. */
        void tryDisplay( int const verbose=defaultVerbose );

        // other utilities ...

        /** create or return a projector for this soln and data, or throw if impossible.
         *
         * \b Note: initial impl copies our MCsoln -- could also do it with a single
         *          \em shared_ptr<MCsoln> I think.
         *
         * \c args can be used to override/add projector arguments.
         * If args are not supplied, then a default set will be provided to
         * <em>agree as much as possible</em> with settings of this solver
         */
        //std::shared_ptr<MCprojProgram> projector(std::string args);
        MCprojProgram* projector(std::string args);

        ::opt::MCsolveArgs const& args() const {return *this;}
        /** utility to save binary Eigen data files.
         * It's especially hard to create compact Eigen SparseM data files.
         * So one way is to write a libsvm-format text file, then run with
         * no --yFile spec (to read libsvm-format), and then \c savex to write
         * a shorter, sparse-binary-Eigen file.
         * \detail
         * These ultimately invoke \c eigen_io_bin* routines of \ref printing.h */
        void savex( std::string fname ) const;
        void savey( std::string fname ) const;                  ///< save binary 'y' data
        /** convert to x data to higher (quadratic) dimensionality.
         * To each existing x.row() append the 'outer-product' as additional
         * dimensions.
         *
         * For example, if x row is 10 long, get 10 + 10*10 = 110
         * dimensional x vectors by appending all products x[i]x[j] for i,j
         * in 0,9 as separate dimensions.
         *
         * Beware -- this might use a LOT of memory.
         * There is no "kernel support" in MCFilter yet! */
        void quadx();
        //::MCsolver const& solver() const {return *this;}
    private:
      MCprojProgram * projProg;
      std::shared_ptr<::MCxyData> xy;
    };

}//opt::
#endif // MCSOLVEPROG_HPP
