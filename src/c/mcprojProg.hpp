#ifndef MCPROJPROG_HPP
#define MCPROJPROG_HPP
/** \file
 * Encapsulate mcfilter into library.
 */
#include "mcfilter.h"
#include "parameter-args.h"

#include <vector>
#include <boost/dynamic_bitset.hpp>
#include <memory>

class MCxyData;

namespace opt {
  
  /** high level MCfilter api, as used in mcproj executable */
  class MCprojProgram : private ::opt::MCprojArgs, public ::MCfilter
  {
    typedef ::opt::MCprojArgs A;
    typedef ::MCfilter F;

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
    MCprojProgram( ::MCsoln const& soln, std::shared_ptr<::MCxyData> data = nullptr,  std::string mod=std::string());
    ~MCprojProgram();
    
    void tryRead( int const verbose=defaultVerbose );            ///< read x,y data, and/or solution or throw
    void tryProj( int const verbose=defaultVerbose );            ///< project data and fiter classes
    void trySave( int const verbose=defaultVerbose );            ///< output fesible classes to outFile (or cout)
    
    void readData(std::string const& xfile);
    void setData(std::shared_ptr<::MCxyData> data);

    ::opt::MCprojArgs const& args() const {return *this;}

    ::MCfilter const& filter() const {return *this;}

  private:
    
    // x examples, from A::xFile, is required, and can be sparse or dense
    std::shared_ptr<::MCxyData> xy;
    /** \b Final projection data [output].
     * For each example row of \c x{Dense|Sparse},
     * raw output is a bitset of feasible classes.
     * ... TBD per projection versions ?
     */
    std::vector<boost::dynamic_bitset<>> feasible;
  };
  
}//opt::
#endif // MCPROJPROG_HPP
