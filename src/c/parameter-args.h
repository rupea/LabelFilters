#ifndef PARAMETER_ARGS_H
#define PARAMETER_ARGS_H

#include "parameter.h"
#include <boost/program_options.hpp>

namespace opt {
    namespace po = boost::program_options;

    /// \name helpers
    ///@{
    void helpUsageDummy( std::ostream& os );

#if 0
    /** retrieve po::options_description for \c param_struct with \em standard default values.
     * \deprecated */
    void mcParameterDesc( po::options_description & desc );
#endif

    /** Retrieve po::options_description with user-specified default values.
     * - Especially if continuing previous run
     * - This form allows command-line settings to \em over-ride existing run settings
     *   - e.g. read from a configuration file (or a .soln or .mc file?)
     * \detail
     * This approach seems easier than using the 'next' feature allowing
     * chained boost variables_map objects to override defaults.
     */
    void mcParameterDesc( po::options_description & desc, param_struct const& p );

    /** vm --> parms */
    void extract( po::variables_map const& vm, param_struct & parms );
    ///@}

    /** translate program arguments --> \c param_struct.
     * \pre \b \c parms has been initialized, perhaps as \c parms=set_default_params()
     * Returned args might be passed along to a second stage of argument parsing?
     * \return unparsed / positional arguments.
     * (only used in demo programs?)
     */
    std::vector<std::string> mcArgs( int argc, char**argv, param_struct & parms
                                     , void(*usageFunc)(std::ostream&)=helpUsageDummy );

    /** wrap program options for standalone mcsolve executable */
    struct MCsolveArgs {
        MCsolveArgs();
        /// \name lua api
        //@{
        void argsParse( int argc, char**argv );
        void init( po::options_description & desc );
        static std::string defaultHelp();
        //@}
        /** internal helper for argsParse errors */
        static void helpUsage( std::ostream& os );
        /// \name argsParse settings
        //@{
        param_struct parms;     ///< solver parameters, \ref parameter.h
        std::string xFile;      ///< x data file name (io via ???)
        std::string yFile;      ///< y data file name (io via eigen_io_binbool)
        std::string solnFile;   ///< solution file basename
        std::string outFile;    ///< solution file basename
        bool outBinary;         ///< outFile format
        bool outText;           ///< outFile format
        bool outShort;          ///< outFile format
        bool outLong;           ///< outFile format
        bool xnorm/*=false*/;   ///< normalize x dims across examples to mean and stdev of 1.0
        //@}
    };

    namespace MCproj { // TODO XXX CHECKME 
        std::string defaultHelp();
        void init( po::options_description & desc );
        void argsParse( int argc, char**argv );
    }//MCproj::

}//opt::

#endif // PARAMETER_ARGS_H
