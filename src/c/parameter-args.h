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

}//opt::

#endif // PARAMETER_ARGS_H
