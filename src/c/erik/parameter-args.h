#ifndef PARAMETER_ARGS_H
#define PARAMETER_ARGS_H

#include "parameter.h"
#include <boost/program_options.hpp>

namespace opt {
    namespace po = boost::program_options;

    /// \name helpers
    ///@{
    void helpUsageDummy( std::ostream& os );

    /** retrieve po::options_description for \c param_struct */
    void mcParameterDesc( po::options_description & desc );
    ///@}

    /** translate program arguments --> \c param_struct.
     * - other args can be passed along to a second stage of argument parsing
     * \return unparsed / positional arguments. */
    std::vector<std::string> mcArgs( int argc, char**argv, param_struct & parms
                                     , void(*usageFunc)(std::ostream&)=helpUsageDummy );

}//opt::

#endif // PARAMETER_ARGS_H
