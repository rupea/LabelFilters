#ifndef PARAMETER_ARGS_H
#define PARAMETER_ARGS_H

#include "parameter.h"
#include <boost/program_options.hpp>

namespace opt {
    namespace po = boost::program_options;

    /** translate program arguments --> \c param_struct.
     * \return unparsed / positional arguments. */
    std::vector<std::string> argsParse( int argc, char**argv, struct param_struct& parms );

}//opt::

#endif // PARAMETER_ARGS_H
