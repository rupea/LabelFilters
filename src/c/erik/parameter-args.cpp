
#include "parameter-args.h"

using namespace std;
namespace opt {
    using namespace boost::program_options;

    static void initParameterDesc( po::options_description & desc ){
        desc.add_options()
            ("no_projections", value<uint32_t>()->
 
    std::vector<std::string> argsParse( int argc, char**argv, struct param_struct& parms ){
        parsed_options parsed = 
            command_line_parse(argc,argv).options(desc).allow_unregistered().run();
        return collect_unrecognized( parsed.options, include_positional );
    }


}//opt::

