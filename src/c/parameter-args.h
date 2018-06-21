#ifndef PARAMETER_ARGS_H
#define PARAMETER_ARGS_H

#include "parameter.h"
#include <boost/program_options.hpp>

namespace opt {
    namespace po = boost::program_options;

    /// \name helpers
    ///@{
    void helpUsageDummy( std::ostream& os );

    /** split a std::string \c cmdline into a vector of items,
     * handling whitespace, backslash-escapes and '" quotes.
     *
     * \c haveProgName default [true,1] will SKIP argv[0].
     *          Use \c cmdSplit(str,0) to avoid missing the first option if you need
     *          to parse something like str="--opt1=1 --opt2=2" and NOT skip opt1.
     */
    std::vector<std::string> cmdSplit( std::string cmdline, bool haveProgName=true );

    std::string helpMcParms();     ///< return the boost::program_options help string

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
        MCsolveArgs();                          ///< construct empty w/ default parms
        MCsolveArgs(int argc, char**argv);      ///< construct and parse in 1 step
        /// \name lua api
        //@{
        void parse( int argc, char**argv );
        static std::string defaultHelp();
        //@}
        /// \name helpers
        //@{
        void init( po::options_description & desc ); ///< checks for parse errors
        static void helpUsage( std::ostream& os );
        //@}
        /// \name parse settings
        //@{
        param_struct parms;     ///< solver parameters, \ref parameter.h
        std::string xFile;      ///< x data file name (io via ???)
        std::string yFile;      ///< y data file name (io via eigen_io_binbool)

        std::string solnFile;   ///< solution file basename
        std::string outFile;    ///< output[.soln] file basename
        bool outBinary;         ///< outFile format
        bool xnorm/*=false*/;   ///< normalize x dims across examples to mean and stdev of 1.0
        bool xunit/*=false*/;   ///< normalize each x example to unit length
        double xscale;          ///< multiply each x example by a constant
        //bool xquad;             ///< add quadratic dimensions to each x example
        int verbose;            ///< verbosity
        //@}
    };
    
}//opt::

namespace po = boost::program_options;
struct MCxyDataArgs {
  static po::options_description getDesc();
  
  MCxyDataArgs();
  MCxyDataArgs(po::variables_map const& vm);
  void extract(po::variables_map const& vm);
  
  std::vector<std::string> xFiles;      ///< x data file name (io via ???)
  std::vector<std::string> yFiles;      ///< optional. If present, same length as xFiles.
  std::string normData;
  uint rmRareF; /*=0*/  // remove features with fewer than rmRareF non-zero entries.   
  uint rmRareL; /*=0*/  // remove lables with fewer than rmRareL non-zero entries.   
  bool xnorm/*=false*/;   ///< normalize x dims across examples to mean and stdev of 1.0
  int center; ///< center the data when col-normalizing. -1: center for dense data don't center for sparse data. 0: don't center, 1:center. (Default -1) 
  bool xunit/*=false*/;   ///< normalize each x example to unit length
  double xscale;          ///< multiply each x example by a constant
  
};

struct MCprojectorArgs {      
  static po::options_description getDesc();
  
  MCprojectorArgs();
  MCprojectorArgs(po::variables_map const& vm);
  void extract(po::variables_map const& vm);
  
  std::vector<std::string> lfFiles;   ///< files to read the trained label filters from
  
  // feasible set changes with each dataset/nprojections/lablel filter. Need some naming convention
  //      std::string feasibleFile;    ///< outputs feasible set in feasibleFile. [.proj] file basename (or cout)
  // bool feasibleBinary;         ///< outFile format
  // bool feasibleDense;          ///< outFile format
  
  std::vector<int> nProj;    ///< numbers of filters (projections) to use. -1 means use all filters. -2 means {0,1,2,...,no_fiters}
};

struct MCclassifierArgs{
  static po::options_description getDesc();
  
  MCclassifierArgs();
  MCclassifierArgs(po::variables_map const& vm);
  void extract(po::variables_map const& vm);
  
  std::vector<std::string> modelFiles;
  
  // prediction set changes with every dataset/nprojection/label filter/model file/etc. Need some naming convention
  // std::string predsFile;
  
  double keep_thresh;
  uint32_t keep_top;
  double threshold;
  uint32_t min_labels;
  
};  
#endif // PARAMETER_ARGS_H
