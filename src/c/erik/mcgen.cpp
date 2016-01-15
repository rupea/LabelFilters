/** \file
 * test case generator for multi-class label filter.
 */

#include <cstdint>
#include <iosfwd>
#include <string>

// fwd and function declarations
namespace mcgen {
    namespace opt {
        /** internal view of program parameters + dependent constants */
        struct Parms;

        /** translate program arguments --> \c Parms */
        void argsParse( int argc, char**argv, struct Parms& parms );

        /** program usage */
        void help( std::ostream& os );

    }//opt::
}//mcgen::

// structure declarations
namespace mcgen {
    namespace opt {
        class Parms {
        public:
            Parms();                        ///< construct with defaults, and call init()

            /// \name core settings
            //@{
            uint32_t axes;                  ///< [3] # projection axes to separate examples
            bool     ball;                  ///< [*] true = examples on unit ball, false = rotated unit hypercube
            uint32_t dim;                   ///< [3] dimension of each training example
            int32_t  margin;                ///< [+axes] |margin|<=a 0=none, +ve=separable, -ve=non-separable
            uint32_t parts;                 ///< [2] each axis separates into parts, nClass = parts ^ axes [2^3]
            bool     trivial;               ///< [1]: soln is first \c axes unitvecs (skew,rot=0), 0: use skew,rot!=0
            uint32_t skew;                  ///< [0] skew < axes dims toward (1,0,0,...) of trivial soln
            uint32_t rot;                   ///< [0] then project into axes < rot < dim and do a random rotation
            //@}
            /// \name non-core settings set via init()
            //@{
            uint32_t nClass;                ///< = parts ^ axes
            uint32_t nxStd;                 ///< pre-determined examples 1 per nClass + (margin!=0? 2*|margin|*a*(parts-1)
            uint32_t nx;                    ///< default (and silent minimum) is max(4*axes, nxStd)
            //@}

            /** Must call after after changing core variables.
             * - does not change \c margin (take care when changing \c axes).
             * \throw runtime_error for crazy settings. */
            void init();

            /** translate Parms core settings into a canonical string.
             * - minimal is "a[axes]d[dim]p[parts]"
             * - default is "a3d3p2", for 3 axes, 3 dims, 2 parts per axis;
             *   -  i.e. one separating plane per axis, for 8 separable classes. */
            std::string str();
        };
    }//opt::
}//mcgen

// -------- implementation --------

#include <boost/program_options.hpp>

// declare implementation details
namespace mcgen {
    namespace opt {
        namespace po = boost::program_options;

        // internal helper declarations
        static void init( po::options_description & desc );                     ///< initialize po
        static void helpUsage( std::ostream& os );                              ///< intro help
        //static void help( std::ostream& os, po::options_description & desc );   ///< options help
        static void helpExamples( std::ostream& os );                           ///< closing comments
    }//opt::
}//mcgen

// define implementation & details
#include <iostream>
namespace mcgen {
    namespace opt {
        // implementations
        using namespace std;
        using namespace boost;

        Parms::Parms()
            : axes( 3U )
              , ball( true )
              , dim( 3U )
              , margin( 3U )
              , parts( 2U )
              , trivial( true )
              , skew( 0U )
              , rot( 0U )
              , nClass( 0U ) // set via init()
              , nxStd( 0U )     // set via init()
              , nx( 0U )        // set via init()
        {
            init();
        }
        void Parms::init(){
            nClass = parts;
            //cout<<" nClass="<<parts; cout.flush();
            for(uint32_t i=1U; i<axes; ++i){
                nClass *= parts;
                //cout<<" .."<<i<<".."<<nClass; cout.flush();
            }
            //cout<<endl;
            nxStd = nClass;
            if( margin != 0 ){
                uint32_t m = static_cast<uint32_t>((margin>0? margin: -margin));
                nxStd += 2 * m *axes * (parts-1U);
            }
            uint32_t nxMinimum = std::max(4*axes, nxStd);
            nx = std::max( nx, nxMinimum );                     // nx must be at least some minimum value
            // - 1st of nClass nxStd examples are ~ [farthest] class centers
            // - rest of nxStd are margin-tightening examples
            // - remaining (nx-nxStd) randomly scattered over example space.
        }

        static void helpUsage( std::ostream& os ){
            os<<
                "\n Usage:"
                "\n    mcgen [options]"
                "\nFunction: generate test problems for multiclass label filtering"
                "\n    produces training file mcgen-OPT.train.txt"
                "\n    produces suggested projection vector matrix mcgen-OPT.soln.txt"
                <<endl;
        }
        static void init( boost::program_options::options_description & desc ){
            desc.add_options()
                ("help,h", "help")

                ("axes,a", po::value<uint32_t>()->implicit_value(3U)
                 , "[3] projection|soln axes, 2 <= a <= d")
                ("ball,b", po::value<bool>()->implicit_value(true)->default_value(true)
                 , "[*] examples on unit ball")
                ("cube,c", po::value<bool>()->implicit_value(false)->default_value(false)
                 , "... or examples in [deformed? r,s] unit cube")
                ("dim,d", po::value<uint32_t>()->implicit_value(3U)
                 , "[3] dimension of training data, d >= 2")
                ("margin,m", po::value<uint32_t>()
                 , "[a] |m|<=a, add 2|m|a(p-1) margin tightness examples."
                 "  +ve: separable, -ve: non-separable, 0: no margin-examples"
                 )
                ("parts,p", po::value<uint32_t>()->implicit_value(2U)->default_value(2U)
                 , "[2] p>=2 parts to divide each -a axis into. # classes = p^a [8]")

                ("rot,r", po::value<uint32_t>()
                 , "after t+s, move into a<=r<=d dims & rotate randomly (keep lengths,angles)")
                ("skew,s", po::value<uint32_t>()
                 , "after t, rot 0<s<a-1 dims (2..s+1) 1st unit (1,0,0...) (axes-->non-orthog)")
                ("trivial,t"
                 , "[*] soln: unit vectors in 1st -a dimensions form projection axes")

                ("examples,x", po::value<uint32_t>()
                 , "[0] How many, silently adjusted up to max(a*4, 2*<pre-determined examples>")
                ;
        }
        static void helpExamples( std::ostream& os ){
            os <<"Notes:"
                "\n - OPT string for filename is options in alphabetical order"
                "\n   with a, d, p (axes, dim, parts) required, and defaults omited"
                "\n   - Ex. defaults correspond to an options string of a3d3p2"
                "\n - Pre-determined examples (#)"
                "\n   - 1 example/class is a 'far center' (# = p^a = #_classes)"
                "\n   - m!=0 generates a*2*|m| more examples per axis (# += 2ad)"
                "\n - Ex. default a=3 p=2 m=a"
                "\n       # = 2^3 + 2*3*3*(2-1) = 8 + 36 = 44 predetermined points"
                "\n       default training examples (x) = max(4*3, 44) = 44"
                "\n - Ex. a=3 p=2 m=0 ---> # = 8 , default (x) = max(4*3, 8) = 12"
                "\n - s skewed signal still in 1st a dims"
                "\n - r moves signal into 1st a<=r<=d dims, higher dims ~ noise"
                ;
        }
        /** \c p might not reflect values it would have after init() */
        std::string Parms::str(){
            Parms def;          // get default core values (well, some have been inited to nonzero)
            ostringstream s;
            s<<'a'<<axes;                               // non-optional
            if( ball != def.ball ) s<<(ball? 'b': 'c'); // optional
            s<<'d'<<dim;                                // non-optional
            if( margin != static_cast<int32_t>(axes) ) s<<'m'<<margin;
            //if( parts != def.parts ) s<<'p'<<parts;     // optional
            s<<'p'<<parts;                              // non-optional
            if( ! trivial ){
                assert( skew > 0U || rot > 0U );
                if( skew != 0U ) s<<'s'<<skew;
                if( rot != 0U ) s<<'r'<<rot;
            }
            // nx is tricky: it is non-core, but default value is set by init()
            def = *this;
            def.init();          // default value is now cp.nx
            if( nx != def.nx ) s<<'x'<<def.nx;
            return s.str();
        }
        void argsParse( int argc, char**argv, struct Parms& parms ){
#define ARGSDEBUG 1            
#if ARGSDEBUG > 0
            cout<<" argsParse( argc="<<argc<<", argv, ... )"<<endl;
            for( int i=0; i<argc; ++i ) {
                cout<<"    argv["<<i<<"] = "<<argv[i]<<endl;
            }
#endif
            try {
                po::options_description desc("Options");
                init( desc );                    // create a description of the options

                po::variables_map vm;
                //po::store( po::parse_command_line(argc,argv,desc), vm );
                // Need more control, I think...
                {
                    po::parsed_options parsed
                        = po::command_line_parser( argc, argv )
                        .options( desc )
                        .positional( po::positional_options_description() ) // empty, none allowed.
                        //.allow_unregistered()
                        .run();
                    po::store( parsed, vm );
                }

                po::notify(vm);

                if( vm.count("help") ) {
                    helpUsage( cout );
                    cout<<desc<<endl;
                    helpExamples(cout);
                    return;
                }
#if 0
                if( vm.count("swap") )
                {
                    opt.swap = vm["swap"].as<uint32_t>();
                    //if( opt.swap > MAXONDISKALG )
                    //{
                    //    cout<<"Unknown block-swapping algorithm # "<<opt.swap<<endl;
                    //    throw "Unknown block-swapping algorithm #";
                    //}
                }

                opt.time = 0U;          // replace NWRITES
                opt.traces = 0U;        // use traces unless other args (--time) force us not to
                opt.awPercent = 0U;
                opt.ek23 = vm.count("ek23");
                opt.tp25 = vm.count("tp25");
                opt.idmap = vm.count("idmap");

                if( vm.count("time") )  // if --time (or -t) option appeared, only OK for fake workload
                {
                    opt.time = vm["time"].as<float>()*SECONDSTOTIME;
                    cout<<" opt.time represents "<<opt.time*TIMETOSECONDS<<" s"<<endl;
                    //if( vm.count("traces") && vm["traces"].as<bool>() )
                    //{
                    //cout<<" our TracePlay doesn't know how to shut off after a given --time="
                    //cout<<" our TracePlay MIGHT know how to shut off after a given --time="
                    //    <<opt.time*TIMETOSECONDS<<endl;
                    //throw "oops";
                    //}
                    //opt.traces = 0;
                }
                if( vm.count("async") )  // if --time (or -t) option appeared, only OK for fake workload
                {
                    opt.awPercent = vm["async"].as<uint32_t>();
                    if( opt.awPercent > 100U )
                    {
                        cout<<" --async="<<opt.awPercent<<" \%-of-writes-async must be in range [0,100]"<<endl;
                        throw("oops");
                    }
                }

                opt.nTimes = vm["times"].as<uint32_t>();
                opt.tMon = vm["tMon"].as<float>();

                opt.wol  = vm["wol"].as<uint32_t>();
                if( opt.wol > 3U )
                {
                    cout<<" --wol="<<opt.wol<<" out of range [0,3]"<<endl;
                    throw " --wol should be 0,1,2,3";
                }
                if( opt.swap == 0U && opt.wol == 1U )
                {
                    cout<<"--wol ineffective with --swap=0 (NEVERSWAP)"<<endl;
                    throw("oops");
                }

                opt.nio = 0U;
                if( vm.count("nio") )   // --nio (or -n) option appeared, only OK for trace files
                {
                    opt.nio = vm["nio"].as<uint32_t>(); 
                    // BOTH WorkModel and current TwrkWrite DO accept --nio switch
                }

                if( vm.count("traces") )        // if --traces (or -T) appeared as an option...
                {
                    opt.traces = vm["traces"].as<bool>();
                }
                { // --traces with missing --time / --nio specs get some low default values
                    if(  opt.traces && opt.nio == 0U )
                    {
                        //opt.nio = 20U;   // something small
                        //cout<<" TracePlay will use low default value of --nio="<<opt.nio<<endl;
                        cout<<" pa21 using whole trace files, --nio="<<opt.nio<<endl;
                    }
                    if( !opt.traces && opt.time == 0U && opt.nio == 0U )
                    {
                        opt.time = 1U*SECONDSTOTIME;   // something small
                        cout<<" WorkModel will use low default value of --time="
                            <<opt.time*TIMETOSECONDS<<" (seconds)"<<endl;
                    }
                    // TODO -- fix loadGen.cpp so that --time=0 makes it go "forever". for now kludge it
                    //         by using some high value (e.g. 50000 seconds)
                    if( !opt.traces && opt.nio > 0U && opt.time == 0U )
                    {
                        opt.time = 50000U * SECONDSTOTIME;
                        cout<<" WorkModel with --nio will use some high-ish value --time="
                            <<opt.time*TIMETOSECONDS<<" (seconds)"<<endl;
                    }
                }

                // ... ADD MORE OPTIONS HERE ...

                // ... check for more unsupported combinations? ...

                if( vm.count("dryrun") )
                {
                    cout<<asOptions( opt )<<endl;
                    exit(0);
                }
#endif
            }
            catch(po::error& e)
            {
                cerr<<"Invalid argument: "<<e.what()<<endl;
                throw;
            }
            catch(...)
            {
                cerr<<"Command-line parsing exception of unknown type!"<<endl;
                throw;
            }
            return;
        }
    }//opt::


    //namespace po=boost::program_options;
    void help( std::ostream& os ){
        boost::program_options::options_description desc("mcgen Options");
        opt::init(desc);
        opt::helpUsage(os);
        // opt::help(os,desc);
        os<<desc<<std::endl;
        opt::helpExamples(os);
    }

}//mcgen

// -------- main program --------

using namespace std;
using namespace mcgen;
int main(int argc, char** argv)
{
    opt::Parms p;
    cout<<" default canonical args: "<<p.str()
        <<" nClass="<<p.nClass<<" nxStd="<<p.nxStd<<" nx="<<p.nx<<endl;
    opt::argsParse( argc, argv, p );
    string canonicalArgs = p.str();
    cout<<"         canonical args: "<<canonicalArgs
        <<" nClass="<<p.nClass<<" nxStd="<<p.nxStd<<" nx="<<p.nx<<endl;
    string trainFile;   // filename for training data file
    string axesFile;    // filename for 'ideal' projection axes
    {
        ostringstream os;
        os<<"mcgen-"<<canonicalArgs<<".train.txt";
        trainFile = os.str();
    }
    {
        ostringstream os;
        os<<"mcgen-"<<canonicalArgs<<".axes.txt";
        axesFile = os.str();
    }
    p.init();
    // 1. generate the trivial axis solution
    // 2. generate trivial solutions: 1 central example per class
    // 3. generate trivial solutions: according the p.margin
    // 4. generate random examples (retry if any fail +ve p.margin)
    // 5. generate labels (use non-native label if in -ve margin region)
    // 6. generate any skew/rot transformation data
    // 7. apply transforms to each of p.axes projection axes, write trainFile
    // 8. apply transforms to each training examples, write axesFile
    // [9. generate a usable MCsoln file with the 'ideal' solution]
    cout<<"\nGoodbye"<<endl;
}

