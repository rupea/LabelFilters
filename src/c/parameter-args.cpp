
#include "parameter-args.h"
#include <iostream>
#include <cstdint>

#define ARGSDEBUG 1

using namespace std;
using namespace boost::program_options;

namespace opt {

    void helpUsageDummy( std::ostream& os ){
        os<<" Usage: foo [options]";
        return;
    }

#if 0
    void mcParameterDesc( po::options_description & desc )
    {
        desc.add_options()
            ("help,h", "")
            //("verbose,v", "")
            ("proj", value<uint32_t>()->default_value(5U) , "# of projections")
            ("C1", value<double>()->default_value(10.0), "~ label in correct [l,u]")
            ("C2", value<double>()->default_value(1.0), "~ label outside other [l,u]")
            ("maxiter", value<uint32_t>()->default_value(1000000U), "max iterations per projection")
            ("batchsize,b", value<uint32_t>()->default_value(100), "batch size")
            ("update,u", value<std::string>()->default_value("BATCH"), "[*]BATCH | SAFE : gradient update type")
            ("eps", value<double>()->default_value(1.e-4), "unused cvgce threshold")
            ("eta", value<double>()->default_value(0.1), "initial learning rate")
            ("etatype", value<std::string>()->default_value("LINEAR")
             , "CONST | SQRT | LIN | 3_4 : learning rate schedule")
            ("etamin", value<double>()->default_value(0.0), "learning rate limit")
            ("optlu", value<uint32_t>()->default_value(10000), "expensive exact {l,u} soln period")
            ("treorder", value<uint32_t>()->default_value(1000), "reorder iteration period")
            ("reorder", value<std::string>()->default_value("AVG")
             , "Permutation re-ordering: AVG projected means | PROJ projected means | MID range midpoints."
             " If --tavg=0, default is PROJ")
            ("report", value<uint32_t>()->default_value(1000U), "period for reports about latest iter")
            ("tavg", value<uint32_t>()->default_value(0U), "averaging start iteration")
            ("reportavg", value<uint32_t>()->default_value(0U), "period for reports about avg, expensive")
            ("reweight", value<std::string>()->default_value("LAMBDA")
             , "NONE | LAMBDA | ALL lambda reweighting method")
            ("wt_by_nclasses", value<bool>()->implicit_value(true)->default_value(false), "?")
            ("wt_class_by_nclasses", value<bool>()->implicit_value(true)->default_value(false), "?")
            ("negclass", value<uint32_t>()->default_value(0U)
             , "# of negative classes used at each iter, 0 ~ all classes")
            ("remove_constraints", value<bool>()->implicit_value(true)->default_value(false)
             , "after each projection, remove constraints")
            ("remove_class", value<bool>()->implicit_value(true)->default_value(false)
             , "after each projection, remove already-separated classes")
            ("threads", value<uint32_t>()->default_value(0U), "# threads, 0 ~ use OMP_NUM_THREADS")
            ("seed", value<uint32_t>()->default_value(0U), "random number seed")
            ("tgrad", value<uint32_t>()->default_value(0U), "iter period for finite difference gradient test")
            ("ngrad", value<uint32_t>()->default_value(1000U), "directions per gradient test")
            ("grad", value<double>()->default_value(1.e-4), "step size per gradient test")
            ("resume", value<bool>()->implicit_value(true)->default_value(false), "resume an existing soln?")
            ("reoptlu", value<bool>()->implicit_value(true)->default_value(false), "reoptimize {l,u} bounds of existing soln?")
            ;
    }
#endif

    /** When re-using an old run, initial \c p parms come from the MCsoln (.soln or .mc) file.
     * In this case, command-line args override the MCsoln values, and not the standard default
     * values.  Probably this should be the \b only \c mcParameterDesc function, with \c p
     * using the "official" set_default_params() in case a --solnfile is not supplied !!!
     */
    void mcParameterDesc( po::options_description & desc, param_struct const& p )
    {
        desc.add_options()
            ("help,h", "")
            //("verbose,v", "")
            ("proj", value<uint32_t>()->default_value(p.no_projections) , "# of projections")
            ("C1", value<double>()->default_value(p.C1), "~ label in correct [l,u]")
            ("C2", value<double>()->default_value(p.C2), "~ label outside other [l,u]")
            ("maxiter", value<uint32_t>()->default_value(p.max_iter), "max iterations per projection")
            ("batchsize,b", value<uint32_t>()->default_value(p.batch_size), "batch size")
            ("update,u", value<std::string>()->default_value(tostring(p.update_type)), "BATCH | SAFE : gradient update type")
            ("eps", value<double>()->default_value(p.eps), "unused cvgce threshold")
            ("eta", value<double>()->default_value(p.eta), "initial learning rate")
            ("etatype", value<std::string>()->default_value(tostring(p.eta_type))
             , "CONST | SQRT | LIN | 3_4 : learning rate schedule")
            ("etamin", value<double>()->default_value(p.min_eta), "learning rate limit")
            ("optlu", value<uint32_t>()->default_value(p.optimizeLU_epoch), "expensive exact {l,u} soln period")
            ("treorder", value<uint32_t>()->default_value(p.reorder_epoch), "reorder iteration period")
            ("reorder", value<std::string>()->default_value(tostring(p.reorder_type))
             , "Permutation re-ordering: AVG projected means | PROJ projected means | MID range midpoints."
             " If --tavg=0, default is PROJ")
            ("report", value<uint32_t>()->default_value(p.report_epoch), "period for reports about latest iter")
            ("tavg", value<uint32_t>()->default_value(p.avg_epoch), "averaging start iteration")
            ("reportavg", value<uint32_t>()->default_value(p.report_avg_epoch), "period for reports about avg, expensive")
            ("reweight", value<std::string>()->default_value(tostring(p.reweight_lambda))
             , "NONE | LAMBDA | ALL lambda reweighting method")
            ("wt_by_nclasses", value<bool>()->implicit_value(true)->default_value(p.ml_wt_by_nclasses), "?")
            ("wt_class_by_nclasses", value<bool>()->implicit_value(true)->default_value(p.ml_wt_class_by_nclasses), "?")
            ("negclass", value<uint32_t>()->default_value(p.class_samples)
             , "# of negative classes used at each iter, 0 ~ all classes")
            ("remove_constraints", value<bool>()->implicit_value(true)->default_value(p.remove_constraints)
             , "after each projection, remove constraints")
            ("remove_class", value<bool>()->implicit_value(true)->default_value(p.remove_class_constraints)
             , "after each projection, remove already-separated classes(?)")
            ("threads", value<uint32_t>()->default_value(p.num_threads), "# threads, 0 ~ use OMP_NUM_THREADS")
            ("seed", value<uint32_t>()->default_value(p.seed), "random number seed")
            ("tgrad", value<uint32_t>()->default_value(p.finite_diff_test_epoch), "iter period for finite difference gradient test")
            ("ngrad", value<uint32_t>()->default_value(p.no_finite_diff_tests), "directions per gradient test")
            ("grad", value<double>()->default_value(p.finite_diff_test_delta), "step size per gradient test")
            ("resume", value<bool>()->implicit_value(true)->default_value(p.resume), "resume an existing soln?")
            ("reoptlu", value<bool>()->implicit_value(true)->default_value(p.reoptimize_LU), "reoptimize {l,u} bounds of existing soln?")
            ;
    }


    void extract( po::variables_map const& vm, param_struct & parms ){
        //if( vm.count("axes") ) { parms.axes = vm["axes"].as<uint32_t>(); }
        //if( vm.count("proj") )
        parms.no_projections        = vm["proj"].as<uint32_t>();
        parms.C1	                =vm["C1"].as<double>();   //10.0;
        parms.C2	                =vm["C2"].as<double>();   //1.0;
        parms.max_iter	        =vm["maxiter"].as<uint32_t>(); //1e6;
        parms.batch_size	        =vm["batchsize"].as<uint32_t>(); //100;
        parms.update_type 	        = MINIBATCH_SGD;        //MINIBATCH_SGD;
        if(vm.count("update")) fromstring( vm["update"].as<string>(), parms.update_type );
        parms.eps	                =vm["eps"].as<double>(); //1e-4;
        parms.eta	                =vm["eta"].as<double>(); //0.1;
        parms.eta_type 	        = ETA_LIN;
        if(vm.count("etatype")) fromstring( vm["etatype"].as<string>(), parms.eta_type );
        parms.min_eta	        =vm["etamin"].as<double>(); // 0;
        parms.optimizeLU_epoch	=vm["optlu"].as<uint32_t>(); //10000; // expensive
        parms.reorder_epoch	        =vm["treorder"].as<uint32_t>(); //1000;
        parms.reorder_type 	= REORDER_AVG_PROJ_MEANS; // defaults to REORDER_PROJ_MEANS if averaging is off
        if(vm.count("order")) fromstring( vm["reorder"].as<string>(), parms.reorder_type );
        parms.report_epoch	        =vm["report"].as<uint32_t>(); //1000;
        parms.report_avg_epoch	=vm["reportavg"].as<uint32_t>(); //0; // this is expensive so the default is 0
        parms.avg_epoch	        =vm["tavg"].as<uint32_t>(); //0;
        if(parms.avg_epoch == 0U && parms.reorder_type == REORDER_AVG_PROJ_MEANS )
            parms.reorder_type = REORDER_PROJ_MEANS;
        parms.reweight_lambda 	= REWEIGHT_LAMBDA;
        if( vm.count("reweight") ) fromstring( vm["reweight"].as<string>(), parms.reweight_lambda );
        parms.class_samples 	=vm["negclass"].as<uint32_t>(); // 0;
        parms.ml_wt_by_nclasses 	=vm["wt_by_nclasses"].as<bool>(); // false;
        parms.ml_wt_class_by_nclasses 	=vm["wt_class_by_nclasses"].as<bool>(); // false;
        parms.remove_constraints 	=vm["remove_constraints"].as<bool>(); // false;
        parms.remove_class_constraints 	=vm["remove_class"].as<bool>(); // false;
        parms.num_threads 	                =vm["threads"].as<uint32_t>(); // 0;          // use OMP_NUM_THREADS
        parms.seed 	                        =vm["seed"].as<uint32_t>(); // 0;
        parms.finite_diff_test_epoch	=vm["tgrad"].as<uint32_t>(); //0;
        parms.no_finite_diff_tests	        =vm["ngrad"].as<uint32_t>(); //1000;
        parms.finite_diff_test_delta	=vm["grad"].as<double>(); //1e-4;
        parms.resume 	                =vm["resume"].as<bool>(); // false;
        parms.reoptimize_LU 	        =vm["reoptlu"].as<bool>(); // false;
    }

    std::vector<std::string> mcArgs( int argc, char**argv, param_struct & parms
                                     , void(*usageFunc)(std::ostream&)/*=helpUsageDummy*/ )
    {
#if ARGSDEBUG > 0
        cout<<" argsParse( argc="<<argc<<", argv, ... )"<<endl;
        for( int i=0; i<argc; ++i ) {
            cout<<"    argv["<<i<<"] = "<<argv[i]<<endl;
        }
#endif
        vector<string> ret;
        po::options_description desc("Options");
        mcParameterDesc( desc, parms );                 // <-- parms MUST be fully initialized by caller !

        po::variables_map vm;
        try {

#if 0
            //po::store( po::parse_command_line(argc,argv,desc), vm );
            po::store( po::command_line_parser(argc,argv)
                       .options(desc)
                       .run(),
                       vm );
#else
            // Need more control, I think...
            po::parsed_options parsed
                = po::command_line_parser( argc, argv )
                .options( desc )
                //.positional( po::positional_options_description() ) // empty, none allowed.
                .allow_unregistered()
                .run();
            po::store( parsed, vm );
#endif

            if( vm.count("help") ) {
                (*usageFunc)( cout );       // some customizable Usage intro
                cout<<desc<<endl;           // param_struct options
                //helpExamples(cout);
                //return ret;
                exit(0);
            }

            po::notify(vm); // at this point, raise any exceptions for 'required' args

            // In your custom program, you would interrogate vm HERE to get additional
            // variables particular to your program.

            extract( vm, parms );

            if( vm.count("help") ) {
                exit(0);
            }

            ret = collect_unrecognized( parsed.options, include_positional );
            return ret;
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
        return ret;
    }

    //
    // ----------------------- MCsolveArgs --------------------
    //

    void MCsolveArgs::helpUsage( std::ostream& os ){
        os  <<" Function:"
            <<"\n    Determine a number (--proj) of projection lines. Any example whose projection"
            <<"\n    lies outside bounds for that class is 'filtered'.  With many projecting lines,"
            <<"\n    potentially many classes can be filtered out, leaving few potential classes"
            <<"\n    for each example."
            <<"\n Usage:"
            <<"\n    mcsolve --xfile=... --yfile=... [--solnfile=...] [--output=...] [solver args...]"
            <<"\n where solver args guide the optimization procedure"
            <<"\n Without solnfile[.soln], use random initial conditions."
            <<"\n xfile is a plain eigen DenseM or SparseM (always stored as float)"
            <<"\n yfile is an Eigen SparseMb matrix of bool storing only the true values,"
            <<"\n       read/written via 'eigen_io_binbool'"
            <<endl;
    }
    void MCsolveArgs::init( po::options_description & desc ){
        desc.add_options()
            ("xfile,x", value<string>()->required(), "x data (row-wise nExamples x dim)")
            ("yfile,y", value<string>()->required(), "y data (slc/mlc SparseMb)")
            ("solnfile,s", value<string>()->default_value(string("")), "solnfile[.soln] starting solver state")
            ("output,o", value<string>()->default_value(string("mc")), "output[.soln] file base name")
            (",B", value<bool>(&outBinary)->implicit_value(true)->default_value(true),"B|T output BINARY")
            (",T", value<bool>(&outText)->implicit_value(true)->default_value(false),"B|T output TEXT")
            (",S", value<bool>(&outShort)->implicit_value(true)->default_value(true),"S|L output SHORT")
            (",L", value<bool>(&outLong)->implicit_value(true)->default_value(false),"S|L output LONG")
            ("threads,t", value<uint32_t>()->default_value(1U), "threads")
            ("xnorm", value<bool>()->implicit_value(true)->default_value(false), "col-normalize x dimensions (mean=stdev=1)")
            ;
    }
    MCsolveArgs::MCsolveArgs()
        : parms(set_default_params())
    // argsParse output...
    , xFile()
        , yFile()
        , solnFile()
        , outFile()
        , outBinary(true)
        , outText(false)
        , outShort(true)
        , outLong(false)
        , xnorm(false)
#if 0 // moved to standalone.h class MCsolveProgram
        , pmcs(nullptr)     // used during argsParse AND trySolve,trySave,tryDisplay
        // tryRead output...
        , xDense()
        , denseOk(false)
        , xSparse()
        , sparseOk(false)
        , y()               // SparseMb
#endif
        {}

    void MCsolveArgs::argsParse( int argc, char**argv ){
#if ARGSDEBUG > 0
        cout<<" argsParse( argc="<<argc<<", argv, ... )"<<endl;
        for( int i=0; i<argc; ++i ) {
            cout<<"    argv["<<i<<"] = "<<argv[i]<<endl;
        }
#endif
        bool keepgoing = true;
        try {
            po::options_description descAll("Allowed options");
            po::options_description descMcsolve("mcsolve options");
            init( descMcsolve );                        // create a description of the options
            po::options_description descParms("solver args");
            opt::mcParameterDesc( descParms, parms );   // add the param_struct options
            descAll.add(descMcsolve).add(descParms);

            po::variables_map vm;
            //po::store( po::parse_command_line(argc,argv,desc), vm );
            // Need more control, I think...
            {
                po::parsed_options parsed
                    = po::command_line_parser( argc, argv )
                    .options( descAll )
                    //.positional( po::positional_options_description() ) // empty, none allowed.
                    //.allow_unregistered()
                    .run();
                po::store( parsed, vm );
            }

            if( vm.count("help") ) {
                helpUsage( cout );
                cout<<descAll<<endl;
                //helpExamples(cout);
                keepgoing=false;
            }

            po::notify(vm); // at this point, raise any exceptions for 'required' args

            cerr<<"msolve args..."<<endl;
            assert( vm.count("xfile") );
            assert( vm.count("yfile") );
            assert( vm.count("solnfile") );
            assert( vm.count("output") );

            xFile=vm["xfile"].as<string>();
            yFile=vm["yfile"].as<string>();
            solnFile=vm["solnfile"].as<string>();
            outFile=vm["output"].as<string>();
            xnorm=vm["xnorm"].as<bool>();

            if( solnFile.rfind(".soln") != solnFile.size() - 5U ) solnFile.append(".soln");
            if( outFile .rfind(".soln") != outFile .size() - 5U ) outFile .append(".soln");

            if( outBinary == outText ) throw std::runtime_error(" Only one of B|T, please");
            if( outShort == outLong ) throw std::runtime_error(" Only one of S|L, please");

            // ISSUE: --solnfile should set INITIAL parms, and commandline
            //        should OVERRIDE (not overwrite) those.
            // Currently 'extract' set ALL parameters to "default or commandline"
            // Need another function to "modify" according to supplied parameters XXX
            cerr<<"opt::extract..."<<endl;
            opt::extract(vm,parms);         // retrieve McSolver parameters
        }
        catch(po::error& e)
        {
            cerr<<"Invalid argument: "<<e.what()<<endl;
            throw;
        }catch(std::exception const& e){
            cerr<<"Error: "<<e.what()<<endl;
            throw;
        }catch(...){
            cerr<<"Command-line parsing exception of unknown type!"<<endl;
            throw;
        }
        if( ! keepgoing ) exit(0);
#if 0 && ARGSDEBUG > 0
        // Good, boost parsing does not touch argc/argv
        cout<<" DONE argsParse( argc="<<argc<<", argv, ... )"<<endl;
        for( int i=0; i<argc; ++i ) {
            cout<<"    argv["<<i<<"] = "<<argv[i]<<endl;
        }
#endif
        return;
    }

    // static fn -- no 'this'
    std::string MCsolveArgs::defaultHelp(){
        std::ostringstream oss;
        try {
            MCsolveArgs mcsa;                                   // make sure we generate *default* help
            po::options_description descAll("Allowed options");
            po::options_description descMcsolve("mcsolve options");
            mcsa.init( descMcsolve );                           // create a description of the options
            po::options_description descParms("solver args");
            opt::mcParameterDesc( descParms, mcsa.parms );      // add the param_struct options
            descAll.add(descMcsolve).add(descParms);

            helpUsage( oss );
            oss<<descAll<<endl;
        }catch(po::error& e){
            cerr<<"Invalid argument: "<<e.what()<<endl;
            throw;
        }catch(std::exception const& e){
            cerr<<"Error: "<<e.what()<<endl;
            throw;
        }catch(...){
            cerr<<"Command-line parsing exception of unknown type!"<<endl;
            throw;
        }
        return oss.str();
    }

    //
    // ----------------------- xxx --------------------
    //

}//opt::
