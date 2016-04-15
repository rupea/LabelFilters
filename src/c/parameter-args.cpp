
#include "parameter-args.h"
#include <iostream>
#include <cstdint>
#include <sstream>

#include <boost/program_options/parsers.hpp>    // split_unix

#define ARGSDEBUG 1

using namespace std;
using namespace boost::program_options;

namespace opt {

    std::vector<std::string> cmdSplit( std::string cmdline, bool haveProgName/*=true*/ )
    {
        if( ! haveProgName ){ // insert a dummy program name, which gets ignored
            cmdline.insert(0,"foo ");
        }
        // defaults sep,quote,escape OK: " \t", "'\"", "\\"
        return boost::program_options::split_unix( cmdline );
    }

    void helpUsageDummy( std::ostream& os ){
        os<<" Usage: foo [options]";
        return;
    }

    /** When re-using an old run, initial \c p parms come from the MCsoln (.soln or .mc) file.
     * In this case, command-line args override the MCsoln values, and not the standard default
     * values.  Probably this should be the \b only \c mcParameterDesc function, with \c p
     * using the "official" set_default_params() in case a --solnfile is not supplied !!!
     *
     * When \em everything has a [client-specified] default, it simplifies
     * using a variables_map, because you don't need to first check vm.count("foo").
     */
    void mcParameterDesc( po::options_description & desc, param_struct const& p )
    {
        po::options_description main("Main options");
        main.add_options()
            ("proj", value<uint32_t>()->default_value(p.no_projections) , "# of projections")
            ("C1", value<double>()->default_value(p.C1), "~ label in correct [l,u]")
            ("C2", value<double>()->default_value(p.C2), "~ label outside other [l,u]")
            ("maxiter", value<uint32_t>()->default_value(p.max_iter), "max iterations per projection")
            ("eta0", value<double>()->default_value(p.eta), "initial learning rate")
            ("seed", value<uint32_t>()->default_value(p.seed), "random number seed")
            ("threads", value<uint32_t>()->default_value(p.num_threads), "# threads, 0 ~ use OMP_NUM_THREADS")
            ("resume", value<bool>()->implicit_value(true)->default_value(p.resume), "resume existing soln?")
            ("reoptlu", value<bool>()->implicit_value(true)->default_value(p.reoptimize_LU), "reoptimize existing [l,u]?")
            ("sample", value<uint32_t>()->default_value(p.class_samples)
             , "Cap -ve classes for chunked gradient estimate.\nTry 100 or 1000 to speed up. 0 ~ all")
            ;
        po::options_description dev("Development options");
        dev.add_options()
            ("update,u", value<std::string>()->default_value(tostring(p.update_type)), "BATCH | SAFE : gradient update type\nBATCH implies default batchsize of max(1000,nExamples)")
            ("batchsize,b", value<uint32_t>()->default_value(p.batch_size), "batch size: !=1 => BATCH, 0~full gradient, [1000 if BATCH]")
            ("eps", value<double>()->default_value(p.eps), "unused cvgce threshold")
            //("eta", value<double>()->default_value(p.eta), "initial learning rate")
            // ... ohoh: eta cannot [fully] be a prefix of any other option
            ("etatype", value<std::string>()->default_value(tostring(p.eta_type))
             , "CONST | SQRT | LIN | 3_4 : learning rate schedule")
            ("etamin", value<double>()->default_value(p.min_eta), "learning rate limit")
            ("optlu", value<uint32_t>(), "expensive exact {l,u} soln period [0=none, def=once, at maxiter")
            ("treorder", value<uint32_t>()->default_value(p.reorder_epoch), "reorder iteration period")
            ("reorder", value<std::string>()->default_value(tostring(p.reorder_type))
             , "Permutation re-ordering: AVG projected means | PROJ projected means | MID range midpoints."
             " If --avg=0, default is PROJ")
            ("treport", value<uint32_t>()->default_value(p.report_epoch), "period for reports about latest iter")
            ("avg", value<uint32_t>(), "averaging start iteration [def. nExamples]")
            ("tavg", value<uint32_t>()->default_value(p.report_avg_epoch), "period for reports about avg, expensive")
            ("reweight", value<std::string>()->default_value(tostring(p.reweight_lambda))
             , "NONE | LAMBDA | ALL lambda reweighting method")
            ("remove_constraints", value<bool>()->implicit_value(true)->default_value(p.remove_constraints)
             , "after each projection, remove constraints")
            ("remove_class", value<bool>()->implicit_value(true)->default_value(p.remove_class_constraints)
             , "after each projection, remove already-separated classes(?)")
            ("wt_by_nclasses", value<bool>()->implicit_value(true)->default_value(p.ml_wt_by_nclasses), "UNTESTED")
            ("wt_class_by_nclasses", value<bool>()->implicit_value(true)->default_value(p.ml_wt_class_by_nclasses), "UNTESTED")
            ;
        // Add generic, main and development option groups
        desc.add_options()
            ("help,h", "")
            //("verbose,v", "")
            ;
        desc.add(main).add(dev);

#if GRADIENT_TEST /*|| others?*/
        po::options_description compiled("Compile-time options (enabled by compile flags)");
#if GRADIENT_TEST
        compiled_add_options()
            ("tgrad", value<uint32_t>()->default_value(p.finite_diff_test_epoch), "iter period for finite difference gradient test")
            ("ngrad", value<uint32_t>()->default_value(p.no_finite_diff_tests), "directions per gradient test")
            ("grad", value<double>()->default_value(p.finite_diff_test_delta), "step size per gradient test")
            ;
#endif
        desc.add(compiled);
#endif
    }

    void extract( po::variables_map const& vm, param_struct & parms ){
        //if( vm.count("axes") ) { parms.axes = vm["axes"].as<uint32_t>(); }
        //if( vm.count("proj") )
        parms.no_projections            =vm["proj"].as<uint32_t>();
        //parms.C1	                =vm["C1"].as<double>();         //1.0; // LATER: *nClasses
        parms.C2	                =vm["C2"].as<double>();         //1.0;
        parms.max_iter	                =vm["maxiter"].as<uint32_t>();  //1e8;
        parms.eta	                =vm["eta0"].as<double>();       //0.1;
        parms.seed 	                =vm["seed"].as<uint32_t>(); // 0;
        parms.num_threads 	        =vm["threads"].as<uint32_t>(); // 0;          // use OMP_NUM_THREADS
        parms.resume 	                =vm["resume"].as<bool>(); // false;
        parms.reoptimize_LU 	        =vm["reoptlu"].as<bool>(); // false;
        parms.class_samples 	        =vm["sample"].as<uint32_t>(); // 0;
        // Development options
        fromstring( vm["update"].as<string>(), parms.update_type );
        parms.batch_size	        =vm["batchsize"].as<uint32_t>(); //100;
        parms.eps	                =vm["eps"].as<double>(); //1e-4;
        fromstring( vm["etatype"].as<string>(), parms.eta_type );
        parms.min_eta	                =vm["etamin"].as<double>(); // 0;
        //parms.optimizeLU_epoch	        =vm["optlu"].as<uint32_t>(); //10000; // expensive
        parms.reorder_epoch	        =vm["treorder"].as<uint32_t>(); //1000;
        fromstring( vm["reorder"].as<string>(), parms.reorder_type );
        parms.report_epoch	        =vm["treport"].as<uint32_t>(); //1000;
        parms.report_avg_epoch	        =vm["tavg"].as<uint32_t>(); //0; // this is expensive so the default is 0
        //parms.avg_epoch	                =vm["avg"].as<uint32_t>(); //0;
        fromstring( vm["reweight"].as<string>(), parms.reweight_lambda );
        parms.ml_wt_by_nclasses 	=vm["wt_by_nclasses"].as<bool>(); // false;
        parms.ml_wt_class_by_nclasses 	=vm["wt_class_by_nclasses"].as<bool>(); // false;
        parms.remove_constraints 	=vm["remove_constraints"].as<bool>(); // false;
        parms.remove_class_constraints 	=vm["remove_class"].as<bool>(); // false;
        // Compile-time options
#if GRADIENT_TEST
        parms.finite_diff_test_epoch	=vm["tgrad"].as<uint32_t>(); //0;
        parms.no_finite_diff_tests	=vm["ngrad"].as<uint32_t>(); //1000;
        parms.finite_diff_test_delta	=vm["grad"].as<double>(); //1e-4;
#endif
        // NEW - some options have side effects that we may not be able to
        //       set up just yet.   Set some flag values if need to correct things later.
        // but this ALWAYS might change if nExamples=x.rows() > 1000U
        //
        // This must be kept in sync with paramter.cpp routine param_finalize(nTrain,nClass,param_struct&)
        //
        // Note: we MIGHT skip this here, but let's set things just in case we can determine
        //       final values at this point ...
        //

        // batch size effects ... NOTE: if batch size != 1 it MIGHT change later
        if( vm.count("batchsize") && parms.batch_size != 1 && parms.update_type != MINIBATCH_SGD ){
            cout<<"--batch_size != 1 implies --update=BATCH"<<endl;
            parms.update_type = MINIBATCH_SGD;
        }
        if( parms.update_type == MINIBATCH_SGD && vm.count("batchsize") == 0 ){
            cout<<"--update_type=BATCH implies --batchsize=1000"<<endl;
            parms.batch_size = 1000U;
        }
        if( parms.update_type != MINIBATCH_SGD ){
            assert( parms.batch_size == 1U );
        }
        // batch_size == 0     ? ---> later batch_size = nTrain (nExamples)
        // batch_size > nTrain ? ---> later batch_size = nTrain
        // batch_size > 1      ? ---> max_iter /= batch_size, UNLESS explicit max_iter
        if( vm.count("maxiter") == 0U && parms.batch_size != 1 ){
            parms.max_iter = 0U;                // FLAG for max_iter/batch_size
        }

        // This side-effect is OK to do now...
        if( vm.count("etatype") == 0 && parms.avg_epoch == 0 ){
            parms.eta_type = ETA_SQRT;
        }
        // This is the logic THAT WE CAN'T EVALUATE fully correctly just yet...        
        //    ... so just set flag values for "auto", if not specified ...

        parms.optimizeLU_epoch = numeric_limits<uint32_t>::max(); // FLAG for 'max_iter"
        if( vm.count("optlu") ) {
            parms.optimizeLU_epoch       = vm["optlu"].as<uint32_t>(); // expensive
        }

        parms.C1	      = vm["C1"].as<double>();
        if( vm.count("C1") == 0 ){
            parms.C1                    = -parms.C1; // FLAG for C1 *= -nClasses, later
        }

        parms.avg_epoch = numeric_limits<uint32_t>::max(); // FLAG for nExamples (x.rows())
        if( vm.count("avg") ){
            parms.avg_epoch	        = vm["avg"].as<uint32_t>(); //0;
            if(parms.avg_epoch == 0U && parms.reorder_type == REORDER_AVG_PROJ_MEANS ){
                cout<<"--avg=0 => --reorder=AVG changed to --reorder=MEANS"<<endl;
                parms.reorder_type = REORDER_PROJ_MEANS;
            }
        }
#if 0
        if( parms.batch_size > 1U ){
            if( vm.count("max_iter") == 0U ){
                parms.max_iter /= parms.batch_size();
                if( parms.max_iter < 1U ) parms.max_iter = 1U;
            }
            if( vm.count("treport") == 0U ){
                parms.report_epoch /= parms.batch_size();
            }
        }
#endif
    }

    std::string helpMcParms(){
        ostringstream oss;
        {
            po::options_description desc("MCFilter solver options");
            mcParameterDesc( desc, set_default_params());
            oss<<desc;
        }
        return oss.str();
    }
    std::vector<std::string> mcArgs( int argc, char**argv, param_struct & parms
                                     , void(*usageFunc)(std::ostream&)/*=helpUsageDummy*/ )
    {
#if ARGSDEBUG > 0
        cout<<" mcArgs( argc="<<argc<<", argv, ... )"<<endl;
        for( int i=0; i<argc; ++i ) {
            cout<<"    argv["<<i<<"] = "<<argv[i]<<endl;
        }
#endif
        vector<string> ret;
        po::options_description desc("Options");

        // define defaults equivalent to current parms
        mcParameterDesc( desc, parms ); // <-- parms MUST be fully initialized by caller !

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

            extract( vm, parms ); // MODIFY parms to match anything from vm

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
        os  <<" Function:   Solve for Multi-Class Label Filter projection lines"
            <<"\n    Determine a number (--proj) of projection lines. Any example whose projection"
            <<"\n    lies outside bounds for that class is 'filtered'.  With many projecting lines,"
            <<"\n    potentially many classes can be filtered out, leaving few potential classes"
            <<"\n    for each example."
            <<"\n Usage:"
            <<"\n    <mcsolve> --xfile=... --yfile=... [--solnfile=...] [--output=...] [ARGS...]"
            <<"\n where ARGS guide the optimization procedure"
            <<"\n - Without solnfile[.soln], use random initial conditions."
            <<"\n - xfile is a plain eigen DenseM or SparseM (always stored as float)"
            <<"\n - yfile is an Eigen SparseMb matrix of bool storing only the true values,"
            <<"\n         read/written via 'eigen_io_binbool'"
            <<endl;
    }
    void MCsolveArgs::init( po::options_description & desc ){
        desc.add_options()
            ("xfile,x", value<string>()->required(), "x data (row-wise nExamples x dim)")
            ("yfile,y", value<string>()->default_value(string("")), "y data (if absent, try reading as libsvm format)")
            ("solnfile,s", value<string>()->default_value(string("")), "solnfile[.soln] starting solver state")
            ("output,o", value<string>()->default_value(string("mc")), "output[.soln] file base name")
            (",B", value<bool>(&outBinary)->implicit_value(true)->default_value(true),"B|T output BINARY")
            (",T", value<bool>(&outText)->implicit_value(true)->default_value(false),"B|T output TEXT")
            (",S", value<bool>(&outShort)->implicit_value(true)->default_value(true),"S|L output SHORT")
            (",L", value<bool>(&outLong)->implicit_value(true)->default_value(false),"S|L output LONG")
            ("xnorm", value<bool>()->implicit_value(true)->default_value(false), "col-normalize x dimensions (mean=stdev=1)\n(forces Dense x)")
            ("xunit", value<bool>()->implicit_value(true)->default_value(false), "row-normalize x examples")
            ("xscale", value<double>()->default_value(1.0), "scale each x example.  xnorm, xunit, xscal applied in order, during read.")
            // xquad ?

            //("threads,t", value<uint32_t>()->default_value(1U), "TBD: threads")
            ("verbose,v", value<int>(&verbose)->implicit_value(1)->default_value(0), "--verbosity=-1 may reduce output")
            ;
    }
    MCsolveArgs::MCsolveArgs()
        : parms(set_default_params())   // parameter.h parses these options
          // MCsolveArgs::parse output...
          , xFile()
          , yFile()
          , solnFile()
          , outFile()
          , outBinary(true)
          , outText(false)
          , outShort(true)
          , outLong(false)
          , xnorm(false)
          , xunit(false)
          , xscale(1.0)
          //, threads(0U)           // unused?
          , verbose(0)            // cmdline value can be -ve to reduce output
        {}

    MCsolveArgs::MCsolveArgs(int argc, char**argv)
        : MCsolveArgs()
    {
        this->parse(argc,argv);
    }

    void MCsolveArgs::parse( int argc, char**argv ){
#if ARGSDEBUG > 0
        cout<<" parse( argc="<<argc<<", argv, ... )"<<endl;
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
            //assert( vm.count("yfile") );
            assert( vm.count("solnfile") );
            assert( vm.count("output") );

            xFile=vm["xfile"].as<string>();
            yFile=vm["yfile"].as<string>();
            solnFile=vm["solnfile"].as<string>();
            outFile=vm["output"].as<string>();
            xnorm=vm["xnorm"].as<bool>();
            xunit=vm["xunit"].as<bool>();
            if( vm.count("xscale") ) xscale=vm["xscale"].as<double>();
            //threads=vm["threads"].as<uint32_t>();
            verbose=vm["verbose"].as<int>();

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
        }
        //catch(std::exception const& e){
        //    cerr<<"Error: "<<e.what()<<endl;
        //    throw;
        //}catch(...){
        //    cerr<<"Command-line parsing exception of unknown type!"<<endl;
        //    throw;
        //}
        if( ! keepgoing ) exit(0);
#if 0 && ARGSDEBUG > 0
        // Good, boost parsing does not touch argc/argv
        cout<<" DONE parse( argc="<<argc<<", argv, ... )"<<endl;
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
    // ----------------------- MCprojArgs --------------------
    //

    void MCprojArgs::helpUsage( std::ostream& os ){
        os  <<" Function:"
            <<"\n    apply .soln {w,l,u} projections to examples 'x' and print eligible class"
            <<"\n    assignments of each example (row of 'x')"
            <<"\n Usage:"
            <<"\n    <mcproj> --xfile=... --solnfile=... [other args...]"
            <<"\n- xfile is a plain eigen DenseM or SparseM (always stored as float)"
            <<"\n- soln is a .soln file, such as output by mcgen or mcsolve"
            <<"\n- <mcsoln> and <mcproj> must agree on --xnorm option. (TBD: detect & forbid)"
            <<"\n  - Perhaps this means the training --xnorm transform needs to be stored"
            <<"\n    as part of the .soln, and applied during <mcproj> on the test/validation set"
            <<"\n  - OR combine the xnorm and weights_avg transforms, adding a"
            <<"\n    translation step to the vanilla matrix multiply?"
            <<"\n  - OR just deprecate '--xnorm' and leave input 'x' xform up to the user?"
            <<endl;
    }
    void MCprojArgs::init( po::options_description & desc ){
        desc.add_options()
            ("xfile,x", value<string>()->required(), "x data (row-wise nExamples x dim)")
            ("solnfile,s", value<string>()->required(), "solnfile[.soln] starting solver state")
            ("output,o", value<string>()->default_value(string("")), "output[.proj] file base name [def. cout]. output.proj has boolean class data.")
            ("proj,p", value<uint32_t>()->implicit_value(0U)->default_value(0U), "**TBD** Use up to --proj projections for output file [0=all] **TBD")
            (",B", value<bool>(&outBinary)->implicit_value(true),"(T) T|B output BINARY")
            (",T", value<bool>(&outText)->implicit_value(true),"(T) T|B output TEXT")
            (",S", value<bool>(&outSparse)->implicit_value(true),"(S) S|D output SPARSE")
            (",D", value<bool>(&outDense)->implicit_value(true),"(S) S|D output DENSE")
            ("yfile,y", value<string>(), "[y] validation data (slc/mlc/SparseMb)")
            ("Yfile,Y", value<string>(), "**TBD** [y] validation data - per projection analysis")
            ("xnorm", value<bool>()->implicit_value(true)->default_value(false), "Uggh. col-normalize x dimensions (mean=stdev=1)")
            ("xunit", value<bool>()->implicit_value(true)->default_value(false), "row-normalize x examples")
            ("xscale", value<double>()->default_value(1.0), "scale each x example.  xnorm, xunit, xscal applied in order, during read.")
            // xquad ?
            ("help,h", value<bool>()->implicit_value(true), "this help")
            //("threads,t", value<uint32_t>()->default_value(1U), "TBD: threads")
            ("verbose,v", value<int>(&verbose)->implicit_value(1)->default_value(0), "--verbosity=-1 may reduce output. v=1: validate y per projection")
            ;
    }
    MCprojArgs::MCprojArgs()
        : // MCprojArgs::parse output...
            xFile()
            , solnFile()
            , outFile()
            , maxProj(0U)
            , outBinary(false)
            , outText(true)
            , outSparse(true)
            , outDense(false)
            , yPerProj(false)
            , yFile()
            , xnorm(false)
            , xunit(false)
            , xscale(1.0)
            , verbose(0)
        {}

    MCprojArgs::MCprojArgs(int argc, char**argv)
        : MCprojArgs()
    {
        this->parse(argc,argv);
    }
    /** helper class for alternate form of constructor.
     * This is a simple white-space tokenizer, not for
     * hard-core robustness (no ' " treatment, ...)
     */
struct MkArgcArgv : public std::vector<char*>
    {
        MkArgcArgv( std::string cmd ) {
            istringstream iss(cmd);
            std::string token;
            while(iss >> token) {
                char *arg = new char[token.size() + 1];
                copy(token.begin(), token.end(), arg);
                arg[token.size()] = '\0';
                push_back(arg);
            }
            push_back(nullptr);
        }
        ~MkArgcArgv(){
            for(size_t i = 0; i < size(); ++i){
                delete[] (*this)[i];
                // (*this)[i] = nullptr;
            }
        }
    };
    MCprojArgs::MCprojArgs(std::string args)
        : MCprojArgs()
    {
        MkArgcArgv a(args);
        this->parse( a.size()-1U, &a[0] );
    }


    void MCprojArgs::parse( int argc, char**argv ){
#if ARGSDEBUG > 0
        cout<<" parse( argc="<<argc<<", argv, ... )"<<endl;
        for( int i=0; i<argc; ++i ) {
            cout<<"    argv["<<i<<"] = "<<argv[i]<<endl;
        }
#endif
        bool keepgoing = true;
        try {
            po::options_description descMcproj("Allowed projections options");
            init( descMcproj );                        // create a description of the options


            outBinary = outText = outSparse = outDense = false;

            po::variables_map vm;
            //po::store( po::parse_command_line(argc,argv,desc), vm );
            // Need more control, I think...
            {
                po::parsed_options parsed
                    = po::command_line_parser( argc, argv )
                    .options( descMcproj )
                    //.positional( po::positional_options_description() ) // empty, none allowed.
                    //.allow_unregistered()
                    .run();
                po::store( parsed, vm );
            }

            if( vm.count("help") ) {
                helpUsage( cout );
                cout<<descMcproj<<endl;
                //helpExamples(cout);
                keepgoing=false;
            }

            po::notify(vm); // at this point, raise any exceptions for 'required' args

            cerr<<"mcproj args..."<<endl;
            assert( vm.count("xfile") );
            assert( vm.count("solnfile") );

            xFile=vm["xfile"].as<string>();
            solnFile=vm["solnfile"].as<string>();
            outFile=vm["output"].as<string>();
            maxProj=vm["proj"].as<uint32_t>();
            xnorm=vm["xnorm"].as<bool>();
            xunit=vm["xunit"].as<bool>();
            xscale=vm["xscale"].as<double>();
            verbose=vm["verbose"].as<int>();

            if( solnFile.rfind(".soln") != solnFile.size() - 5U ) solnFile.append(".soln");
            if( outFile.size() && outFile .rfind(".proj") != outFile .size() - 5U ) outFile.append(".proj");

            //{cout<<" -"; if(outBinary) cout<<"B"; if(outText)   cout<<"T"; if(outSparse) cout<<"S"; if(outDense)  cout<<"D";}
            //if(vm.count("-B")) outBinary=true;
            //if(vm.count("-T")) outText=true;
            //if(vm.count("-S")) outSparse=true;
            //if(vm.count("-D")) outDense=true;
            //{cout<<" -"; if(outBinary) cout<<"B"; if(outText)   cout<<"T"; if(outSparse) cout<<"S"; if(outDense)  cout<<"D";}
            if( !outBinary && !outText ) outText = true; // default
            if( !outSparse && !outDense ) outDense = true; // default
            //{cout<<" -"; if(outBinary) cout<<"B"; if(outText)   cout<<"T"; if(outSparse) cout<<"S"; if(outDense)  cout<<"D";}
            if( outBinary == outText ) throw std::runtime_error(" Only one of B|T, please");
            if( outSparse == outDense ) throw std::runtime_error(" Only one of S|D, please");

            yPerProj = false;
            yFile=string("");
            if( vm.count("yfile") && vm.count("Yfile") )
                throw std::runtime_error("Can only specify one of --yfile (shorter) or --Yfile (longer output)");
            if( vm.count("yfile") ){
                yFile = vm["yfile"].as<string>();
            }
            if( vm.count("Yfile") ){
                yPerProj = true;
                yFile = vm["Yfile"].as<string>();
            }
            if( solnFile.rfind(".soln") != solnFile.size() - 5U ) solnFile.append(".soln");

            // projections operation doesn't need solver parms
            //opt::extract(vm,parms);         // retrieve McSolver parameters
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
        if( ! keepgoing ) exit(0);
        return;
    }

    // static fn -- no 'this'
    std::string MCprojArgs::defaultHelp(){
        std::ostringstream oss;
        try {
            MCprojArgs proj;                    // make sure we generate *default* help
            po::options_description descMcproj("Allowed projection options");
            proj.init( descMcproj );            // create a description of the options

            helpUsage( oss );
            oss<<descMcproj<<endl;
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

}//opt::
