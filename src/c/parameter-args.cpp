
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
	  ("C1", value<double>()->required(), "~ label in correct (l,u)")
	  ("C2", value<double>()->required(), "~ label outside other (l,u)")
	  ("nfilters", value<uint32_t>()->default_value(p.no_projections) , "# of label filters")
	  ("maxiter", value<uint32_t>(), "max iterations per projection. [1e8/batch_size]")
	  ("eta0", value<double>()->default_value(p.eta), "initial learning rate")
	  ("seed", value<uint32_t>()->default_value(p.seed), "random number seed. 0 - initizize rn using current time, 1 - do not initialize rng")
	  ("threads", value<uint32_t>()->default_value(p.num_threads), "# threads, 0 ~ use OMP_NUM_THREADS")
	  ("resume", value<bool>()->implicit_value(true)->default_value(p.resume), "add more filters to a previous solution")
	  ("reoptlu", value<bool>()->implicit_value(true)->default_value(p.reoptimize_LU), "reoptimize the intevals [l,u] for previously learned filter directions.")
	  ("sample", value<uint32_t>()->default_value(p.class_samples)
	   , "how many negative classes to use for gradient estimate.\nTry 100 or 1000 to speed up. 0 ~ all")
	  ;
        po::options_description dev("Development options");
        dev.add_options()
	  ("update,u", value<std::string>()->default_value(tostring(p.update_type)), "PROJECTED | MINIBATCH : gradient update type\nPROJECTED implies default batchsize of 1")
	  ("batchsize,b", value<uint32_t>(), "batch size. 0~full gradient, [1 1000 if BATCH]")
	  ("etatype", value<std::string>()
	   , "CONST | SQRT | LIN | 3_4 : learning rate schedule [3_4 if using averaged gradiend, LIN if not]")
	  ("etamin", value<double>()->default_value(p.min_eta), "minimum learning rate limit")
	  ("toptlu", value<uint32_t>(), "expensive exact {l,u} optmization period [once, at maxiter]")
	  ("treorder", value<uint32_t>()->default_value(p.reorder_epoch), "reorder iteration period")
	  ("reorderby", value<std::string>()->default_value(tostring(p.reorder_type))
	   , "Permutation re-ordering: PROJ mean of projected instances | MID range midpoints.")
	  ("treport", value<uint32_t>(), "report the objective value every treport iteratoins. 0 for no reporting. [maxiter/10]")
	  ("avgstart", value<uint32_t>(), "averaging start iteration [max(nExamples,dim)]")
	  //	  ("tavg", value<uint32_t>()->default_value(p.report_avg_epoch), "period for reports about avg, expensive")
	  ("reweight", value<std::string>()->default_value(tostring(p.reweight_lambda))
	   , "NONE | LAMBDA | ALL lambda reweighting method")
	  ("remove_constraints", value<bool>()->implicit_value(true)->default_value(p.remove_constraints)
	   , "after each projection, remove constraints involving labels already eliminated for the example")
	  //	  ("remove_class", value<bool>()->implicit_value(true)->default_value(p.remove_class_constraints)
	  //	   , "after each projection, remove already-separated classes(?)")
	  //	  ("wt_by_nclasses", value<bool>()->implicit_value(true)->default_value(p.ml_wt_by_nclasses), "UNTESTED")
	  //	  ("wt_class_by_nclasses", value<bool>()->implicit_value(true)->default_value(p.ml_wt_class_by_nclasses), "UNTESTED")
            ;
        // Add generic, main and development option groups	  
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
      parms.no_projections            =vm["nfilters"].as<uint32_t>();
      parms.C1	                =vm["C1"].as<double>();  //required
      parms.C2	                =vm["C2"].as<double>();  //required
      if (vm.count("maxiter"))
	parms.max_iter	                =vm["maxiter"].as<uint32_t>();  
      parms.eta	                =vm["eta0"].as<double>();       //0.1;
      parms.seed 	                =vm["seed"].as<uint32_t>(); // 0;
      parms.num_threads 	        =vm["threads"].as<uint32_t>(); // 0;          // use OMP_NUM_THREADS
      parms.resume 	                =vm["resume"].as<bool>(); // false;
      parms.reoptimize_LU 	        =vm["reoptlu"].as<bool>(); // false;
      parms.class_samples 	        =vm["sample"].as<uint32_t>(); // 0;
      // Development options
      fromstring( vm["update"].as<string>(), parms.update_type );
      if (vm.count("batchsize"))
	{
	  parms.default_batch_size = false;
	  parms.batch_size	        =vm["batchsize"].as<uint32_t>(); //100;
	}
      if (vm.count("etatype"))
	fromstring( vm["etatype"].as<string>(), parms.eta_type );
      
      parms.min_eta	                =vm["etamin"].as<double>(); // 0;
      if (vm.count("toptlu"))
	{
	  parms.optimizeLU_epoch	        =vm["toptlu"].as<uint32_t>(); // expensive
	  parms.default_optimizeLU_epoch = false;
	}
      
      parms.reorder_epoch	        =vm["treorder"].as<uint32_t>(); //1000;
      fromstring( vm["reorderby"].as<string>(), parms.reorder_type );
      if (vm.count("treport"))
	{	  
	  parms.report_epoch	        =vm["treport"].as<uint32_t>(); //;
	  parms.default_report_epoch = false;
	}
      if (vm.count("avgstart"))
	{
	  parms.avg_epoch	        =vm["avgstart"].as<uint32_t>();
	  parms.default_avg_epoch = false;
	}      
      fromstring( vm["reweight"].as<string>(), parms.reweight_lambda );
      //parms.ml_wt_by_nclasses 	=vm["wt_by_nclasses"].as<bool>(); // false;
      //parms.ml_wt_class_by_nclasses 	=vm["wt_class_by_nclasses"].as<bool>(); // false;

      parms.remove_constraints 	=vm["remove_constraints"].as<bool>(); // false;
      //parms.remove_class_constraints 	=vm["remove_class"].as<bool>(); // false;
      
      parms.verbose = vm["verbose"].as<int>();

        // Compile-time options
#if GRADIENT_TEST
        parms.finite_diff_test_epoch	=vm["tgrad"].as<uint32_t>(); //0;
        parms.no_finite_diff_tests	=vm["ngrad"].as<uint32_t>(); //1000;
        parms.finite_diff_test_delta	=vm["grad"].as<double>(); //1e-4;
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

            po::parsed_options parsed
                = po::command_line_parser( argc, argv )
                .options( desc )
                //.positional( po::positional_options_description() ) // empty, none allowed.
                .allow_unregistered()
                .run();
            po::store( parsed, vm );

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
            <<"\n - Without solnfile, use random initial conditions."
            <<"\n - xfile is a plain eigen DenseM or SparseM (always stored as float)"
            <<"\n - yfile is an Eigen SparseMb matrix of bool storing only the true values,"
            <<"\n         read/written via 'eigen_io_binbool'"
            <<endl;
    }
    void MCsolveArgs::init( po::options_description & desc ){
        desc.add_options()
	  ("xfile,x", value<string>(&xFile)->required(), "x data (row-wise nExamples x dim)")
	  ("yfile,y", value<string>(&yFile)->default_value(string("")), "y data (if absent, try reading as libsvm format)")
	  ("solnfile,s", value<string>(&solnFile)->default_value(string("")), "solnfile starting solver state")
	  ("output,o", value<string>(&outFile)->default_value(string("mc")), "output file base name")
	  (",B", value<bool>(&outBinary)->implicit_value(true)->default_value(false),"output solution in  BINARY (default: TEXT")
	  ("xnorm", value<bool>(&xnorm)->implicit_value(true)->default_value(false), "col-normalize x dimensions (mean=stdev=1)\n(forces Dense x)")
	  ("xunit", value<bool>(&xunit)->implicit_value(true)->default_value(false), "row-normalize x examples")
	  ("xscale", value<double>(&xscale)->default_value(1.0), "scale each x example.  xnorm, xunit, xscal applied in order, during read.")
	  // xquad ?
	  ("help,h", "")
	  
	  //("threads,t", value<uint32_t>()->default_value(1U), "TBD: threads")
	  ("verbose,v", value<int>(&verbose)->implicit_value(1)->default_value(1), "--verbosity=-1 may reduce output")
	  ;
    }
    MCsolveArgs::MCsolveArgs()
        : parms(set_default_params())   // parameter.h parses these options
          // MCsolveArgs::parse output...
          , xFile()
          , yFile()
          , solnFile()
          , outFile()
          , outBinary(false)
          , xnorm(false)
          , xunit(false)
          , xscale(1.0)
          //, threads(0U)           // unused?
          , verbose(1)            // cmdline value can be -ve to reduce output
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
    try {
      po::options_description descAll("Allowed options");
      po::options_description descMcsolve("mcsolve options");
      init( descMcsolve );                        // create a description of the options
      po::options_description descParms("solver args");
      opt::mcParameterDesc( descParms, parms );   // add the param_struct options
      descAll.add(descMcsolve).add(descParms);
      
      po::variables_map vm;
      {
	po::parsed_options parsed
	  = po::command_line_parser( argc, argv )
	  .options( descAll )
	  .run();
	po::store( parsed, vm );
      }
      
      if( vm.count("help") ) {
	helpUsage( cout );
	cout<<descAll<<endl;
	//helpExamples(cout);
	exit(0);
      }
      
      po::notify(vm); // at this point, raise any exceptions for 'required' args
      
      assert( vm.count("xfile") );
      
      opt::extract(vm,parms);         // retrieve McSolver parameters
      finalize_default_params(parms);      
    }
    catch(po::error& e)
        {
            cerr<<"Invalid argument: "<<e.what()<<endl;
            throw;
        }
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
	("solnfile,s", value<string>(&solnFile)->default_value(string("")), "file with the saved label filter")
	("xfile,x", value<string>(&xFile)->default_value(string("")), "x data (row-wise nExamples x dim)")
	("output,o", value<string>(&outFile)->default_value(string("")), "file to output the feasible classes after filter is applied.")
	("proj,p", value<uint32_t>()->implicit_value(0U)->default_value(0U), "use up to --proj projections [0=all]")
	("outBinary,B", value<bool>(&outBinary)->implicit_value(true)->default_value(false), "output feasible classed in  BINARY format")
	("outDense,D", value<bool>(&outDense)->implicit_value(true)->default_value(false),"output feasible classes in DENSE format (matrix or 0|1)")
	("xnorm", value<bool>(&xnorm)->implicit_value(true)->default_value(false), "Uggh. col-normalize x dimensions (mean=stdev=1)")
	("xunit", value<bool>(&xunit)->implicit_value(true)->default_value(false), "row-normalize x examples")
	("xscale", value<double>(&xscale)->default_value(1.0), "scale each x example.  xnorm, xunit, xscal applied in order, during read.")
	// xquad ?
	("help,h", "this help")
	//("threads,t", value<uint32_t>()->default_value(1U), "TBD: threads")
	("verbose,v", value<int>(&verbose)->implicit_value(1)->default_value(0), "--verbosity=-1 may reduce output.")
	;
    }
    MCprojArgs::MCprojArgs()
        : // MCprojArgs::parse output...
            xFile()
            , solnFile()
            , outFile()
            , maxProj(0U)
            , outBinary(false)
            , outDense(false)
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
        try {
            po::options_description descMcproj("Allowed projections options");
            init( descMcproj );                        // create a description of the options


            po::variables_map vm;
            {
	      po::parsed_options parsed
		= po::command_line_parser( argc, argv )
		.options( descMcproj )
		.run();
	      po::store( parsed, vm );
            }
	    
            if( vm.count("help") ) {
	      helpUsage( cout );
	      cout<<descMcproj<<endl;
	      //helpExamples(cout);
	      exit(0);
            }

            po::notify(vm); // at this point, raise any exceptions for 'required' args
	    
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
