
#include "parameter-args.h"
#include <iostream>
#include <cstdint>
#include <sstream>
#include <boost/program_options.hpp>
#include <boost/program_options/parsers.hpp>    // split_unix

using namespace std;
namespace po = boost::program_options;

using namespace po;

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
	  ("remove_constraints", value<bool>()->implicit_value(true)->default_value(p.remove_constraints)
	   , "after each projection, remove constraints involving incorrect labels already eliminated for the example")
	  ("remove_class_constraints", value<bool>()->implicit_value(true)->default_value(p.remove_class_constraints)
	   , "after each projection, remove constraints involving correct labels already eliminated for the example")
	  ("adjustC", value<bool>()->implicit_value(true)->default_value(p.adjust_C)
	   , "adjust C1 and C2 to acount for the removed constraints")
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

      parms.remove_constraints 	=vm["remove_constraints"].as<bool>(); // true;
      parms.remove_class_constraints =vm["remove_class_constraints"].as<bool>(); // false;
      parms.adjust_C = vm["adjustC"].as<bool>(); //true

      //parms.ml_wt_by_nclasses 	=vm["wt_by_nclasses"].as<bool>(); // false;
      //parms.ml_wt_class_by_nclasses 	=vm["wt_class_by_nclasses"].as<bool>(); // false;
      
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

} //opt::


//
// ----------------------- MCprojArgs --------------------
//

po::options_description MCprojectorArgs::getDesc(){
  po::options_description projopt("Label Filter Options");
  projopt.add_options()
    ("lfFiles,f", value<std::vector<std::string>>()->multitoken()->zero_tokens()->composing(), "label filter files") 
    ("nProj", value<std::vector<int>>()->multitoken()->composing(), "number of filters to apply. 0 = no filters, -1 = all filters. -2 = {0,1,2,..., all filters}")
    ;
  return projopt;
}

MCprojectorArgs::MCprojectorArgs()
  : 
  lfFiles()	    
  , nProj({-1})
{}
  
MCprojectorArgs::MCprojectorArgs(po::variables_map const& vm)
  : MCprojectorArgs()
{
  this->extract(vm);
}
  
void MCprojectorArgs::extract(po::variables_map const& vm)
{
  if (vm.count("lfFiles"))
    {
      lfFiles = std::vector<std::string>(vm["lfFiles"].as<std::vector<std::string>>());
    }
    
  if (vm.count("nProj"))
    {
      nProj = std::vector<int>(vm["lfFiles"].as<std::vector<int>>());
    }
}
  
  
  
po::options_description MCxyDataArgs::getDesc(){
  po::options_description dataopt("Data Options");
  dataopt.add_options()
    ("xFiles,x", value<std::vector<std::string>>()->multitoken()->composing()->required(), "Data files. LibSVM, XML or binary dense/sparse formats. LibSVM and XML formats also contain the labels. Binary formats don't contain labels.") 
    ("yFiles,y", value<std::vector<std::string>>()->multitoken()->zero_tokens()->composing(), "Lables for the data. Must be one file used with all xFiles, or the same number and in same order as xFiles.")
    ("rmRareF", value<uint>()->default_value(0U), "Remove features with fewer than this many non-zero values.")
    ("rmRareL", value<uint>()->default_value(0U), "Remove labels that appear fewer than this many times.")
    ("xnorm", value<bool>()->implicit_value(true)->default_value(false), "Col-normalize the data(mean=0, stdev=1)")
    ("center", value<bool>()->implicit_value(true), "Remove the mean when col-normalizing. Default true for dense data and false for sparse data.")
    ("normdata", value<std::string>()->default_value(""), "Data to compute statistics for col-normalization and/or for rare feature and rare labels remova. If empty, statistis are calculated on the first xFiles argument")
    ("xunit", value<bool>()->implicit_value(true)->default_value(false), "Row-normalize data to unit length")
    ("xscale", value<double>()->default_value(1.0), "scale each x example. rare feature removal, xnorm, xunit, xscal applied in order.")
    ;
  return dataopt;
}
  
MCxyDataArgs::MCxyDataArgs()
  : xFiles()
  , normData()
  , rmRareF(0U)
  , rmRareL(0U)
  , xnorm(false)
  , center(-1)
  , xunit(false)
  , xscale(1.0)
{}
  
MCxyDataArgs::MCxyDataArgs(po::variables_map const& vm)
  : MCxyDataArgs()
{
  this->extract(vm);
}
  
void MCxyDataArgs::extract(po::variables_map const& vm)
{ 
  if (vm.count("xFiles"))
    {
      xFiles = std::vector<std::string>(vm["xFiles"].as<std::vector<std::string>>());
    }
  if (vm.count("yFiles"))
    {
      yFiles = std::vector<std::string>(vm["xFiles"].as<std::vector<std::string>>());
    }
    
  if (yFiles.size()!=0 && yFiles.size()!=1 && yFiles.size() != xFiles.size())
    {
      throw std::runtime_error("Different numbers of xFiles and yFiles");
    }

  if (vm.count("rmRareF"))
    {
      rmRareF = vm["rmRareF"].as<uint>();
    }

  if (vm.count("rmRareL"))
    {
      rmRareL = vm["rmRareL"].as<uint>();
    }

  if (vm.count("xnorm"))
    {
      xnorm = vm["xnorm"].as<bool>();
    }

  if (vm.count("center"))
    {
      center = vm["center"].as<bool>()?1:0;
    }

  if (vm.count("normdata"))
    {
      normData = vm["normdata"].as<std::string>();
    }    
  if (vm.count("xunit"))
    {
      xunit = vm["xunit"].as<bool>();
    }
  if (vm.count("xscale"))
    {
      xunit = vm["xscale"].as<double>();
    }
}

po::options_description MCclassifierArgs::getDesc(){
  po::options_description classopt("Classifier Options");
  classopt.add_options()
    ("modelFiles", value<std::vector<std::string>>()->multitoken()->composing()->required(), "Files with saved models. Binary dense/sparse format or text sparse format")       
    ("keep_top", value<uint32_t>()->default_value(10), "Keep at least keep_top predictions for each instance. Others are considered -inf. (i.e. can not calculate Prec@20 if keep_top < 20")
    ("keep_thresh", value<double>()->default_value(0.0), "Keep all predictions larger than keep_thresh. Others are considered -inf")
    ("threshold", value<double>()->default_value(0.0), "Threshold to use when assigning labels to an instance. Used when calculating precision/recall")
    ("min_labels", value<uint32_t>()->default_value(1), "Each example will have at least min_labels assigned (even if they are below the threshold). Used when calculating precision/recall") 
    ;
  return classopt;
}

MCclassifierArgs::MCclassifierArgs()
  : modelFiles()
  , keep_thresh(0.0)
  , keep_top(10)
  , threshold(0.0)
  , min_labels(1)
{}

MCclassifierArgs::MCclassifierArgs(po::variables_map const& vm)
  : MCclassifierArgs()
{
  this->extract(vm);
}

void MCclassifierArgs::extract(po::variables_map const& vm)
{ 
  if (vm.count("modelFiles"))
    {
      modelFiles = std::vector<std::string>(vm["modelFiles"].as<std::vector<std::string>>());
    }
    
  if (vm.count("keep_top"))
    {
      keep_top = vm["keep_top"].as<uint32_t>();
    }
  if (vm.count("min_labels"))
    {
      min_labels = vm["min_labels"].as<uint32_t>();
    }
    
  if (min_labels > keep_top)
    {
      throw std::runtime_error("Asked to predict at least " + std::to_string(min_labels) + " per example, but keep_top is set to " + std::to_string(keep_top));
    }
    
  if (vm.count("keep_thresh"))
    {
      keep_thresh = vm["keep_thresh"].as<double>();
    }
  if (vm.count("threshold"))
    {
      threshold = vm["threshold"].as<double>();
    }
  if (threshold < keep_thresh)
    {
      throw std::runtime_error("Asked to use threshold " + std::to_string(threshold) + " but keep_thresh is set to " + std::to_string(keep_thresh));
    }    
}

