
#include "parameter-args.h"
#include <iostream>
#include <cstdint>

namespace opt {
    using namespace std;
    using namespace boost::program_options;
    //namespace po = boost::program_options;

    void helpUsageDummy( std::ostream& os ){
        os<<" Usage: foo [options]";
        return;
    }

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
            ("wt_by_nclasses", value<bool>()->default_value(false), "?")
            ("wt_class_by_nclasses", value<bool>()->default_value(false), "?")
            ("negclass", value<uint32_t>()->default_value(0U)
             , "# of negative classes used at each iter, 0 ~ all classes")
            ("remove_constraints", value<bool>()->default_value(false)
             , "after each projection, remove constraints")
            ("remove_class", value<bool>()->default_value(false)
             , "after each projection, remove already-separated classes")
            ("threads", value<uint32_t>()->default_value(0U), "# threads, 0 ~ use OMP_NUM_THREADS")
            ("seed", value<uint32_t>()->default_value(0U), "random number seed")
            ("tgrad", value<uint32_t>()->default_value(0U), "iter period for finite difference gradient test")
            ("ngrad", value<uint32_t>()->default_value(1000U), "directions per gradient test")
            ("grad", value<double>()->default_value(1.e-4), "step size per gradient test")
            ("resume", value<bool>()->default_value(false), "resume an existing soln?")
            ("reoptlu", value<bool>()->default_value(false), "reoptimize {l,u} bounds of existing soln?")
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
#define ARGSDEBUG 1            
#if ARGSDEBUG > 0
        cout<<" argsParse( argc="<<argc<<", argv, ... )"<<endl;
        for( int i=0; i<argc; ++i ) {
            cout<<"    argv["<<i<<"] = "<<argv[i]<<endl;
        }
#endif
        vector<string> ret;
        po::options_description desc("Options");
        mcParameterDesc( desc );

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
}//opt::

