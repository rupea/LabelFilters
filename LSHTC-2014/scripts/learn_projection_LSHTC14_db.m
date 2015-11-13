#!/home/mlshack/alex/Programs/octave-3.8.2/bin/octave -qf

### wrapper octave script to learn the projections. It learns the projection and saves the 
### learned w , minproj and maxproj to disk.

global __DS_VERBOSE = false;

srcdir = "~/Research/mcfilter/src";

#addpath("../scripts/")
addpath([srcdir "/octave/"])
addpath([srcdir "/libsvm-3.17/matlab/"])
addpath([srcdir "/liblinear-1.94/matlab/"])

%addpath("/bigml/alex/Research/mcfilter/LSHTC-2014/scripts")
addpath("~/Research/mcfilter/scripts")
addpath("~/Research/mcfilter/ds_scripts")

#addpath("~/Programs/gperftools-2.1/install/lib/")


proj_params = default_proj_params();

## override the default params for this dataset
proj_params.exp_name = "LSHTC14";
proj_params.C1 = 2000000;
proj_params.C2 = 100;
proj_params.max_iter = 1e5;
proj_params.batch_size = 1000;
proj_params.reorder_epoch = 1000;
proj_params.report_epoch = 1000;
proj_params.eta = 1;
proj_params.min_eta = 0;
proj_params.no_projections = 5;
proj_params.remove_constraints = true;
proj_params.remove_class_constraints = false;
proj_params.reweight_lambda = 1;
proj_params.onlycorrect = false;
proj_params.ova_preds_file = "";
proj_params.ova_preds_query = "";
proj_params.relearn_projection = false;
proj_params.resume = false;
proj_params.restarts = 1;
proj_params.data_file = "";
proj_params.data_query = "";
proj_params.plot_objval = true;
proj_params.obj_plot_file = "";
proj_params.obj_plot_dir = "./";
proj_params.obj_plot_name_fields = "data_params.data.min_ex_per_class C1 C2 no_projections max_iter avg_epoch optimizeLU_epoch __HASH";
proj_params.log_file = "";
proj_params.log_dir = "./";
proj_params.log_name_fields = "data_params.data.min_ex_per_class C1 C2 no_projections max_iter avg_epoch optimizeLU_epoch __HASH";
proj_params.projection_file = "";
proj_params.projection_dir = "results/";
proj_params.projection_name_fields = "data_params.data.min_ex_per_class C1 C2 no_projections max_iter avg_epoch optimizeLU_epoch __HASH";
proj_params.C1multiplier = 1;
proj_params.seed = 2131;

## process command line arguments
arg_list = argv(); 
firstarg = 1;
database = "";
if (!strncmp(arg_list{1},"-", 1))
  database = arg_list{1};
  firstarg = 2;
endif 
arg_list = arg_list(firstarg:end);
proj_params=process_proj_params_args(arg_list, length(arg_list), proj_params);

learn_projection_db(proj_params, database);
