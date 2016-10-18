#!/home/mlshack/alex/Programs/octave-3.8.2/bin/octave -qf

### wrapper octave script to learn the projections. It learns the projection and puts the learned learned w , minproj and maxproj in the exp database

global __DS_VERBOSE = false;

addpath("~/Research/mcfilter/scripts")


data_params.type = "data_file";
data_params.original_train.name = "ipc";
data_params.min_ex_per_class = 0;
data_params.min_word_count = 1;
data_params.coding = "none";
data_params.normalization = "row";

proj_params = default_proj_params();

## override the default params for this dataset
proj_params.exp_name = "ipc_full";
proj_params.C1 = 200000;
proj_params.C2 = 100;
proj_params.max_iter = 1e5;
proj_params.optimizeLU_epoch = proj_params.max_iter;
proj_params.seed = 2131;
proj_params.data_query = struct2query(data_params);


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
proj_params.plot_objval = false;
proj_params.obj_plot_file = "";
proj_params.obj_plot_dir = "./obj_plots";
proj_params.obj_plot_name_fields = "data_params.data.min_ex_per_class data_params.data.min_word_count ova_preds_parms.C C1 C2 no_projections max_iter avg_epoch eta_type optimizeLU_epoch __HASH";
proj_params.log_file = "";
proj_params.log_dir = "./log_files";
proj_params.log_name_fields = "data_params.data.min_ex_per_class data_params.data.min_word_count ova_preds_parms.C C1 C2 no_projections max_iter avg_epoch eta_type optimizeLU_epoch __HASH";
proj_params.projection_file = "";
proj_params.projection_dir = "results/";
proj_params.projection_name_fields = "data_params.data.min_ex_per_class data_params.data.min_word_count ova_preds_parms.C C1 C2 no_projections max_iter avg_epoch eta_type optimizeLU_epoch __HASH";
proj_params.C1multiplier = false;

## process command line arguments
arg_list = argv(); 
proj_params=process_proj_params_args(arg_list, length(arg_list), proj_params);

## it would be nice to do this in the default params, but we would need 
## lazy evaluation for this

## override the defaults with dataset specific 
## need to be careful not to override the parameters already set through 
## command line arguments arguments

time_str = num2str(time());   
   
if (isempty(proj_params.projection_file))
  proj_params.projection_file = sprintf("results/wlu__%s__C1_%g__C2_%g__%s.mat",proj_params.exp_name, proj_params.C1, proj_params.C2,time_str);
  if (!exist("results","dir"))
    mkdir("results");
  endif
endif 
## if (isempty(proj_params.obj_plot_file))
##   proj_params.obj_plot_file = sprintf("obj_plots/objective_plot__%s__C1_%g__C2__%g__%s.pdf", proj_params.exp_name, proj_params.C1, proj_params.C2, time_str);
##   if (!exist("obj_plots","dir"))
##     mkdir("obj_plots");
##   endif
## endif 
if (isempty(proj_params.log_file))
  proj_params.log_file = sprintf("log_files/%s__C1_%g__C2_%g.log",proj_params.exp_name, proj_params.C1, proj_params.C2);
  if (!exist("log_files","dir"))
    mkdir("log_files");
  endif
endif 

## get the default parameters 
## because these depend on other parameters we need to do it after 
## parsing the input arguments and setting the 
## file default params
proj_params = default_proj_param_strings(proj_params);


learn_projection_db(proj_params, database);
