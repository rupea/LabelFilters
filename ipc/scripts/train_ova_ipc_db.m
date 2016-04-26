#!/home/mlshack/alex/Programs/octave-3.8.2/bin/octave -qf

### wrapper octave script to learn the ova models. It learns the models and saves them
### to disk

global __DS_VERBOSE = false;

srcdir = "~/Research/mcfilter/src";

#addpath("../scripts/")
addpath([srcdir "/octave/"])
addpath([srcdir "/libsvm-3.17/matlab/"])
addpath([srcdir "/liblinear-1.94/matlab/"])

%addpath("/bigml/alex/Research/mcfilter/LSHTC-2014/scripts")
addpath("/bigml/alex/Research/mcfilter/scripts")
addpath("/bigml/alex/Research/mcfilter/ds_scripts")

#addpath("~/Programs/gperftools-2.1/install/lib/")

arg_list = argv();

input_params.data_query = "";
input_params.data_file = "";
input_params.C = 30;
input_params.keep_out = true;
input_params.threshold = [];
input_params.sparsemodel = false;
input_params.wfilemap = true;
input_params.solver = "liblinear";
input_params.solverparam = "-s 3";
input_params.n_batches = 2000;
input_params.min_batch_size = 10;
input_params.retrain = false;
input_params.filter_query = "";
input_params.filter_file = "";
input_params.filter_name = "";
input_params.filter_name_fields = "C1 C2";
input_params.filename = "";
input_params.filename_fields = "C data.min_ex_per_class data.min_word_count filter.C1 filter.C2 __HASH";

input_params = process_proj_params_args(arg_list,length(arg_list), input_params)

train_ova_db(input_params);