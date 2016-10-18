#!/home/ml/alex/Programs/octave-3.8.2/bin/octave -qf

### wrapper octave script to learn the ova models. It learns the models and saves them
### to disk

global __DS_VERBOSE = false;


## dafaults for SVMs 

## parameters that change the model or what is recorded
svm_ova_default.C = 30;
svm_ova_default.keep_out = false;
svm_ova_default.threshold = 0;
svm_ova_default.sparsemodel = false;
svm_ova_default.wfilemap = true;
svm_ova_default.solver = "liblinear";
svm_ova_default.solverparam = "-s 3";

## parameters that do not influence the model, and parameters that indicate the 
## training data and filter (if applicable) . The training data and filter will 
## be included in the EBD entry automatically

svm_additional_default.data_query = "";
svm_additional_default.data_file = "";
svm_additional_default.n_batches = 2000;
svm_additional_default.min_batch_size = 10;
svm_additional_default.retrain = false;
svm_additional_default.filter_query = "";
svm_additional_default.filter_file = "";
svm_additional_default.filter_name = "";
svm_additional_default.filter_name_fields = "C1 C2";
svm_additional_default.filename = "";
svm_additional_default.filename_fields = "C data.min_ex_per_class data.min_word_count filter.C1 filter.C2 __HASH";
svm_additional_default.node_requests = "";


### DEFAULTS FOR NB ############
## parameters that change the model or what is recorded
nb_ova_default.alpha = 1;
nb_ova_default.wfilemap = true;  # can only be true
## nb_ova_default.keep_out = false;  # not implemented 
## nb_ova_default.threshold = 0; # not implemented 
## nb_ova_default.sparsemodel = false; # not implemented

## parameters that do not influence the model, and parameters that indicate the 
## training data and filter (if applicable) . The training data and filter will 
## be included in the EBD entry automatically

nb_additional_default.data_query = "";
nb_additional_default.data_file = "";
nb_additional_default.retrain = false;
nb_additional_default.filename = "";
nb_additional_default.filename_fields = "alpha data.min_ex_per_class data.min_word_count __HASH";
nb_additional_default.node_requests = "";
##filtering  not implemented with naive bayes
# nb_additional_default.filter_query = "";
# nb_additional_default.filter_file = "";
# nb_additional_default.filter_name = "";
# nb_additional_default.filter_name_fields = "C1 C2";


srcdir = "~/Research/mcfilter/src";

addpath([srcdir "/octave/"])
addpath([srcdir "/libsvm-3.17/matlab/"])
addpath([srcdir "/liblinear-1.94/matlab/"])

addpath("~/Research/mcfilter/scripts")
addpath("~/Research/mcfilter/edb_scripts")
addpath("~/Research/mcfilter/edb_scripts/jsonlab-1.2")

#addpath("~/Programs/gperftools-2.1/install/lib/")

arg_list = argv();

if (length(arg_list) < 1 )
  error("No model type provided.")
endif

model_type = arg_list(1);

if (strcmp(model_type,"svm"))
  ova_params = svm_ova_default;
  additional_params = svm_additional_default;
  ova_params.model_type = "svm";
elseif (strcmp(model_type,"nb"))
  ova_params = nb_ova_default;
  additional_params = nb_additional_default;
  ova_params.model_type = "nb";
  ova_params.bias = 1; #indicate that there is a bias column, and what the bias should be
else
  error(["Ova model type " model_type " is not implemented."]);
endif
  
arg_list = arg_list(2:end);

[ova_params, additional_params] = process_proj_params_args(arg_list,length(arg_list), ova_params, additional_params)

train_ova_db(ova_params, additional_params);
