#!/home/mlshack/alex/Programs/octave-3.8.2/bin/octave -qf

global __DS_VERBOSE = true;

addpath("~/Research/mcfilter/src/libsvm-3.17/matlab") ## for read_sparse_ml
addpath("~/Research/mcfilter/src/octave")
addpath("/bigml/alex/Research/mcfilter/LSHTC-2014/scripts")
addpath("/bigml/alex/Research/mcfilter/ds_scripts")



datadir = "../data/"

data_params.min_ex_per_class = 0;
data_params.min_word_count = 10;
data_params.weighting = "tfidf";
data_params.normalization = "row";
data_params.recreate = false;
data_params.type = "data_file";
data_params.original_data.name = "LSHTC14";
data_params.original_data.type = "original_train";
data_params.original_data.format = "svm";
data_params.filename = [datadir "LSHTC14train_minclass" num2str(data_params.min_ex_per_class) "_minfeat" num2str(data_params.min_word_count) "_weighting_" data_params.weighting "_normalization_" data_params.normalization ".mat"];

db_data_params = rmfield(data_params, ["filename";"recreate"]);

db_data_entries = ds_query(db_data_params);
if (length(db_data_entries) > 1 )
  error("More than one existing db entry matches the data file parameters.");
endif
if (length(db_data_entries) == 1)
  db_data_entries = db_data_entries{1};
  if (exist(db_data_entries.db_path))
    ds_unlink(db_data_entries.db_path);
  endif
  if (data_params.recreate)
    ds_remove(db_data_params);
    db_data_entries = {};
  else
    ds_get(db_data_params);
    data_params.filename = db_data_entries.db_path;    
  endif
endif


## we need to get the original file
if (!exist(data_params.filename,"file"))
  db_original_entries = ds_query(data_params.original_data);
  if (length(db_original_entries) == 1 )
    db_original_entries = db_original_entries{1};
    if (!exist(db_original_entries.db_path))
      ds_get(data_params.original_data);
    endif
    data_params.original_data = db_original_entries.params;
  elseif (length(db_original_entries) == 0)
    error("Query for the original data file returned no matches");
  else
    error("Query for the original data file returned more than one match");
  endif
else
  db_original_entries.db_path = "foo"; # this is not needed since the data file exists already and will 
                                       # not be recalculated
endif

## still need to call this even if the file already exists since this loads the 
## file
[y,x, map, map_ignored, idf]=prepare_LSHTC14(data_params.filename, db_original_entries.db_path, data_params.min_ex_per_class, data_params.min_word_count, data_params.weighting, data_params.normalization);

db_data_params.ninstances = size(x,1);
db_data_params.nfeatures = size(x,2);
db_data_params.nlabels = size(y,2); 

if (exist(data_params.filename, "file") && length(db_data_entries) == 0)
  ## the data file does not exist in the DB. Put it there.
  ds_add(data_params.filename, db_data_params);
endif




trial_params.trial = 1;
trial_params.seed = trial_params.trial;
trial_params.type ="random_split";
trial_params.train_fraction = 0.80;
trial_params.stratified = false;
trial_params.recreate = false || data_params.recreate;
trial_params.filename = [regexp(data_params.filename,"(?<fname>.*).mat$","names").fname "_trial" num2str(trial_params.trial) ".mat"];
trial_params.data = db_data_params;

db_trial_params = rmfield(trial_params, ["filename";"recreate"]);
db_trial_entries = ds_query(db_trial_params);
if (length(db_trial_entries) > 1 )
  error("Query returned more than one random split file matching the parameters.");
endif
if (length(db_trial_entries) == 1)
  db_trial_entries = db_trial_entries{1};
  if (exist(db_trial_entries.db_path))
    ds_unlink(db_trial_entries.db_path);
  endif
  if (trial_params.recreate)
    ds_remove(db_trial_params);    
    db_trial_entries = {};
  else
    ds_get(db_trial_params);
    trial_params.filename = db_trial_entries.db_path;    
    load(trial_params.filename);
    return;
  endif
endif


## generate the train test split
old_state=rand("state");
if (trial_params.seed > 0)
  rand("state",trial_params.seed);
else
  rand("state","reset");
end
n=size(x,1);
n_train = floor(trial_params.train_fraction*n);
perm=randperm(n);
y_tr = y(perm(1:n_train),:);
x_tr = x(perm(1:n_train),:);
y_te = y(perm((n_train+1):end),:);
x_te = x(perm((n_train+1):end),:);

rand("state",old_state);

save(trial_params.filename, "-v6" ,"y_tr","x_tr","y_te","x_te")

db_trial_params.trainsize = size(x_tr,1);
db_trial_params.testsize = size(x_te,1);

if (exist(trial_params.filename, "file") && length(db_trial_entries) == 0)
  ## the trial data function is not in the database. Put it there. 
  ds_add(trial_params.filename, db_trial_params);
endif

