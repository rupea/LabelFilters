srcdir = "~/Research/mcfilter/src"

addpath([srcdir "/octave/"])
addpath([srcdir "/libsvm-3.17/matlab/"])

addpath("~/Programs/gperftools-2.1/install/lib/")

exp_name = "LSHTC14train_minclass10_minfeat_10"

[y,x]=prepare_LSHTC14(0.8,10,10,1);

trial=1;
seed=trial;
## generate the train test split
old_state=rand("state");
if (seed > 0)
  rand("state",seed);
else
  rand("state","reset");
end
n=size(x,1);
n_train = floor(train_fraction*n);
perm=randperm(n);
y_tr = y(perm(1:n_train),:);
x_tr = x(perm(1:n_train),:);
y_te = y(perm((n_train+1):end),:);
x_te = x(perm((n_train+1):end),:);

rand("state",old_state);

save(["../data/" exp_name "_trial" trial ".mat"], "-v6" ,"y_tr","x_tr","y_te","x_te")

trial_expname = [exp_name "_trial" trial]

mkdir("svm_results");
[out, out_tr] = perform_parallel_multilabel_svm(trial_expname, 1, "", size(y_tr,2), "../data", "", "", "", false);
