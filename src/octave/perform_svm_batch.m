function perform_svm_batch(exp_name,exp_dir,C,class_idx_start,class_idx_end)

svm_dir = '../libsvm-3.17/';
addpath([svm_dir "/matlab/"]);

%loading data
load([exp_dir exp_name ".mat"], "-v6");

probs=0;
dispflag=0;

classes = sort(unique(tr_label));
nClasses = numel(classes);
n = size(x_orig,1);
m = size(xtest_orig,1);
out = zeros(m,class_idx_end-class_idx_start);

for c = 1 : class_idx_end-class_idx_start % for all classes
    disp(c + class_idx_start);
    tmpY = double(tr_label==classes(c + class_idx_start));
    tmpY(tmpY==0) = -1;
    assert(numel(unique(tmpY))==2);

    if (probs)
        svmparams=sprintf("-q -t 0 -c %f -b 1", C);
    else
        svmparams=sprintf("-t 0 -c %f", C);
    end

    disp("here");

    disp(size(x_orig));
    disp(size(tmpY));
    disp(svmparams);

    model = svmtrain(tmpY,x_orig, svmparams);

    disp("here");

    if (probs)
        svmparams=sprintf("-q -b 1", C);
    else
        svmparams=sprintf("-q", C);
    end

    disp("here");
    [foo, bar, preds] = svmpredict(zeros(m,1), xtest_orig, model, svmparams);

    out(:,c)=preds;
    disp("here");
    
    svm_models{c} = model;
    disp("here");
    if (dispflag)
        fprintf('%d of %d done\r',c,nClasses);
    end
end

filename = ["svm_results/svm_" exp_name "_" num2str(C) "_" num2str(class_idx_start) "_" num2str(class_idx_end) ".mat"];
save(filename, 'out', 'svm_models', 'class_idx_start', 'class_idx_end');


if (dispflag)
    fprintf('\n');
end

return;

