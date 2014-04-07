### run one vs all SVM for a batch of classes. 
### Project training points on a learned w and filter the instances before training the svm for each class
### TO DO: unify with perform_svm_batch.m (use projected file = "" to turn off projection)

function perform_projected_svm_batch(exp_name,exp_dir,projection_file,projection_dir, project_str,C,class_idx_start,class_idx_end)

%loading data
load([exp_dir exp_name ".mat"], "-v6");
load([projection_dir projection_file ".mat"]);

probs=0;
dispflag=0;

classes = sort(unique(tr_label));
nClasses = numel(classes);
n = size(x_orig,1);
m = size(xtest_orig,1);
out = zeros(m,class_idx_end-class_idx_start);

xproj=x_orig*w;
xtestproj=xtest_orig*w;


for c = 1 : class_idx_end-class_idx_start % for all classes
  % get only the examples that fit within the projection  
  class=c+class_idx_start;
  disp(c + class_idx_start);
  in_range = ones(size(xproj,1),1);
  for j = 1 : size(xproj,2),
    in_range  = and(in_range, xproj(:,j) >= min_proj(class,j) &  xproj(:,j) <= max_proj(class,j));
  end
  tmpX=x_orig(in_range,:); 
  tmpY=tr_label(in_range);
  tmpY = double(tmpY==classes(class));
  tmpY(tmpY==0) = -1;
  assert(numel(unique(tmpY))==2);

    if (probs)
        svmparams=sprintf("-q -t 0 -c %f -b 1", C);
    else
        svmparams=sprintf("-t 0 -c %f", C);
    end

    disp(size(tmpX));
    disp(size(tmpY));
    disp(svmparams);

    model = svmtrain(tmpY,tmpX, svmparams);

    if (probs)
        svmparams=sprintf("-q -b 1", C);
    else
        svmparams=sprintf("-q", C);
    end


  in_range = ones(size(xtestproj,1),1);
  for j = 1 : size(xtestproj,2),
    in_range  = and(in_range, xtestproj(:,j) >= min_proj(class,j) &  xtestproj(:,j) <= max_proj(class,j));
  end
  out(:,c)=-9999;
  tmpXtest = xtest_orig(in_range,:);

  
  [foo, bar, preds] = svmpredict(zeros(size(tmpXtest,1),1), tmpXtest, model, svmparams);

  disp(size(preds));
    out(in_range,c) = preds;
    
    svm_models{c} = model;
    if (dispflag)
        fprintf('%d of %d done\r',c,nClasses);
    end
end

filename = ["svm_results/svm_" exp_name "_" num2str(C) "_" num2str(class_idx_start) "_" num2str(class_idx_end) "_projected_" project_str ".mat"];
save(filename, 'out', 'svm_models', 'class_idx_start', 'class_idx_end');


if (dispflag)
    fprintf('\n');
end

return;

