### run one vs all SVM for a batch of classes. 
### Project training points on a learned w and filter the instances before training the svm for each class. If project_file=="" no projection is done.


function perform_projected_multilabel_svm_batch(exp_name,exp_dir,projection_file,projection_dir, project_str,C,class_idx_start,class_idx_end)

  %%loading data
  load([exp_dir exp_name ".mat"], "-v6");

  if (isempty(projection_file)),
    projection = false;
  else
    projection = true;
  end
  
  if (projection && ~exist([projection_dir projection_file ".mat"], "file"))
    error("Projection file does not exist");
    projection = false;
  end
  
  if ( ~projection )
    if (~exist("project_str","var") || isempty(project_str))
      project_str = "full";
    end
  end
  
  
  probs=0;
  dispflag=0;
  
  nClasses = size(y_tr, 2);
  n = size(x_tr,1);
  m = size(x_te,1);
  out_tr = zeros(n,class_idx_end-class_idx_start+1);
  out = zeros(m,class_idx_end-class_idx_start+1);
  svm_models = cell(class_idx_end-class_idx_start+1,1);

  if (projection)
    xproj=x_tr*w;
    xtestproj=x_te*w;
  end

  for class = class_idx_start:class_idx_end % for all classes
    %% get only the examples that fit within the projection  
    disp(class);
    in_range = logical(ones(n,1));
    if (projection)
      for j = 1 : size(xproj,2),
	in_range  = and(in_range, xproj(:,j) >= min_proj(class,j) &  xproj(:,j) <= max_proj(class,j));
      end
    end
    tmpX=x_tr(in_range,:); 
    tmpY=y_tr(in_range,class);
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
    
    ## make predictions on the training set
    out_idx=class-class_idx_start+1;
    out_tr(:,out_idx)=-inf;
    [foo, bar, preds_tr] = svmpredict(zeros(size(tmpX,1),1), tmpX, model, svmparams);
    out_tr(in_range,out_idx) = preds_tr;
    
    ## make prediction on the test set
    in_range = logical(ones(m,1));
    ## if projection is performed identify the examples that are in the range of the class
    if (projection)
      for j = 1 : size(xtestproj,2),
	in_range  = and(in_range, xtestproj(:,j) >= min_proj(class,j) &  xtestproj(:,j) <= max_proj(class,j));
      end
    end
    out(:,out_idx)=-inf;
    tmpXtest = x_te(in_range,:);
    
    [foo, bar, preds] = svmpredict(zeros(size(tmpXtest,1),1), tmpXtest, model, svmparams);
    
    out(in_range,out_idx) = preds;
    
    svm_models{out_idx} = model;
    if (dispflag)
      fprintf("%d of %d done\r",out_idx,class_idx_end-class_idx_start+1);
    end
    
  end
  
  filename = ["svm_results/svm_" exp_name "_" num2str(C) "_" num2str(class_idx_start) "_" num2str(class_idx_end) "_projected_" project_str ".mat"];
  save(filename, 'out', 'out_tr', 'svm_models', 'class_idx_start', 'class_idx_end');
  
  
  if (dispflag)
    fprintf('\n');
  end
  
  return;
end
