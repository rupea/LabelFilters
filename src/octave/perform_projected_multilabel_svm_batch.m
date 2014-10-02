### run one vs all SVM for a batch of classes. 
### Project training points on a learned w and filter the instances before training the svm for each class. If project_file==[] no projection is done.


function perform_projected_multilabel_svm_batch(exp_name,exp_dir,projection_file,projection_dir, project_str, C, class_idx_start,class_idx_end, threshold = [], solver="libsvm", solverparams = "-t 0")

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
    if (~exist("project_str","var") || isempty(project_str)) || strcmp(project_str,"")
      project_str = "full";
    end
  end

  nClasses = size(y_tr, 2);
  n = size(x_tr,1);
  m = size(x_te,1);
  svm_models = cell(class_idx_end-class_idx_start+1,1);
  
  if (isempty(threshold))
    thresh_str = "none";
    out_tr = zeros(n,class_idx_end-class_idx_start+1);
    out = zeros(m,class_idx_end-class_idx_start+1);
  else
    if (ischar(threshold))
      threshold = str2num(threshold);
    end
    thresh_str = num2str(threshold);
    out_tr = sparse(n,class_idx_end-class_idx_start+1);
    out = sparse(m,class_idx_end-class_idx_start+1);
  end
  
  probs=0;
  dispflag=0;
  

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
    
    disp(size(tmpX));
    disp(size(tmpY));
    
    svmparams = sprintf("%s -c %f", solverparams, C);
    if (probs)
      svmparams = sprintf("%s -b 1", svmparams);
    end
    disp(svmparams);    

    if (strcmp(solver,"libsvm"))      
      model = svmtrain(tmpY,tmpX, svmparams);
    elseif (strcmp(solver,"liblinear"))
      model = train(tmpY, tmpX, svmparams);
    end
    
    svmparams = "";
    if (probs)
      svmparams=sprintf("-b 1", C);
    end
    
    ## make predictions on the training set
    out_idx=class-class_idx_start+1;
    if (strcmp(solver,"libsvm"))
      [foo, bar, preds_tr] = svmpredict(zeros(size(tmpX,1),1), tmpX, model, svmparams);
    elseif (strcmp(solver,"liblinear"))
      [foo, bar, preds_tr] = predict(zeros(size(tmpX,1),1), tmpX, model, svmparams);      
    end
    if ( ~isempty(threshold))
      pos = preds_tr > threshold;
      idx = ((1:n)(in_range))(pos);
      out_tr(idx,out_idx) = preds_tr(pos); # could also make it 1
    else    
      out_tr(in_range,out_idx) = preds_tr;
      out_tr(~in_range,out_idx)=-inf;
    end

    ## make prediction on the test set
    in_range = logical(ones(m,1));
    ## if projection is performed identify the examples that are in the range of the class
    if (projection)
      for j = 1 : size(xtestproj,2),
	in_range  = and(in_range, xtestproj(:,j) >= min_proj(class,j) &  xtestproj(:,j) <= max_proj(class,j));
      end
    end

    tmpXtest = x_te(in_range,:);
    if (strcmp(solver,"libsvm"))
      [foo, bar, preds] = svmpredict(zeros(size(tmpXtest,1),1), tmpXtest, model, svmparams);
    elseif (strcmp(solver,"liblinear"))
      [foo, bar, preds] = predict(zeros(size(tmpXtest,1),1), tmpXtest, model, svmparams);
    end
    if ( ~isempty(threshold))
      pos = preds > threshold;
      idx = ((1:m)(in_range))(pos);
      out(idx,out_idx) = preds(pos); # could also make it 1
    else    
      out(in_range,out_idx) = preds;
      out(~in_range,out_idx)=-inf;
    end
    
    svm_models{out_idx} = model;
    if (dispflag)
      fprintf("%d of %d done\r",out_idx,class_idx_end-class_idx_start+1);
    end
    
  end
  
  filename = ["svm_results/svm_" exp_name "_" num2str(C) "_" num2str(class_idx_start) "_" num2str(class_idx_end) "_threshold" thresh_str "_projected_" project_str ".mat"];
  save(filename, "out", "out_tr", "svm_models", "class_idx_start", "class_idx_end", "solver", "solverparams");
  
  
  if (dispflag)
    fprintf('\n');
  end
  
  return;
end
