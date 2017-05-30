### run one vs all SVM for a batch of classes. 
### Project training points on a learned w and filter the instances before training the svm for each class. If project_file==[] no projection is done.


function perform_projected_multilabel_svm_batch(out_file, data_file, ...
						projection_file, ...
						C, class_idx_start,class_idx_end, ...
						threshold = [], ...
						solver="liblinear", solverparams = "-s 3", ...
						weights_threshold = 0, ...
						sparsemodel = false, keep_out = false)

  %%loading data
  load(data_file, "-v6");

  if (isempty(projection_file)),
    projection = false;
  else
    projection = true;
    if (~exist(projection_file, "file"))
      error("Projection file does not exist");
      projection = false;      
    else
      load(projection_file, "w", "min_proj", "max_proj");
    end
  end
  
  n = size(x_tr,1);
  m = size(x_te,1);
  svm_models = cell(class_idx_end-class_idx_start+1,1);
  
  if (keep_out)
    if (isempty(threshold))
      out_tr = zeros(n,class_idx_end-class_idx_start+1);
      out = zeros(m,class_idx_end-class_idx_start+1);
    else
      if (ischar(threshold))
	threshold = str2num(threshold);
      end
      out_tr = sparse(n,class_idx_end-class_idx_start+1);
      out = sparse(m,class_idx_end-class_idx_start+1);
    end
  else
    out_tr = [];
    out = [];
  endif

  if (projection)
    xproj=x_tr*w;
    xtestproj=x_te*w;
  end

  for class = class_idx_start:class_idx_end % for all classes
    %% get only the examples that fit within the projection  
    in_range = logical(ones(n,1));
    if (projection)
      for j = 1 : size(xproj,2),
	in_range  = and(in_range, xproj(:,j) >= min_proj(class,j) &  xproj(:,j) <= max_proj(class,j));
      end
    end
    tmpX=x_tr(in_range,:); 
    if (size(y_tr,2)==1)
      %% multiclass problem with label vector
      %% assumes that classes are 1:noClasses
      tmpY=y_tr(in_range);
      tmpY=double(tmpY==class); 
    else
      %% multiclass or multilabel problem with label indicator matrix
      tmpY=y_tr(in_range,class);
    end
    tmpY(tmpY==0) = -1;

    assert(numel(unique(tmpY))<=2);

    if (size(tmpX,1) == 0)
      %% all trainig examples have been filtered out
      %% just create a classifier that alwasy predicts 0
      tmpX = sparse(2,size(x_tr,2));
      tmpY = [-1;1];
    end
    

    if (numel(unique(tmpY)) == 1)
      %% one label is missing. 
      %% liblinear treats it as a one class problem, which might be fine except that it will always make positive predictiosn(even if the labels are -1).
      %% maybe these classes should be removed, but for now, just add an all zero document with the missing label.      
      %% this could create problems down the road, but it is an easier fix now
      %% maybe a better fix is for these models to always predict -9999
      %% could do this if we introduce a bias
      tmpX = [sparse(1,size(x_tr,2));tmpX];
      tmpY = [-unique(tmpY); tmpY];
    endif
    
    svmparams = sprintf("%s -c %f", solverparams, C);

    if (strcmp(solver,"libsvm"))      
      model = svmtrain(tmpY,tmpX, svmparams);
    elseif (strcmp(solver,"liblinear"))
      model = train(tmpY, tmpX, svmparams);
    end
    
    # should put a check that we have trained a linear model 
    if (weights_threshold > 0)
       model.w(abs(model.w) < weights_threshold) = 0;
       ## s = sum(abs(model.w))
       ## sm  = sort(abs(model.w), "descend");
       ## i = find(cumsum(sm)/s > weights_threshold,1)
       ## sm(i)       
       ## cumsum(sm)(i)
       ## model.w(abs(model.w) < sm(i)) = 0;
    endif
    
    svmparams = "";

    out_idx=class-class_idx_start+1;

    if (keep_out)    
      ## make predictions on the training set
      ## make predictions on the entire set, do not filter according to the projection
      if (strcmp(solver,"libsvm"))
	[foo, bar, preds_tr] = svmpredict(zeros(size(x_tr,1),1), x_tr, model, svmparams);
      elseif (strcmp(solver,"liblinear"))
	[foo, bar, preds_tr] = predict(zeros(size(x_tr,1),1), x_tr, model, svmparams);      
      end
      if ( ~isempty(threshold))
	pos = preds_tr > threshold;
	idx = (1:n)(pos);
	out_tr(idx,out_idx) = preds_tr(pos); # could also make it 1
      else    
	out_tr(:,out_idx) = preds_tr;
      end

      ## make prediction on the test set
      ## make predictions on the entire set, do not filter according to the projection
      
      if (strcmp(solver,"libsvm"))
	[foo, bar, preds] = svmpredict(zeros(size(x_te,1),1), x_te, model, svmparams);
      elseif (strcmp(solver,"liblinear"))
	[foo, bar, preds] = predict(zeros(size(x_te,1),1), x_te, model, svmparams);
      end
      if ( ~isempty(threshold))
	pos = preds > threshold;
	idx = (1:m)(pos);
	out(idx,out_idx) = preds(pos); # could also make it 1
      else    
	out(:,out_idx) = preds;
      end
    endif
    
    if (sparsemodel)
      model.w = sparse(model.w');
    endif

    svm_models{out_idx} = model;
      
  end
  
  save(out_file, "-v6", "out", "out_tr", "svm_models", "class_idx_start", "class_idx_end", "solver", "solverparams", "sparsemodel");
      
  return;
end
