function [out_final, out_final_tr, svm_models_final] = multilabel_svm_merge_batches(filename, label_range, noClasses, cur_file, sparsemodel_final = false, keep_out = true)

  function y = tosingle(x)
    y=x;
    y.w = single(x.w);
  end

  function y = tosparse(x)
    y=x;
    y.w = sparse(y.w);
  end

  function y = todense(x)
    y=x;
    y.w = full(y.w);
  end

  display("Merging files ...");
  nfiles = length(label_range) - 1
  
  nClasses = label_range(end)-1;
  svm_models_final = cell(nClasses,1);

  if (keep_out)
    out_cell = cell(nfiles,1);
    nnz_out = 0;
    out_tr_cell = cell(nfiles,1);
    nnz_out_tr = 0;
  else
    out_final = [];
    out_final_tr = [];
  endif
  

  for lbl_idx = 1 : nfiles
    load(cur_file(lbl_idx),"out","out_tr", "svm_models", "class_idx_start", "class_idx_end", "solver", "solverparams", "sparsemodel");
    if (keep_out)       
      out_tr_cell{lbl_idx} = out_tr;
      nnz_out_tr = nnz_out_tr + nnz(out_tr); 
      out_cell{lbl_idx}=out;
      nnz_out = nnz_out + nnz(out);
    endif
    if (sparsemodel_final && !sparsemodel)
      svm_models = cellfun("tosparse",svm_models, "UniformOutput", false);
    endif
    if (!sparsemodel_final && !sparsemodel)
      svm_models = cellfun("tosingle",svm_models, "UniformOutput", false);
    endif
    
    svm_models_final(class_idx_start:class_idx_end) = svm_models;

  end

  display("Done loading...");
  
  if (keep_out)
    n = size(out_cell{1},1);
    n_tr = size(out_tr_cell{1},1);
    
    if (issparse(out_cell{1}))
      out_final = spalloc(n,noClasses,nnz_out);
      out_finatr = spalloc(n_tr,noClasses,nnz_out_tr);
    else
      out_final = zeros(n,noClasses);
      out_final_tr = zeros(n_tr,noClasses);
    end
    
    for lbl_idx = 1 : nfiles
      start_range = label_range(lbl_idx);
      end_range = label_range(lbl_idx+1)-1;
      out_final(:,start_range:end_range)=out_cell{lbl_idx};
      out_final_tr(:,start_range:end_range)=out_tr_cell{lbl_idx};
    end
  endif
  ## save in octave binary format because saving in matlab v6 format gives errors
  save(filename, "-binary", "out_final", "out_final_tr", "svm_models_final", "solver", "solverparams", "sparsemodel_final");    
  return ;
end
