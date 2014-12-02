function [out_final, out_final_tr, svm_models_final] = multilabel_svm_merge_batches(filename, label_range, noClasses, cur_file)

  function y = tosingle(x)
    y=x;
    y.w = single(x.w);
  end

  display("Merging files ...");
  nfiles = length(label_range) - 1
  
  nClasses = label_range(end)-1;
  svm_models_final = cell(nClasses,1);

  out_cell = cell(nfiles,1);
  nnz_out = 0;
  out_tr_cell = cell(nfiles,1);
  nnz_out_tr = 0;
  
  for lbl_idx = 1 : nfiles
    load(cur_file(lbl_idx),"out","out_tr", "svm_models", "class_idx_start", "class_idx_end", "solver", "solverparams");
    out_tr_cell{lbl_idx} = out_tr;
    nnz_out_tr = nnz_out_tr + nnz(out_tr); 
    out_cell{lbl_idx}=out;
    nnz_out = nnz_out + nnz(out);
    svm_models_final(class_idx_start:class_idx_end) = cellfun("tosingle",svm_models, "UniformOutput", false);
  end

  display("Done loading...");
  
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

  save(filename, "-v6", "out_final", "out_final_tr", "svm_models_final", "solver", "solverparams");    
  return ;
end
