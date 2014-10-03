function [out_final, out_final_tr] = svm_merge_batches(filename, label_range, noClasses, cur_file)

  display("Merging files ...");
  nfiles = length(label_range) - 1
  
  out_cell = cell(nfiles,1);
  nnz_out = 0;
  if (nargout == 2)
    out_tr_cell = cell(nfiles,1);
    nnz_out_tr = 0;
  end

  for lbl_idx = 1 : nfiles
    if (nargout == 2)
      load(cur_file(lbl_idx),"out","out_tr");
      out_tr_cell{lbl_idx} = out_tr;
      nnz_out_tr = nnz_out_tr + nnz(out_tr); 
    else
      load(cur_file(lbl_idx),"out");
    end
    out_cell{lbl_idx}=out;
    nnz_out = nnz_out + nnz(out);
  end

  display("Done loading...");
  
  n = size(out_cell{1},1);
  if (nargout == 2)
    n_tr = size(out_tr_cell{1},1);
  end

  if (issparse(out_cell{1}))
    out_final = spalloc(n,noClasses,nnz_out);
    if (nargout == 2)
      out_finatr = spalloc(n_tr,noClasses,nnz_out_tr);
    end
  else
    out_final = zeros(n,noClasses);
    if (nargout == 2)
      out_final_tr = zeros(n_tr,noClasses);
    end
  end

  for lbl_idx = 1 : nfiles
    start_range = label_range(lbl_idx);
    end_range = label_range(lbl_idx+1)-1;
    out_final(:,start_range:end_range)=out_cell{lbl_idx};
    if (nargout == 2)
      out_final_tr(:,start_range:end_range)=out_tr_cell{lbl_idx};
    endif
  end  

  if (nargout == 2)
    save(filename, "-v6", 'out_final', 'out_final_tr');    
  else
    save(filename, "-v6", 'out_final');        
  end
  return ;
end
