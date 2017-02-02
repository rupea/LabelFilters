function [out_final, out_final_tr, svm_models_final] = multilabel_svm_merge_batches(filename, label_range, noClasses, cur_file, sparsemodel_final = false, keep_out = true, wfilemap = false)

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

  function y = removew(x)
    y=x;
    y.w = [];
  end
  
  function cnt = writew(x,fid)	   
    cnt = fwrite(fid,x.w,"single");
  end
  
  function ret = getSparse(x)
    if (size(x.w,1)==1)   
       x.w=x.w';
    endif
    [idx, foo, val] = find(x.w);
    ret.cidx = length(idx);
    ret.ridx = (idx - 1)';
    ret.val = val';
    ret.nrows = size(x.w,1);
  end
  
  wmapfilename = "";
  if (wfilemap)
    wmapfilename = [filename ".wmap"] 
    wfile = fopen(wmapfilename, "w");        
  endif

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

  if (wfilemap && sparsemodel_final)
     vals = [];
     cidx = [0];
     ridx = [];
  endif

  for lbl_idx = 1 : nfiles
    load(cur_file(lbl_idx),"out","out_tr", "svm_models", "class_idx_start", "class_idx_end", "solver", "solverparams", "sparsemodel");
    if (!exist("sparsemodel","var"))
      sparsemodel = false;
    endif
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
    if (!sparsemodel_final && sparsemodel)
      svm_models = cellfun("todense",svm_models, "UniformOutput", false);
      svm_models = cellfun("tosingle",svm_models, "UniformOutput", false);
    endif 
    
    if (!wfilemap)
      svm_models_final(class_idx_start:class_idx_end) = svm_models;      
    else
      if (!sparsemodel_final)
	cellfun("writew",svm_models, {wfile}); 
      else
	spvals = cellfun("getSparse", svm_models,"UniformOutput", false);
	vals = [vals cat(2,cell2mat(spvals).val)];
	ridx = [ridx cat(2,cell2mat(spvals).ridx)];
	cidx = [cidx (cumsum(cat(2,cell2mat(spvals).cidx)) + cidx(end))];
	nrows = spvals{1}.nrows;
      end
      svm_models_final(class_idx_start:class_idx_end) = cellfun("removew",svm_models,"UniformOUtput",false);
    endif    
  end
  display("Done loading...");
  
  if (wfilemap && sparsemodel_final)
     ncols = length(cidx) - 1;
     fwrite(wfile, [ncols nrows] , "int32");
     fwrite(wfile,cidx,"int32");
     fwrite(wfile,ridx, "int32");
     fwrite(wfile,vals,"single");	 
  end

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
  if (wfilemap)
    fclose(wfile);
  endif
  save(filename, "-binary", "out_final", "out_final_tr", "svm_models_final", "solver", "solverparams", "sparsemodel_final", "wfilemap", "wmapfilename");    

  # delete the partial files
  for lbl_idx = 1 : nfiles
    unlink(cur_file(lbl_idx));
  end
  return ;
end
