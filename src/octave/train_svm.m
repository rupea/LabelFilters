function [svm_filename] = train_svm(params)

  [data_dir, data_file] = fileparts(params.data_file);
  if (isempty(data_dir))
    data_dir = "./";
  else
    data_dir = [data_dir "/"];
  endif


  if (!isfield(params,"threshold") || isempty(params.threshold))
    params.threshold = [];  #only used when making predictions. Should remove it since predictions are not ususally made.
    thresh_str = "none";
  else
    thresh_str = num2str(params.threshold);
  end


  ## add a hash of the params to make a unique file name.
  params.filter_name = [params.filter_name "__" md5sum(cell2mat(strsplit(disp(params),"\n")),true)];

  svm_filename = ["svm_results/svm_" data_file "_C" num2str(params.C) "_threshold" thresh_str "_projected_" params.filter_name ".mat"]  

  mkdir("svm_results");  
  perform_parallel_projected_multilabel_svm(data_file, params.C, [], data_dir, params.retrain, params.filter_file, params.filter_name, params.threshold, params.solver, params.solverparam, params.n_batches, params.min_batch_size, params.sparsemodel, params.keep_out, params.wfilemap);      

  if (!isempty(params.filename) && !strcmp(params.filename,svm_filename))
    movefile(svm_filename,params.filename,"f")
    if (params.wfilemap)
      movefile([svm_filename ".wmap"], [params.filename ".wmap"], "f");
    endif
    svm_filename = params.filename;
  endif
  return;
end
