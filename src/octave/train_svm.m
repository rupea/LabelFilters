function [svm_filename] = train_svm(params)

  if (!isfield(params,"threshold") || isempty(params.threshold))
    params.threshold = [];  #only used when making predictions. Should remove it since predictions are not ususally made.
  end


  if (isfield(params, "bias") && params.bias > 0 && strcmp(params.solver, "liblinear"))
     params.solverparam = [params.solverparam " -B " num2str(params.bias)]
  endif

  if (isempty(params.filename))
     error("No output filename provided");
  endif

  perform_parallel_projected_multilabel_svm(params.filename, params.data_file, params.C, -1, params.retrain, params.filter_file, params.threshold, params.solver, params.solverparam, params.weights_threshold, params.n_batches, params.min_batch_size, params.sparsemodel, params.keep_out, params.wfilemap);

  return;
end
