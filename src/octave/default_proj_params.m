function [ proj_params ] = default_proj_params() 
  proj_params.exp_name = "projection";
  proj_params.C1 = 200000;
  proj_params.C2 = 100;
  proj_params.max_iter = 1e5;
  proj_params.batch_size = 1000;
  proj_params.avg_epoch = 0;
  proj_params.reorder_epoch = 1000;
  proj_params.update_type = "minibatch";
  proj_params.reorder_type = "avg_proj_means";
  proj_params.report_epoch = 0;
  proj_params.report_avg_epoch = 0;
  proj_params.optimizeLU_epoch = 0;
  proj_params.eta = 0.1;
  proj_params.eta_type = "lin";
  proj_params.min_eta = 0;
  proj_params.no_projections = 5;
  proj_params.remove_constraints = true;
  proj_params.remove_class_constraints = false;
  proj_params.reweight_lambda = true;
  proj_params.plot_objval = true;
  proj_params.onlycorrect = false;
  proj_params.ova_preds_file = "";
  proj_params.ova_preds_query = "";
  proj_params.relearn_projection = false;
  proj_params.resume = false;
  proj_params.resume_from = "";
  proj_params.resumed_from = "";
  proj_params.reoptimize_LU = false;
  proj_params.reoptimize_LU_query = "";
  proj_params.reoptimize_LU_file = "";
  proj_params.restarts = 1;
  proj_params.data_file = "";
  proj_params.data_query = "";
  proj_params.obj_plot_file = "";
  proj_params.projection_file = "";
  proj_params.log_file = "";
  proj_params.C1multiplier = false;
  proj_params.seed = 0;
  proj_params.num_threads = 0;
  proj_params.finite_diff_test_epoch = 0;
  proj_params.no_finite_diff_tests = 1000;
  proj_params.finite_diff_test_delta = 1e-4;
  proj_params.class_samples = 0;
  return;
end
