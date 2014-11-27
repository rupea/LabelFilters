function [proj_params] = default_proj_param_strings(proj_params);

  if (isempty(proj_params.data_file))
    proj_params.data_file = sprintf("%s.mat", proj_params.exp_name);
  endif
  if (isempty(proj_params.projection_file))
    proj_params.projection_file = sprintf("results/wlu_%s_C1_%d_C2_%d.mat",proj_params.exp_name, proj_params.C1, proj_params.C2);
  endif
  if (isempty(proj_params.obj_plot_file))
    proj_params.obj_plot_file = sprintf("objective_plot_%s_C1_%d_C2_%d.pdf", proj_params.exp_name, proj_params.C1, proj_params.C2);
  endif
  if (isempty(proj_params.ova_preds_file))
    proj_params.ova_preds_file = sprintf("svm_results/svm_%s.mat", proj_params.exp_name);
  endif

  return;
end

  