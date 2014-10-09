function [w, min_proj, max_proj, obj_val] = \
      project_tests(x,tr_label, parameters, exp_name,restarts, resume, plot_objval= true)
  
  %% if we have multiple restarts we can't resume the computation
  if (restarts != 1)
    resume = 0
  end
  
  
  C1 = parameters.C1;
  C2 = parameters.C2;
  
  disp("-----------------------------------------------------");
  best_obj=Inf;
  for r=1:restarts
    disp(sprintf("restart %d\n", r));
    if ( resume && exist(sprintf("results/wlu_%s_C1_%d_C2_%d.mat",exp_name, C1, C2), "file"))
      load(sprintf("results/wlu_%s_C1_%d_C2_%d.mat",exp_name, C1, C2))
      [w,min_proj,max_proj, obj_val]=oct_find_w(x,tr_label,parameters,w,min_proj, max_proj);
    else 
      w=init_w(2, x, tr_label, size(x,2), parameters.no_projections);
      [w,min_proj,max_proj, obj_val]=oct_find_w(x,tr_label,parameters,w);
    endif    
    
    if (best_obj > obj_val(length(obj_val)))
      best_w = w;
      best_min_proj = min_proj;
      best_max_proj = max_proj;
      best_obj_val = obj_val;
      best_obj = obj_val(length(obj_val));
    end
  end
  
  w = best_w;
  min_proj = best_min_proj;
  max_proj = best_max_proj;
  obj_val = best_obj_val;
  
  %% plot the objective value vs iteration 	   
  if (plot_objval)
    figure("visible","off");
    plot(obj_val, "b", "LineWidth", 3);
    
    print("-dpdf", sprintf("objective_plot_%s_C1_%d_C2_%d.pdf", exp_name,C1, C2))
  end
  
  save(sprintf("results/wlu_%s_C1_%d_C2_%d.mat",exp_name, C1, C2), "w", "min_proj", "max_proj" , "obj_val");
  
end
