function [w, min_proj, max_proj, obj_val, w_noavg, min_proj_noavg, max_proj_noavg, obj_val_noavg] = ...
      learn_projections(parameters, x_tr=[],y_tr=[])

  
  %% if we have multiple restarts we can't resume the computation
  # if (parameters.restarts != 1)
  #   parameters.resume = 0
  # end

  ## if parameters.resume is true, it assumes that parameters.projection_file exists
  ## and will resume the computation from this file
  ## parameters.projection_file will be overwritten

  if (!parameters.relearn_projection && exist(parameters.projection_file,"file"))
    load(parameters.projection_file, "w", "min_proj", "max_proj", "obj_val", "w_noavg", "min_proj_noavg", "max_proj_noavg", "obj_val_noavg");
    if (!parameters.resume || parameters.no_projections <= size(w,2))
      return;
    endif
  endif


  if (isempty(x_tr))
    if (!isfield(parameters, "data_file") || !exist(parameters.data_file,"file"))
      error("Learn_projections is called without a dataset and without a data file");
    endif
    load(parameters.data_file, "x_tr","y_tr")
  endif

  if (parameters.onlycorrect)
    if (!isfield(parameters, "ova_preds_file") || !exist(parameters.ova_preds_file,"file"))
      warning("Learn_projections is called with onlycorrect but without without a ova predictions file. Disabling onlycorrect.");
      parameters.onlycorrect=false;
    endif
    if (size(y_tr,2)!=1)
      %% currently the ability to eliminate the mistakes from the training of 
      %% the projection only works for the multiclass problems.
      warning("Learn_projections: onlycorrect is not implemented with multilabl problems. Disabling onlycorrect");
      parameters.onlycorrect=false;
    endif
  endif

  if (parameters.onlycorrect)
    load(parameters.ova_preds_file, "out_tr");
    noties = (out_tr + 1e-9 * randn(size(out_tr)))';
    [ignore,pred] = max(noties, [],1);
    if size(y_tr,1) == size(pred,1)
      correct=(y_tr == pred);
    else
      correct=(y_tr == (pred'));
    endif
    x_tr_proj=x_tr(correct,:);
    y_tr_proj=y_tr(correct);
  else    
    x_tr_proj=x_tr;
    y_tr_proj=y_tr;  
  endif

  if (parameters.C1multiplier == 1)
    if (size(y_tr_proj,2) == 1)
      noClasses = max(y_tr_proj);
    else
      noClasses = size(y_tr_proj,2);
    endif
    parameters.C1 = parameters.C2*parameters.C1*noClasses;
  endif  

  if (parameters.C1multiplier == 2)
    if (size(y_tr_proj,2) == 1)
      noClasses = max(y_tr_proj);
    else
      noClasses = size(y_tr_proj,2);
    endif
    parameters.C1 = parameters.C1*noClasses;
  endif  

  ## consistency checks for resume
  if (parameters.resume)
    if(exist(parameters.resume_from,"file"))
      load(parameters.resume_from, "w", "min_proj", "max_proj", "obj_val", "w_noavg", "min_proj_noavg", "max_proj_noavg", "obj_val_noavg");
    else
      warning("Resume is true, but resume_from file does not exist. Starting from scratch");
      parameters.resume = false;
    endif
  endif

  if (!exist("w","var"))
    parameters.resume = false;
  endif

  if (parameters.resume)
    assert(size(w,1) == size(x_tr,2))
    if (size(y_tr,2) == 1) #multi-class problem
      assert(size(min_proj,1) == max(y_tr));
    else
      assert(size(min_proj,1) == size(y_tr,2));
    endif
  endif

  disp("-----------------------------------------------------");
  best_obj=Inf;
  for r=1:parameters.restarts
    disp(sprintf("restart %d\n", r));
    if ( parameters.resume )
      if (exist("w_noavg","var"))
	[w,min_proj,max_proj, obj_val, w_noavg, min_proj_noavg, max_proj_noavg, obj_val_noavg]=oct_find_w(x_tr_proj,y_tr_proj,parameters,w,min_proj, max_proj, obj_val, w_noavg, min_proj_noavg, max_proj_noavg, obj_val_noavg);
      elseif (exist("w","var"))
	[w,min_proj,max_proj, obj_val,w_noavg, min_proj_noavg, max_proj_noavg, obj_val_noavg]=oct_find_w(x_tr_proj,y_tr_proj,parameters,w,min_proj, max_proj, obj_val);
      endif
    else 
      #w=init_w(2, x_tr_proj, y_tr_proj, size(x_tr_proj,2), parameters.no_projections);
      [w,min_proj,max_proj, obj_val, w_noavg, min_proj_noavg, max_proj_noavg, obj_val_noavg]=oct_find_w(x_tr_proj,y_tr_proj,parameters);
    endif    
    
    if (isempty(obj_val) || (best_obj > obj_val(length(obj_val))))
      best_w = w;
      best_min_proj = min_proj;
      best_max_proj = max_proj;
      best_obj_val = obj_val;
      ##  store the noavg values for the best run. This is not the best noavg run
      best_w_noavg = w_noavg;
      best_min_proj_noavg = min_proj_noavg;
      best_max_proj_noavg = max_proj_noavg;
      best_obj_val_noavg = obj_val_noavg;
      if (isempty(obj_val))
	best_obj=-1;
      else
	best_obj = obj_val(length(obj_val));
      endif
    end
  end
  
  w = best_w;
  min_proj = best_min_proj;
  max_proj = best_max_proj;
  obj_val = best_obj_val;
  w_noavg = best_w_noavg;
  min_proj_noavg = best_min_proj_noavg;
  max_proj_noavg = best_max_proj_noavg;
  obj_val_noavg = best_obj_val_noavg;
  
  %% plot the objective value vs iteration 	   
  if (parameters.plot_objval)
    figure("visible","off");
    plot(obj_val, "b", "LineWidth", 3);
    
    print("-dpdf", parameters.obj_plot_file)
  end
  
  if (length(parameters.projection_file) >= 255)
    parameters.projection_file = sprintf("%s_%d",strtrunc(parameters.projection_file,248),unidrnd(1e5));
  endif
     

  save(parameters.projection_file, "w", "min_proj", "max_proj" , "obj_val", "w_noavg", "min_proj_noavg", "max_proj_noavg", "obj_val_noavg");

  return; 
end
