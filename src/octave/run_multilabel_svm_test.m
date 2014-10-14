function [class_acc,class_acc_proj,proj_lbl_ignore,proj_lbl_ignore_percent] = ...
      run_multilabel_svm_test(x_tr, y_tr, x_te, y_te, ...
			      datadir, exp_name, ...
			      proj_exp_name, ...
			      ova_params, proj_params, restarts, resume, ...
			      cluster_params, ...
			      full_svm = 1, projected_svm=0, onlycorrect=0, ...
			      relearn_projection = 1)
  

  probs = 0;
  close all
  
  ## maybe the place of the random projections is not here
  ## if random projections are desired they should be done during the 
  ## data preprocessing stage 
  
  ## if ~exist('do_random_projection', 'var')
  ##     do_random_projection = false;
  ## end
  
  ## if ( do_random_projection )
  ##     [R k] = random_project(size(x_tr,2),size(x_tr,1)+size(x_ts,1),1e-6);
  ##     x_tr = x_tr*R;
  ##     x_ts = x_ts*R; % (size(x_orig,1)+1:size(x_orig,1)+size(xtest_orig,1),:);
  ##     fprintf(1,"projection time: %f\n", toc);
  ## end    
  
  
  ## since the parallel svm is loading data from this file, it is safer if 
  ## the data is loaded here too since this will guarantee that the same
  ##  data is used.
  ## on  the other hand this is an extra disk access that might not be necessary
  ## so let's only load the data if it is not passed in
  
  ## it is important that the name of the variables match the ones in the file: x_tr, y_tr, x_te, y_te
  if (isempty(x_tr))
    [y_tr, x_tr, y_te, x_te] = load([datadir exp_name ".mat"], "y_tr","x_tr","y_te","x_te");
  end
  
  if(size(y_tr,2)==1)
    %% multiclass problem with labels in a vector
    %% assumes that the labels are 1:noClasses
    noClasses = max(y_tr);
  else
    noClasses=size(y_tr,2);
  end
   
  if( full_svm ) 
    %% performing svm in parallel: if exists uses previously trained models. Otherwise, trains the svm on cluster
    force_retrain = false;
    [out, out_tr] = perform_parallel_projected_multilabel_svm(exp_name, ova_params.C, noClasses, datadir, force_retrain,  "", "full", ova_params.threshold, ova_params.solver, ova_params.solverparam, cluster_params.n_batches, cluster_params.min_batch_size);
    [class_acc acc class_F1 F1 top5 top10] = ...
	evaluate_ova_model(y_te, [], out, ova_params.threshold);
  endif

  if (onlycorrect && full_svm && size(y_tr,2)==1)
    %% onlycorrect makes sense only if the full_svm has been trained
 
    %% currently the ability to eliminate the mistakes from the training of 
    %% the projection only works for the multiclass problems.

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
  
  tic;
  fprintf(1,"performing the projection ... \n");

  %% learn the projection
  projectionfile=sprintf("results/wlu_%s_C1_%d_C2_%d.mat",proj_exp_name, proj_params.C1, proj_params.C2);      
  if ( relearn_projection || ~exist(projectionfile, "file") )
    [w, min_proj, max_proj] = learn_projections(x_tr_proj, y_tr_proj, proj_params, proj_exp_name, restarts, resume, proj_params.plot_objval);
    relearn_prjection = true;
  else
    load(projectionfile,"w","max_proj","min_proj");
  end

  disp(norm(w));
  for j=1:size(w,2)-1
    for k=j+1:size(w,2)
      fprintf(1,"W%d * W%d = %f \n", j,k,w(:,j)'*w(:,k)/(norm(w(:,j))*norm(w(:,k))));
    end
  end
  
  fprintf(1,"learn projection time: %f\n", toc);
  tic;

  [projected_labels_te, projected_te] = project_data(x_te, noClasses, w, min_proj, max_proj);

  fprintf(1,"label selection time: %f\n", toc);
  
  disp(size(projected_labels_te));
  
  
  fprintf(1,"**************************************************************************\n");
 
  if ( full_svm )
    fprintf(1,">> class_acc: %f, class_F1: %f, acc: %f, top5: %f, top10: %f\n", class_acc, class_F1, acc, top5, top10);
  end

  fprintf(1,"C1, C2: %g, %g \n", proj_params.C1,proj_params.C2);
  
  for j=1:size(projected_labels_te,3)              
    all_lbl_preds = size(x_te,1)*noClasses;
    proj_lbl_ignore = all_lbl_preds - nnz(projected_labels_te(:,:,j));
        
    [class_acc_proj acc_proj class_F1_proj F1_proj top5_proj top10_proj] = ...
	evaluate_ova_model(y_te, projected_labels_te(:,:,j), out, ova_params.threshold);
           
    proj_lbl_ignore_percent = proj_lbl_ignore * 100 / all_lbl_preds;
        
    fprintf(1,"projection %d: number of predications to ignore is (%d) out of (%d): %f\n", ...
    j, proj_lbl_ignore, all_lbl_preds, proj_lbl_ignore_percent);
  
    fprintf(1,">> class_acc_proj: %f, class_F1_proj: %f, acc_proj: %f, F1_proj: %f, top5_proj: %f, top10_proj: %f\n", ...
    class_acc_proj, class_F1_proj, acc_proj, F1_proj, top5_proj, top10_proj);
    
  end
  
  if (projected_svm)      
    force_retrain = relearn_projection; #if the projection has changed then retrain the svms
    [out_proj_svm] = perform_parallel_projected_multilabel_svm(exp_name, ova_params.C, noClasses, datadir, force_retrain, projectionfile, sprintf("C1_%g_C2_%g",proj_params.C1,proj_params.C2), ova_params.threshold, ova_params.solver, ova_params.solverparam, cluster_params.n_batches, cluster_params.min_batch_size);

    [class_acc_proj_svm acc_proj_svm class_F1_proj_svm F1_proj_svm top5_proj_svm top10_proj_svm] = ...
	evaluate_ova_model(y_te, [], out_proj_svm, ova_params.threshold);
    
    fprintf(1,">> class_acc_proj_svm: %f, class_F1_proj_svm: %f, acc_proj_svm: %f, F1_proj_svm: %f, top5_proj_svm: %f, top10_proj_svm: %f\n", ...
	    class_acc_proj_svm, class_F1_proj_svm, acc_proj_svm, F1_proj_svm, top5_proj_svm, top10_proj_svm);
    
    
    [class_acc_proj_svm_proj acc_proj_svm_proj class_F1_proj_svm_proj F1_proj_svm_proj top5_proj_svm_proj top10_proj_svm_proj] = ...
	evaluate_ova_model(y_te, projected_labels_te(:,:,end), out_proj_svm, ova_params.threshold);
    
    fprintf(1,">> class_acc_proj_svm_proj: %f, class_F1_proj_svm_proj: %f, acc_proj_svm_proj: %f, F1_proj_svm_proj: %f, top5_proj_svm_proj: %f, top10_proj_svm_proj: %f\n", ...
	    class_acc_proj_svm_proj, class_F1_proj_svm_proj, acc_proj_svm_proj, F1_proj_svm_proj, top5_proj_svm_proj, top10_proj_svm_proj);
    
  endif   

  fprintf(1,"**************************************************************************\n");    
    
end

 