function [class_acc,class_acc_proj,proj_lbl_ignore,proj_lbl_ignore_percent] = run_svm_test(bestC, ...
x_orig, tr_label, xtest_orig, te_label, do_random_projection, projection_func,exp_name, proj_exp_name, datadir, proj_params, restarts, resume, projected_svm=0)
tic;

iterations = length(proj_params);

probs = 0;
close all

if ~exist('do_random_projection', 'var')
    do_random_projection = false;
end
  
if do_random_projection ,
    [R k] = random_project(size(x_orig,2),size(x_orig,1)+size(xtest_orig,1),1e-6);
    x = x_orig*R;
    xtest = xtest_orig*R; % (size(x_orig,1)+1:size(x_orig,1)+size(xtest_orig,1),:);
    fprintf(1,"projection time: %f\n", toc);
    clear x_orig, xtest_orig;	
else
    x = x_orig;
    clear x_orig;
    
    xtest = xtest_orig;
    clear xtest_orig;
end    

if( ~projected_svm ) 
  %% performing svm in parallel: if exists uses previously trained models. Otherwise, trains the svm on cluster
  %% perform_parallel_svm(exp_name, C, option, exp_dir, force_retrain, tr_label)
  [out] = perform_parallel_svm(exp_name,bestC,"",datadir,false,tr_label);
endif

for iter = 1 : iterations
    tic;
    fprintf(1,"performing the projection ... \n");
    [projected_labels,projected] = project_tests(x,tr_label,xtest,te_label, projection_func, proj_params(iter), proj_exp_name, restarts, resume);   
    
    
    fprintf(1,"projection and label selection time: %f\n", toc);
    tic;
    
    disp(size(projected_labels));
    
    for j=1:size(projected_labels,3)              
      proj_lbl_ignore(iter) = length(find(projected_labels(:,:,j)==0));
      all_lbl_preds = size(projected_labels,1)*size(projected_labels,2);
      
      
      [class_acc_proj(iter) class_F1_proj(iter) acc_proj(iter)] = ...
	  evaluate_svm_model(tr_label,te_label, projected_labels(:,:,j), out);
	    
      fprintf(1,"time to train/predict in svm: %f\n", toc);
      tic;
	      	    
	
      proj_lbl_ignore_percent(iter) = proj_lbl_ignore(iter) * 100 / all_lbl_preds;
    
    [class_acc(iter) class_F1(iter) acc(iter)] = ...
    evaluate_svm_model(tr_label,te_label, [], out);
             
    
    if (projected_svm)      
      projectionfile=sprintf("wlu_%s_C1_%d_C2_%d",proj_exp_name, proj_params(iter).C1, proj_params(iter).C2);
      [out_proj] = perform_parallel_projected_svm(proj_exp_name, bestC, "", datadir, projectionfile, "results/", sprintf("C1_%g_C2_%g",proj_params(iter).C1,proj_params(iter),C2), false, tr_label);
      [class_acc_proj_svm(iter) class_F1_proj_svm(iter) acc_proj_svm(iter)] = ...
	  evaluate_svm_model(tr_label,te_label, [], out_proj);
    endif 
    
    fprintf(1,"**************************************************************************\n");
    fprintf(1,"C1, C2: %g, %g \n", proj_params(iter).C1,proj_params(iter).C2);
    fprintf(1,"number of predications to ignore is (%d) out of (%d): %f\n", ...
    proj_lbl_ignore(iter), all_lbl_preds, proj_lbl_ignore_percent(iter));
  
    fprintf(1,">> class_acc_proj: %f, class_F1_proj: %f, acc_proj: %f\n", ...
    class_acc_proj(iter), class_F1_proj(iter), acc_proj(iter));
    
    fprintf(1,">> class_acc: %f, class_F1: %f, acc: %f\n", class_acc(iter), class_F1(iter), acc(iter));

    if (projected_svm)
      fprintf(1,">> class_acc_proj_svm: %f, class_F1_proj_svm: %f, acc_proj_svm: %f\n", ...
	      class_acc_proj_svm(iter), class_F1_proj_svm(iter), acc_proj_svm(iter));
    endif
    fprintf(1,"**************************************************************************\n");
    
end


	
# fprintf(1,'all class_acc, class_acc_proj, acc, acc_proj, percent_ignored:\n');
# disp([class_acc, class_acc_proj, acc, acc_proj, proj_lbl_ignore_percent]);
# fprintf(1,"mean(class_acc): %f, mean(class_acc_proj): %f\n", mean(class_acc), mean(class_acc_proj));
# fprintf(1,"mean(acc): %f, mean(acc_proj): %f, mean(percent_ignored): %f\n", mean(acc), mean(acc_proj), proj_lbl_ignore_percent(iter));
end

						    