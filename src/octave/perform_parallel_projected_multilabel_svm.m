function [out_final, out_final_tr, svm_models_final] = perform_parallel_projected_multilabel_svm(exp_name, C,noClasses, exp_dir, force_retrain, projection_file = "", project_str = "", threshold = [], solver = "libsvm", solverparam = "-t 0", n_batches = 1000, min_batch_size = 10)

  if ~exist('exp_dir', 'var'),
    exp_dir = '';
  end      

  if ~exist('force_retrain', 'var'),
    force_retrain = false;
  end


  if (isempty(projection_file))
    projection = false;
  else
    projection = true;
  end
  
  if (projection && ~exist(projection_file, "file"))
    error("Projection file does not exist");
    projection = false;
  end
  
  if ( ~projection )
    if (~exist("project_str","var") || isempty(project_str))
      project_str = "full";
    end
  end
  
  if (isempty(threshold))
    thresh_str = "none";
  else
    thresh_str = num2str(threshold);
  end
  
  filename = ["svm_results/svm_" exp_name "_C" num2str(C) "_threshold" thresh_str "_projected_" project_str ".mat"];
  disp(filename);
  
  retrain = true;
  
  if exist(filename,"file") && ~force_retrain
    retrain = false;
    load(filename, "out_final", "out_final_tr", "svm_models_final");
    if ( ~exist("out_final_tr","var") )
      retrain = true;
    end
    if ( ~exist("out_final","var") )
      retrain = true
    end
    if ( ~exist("svm_models_final", "var") )
      retrain = true
    end
  end

  if (retrain)
    batch_length = floor(noClasses / n_batches);  % find the interval of the batch sizes
    if batch_length < min_batch_size,  % make sure the batches are not too small 
      batch_length = min_batch_size;
    end
    if (noClasses < batch_length)
      batch_length = noClasses;
    end
 
    label_range = 1 : batch_length : (noClasses+1);
    label_range(end) = noClasses+1;
    

    file_expr = @(idx) [exp_name "_C" num2str(C) "_" num2str(label_range(idx)) "_" num2str(label_range(idx+1)-1) "_threshold" thresh_str "_projected_" project_str];    
    
    cur_file = @(idx) ["svm_results/svm_" file_expr(idx) ".mat"];    
    
    octave_cmd = @(idx) sprintf("octave -q --path %s --eval \"perform_projected_multilabel_svm_batch('%s','%s', '%s', '%s' ,%f, %d, %d, '%s', '%s', '%s')\"", ...
    path(), exp_name, exp_dir,projection_file, project_str, C, label_range(idx),label_range(idx+1)-1, num2str(threshold), solver, solverparam);
    
    pbs_command = @(idx) ["#!/bin/bash \n" ...
    "#PBS -l nodes=1:ppn=1,walltime=10:00:00 \n" ...
    "#PBS -N svm_" exp_name " \n" ...
    "#PBS -o localhost:${PBS_O_WORKDIR}/svm_results/" file_expr(idx) ".out \n" ...
    "#PBS -e localhost:${PBS_O_WORKDIR}/svm_results/" file_expr(idx) ".err \n" ...
    "#PBS -V \n" ...
    "cd ${PBS_O_WORKDIR} \n" octave_cmd(idx)]; % .${PBS_JOBID}
     
    for lbl_idx = 1 : 1 : length(label_range) -1
	if (~exist(cur_file(lbl_idx),'file') || force_retrain)
	  if (exist(cur_file(lbl_idx),'file'))
	    unlink(cur_file(lbl_idx));
	  endif
          f = ["svm_results/" file_expr(lbl_idx) ".pbs"];       
	  display(f);
          fid = fopen(f, 'w');
          fprintf(fid, pbs_command(lbl_idx)); 
          fclose(fid);
          disp(file_expr(lbl_idx));
          system(["qsub " f]);
	  pause(0.5);
	end
    end       
    
    disp('All jobs submitted. Now wait ...');            
    
    all_exist = false;
    while ~all_exist
        all_exist = true;            
        for lbl_idx = 1 : length(label_range) - 1
            if ~exist( cur_file(lbl_idx) )
                all_exist = false;
                break;
            end                
        end
        pause(10);
    end    
    
    [out_final, out_final_tr, svm_models_final] = multilabel_svm_merge_batches(filename, label_range, noClasses, cur_file);    
    
    disp('all done ...');                  

  end

  return;
end
