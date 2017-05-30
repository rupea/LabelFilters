function [out_final, out_final_tr, svm_models_final] = perform_parallel_projected_multilabel_svm(out_file, data_file, C, noClasses = -1, force_retrain = false, projection_file = "", threshold = [], solver = "liblinear", solverparam = "-s 3", weights_threshold = -1, n_batches = 1000, min_batch_size = 10, sparsemodel = false, keep_out = false, wfilemap = false)

  if (isempty(projection_file))
    projection = false;
  else
    projection = true;
  end
  
  if (projection && ~exist(projection_file, "file"))
    error("Projection file does not exist");
    projection = false;
  end
    
  if (noClasses < 0)
    load(data_file, "-v6");
    if (size(y_tr,2) == 1)
      noClasses = max(cat(1,y_tr,y_te));
    else
      noClasses = size(y_tr,2);
    endif
  endif
  
  
  retrain = true;
  
  if exist(out_file,"file") && ~force_retrain
    retrain = false;
    load(out_file, "out_final", "out_final_tr", "svm_models_final");
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
    

    file_expr = @(idx) [out_file "_" num2str(label_range(idx)) "_" num2str(label_range(idx+1)-1)];    
    
    cur_file = @(idx) [file_expr(idx) ".mat"];    
    
    octave_cmd = @(idx) sprintf("octave -q --path %s --eval \"perform_projected_multilabel_svm_batch('%s', '%s', '%s', %f, %d, %d, '%s', '%s', '%s', %f, %d, %d)\"", ...
    path(), cur_file(idx), data_file, projection_file, C, label_range(idx), label_range(idx+1)-1, num2str(threshold), solver, solverparam, weights_threshold, sparsemodel, keep_out);
    
    pbs_command = @(idx) ["#!/bin/bash \n" ...
    "#PBS -l nodes=1:ppn=1:mlc,walltime=10:00:00 \n" ...
    "#PBS -N svm_" data_file " \n" ...
    "#PBS -o localhost:${PBS_O_WORKDIR}/" file_expr(idx) ".out \n" ...
    "#PBS -e localhost:${PBS_O_WORKDIR}/" file_expr(idx) ".err \n" ...
    "#PBS -V \n" ...
    "cd ${PBS_O_WORKDIR} \n" octave_cmd(idx)]; % .${PBS_JOBID}
     
    for lbl_idx = 1 : 1 : length(label_range) -1
	if (~exist(cur_file(lbl_idx),'file') || force_retrain)
	  if (exist(cur_file(lbl_idx),'file'))
	    unlink(cur_file(lbl_idx));
	  endif
          f = [file_expr(lbl_idx) ".pbs"];       
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
    
    [out_final, out_final_tr, svm_models_final] = multilabel_svm_merge_batches(out_file, label_range, noClasses, cur_file, sparsemodel, keep_out, wfilemap);    
    
    disp('all done ...');                  

  end

  return;
end
