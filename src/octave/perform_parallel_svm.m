function [out_final] = perform_parallel_svm(exp_name, C, option, exp_dir, force_retrain, tr_label)

if ~exist('option', 'var'),
    option = '';
end      
    
if ~exist('exp_dir', 'var'),
    exp_dir = '';
end      

filename = ["svm_results/svm_" exp_name "_" num2str(C) ".mat"];
disp(filename);

if ~exist('force_retrain', 'var'),
    force_retrain = false;
end

if exist(filename,"file") && ~force_retrain
    load(filename);    
else         
    if ~exist('tr_label', 'var')
        load([exp_dir exp_name ".mat"], "-v6", "tr_label");
    end
    
    noClasses = length(unique(tr_label));
    
    if noClasses < 10,
        batch_length = noClasses;
    else
        batch_length = floor(noClasses / 50);  % find the interval of the batch sizes
        if batch_length < 10,  % make sure the batches are not too small 
            batch_length = 10;
        end
    end
 
    label_range = 0 : batch_length : noClasses;
    if label_range(end) > noClasses, label_range(end)= noClasses; end;
    if label_range(end) < noClasses, label_range(end)= noClasses; end;
        
    file_expr = @(idx) [exp_name "_" num2str(C) "_" num2str(label_range(idx)) "_" num2str(label_range(idx+1))];    
    
    cur_file = @(idx) ["svm_results/svm_" file_expr(idx) ".mat"];    
    
    octave_cmd = @(idx) sprintf("octave -q --path %s --eval \"perform_svm_batch('%s','%s',%f,%d,%d)\"", ...
    path(), exp_name,exp_dir,C, label_range(idx),label_range(idx+1));
    
    pbs_command = @(idx) ["#!/bin/bash \n" ...
    "#PBS -l nodes=1:ppn=1,walltime=5:00:00 \n" ...
    "#PBS -N svm_" exp_name " \n" ...
    "#PBS -o localhost:${PBS_O_WORKDIR}/svm_results/" file_expr(idx) ".out \n" ...
    "#PBS -e localhost:${PBS_O_WORKDIR}/svm_results/" file_expr(idx) ".err \n" ...
    "#PBS -V \n" ...
    "cd ${PBS_O_WORKDIR} \n" octave_cmd(idx)]; % .${PBS_JOBID}
     
    for lbl_idx = 1 : 1 : length(label_range) -1
        f = ["svm_results/" file_expr(lbl_idx) ".pbs"];       
	display(f);
        fid = fopen(f, 'w');
        fprintf(fid, pbs_command(lbl_idx)); 
        fclose(fid);
        disp(file_expr(lbl_idx));
        system(["qsub " f]);
    end       
    
    disp('All jobs submitted. Now wait ...');            
    
    exp_name_substr = ["svm_" exp_name](1:10) % we use a substring of the exp_name since we are checking with the table and it mightbe trunctaed ...
    
    all_exist = false;
    while ~all_exist
        for lbl_idx = 1 : length(label_range) - 1
            all_exist = true;            
            if ~exist( cur_file(lbl_idx) )
                all_exist = false;
                break;
            end                
        end
        pause(10);
    end    
    
    out_final = svm_merge_batches(filename, label_range, noClasses, cur_file);    
    
    disp('all done ...');                  

    if ~exist('te_label', 'var')
        load([exp_dir exp_name ".mat"], "-v6", "te_label");
    end
    
    [class_acc class_F1 acc] = evaluate_svm_model(tr_label,te_label, [], out_final);
    
    fprintf(1,">> class_acc: %f, class_F1: %f, acc: %f\n", class_acc, class_F1, acc);
end


return;
