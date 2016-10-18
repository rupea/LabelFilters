function [projection_file_name] = learn_projection_db(proj_params)

  srcdir = "~/Research/mcfilter/src";
  addpath([srcdir "/octave/"])
  addpath([srcdir "/libsvm-3.17/matlab/"])
  addpath([srcdir "/liblinear-1.94/matlab/"])

  addpath("~/Research/mcfilter/edb_scripts")
  addpath("~/Research/mcfilter/edb_scripts/jsonlab-1.2")

  struct_levels_to_print(10);

  if (proj_params.resume && proj_params.reoptimize_LU)
    error("Resume and reoptimize_LU can not both be true");
  endif
  
  ## if the data and/or the ova predictions are specified in the form of 
  ## a db query, get them here so they can be added to the 
  ## proj_params 
  db_data_params = {};
  if (! isempty(proj_params.data_query))
    db_data_params = edb_query(proj_params.data_query);
    if (length(db_data_params) == 1 )
      db_data_params = db_data_params{1};
      proj_params.data_file = db_data_params.path;
      proj_params.data_params = db_data_params;
    elseif (length(db_data_params) == 0)
      error("Query for the data file returned no matches");
    else
      error("Query for the data file returned more than one match");
    endif
  else
    proj_params.data_params = proj_params.data_file;
  endif
  
  db_ova_preds_params = {};
  if (proj_params.onlycorrect)
    if (!isempty(proj_params.ova_preds_query))
      db_ova_preds_params = edb_query(proj_params.ova_preds_query);
      if (length(db_ova_preds_params) == 1 )
	db_ova_preds_params = db_ova_preds_params{1};
	proj_params.ova_preds_file = db_ova_preds_params.path;
	proj_params.ova_preds_params = db_ova_preds_params;
      elseif (length(db_ova_preds_params) == 0)
	error("Query for the ova preds file returned no matches");
      else
	error("Query for the ova preds file returned more than one match");
      endif
    else
      proj_params.ova_preds_params = proj_params.ova_preds_file;
    endif  
  else
    proj_params.ova_preds_query = "";
    proj_params.ova_preds_file = "";
  end
  

  db_reoptimize_LU_params = {};
## if we are reoptimizing LU get the starting file
  if (proj_params.reoptimize_LU)
    if (!isempty(proj_params.reoptimize_LU_query))
      db_reoptimize_LU_params = edb_query(proj_params.reoptimize_LU_query);
      if (length(db_reoptimize_LU_params) == 1 )
	db_reoptimize_LU_params = db_reoptimize_LU_params{1};
	proj_params.reoptimize_LU_file = db_reoptimize_LU_params.path;
	proj_params.reoptimize_LU_params = db_reoptimize_LU_params;
      elseif (length(db_reoptimize_LU_params) == 0)
	error("Query for the file to reoptimize LU from returned no matches");
      else
	error("Query for the file to reoptimize LU from returned more than one match");
      endif
    else
      proj_params.reoptimize_LU_params = proj_params.reoptimize_LU_file;
    endif  
  else
    proj_params.reoptimize_LU_query = "";
    proj_params.reoptmize_LU_file = "";
  end
  
  
  ## if we have a log file, try to get it from the DB to append to it
  ## if we relearn the projection, remove the log file entry from the DB
  ## if the file exists but it is not in the DB, keep it and append to it 
  
  db_log_entries = {};
  if (!strcmp(proj_params.log_file,"stdout"))
    db_proj_log_params = rmfield(proj_params, ["exp_name"; "log_name_fields"; "projection_name_fields"; "obj_plot_name_fields"; "relearn_projection"; "projection_file"; "obj_plot_file"; "ova_preds_file"; "ova_preds_query"; "reoptimize_LU_file"; "reoptimize_LU_query"; "projection_dir"; "log_dir"; "obj_plot_dir"; "data_file"; "data_query"; "resume_from"; "resumed_from"; "resume"]);
    db_proj_log_params.type = "log_file";
    
    db_log_entries = edb_query(db_proj_log_params);
    if (length(db_log_entries) == 1)
      ## remove the file from the current path
      db_log_entries = db_log_entries{1};
      if (proj_params.relearn_projection)
	## remove the log file from the database
	## we start anew 
	ds_remove(db_proj_log_params);
      else
	downloaded_ids = edb_download(db_proj_log_entries); #download/update the local file if necessary
	if (length(downloaded_ids) != 1)
	  warning("Found a previous log file in the db, but had an error downlaoding it. Maybe it was deleted. Continuing with an empty file.");
	endif
	proj_params.log_file = db_log_entries.path;
      endif 
    elseif (length(db_log_entries) > 1)
      error("There was more than one file in the database that matched the query");
    endif

    if (isempty(proj_params.log_file))
      if (isempty(proj_params.log_name_fields))
	proj_params.log_file = "stdout";
	warning("Neither log_file nor log_name_fields were specified. Using stdout for log")
      endif
      fname = edb_name(db_proj_log_params, strsplit(proj_params.log_name_fields));
      proj_params.log_file = fullfile(tilde_expand(proj_params.log_dir),fname);
    endif
      
    if (proj_params.relearn_projection && !strcmp(proj_params.log_file,"stdout") && exist(proj_params.log_file,"file"))
      unlink(proj_params.log_file);
    endif
  endif 
    
  ## all this to create a log 
  if (!strcmp(proj_params.log_file,"stdout"))
    [old_stdout, msg] = fopen(sprintf("tempstdout.%d",getpid()),"w");
    if (old_stdout < 0) 
      error(["Unable to open file " sprintf("tempstdout.%d",getpid()) " : " msg]);
    endif
    [log_file, msg] = fopen(proj_params.log_file, "a");
    if (log_file < 0) 
      error(["Unable to open file " proj_params.log_file " : " msg]);
    endif
    dup2(stdout,old_stdout);
    unlink(sprintf("tempstdout.%d",getpid()));
    dup2(log_file,stdout);
  endif
  
  disp("");
  disp("");
  disp("");
  disp(["------------log at " ctime(time())]);
  
  
  ## query if the result already exists in the database
  db_proj_file_params = rmfield(proj_params, [ "projection_dir"; "log_dir"; "obj_plot_dir";"log_name_fields"; "projection_name_fields"; "obj_plot_name_fields"; "exp_name"; "data_file"; "data_query"; "ova_preds_file"; "ova_preds_query"; "reoptimize_LU_file"; "reoptimize_LU_query"; "projection_file"; "relearn_projection"; "plot_objval"; "obj_plot_file"; "log_file"; "report_epoch"; "report_avg_epoch"; "num_threads"; "finite_diff_test_epoch"; "no_finite_diff_tests"; "finite_diff_test_delta"; "resume_from"; "resume"; "resumed_from"]);
  db_proj_file_params.type = "projection_file";
  
  db_proj_file = edb_query(db_proj_file_params); 
  if (length(db_proj_file) > 1) 
    error("More than one existing db entry matches the projection file parameters.");
  endif
  if (length(db_proj_file) == 1)
    db_proj_file = db_proj_file{1};
    if (proj_params.relearn_projection)
      edb_remove(db_proj_file);
      if (exist(db_proj_file.path, "file"))
	unlink(db_proj_file.path)
      endif
    else
      downloaded_ids = edb_download(db_proj_file); #download/update the local file if necessary
      if (length(downloaded_ids) != 1)
	error("Found the projection file in edb, but had an error downloading it. Maybe it was deleted from edb.");
      endif
      proj_params.projection_file = db_proj_file.path;
    endif
  else
    ## no db entry has been found
    db_proj_file = {};
  endif
  
  ## resumed_from shoud not be used in the query (i.e. if there is a matching entry it is irrelevant if it was resumed and where from)
  ## db_proj_file_params.resumed_from = "";
  
  if (isempty(proj_params.projection_file))
    if (isempty(proj_params.projection_name_fields))
      error("Neither projection_file nor projection_name_fields were specified")
    endif
    fname = edb_name(db_proj_file_params, strsplit(proj_params.projection_name_fields));
    proj_params.projection_file = fullfile(tilde_expand(proj_params.projection_dir),fname);
  endif
  
  
  ## if we already have the file 
  ## and we do not need to relearn the projection
  ## (otherwise the file would have been deleted before)
  ## so we just exit. 
  
  ## there remains a small problem if the file name we are asking for is not
  ## the same as the file name in the database, but ignore this for now
  ## we will solve this by adding a renaming facility 
  ## and rename the file we get with the name we asked for
  if (!isempty(db_proj_file) && exist(db_proj_file.path,"file"))
    ##exit();
    projection_file_name = db_proj_file.path;
    ## update the log file if it is in edb.
    if (!isempty(db_log_entries))
      edb_upload(db_log_entries);
    endif
    return;
  endif  
    
  if (proj_params.resume && !exist(proj_params.projection_file,"file"))
    db_proj_file_resume_params = rmfield(db_proj_file_params,["no_projections";"seed"]);

    db_proj_file_resume = edb_query(db_proj_file_resume_params);
    if (length(db_proj_file_resume) >= 1) 
      no_projs = mygetfields(db_proj_file_resume, {"params.no_projections"});
      [~,max_no_proj_idx] = max(no_projs);
      db_proj_file_resume = db_proj_file_resume{max_no_proj_idx};
      downloaded_ids = edb_download(db_proj_file_resume); #download/update the local file if necessary
      if (length(downloaded_ids) != 1)
	error("Error downloading the previous proj file from edb.");
      endif
      proj_params.resume_from = db_proj_file_resume.path;
      db_proj_file_params.resumed_from = db_proj_file_resume; ## will need to add as an ancestor
    else
      warning("Resume was true, but no file was found to resume from. Starting from scratch");
      proj_params.resume = false;
      proj_params.reoptimize_LU = false;
    endif
  endif
    
  ## download the needed files now, since we know we will need them. 
  if (! isempty(db_data_params))
    downloaded_ids = edb_download(db_data_params); #download/update the local file if necessary
    if (length(downloaded_ids) != 1)
      error("Error downloading the data file from edb.");
    endif
  endif
  
  if (!isempty(db_ova_preds_params))
    downloaded_ids = edb_download(db_ova_preds_params); #download/update the local file if necessary
    if (length(downloaded_ids) != 1)
      error("Error downloading the ova predictions file from edb.");
    endif
  endif
  
  if (!isempty(db_reoptimize_LU_params))
    downloaded_ids = edb_download(db_reoptimize_LU_params); #download/update the local file if necessary
    if (length(downloaded_ids) != 1)
      error("Error downloading the reoptimize_LU file from edb.");
    endif
  endif
    
  proj_params
  [w min_proj max_proj] = learn_projections(proj_params);
  
  if (size(w,2) < proj_params.no_projections)
    ## This protects against resume=true and something going wrong
    if (exist(proj_params.projection_file, "file"))
      unlink(proj_params.projection_file);
    endif
    error("Number of projection components smaller than requested. Something went wrong. Deleting the file.");
  endif
  
  ## add the generated files to the results database
  if (exist(proj_params.projection_file, "file"))
    edb_put(proj_params.projection_file, db_proj_file_params);
    edb_upload(proj_params.projection_file);
  endif
    
  ## add the log file
  
  ## turn of the diary so that the log file does not change any more
  if (!strcmp(proj_params.log_file,"stdout"))
    dup2(old_stdout,stdout);
    ##diary off 
  endif
  
  if (!strcmp(proj_params.log_file,"stdout") && exist(proj_params.log_file,"file"))
    if (isempty(db_log_entries))
      edb_put(proj_params.log_file, db_proj_log_params);
    endif
    edb_upload(proj_params.log_file);
  end
  
  projection_file = proj_params.projection_file;
  return;
end
