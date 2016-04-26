function learn_projection_db(proj_params, database)

  struct_levels_to_print(10);


  if (!exist(".ds","file"))
    if (isempty(database))
      error(sprintf("No DB specified in the command and folder %s is not connected to a DB", pwd()));
    endif
    ds_connect(database);
  endif 
  
  ## if the data and/or the ova predictions are specified in the form of 
  ## a db query, get them here so they can be added to the 
  ## proj_params 
  if (! isempty(proj_params.data_query))
    db_data_params = ds_query(proj_params.data_query);
    if (length(db_data_params) == 1 )
      db_data_params = db_data_params{1};
      proj_params.data_file = db_data_params.db_path;
      proj_params.data_params = db_data_params.params;
    elseif (length(db_data_params) == 0)
      error("Query for the data file returned no matches");
    else
      error("Query for the data file returned more than one match");
    endif
  else
    proj_params.data_params.type = "local_file";
    proj_params.data_params.filename = proj_params.data_file;
  endif
  
  if (proj_params.onlycorrect)
    if (!isempty(proj_params.ova_preds_query))
      db_ova_preds_params = ds_query(proj_params.ova_preds_query);
      if (length(db_ova_preds_params) == 1 )
	db_ova_preds_params = db_ova_preds_params{1};
	proj_params.ova_preds_file = db_ova_preds_params.db_path;
	proj_params.ova_preds_params = db_ova_preds_params.params;
      elseif (length(db_ova_preds_params) == 0)
	error("Query for the ova preds file returned no matches");
      else
	error("Query for the ova preds file returned more than one match");
      endif
    else
      proj_params.ova_preds_params.type = "local_file";
      proj_params.ova_preds_params.filename = proj_params.ova_preds_file;
    endif  
  else
    proj_params.ova_preds_query = "";
    proj_params.ova_preds_file = "";
  end
  

## if we are reoptimizing LU get the starting file
  if (proj_params.reoptimize_LU)
    if (!isempty(proj_params.reoptimize_LU_query))
      db_reoptimize_LU_params = ds_query(proj_params.reoptimize_LU_query);
      if (length(db_reoptimize_LU_params) == 1 )
	db_reoptimize_LU_params = db_reoptimize_LU_params{1};
	proj_params.reoptimize_LU_file = db_reoptimize_LU_params.db_path;
	proj_params.reoptimize_LU_params = db_reoptimize_LU_params.params;
      elseif (length(db_reoptimize_LU_params) == 0)
	error("Query for the file to reoptimize LU from returned no matches");
      else
	error("Query for the file to reoptimize LU from returned more than one match");
      endif
    else
      proj_params.reoptimize_LU_params.type = "local_file";
      proj_params.reoptimize_LU_params.filename = proj_params.ova_preds_file;
    endif  
  else
    proj_params.reoptimize_LU_query = "";
    proj_params.reoptmize_LU_file = "";
  end
  
  
  ## if we have a log file, try to get it from the DB to append to it
  ## if we relearn the projection, remove the log file entry from the DB
  ## if the file exists but it is not in the DB, keep it and append to it 
  
  if (!strcmp(proj_params.log_file,"stdout"))
    db_proj_log_params = rmfield(proj_params, ["exp_name"; "log_name_fields"; "projection_name_fields"; "obj_plot_name_fields"; "relearn_projection"; "projection_file"; "obj_plot_file"; "ova_preds_file"; "ova_preds_query"; "reoptimize_LU_file"; "reoptimize_LU_query"; "projection_dir"; "log_dir"; "obj_plot_dir"; "data_file"; "data_query"; "resume_from"; "resumed_from"; "resume"]);
    db_proj_log_params.type = "log_file";
    
    db_log_entries = ds_query(db_proj_log_params);
    if (length(db_log_entries) == 1)
      ## remove the file from the current path
      db_log_entries = db_log_entries{1};
      if (exist(db_log_entries.db_path,"file"))
	ds_unlink(db_log_entries.db_path);
      endif         
      if (proj_params.relearn_projection)
	## remove the log file from the database
	## we start anew 
	ds_remove(db_proj_log_params);
      else
	ds_get(db_proj_log_params);
	proj_params.log_file = db_log_entries.db_path;
      endif 
    elseif (length(db_log_entries) > 1)
      error("There was more than one file in the database that matched the query");
    endif

    if (isempty(proj_params.log_file))
      if (isempty(proj_params.log_name_fields))
	proj_params.log_file = "stdout";
	warning("Neither log_file nor log_name_fields were specified. Using stdout for log")
      endif
      fname = ds_name(db_proj_log_params, strsplit(proj_params.log_name_fields));
      proj_params.log_file = fullfile(tilde_expand(proj_params.log_dir),fname);
    endif
      
    if (proj_params.relearn_projection && !strcmp(proj_params.log_file,"stdout") && exist(proj_params.log_file,"file"))
      unlink(proj_params.log_file);
    endif
  endif 
  
  ## need this here since we are using the log file below
  ## using autopull now
  ## ds_pull();
  
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
    ##diary(proj_params.log_file);
  endif
  
  disp("");
  disp("");
  disp("");
  disp(["------------log at " ctime(time())]);
  
  
  ## query if the result already exists in the database
  db_proj_file_params = rmfield(proj_params, [ "projection_dir"; "log_dir"; "obj_plot_dir";"log_name_fields"; "projection_name_fields"; "obj_plot_name_fields"; "exp_name"; "data_file"; "data_query"; "ova_preds_file"; "ova_preds_query"; "reoptimize_LU_file"; "reoptimize_LU_query"; "projection_file"; "relearn_projection"; "plot_objval"; "obj_plot_file"; "log_file"; "report_epoch"; "report_avg_epoch"; "num_threads"; "finite_diff_test_epoch"; "no_finite_diff_tests"; "finite_diff_test_delta"; "resume_from"; "resume"; "resumed_from"]);
  db_proj_file_params.type = "projection_file";
  
  db_proj_file = ds_query(db_proj_file_params); 
  if (length(db_proj_file) > 1) 
    error("More than one existing db entry matches the projection file parameters.");
  endif
  if (length(db_proj_file) == 1)
    db_proj_file = db_proj_file{1};
    if (exist(db_proj_file.db_path))
      ds_unlink(db_proj_file.db_path);
    endif
    if (proj_params.relearn_projection)
      ds_remove(db_proj_file_params);
    else
      ds_get(db_proj_file_params); #assumes the that the db has not changed
      proj_params.projection_file = db_proj_file.db_path;
    endif
  else
    ## no db entry has been found
    db_proj_file = [];
  endif

  ## resumed_from shoud not be used in the query (i.e. if there is a matching entry it is irrelevant if it was resumed and where from)
  db_proj_file_params.resumed_from = "";

  if (isempty(proj_params.projection_file))
    if (isempty(proj_params.projection_name_fields))
      error("Neither projection_file nor projection_name_fields were specified")
    endif
    fname = ds_name(db_proj_file_params, strsplit(proj_params.projection_name_fields));
    proj_params.projection_file = fullfile(tilde_expand(proj_params.projection_dir),fname);
  endif

  
  ## needed here because of the potential remove below
  ## should be better structured 
  ##ds_pull(); 
  
  db_obj_plot_file_params = rmfield(proj_params, [ "projection_dir"; "log_dir"; "obj_plot_dir";"log_name_fields"; "projection_name_fields"; "obj_plot_name_fields"; "exp_name"; "ova_preds_file"; "ova_preds_query"; "reoptimize_LU_file"; "reoptimize_LU_query"; "data_file"; "data_query"; "obj_plot_file"; "projection_file"; "log_file"; "relearn_projection"; "plot_objval"; "num_threads"; "finite_diff_test_epoch"; "no_finite_diff_tests"; "finite_diff_test_delta"; "resume_from"; "resume"; "resumed_from"]);
  db_obj_plot_file_params.type = "obj_plot";
  
  db_obj_plot_file = ds_query(db_obj_plot_file_params); 
  if (length(db_obj_plot_file) > 1) 
    error("More than one existing db entry matches the obj plot file parameters.");
  endif
  if (length(db_obj_plot_file) == 1)
    db_obj_plot_file = db_obj_plot_file{1};
    if (exist(db_obj_plot_file.db_path))
      ds_unlink(db_obj_plot_file.db_path);
    endif
    if (proj_params.relearn_projection)
      ds_remove(db_obj_plot_file_params);
    else
      ds_get(db_obj_plot_file_params);      
      proj_params.obj_plot_file = db_obj_plot_file.db_path;
    endif
  else
    ## no db entry has been found
    db_obj_plot_file = [];
  endif

  if (isempty(proj_params.obj_plot_file))
    if (proj_params.plot_objval && isempty(proj_params.obj_plot_name_fields))
      proj_params.plot_objval = false;
      warning("Neither obj_plot_file nor obj_plot_name_fields were specified so plot_objval is set to false")
    endif
    fname = ds_name(db_obj_plot_file_params, strsplit(proj_params.obj_plot_name_fields));
    proj_params.obj_plot_file = fullfile(tilde_expand(proj_params.obj_plot_dir),fname);
  endif

  
  ## we might exit after this, so get all the files that are 
  ## on the transacion list
  ##ds_pull();
  
  ## if we already have the file 
  ## and we do not need to relearn the projection
  ## (otherwise the file would have been deleted before)
  ## so we just exit. 
  
  ## there remains a small problem if the file name we are asking for is not
  ## the same as the file name in the database, but ignore this for now
  ## we will solve this by adding a renaming facility to ds_query and ds_get
  ## and rename the file we get with the name we asked for
  if (!isempty(db_proj_file) && exist(db_proj_file.db_path,"file"))
    exit();
  endif  
    
  if (proj_params.resume && !exist(proj_params.projection_file,"file"))
    db_proj_file_resume_params = rmfield(db_proj_file_params,["no_projections";"seed";"resumed_from"]);

    db_proj_file_resume = ds_query(db_proj_file_resume_params);
    if (length(db_proj_file_resume) >= 1) 
      no_projs = mygetfields(db_proj_file_resume, {"params.no_projections"});
      [~,max_no_proj_idx] = max(no_projs);
      db_proj_file_resume = db_proj_file_resume{max_no_proj_idx};
      if (!exist(db_proj_file_resume.db_path))
	ds_get(db_proj_file_resume.params);
      endif
      proj_params.resume_from = db_proj_file_resume.db_path;
      db_proj_file_params.resumed_from = db_proj_file_resume.params; ## will need to add as an ancestor
    else
      warning("Resume was true, but no file was found to resume from. Starting from scratch");
      proj_params.resume = false;
      proj_params.reoptimize_LU = false;
    endif
  endif
    
  if (! isempty(proj_params.data_query) && !exist(db_data_params.db_path))
    db_data_params_get = ds_get(proj_params.data_query);
    ## need to check if nothing has changed in the db 
    ## between the time we queried for the data file and we 
    ## got the datafile
    if (length(db_data_params_get) == 1 )
      db_data_params_get = db_data_params_get{1};
      if (!isequal(db_data_params_get,db_data_params))
	exit("DB entry for the data has changed between querying and getting it.");
      endif
    elseif (length(db_data_params_get) == 0)
      error("The data file was removed from the DB");
    else
      error("Too many DB entries match the data file query");
    endif
  endif
  
  
  
  if (proj_params.onlycorrect && !isempty(proj_params.ova_preds_query) && !exist(db_ova_preds_params.db_path))
    db_ova_preds_params_get = ds_get(proj_params.ova_preds_query);
    if (length(db_ova_preds_params_get) == 1 )
      db_ova_preds_params_get = db_ova_preds_params_get{1};
      if (!isequal(db_ova_preds_params_get,db_ova_preds_params))
	exit("DB entry for the OVA predictions has changed between querying and getting it.");
      endif
    elseif (length(db_ova_preds_params_get) == 0)
      error("The data file was removed from the DB");
    else
      error("Too many DB entries match the data file query");
    endif
  endif
  
  ##get all the files on the transaction list before the computation starts
  ## using autopull
  ## ds_pull();

  if (proj_params.reoptimize_LU && !isempty(proj_params.reoptimize_LU_query) && !exist(db_reoptimize_LU_params.db_path))
    db_reoptimize_LU_params_get = ds_get(proj_params.reoptimize_LU_query);
    if (length(db_reoptimize_LU_params_get) == 1 )
      db_reoptimize_LU_params_get = db_reoptimize_LU_params_get{1};
      if (!isequal(db_reoptimize_LU_params_get,db_reoptimize_LU_params))
	exit("DB entry for the OVA predictions has changed between querying and getting it.");
      endif
    elseif (length(db_reoptimize_LU_params_get) == 0)
      error("The data file was removed from the DB");
    else
      error("Too many DB entries match the data file query");
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
    ds_add(proj_params.projection_file, db_proj_file_params);
  endif
  
  ##needed here because of the ds_remove inside ds_add
  ## using autopush
  ##ds_push();
  
  if (proj_params.plot_objval && exist(proj_params.obj_plot_file,"file"))
    ds_add(proj_params.obj_plot_file, db_obj_plot_file_params);
  endif
  
  ##needed here because of the ds_remove inside ds_add
  ## using autopush
  ##ds_push();
  
  
  ## add the log file
  
  ## turn of the diary so that the log file does not change any more
  if (!strcmp(proj_params.log_file,"stdout"))
    dup2(old_stdout,stdout);
    ##diary off 
  endif
  
  if (!strcmp(proj_params.log_file,"stdout") && exist(proj_params.log_file,"file"))
    ds_add(proj_params.log_file, db_proj_log_params);
  end
  
  ## using autopush
  ##ds_push();
end
