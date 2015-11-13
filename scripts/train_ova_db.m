function ova_filename = train_ova_db(input_params)

  filename_fields = strsplit(input_params.filename_fields);
  filter_name_fields = strsplit(input_params.filter_name_fields);

  ova_params = rmfield(input_params,["n_batches";"min_batch_size";"data_query";"data_file";"filter_query";"retrain";"filter_file";"filter_name";"filter_name_fields";"filename";"filename_fields"]);
  
  if (!ova_params.keep_out)
    ##threshold is irrelevant
    ova_params = rmfield(ova_params,"threshold");
  endif

  ova_params.type = "ova_model";
  ova_params.model_type = "svm";
  

  if (! isempty(input_params.data_query))
    db_data_entries = ds_query(input_params.data_query);
    if (length(db_data_entries) == 1 )
      db_data_entries = db_data_entries{1};
      input_params.data_file = db_data_entries.db_path;
      ova_params.data = db_data_entries.params;
    elseif (length(db_data_entries) == 0)
      error("Query for the data file returned no matches");
    else
      error("Query for the data file returned more than one match");
    endif
  else
    if (isempty(input_params.data_file))
      error("No data file specified");
    endif
    ova_params.data.type = "local_file";
    ova_params.data.filename = input_params.data_file;
    
    local_filename_fields = filename_fields(!strncmp(filename_fields,"data.",5));
    time_field = any(strcmp(local_filename_fields,"__TIME"));
    local_filename_fields = local_filename_fields(!strcmp(local_filename_fields,"__TIME"));
    hash_field = any(strcmp(local_filename_fields,"__HASH"));
    local_filename_fields = local_filename_fields(!strcmp(local_filename_fields,"__HASH"));
    local_filename_fields(end+1) = "data.type";
    if (time_field)
      local_filename_fields(end+1) = "__TIME";
    endif
    if (hash_field)
      local_filename_fields(end+1) = "__HASH";
    endif
    filename_fields = local_filename_fields;
  endif
  
  input_params.filter_str = input_params.filter_name;
  if (! isempty(input_params.filter_query))
    db_filter_entries = ds_query(input_params.filter_query);
    if (length(db_filter_entries) == 1 )
      db_filter_entries = db_filter_entries{1};
      input_params.filter_file = db_filter_entries.db_path;
      ova_params.filter = db_filter_entries.params;
      if (!isempty(input_params.filter_name))
	input_params.filter_str = input_params.filter_name;
      else
	input_params.filter_str = ds_name(db_filter_entries.params, filter_name_fields);
      endif
    elseif (length(db_filter_entries) == 0)
      error("Query for the data file returned no matches");
    else
      error("Query for the data file returned more than one match");
    endif
  elseif (!isempty(input_params.filter_file))  
    ova_params.filter.type = "local_file";
    ova_params.filter.filename = input_params.filter_file;
    if (!exist(input_params.filter_file))
      error("Required local filter file does not exist");
    endif
    if (isempty(input_params.filter_name))
      input_params.filter_str = "local_file";
    endif
  else
    ova_params.filter.type = "no_filter";
    if (isempty(input_params.filter_name))
      input_params.filter_str = "full";
    endif
  endif
  
  if (strcmp(ova_params.filter.type,"local_file") || strcmp(ova_params.filter.type,"no_filter"))
    local_filename_fields = filename_fields(!strncmp(filename_fields,"filter.",7));
    time_field = any(strcmp(local_filename_fields,"__TIME"));
    local_filename_fields = local_filename_fields(!strcmp(local_filename_fields,"__TIME"));
    hash_field = any(strcmp(local_filename_fields,"__HASH"));
    local_filename_fields = local_filename_fields(!strcmp(local_filename_fields,"__HASH"));
    local_filename_fields(end+1) = "filter.type";
    if (time_field)
      local_filename_fields(end+1) = "__TIME";
    endif
    if (hash_field)
      local_filename_fields(end+1) = "__HASH";
    endif
    filename_fields = local_filename_fields;
  endif
  
  
  db_ova_entries = ds_query(ova_params); 
  if (length(db_ova_entries) > 1) 
    error("More than one existing db entry matches the ova parameters.");
  endif
  if (length(db_ova_entries) == 1)
    db_ova_entries = db_ova_entries{1};
    if (input_params.retrain)
      ds_unlink(db_ova_entreis.db_path);
      ds_remove(ova_params);
      unlink(db_ova_entries.params.filename);
    else
      ova_filename = db_ova_entries.params.filename;
      return;
    endif
  endif 
  
  ## get the data and the filter file if not already local
  if (! isempty(input_params.data_query) && !exist(db_data_entries.db_path))
    db_data_entries_get = ds_get(input_params.data_query);
    ## need to check if nothing has changed in the db 
    ## between the time we queried for the data file and we 
    ## got the datafile
    if (length(db_data_entreis_get) == 1 )
      db_data_entreis_get = db_data_entreis_get{1};
      if (!isequal(db_data_entries_get,db_data_entries))
	exit("DB entry for the data has changed between querying and getting it.");
      endif
    elseif (length(db_data_entries_get) == 0)
      error("The data file was removed from the DB");
    else
      error("Too many DB entries match the data file query");
    endif
  endif
  
  if (!isempty(input_params.filter_query) && !exist(db_filter_entries.db_path))
    db_filter_entries_get = ds_get(input_params.filter_query);
    ## need to check if nothing has changed in the db 
    ## between the time we queried for the data file and we 
    ## got the datafile
    if (length(db_filter_entries_get) == 1 )
      db_filter_entries_get = db_filter_entries_get{1};
      if (!isequal(db_filter_entries_get,db_filter_entries))
	exit("DB entry for the data has changed between querying and getting it.");
      endif
    elseif (length(db_filter_entries_get) == 0)
      error("The data file was removed from the DB");
    else
      error("Too many DB entries match the data file query");
    endif
  endif
  
  
  [data_dir, data_file] = fileparts(input_params.data_file);
  if (isempty(data_dir))
    data_dir = "./"
  else
    data_dir = [data_dir "/"];
  endif
  
  
  if (isempty(input_params.threshold))
    thresh_str = "none";
  else
    thresh_str = num2str(input_params.threshold);
  end
  
  if (isempty(input_params.filename)) 
    input_params.filename = ["svm_results/" ds_name(ova_params, filename_fields)];
  endif
  
  
  if (!isempty(input_params.filename) && exist(input_params.filename,"file"))
    ova_filename = input_params.filename;
  else    
    if (isempty(input_params.filter_name))
      input_params.filter_str = [input_params.filter_str "__" ds_name(ova_params, {"__HASH"})];
    endif      
    ova_filename = ["svm_results/svm_" data_file "_C" num2str(ova_params.C) "_threshold" thresh_str "_projected_" input_params.filter_str ".mat"];  
    mkdir("svm_results");  
    perform_parallel_projected_multilabel_svm(data_file, input_params.C, [], data_dir, input_params.retrain, input_params.filter_file, input_params.filter_str, input_params.threshold, input_params.solver, input_params.solverparam, input_params.n_batches, input_params.min_batch_size, input_params.sparsemodel, input_params.keep_out, input_params.wfilemap);      
  endif
  
  if (!isempty(input_params.filename) && !strcmp(input_params.filename,ova_filename))
    movefile(ova_filename,input_params.filename,"f")
    if (input_params.wfilemap)
      movefile([ova_filename ".wmap"], [input_params.filename ".wmap"], "f");
    endif
    ova_filename = input_params.filename;
  endif
  
  ova_params.filename = ova_filename;

  if (exist(ova_params.filename,"file") && (!input_params.wfilemap || exist([ova_params.filename ".wmap"],"file")))
    ## put a dummy file in the db
    dbfile = [ova_params.filename ".dbstore"];
    fid = fopen(dbfile,"w");
    fputs(fid,ova_params.filename);
    fclose(fid);  
    ds_add(dbfile, ova_params);
  endif
  
  ova_filename = ova_params.filename;
  return;
end
