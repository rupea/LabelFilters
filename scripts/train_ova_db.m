# function to train an OVA model

# ova_params - a struct that holds parameters for training the ova model. These parameters will be added to the EDB. 
#              It should have a field named model_type that indicates the ova model (svm or nb for now)
#              before adding to the EDB, details for the training data and filter (if any) will be appended 

# additional_params - parameters that will not be included in the EDB. This include teh data query or data_file, 
#                     filter_query/file, n_batches and min_batch_size for parallel training of SVMs, whether to retrain
#                     the method, how to name the files, etc. 

function ova_filename = train_ova_db(ova_params, additional_params)
  
  function s=combine_structs(varargin) 
    for i=1:nargin 
      a=varargin{i};
      for [v,n]=a 
	s.(n)=v; 
      endfor 
    endfor 
  endfunction 


  if (isfield(additional_params, "filename_fields"))
    filename_fields = strsplit(additional_params.filename_fields);
  else
    filename_fields = {"__HASH"} #use the hash as a default file name
  endif
  
  if (isfield(additional_params, "filter_name_fields"))
    filter_name_fields = strsplit(additional_params.filter_name_fields);
  else
    filter_name_fields = {};
  endif
  
  ova_params.type = "ova_model";

  
  if (!isfield(ova_params,"keep_out") || !ova_params.keep_out)
    ##threshold is irrelevant
    if (isfield(ova_params, "threshold"))
      ova_params = rmfield(ova_params,"threshold");
    endif
  endif  

  db_data_entries = {};
  if (isfield(additional_params,"data_query") && !isempty(additional_params.data_query))
    db_data_entries = edb_query(additional_params.data_query);
    if (length(db_data_entries) == 1 )
      db_data_entries = db_data_entries{1};
      additional_params.data_file = db_data_entries.path;
      ova_params.data = db_data_entries;
    elseif (length(db_data_entries) == 0)
      error("Query for the data file returned no matches");
    else
      error("Query for the data file returned more than one match");
    endif
  elseif (isfield(additional_params,"data_file") && !isempty(additional_params.data_file))

    ## no more data substructure because we don't support complex keys at this time. 
    #ova_params.data.type = "local_file";
    #ova_params.data.filename = additional_params.data_file;

    ## store the filename in the data param instead 
    ova_params.data = additional_params.data_file;

    ## remove all the data components from the filed name since we do not have a db structure for data
    if (any(strncmp(filename_fields,"data.",5)))
      filename_fields = filename_fields(!strncmp(filename_fields,"data.",5));
      ## no more data substructure because we don't support complex keys at this time. 
      #filename_fields(end+1) = "data.type";
    endif
  else
    error("No data file specified");
  endif
  
  db_filter_entries = {};
  if (isfield(additional_params,"filter_query") && !isempty(additional_params.filter_query))
    db_filter_entries = edb_query(additional_params.filter_query);
    if (length(db_filter_entries) == 1 )
      db_filter_entries = db_filter_entries{1};
      additional_params.filter_file = db_filter_entries.path;
      ova_params.filter = db_filter_entries;
      if (isfield(additional_params,"filter_name") && !isempty(additional_params.filter_name))
	additional_params.filter_name = additional_params.filter_name;
      else
	additional_params.filter_name = edb_name(db_filter_entries, filter_name_fields, false);
      endif
    elseif (length(db_filter_entries) == 0)
      error("Query for the data file returned no matches");
    else
      error("Query for the data file returned more than one match");
    endif
  elseif (isfield(additional_params,"filter_file") && !isempty(additional_params.filter_file))  
    ## no more filter substructure because we don't support complex keys at this time. 
    ##ova_params.filter.type = "local_file";
    ##ova_params.filter.filename = additional_params.filter_file;

    ## store the filter filename in the filter param instead
    ova_params.filter = additional_params.filter_file;

    if (!exist(additional_params.filter_file))
      error("Required local filter file does not exist");
    endif
    if (!isfield(additional_params,"filter_name") || isempty(additional_params.filter_name))
      additional_params.filter_name = "local_file";
    endif
  else
    ## no more filter substructure because we don't support complex keys at this time. 
    ##ova_params.filter.type = "no_filter";
    ova_params.filter = "no_filter";

    if (!isfield(additional_params,"filter_name") || isempty(additional_params.filter_name))
      additional_params.filter_name = "full";
    endif
  endif

  ## no more data substructure because we don't support complex keys at this time. 
  #if (strcmp(ova_params.filter.type,"local_file") || strcmp(ova_params.filter.type,"no_filter"))
  if (!isstruct(ova_params.filter))
    ## remove all the filter components from the filed name since we do not have a db structure for filter
    if (any(strncmp(filename_fields,"filter.",7)))
      filename_fields = filename_fields(!strncmp(filename_fields,"filter.",7));
      ## no more filter substructure because we don't support complex keys at this time. 
      ## filename_fields(end+1) = "filter.type";
    endif
  endif
  
  db_ova_entries = edb_query(ova_params); 
  if (length(db_ova_entries) > 1) 
    error("More than one existing db entry matches the ova parameters.");
  endif
  if (length(db_ova_entries) == 1)
    db_ova_entries = db_ova_entries{1};
    if (additional_params.retrain)
      edb_remove(db_ova_entries);
      if (exist(db_ova_entries.path,"file"))
	unlink(db_ova_entries.path);
      endif
    else
      ## the ova file is not acutally stored in the edb, so we do not need to download it.
      ova_filename = db_ova_entries.filename;
      return;
    endif
  endif 
  
  ## get the data and the filter file if not already local
  if (length(db_data_entries)==1)
    downloaded_ids = edb_download(db_data_entries); #download/update the local file if necessary
    if (length(downloaded_ids) != 1)
      error("Error downloading the data file from edb.");
    endif
  endif

  if (length(db_filter_entries)==1)
    downloaded_ids = edb_download(db_filter_entries); #download/update the local file if necessary
    if (lenght(downloaded_ids) != 1)
      error("Error downloading the filter file from edb.");
    endif
  endif
  
 
  if (isempty(additional_params.filename))
    additional_params.filename = [ova_params.model_type "_results/" edb_name(ova_params, filename_fields)];
  endif
  
  
  if (!isempty(additional_params.filename) && exist(additional_params.filename,"file") && (!ova_params.wfilemap || exist([additional_params.filename ".wmap"],"file")))
    ova_filename = additional_params.filename;
  else    
    params = combine_structs(additional_params, ova_params);
    if ( strcmp(ova_params.model_type,"svm") )
      train_svm(params);
    elseif ( strcmp(ova_params.model_type,"nb") )
      if (!isfield(ova_params,"bias"))
	ova_params.bias = 1; #indicate that there is a bias column, and what the bias should be
      endif
      train_nb(params);      
    else
      error(["Ova model type " ova_params.model_type " is not implemented."]);
    endif
  endif
  
  
  if (exist(additional_params.filename,"file") && (!ova_params.wfilemap || exist([additional_params.filename ".wmap"],"file")))
    ## put a dummy file in the db
    dbfile = [additional_params.filename ".dbstore"];
    fid = fopen(dbfile,"w");
    fputs(fid,additional_params.filename);
    fclose(fid);  
    ova_params.filename = additional_params.filename;
    edb_put(dbfile, ova_params);
    ## no need to upload the file since it is just a dummy file
  endif
  
  ova_filename = additional_params.filename;
  return;
end
