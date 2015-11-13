function [query_return] = ds_query_get(query_struct, operation, entry_type = "")
  global __DS_VERBOSE;
  if (isstruct(query_struct))
    if (!isempty(entry_type))
      query_struct.type = entry_type;
    endif 
    query = struct2query(query_struct);
    ds_query_str= sprintf("if(%s){emit(doc)}", query);
  elseif (strncmp(query_struct,"if",2))
    if (!isempty(entry_type))
      error("Query closed, can not add an entry type");
    endif	  
    ds_query_str = query_struct;
  elseif (regexp(query_struct,'^\s*\(*\s*doc\.')) ## will match any file starting with doc.
    if (!isempty(entry_type))
      query = sprintf("%s && (doc.params.type == \"%s\")",query_struct, entry_type);
    else
      query = query_struct;
    endif
    ds_query_str= sprintf("if(%s){emit(doc)}", query);
  else
    ## must be a local file
    paramstrct.type = "local_file";
    paramstrct.filename = query_struct;
    retstrct.params = paramstrct;
    retstrct.db_path = query_struct;
    query_return = {retstrct};
    return;
  endif
  
  if (strcmp(operation,"get"))
    ## use autopull in order not to block add transactions that might happen
    ## in different processes
    operation = "get --autopull";
  endif
  
  ds_query_command = sprintf("ds %s --map='%s'", operation, ds_query_str);
  [status ds_entries] = system(ds_query_command,1);
  if (status != 0 || __DS_VERBOSE)
    disp(sprintf("ds_query_get exited with status %d",status));
    disp(ds_entries);
  endif
  query_return = parse_ds_return(ds_entries);
end
