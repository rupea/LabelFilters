function [] = ds_remove(query_struct, entry_type = "")
  global __DS_VERBOSE;
  if (isstruct(query_struct))
    if (!isempty(entry_type))
      query_struct.type = entry_type;
    endif 
    query = struct2query(query_struct);
    ds_query_str= sprintf("if(%s){emit(doc)}", query);
  else
    if (strncmp(query_struct,"if",2))
      if (!isempty(entry_type))
	error("Query closed, can not add an entry type");
      endif	  
      ds_query_str = query_struct;
    else
      if (!isempty(entry_type))
	query = sprintf("%s && (doc.params.type == \"%s\")",query_struct, entry_type);
      else
	query = query_struct;
      endif
      ds_query_str= sprintf("if(%s){emit(doc)}", query);
    endif
  endif
  
  ds_remove_command = sprintf("ds remove --y --map='%s'", ds_query_str);
  [status output] = system(ds_remove_command,1);
  if (status != 0 || __DS_VERBOSE)
    disp(sprintf("ds_remove exited with status %d",status));
    disp(output);
  endif
end
