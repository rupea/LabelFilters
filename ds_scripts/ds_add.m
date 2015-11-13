function ds_add(file, add_struct,entry_type = "")
  global __DS_VERBOSE;
  if (!isempty(entry_type))
    add_struct.type = entry_type;
  endif 

  ## remove the previous file(s) from the db if any until 
  ## a versioning system is implemented
  ds_remove(add_struct);

  ds_params = struct2params(add_struct);
  ds_params = sprintf("\{%s\}",ds_params);
  ds_add_command = sprintf("ds add %s --strict --autopush --params='%s'", file, ds_params);
  [status, output] = system(ds_add_command,1);
  if (status != 0 || __DS_VERBOSE)
    disp(sprintf("ds_add exited with status %d",status));
    disp(output);
    if (status != 0)
      disp(["WARNING: ds_add failed for file " file]);
      [foo bar] = system("ds clear",1);
    endif
  endif
end
