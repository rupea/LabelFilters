function [return_struct] = parse_ds_return(ds_return)
  db_entries = strsplit(ds_return,"\n");
  db_entries = db_entries(strncmp(db_entries,"db:",3));
  if (length(db_entries) == 0)
    ## there were no matching db entries
    return_struct = {};
    return;
  endif
  ## ds query return
  db_paramstruct = regexp(db_entries, "\{(?<params>.*)\} -> (?<path>.*)",'names');
  if (isempty(db_paramstruct{1}.params))
    ## did not match the ds query return. Must be ds get return
    db_paramstruct = regexp(db_entries, "-> (?<path>[^:]*):\{(?<params>.*)\}",'names');
  end			    
  return_struct = cellfun("parse_ds_params", db_paramstruct, "UniformOutput", false);
end