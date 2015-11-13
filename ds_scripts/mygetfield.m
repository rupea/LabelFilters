function [vals] = mygetfield(structs,key)
  if (!iscell(structs))
    structs = {structs};
  endif

  idx = index(key,".");
  if (idx)
    firstkey = key(1:(idx-1));
    lastkey = key((idx+1):end);
  else
    firstkey = key;
    lastkey = "";
  endif

  are_struct = cellfun("isstruct",structs);
  have_field = cellfun("isfield", structs, {firstkey});
  alive = are_struct & have_field;

  if (!any(alive))
    vals(!alive) = NA;
    return;
  endif

  if (isempty(lastkey))
    if (isnumeric(getfield(structs(alive){1},firstkey)))
      vals(alive) = cellfun("getfield",structs(alive),{firstkey});
    else
      vals(alive) = cellfun("getfield",structs(alive),{firstkey},"UniformOutput",false);
    endif      
  else
    vals(alive) = mygetfield(cellfun("getfield",structs(alive),{firstkey},"UniformOutput",false),lastkey);
  endif

  if (iscell(vals))
    vals(!alive) = "NA";
  else
    vals(!alive) = NA;
  endif		     
end
