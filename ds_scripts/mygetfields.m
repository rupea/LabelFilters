function [vals] = mygetfields(structs, fields, ascell = false)
  if (!iscell(fields))
    fields = {fields};
  endif

  vals = cellfun("mygetfield",{structs},fields,"UniformOutput",false)';
  if (all(cellfun("isnumeric",vals))&&!ascell)
    vals = cell2mat(vals);
  endif
end
