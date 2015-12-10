function [params] = struct2params(struct,prev_key="")
  params = "";
  comma = "";
  for  [val, key] = struct
    if (isempty(val))
      params = sprintf("%s%s \"%s%s\":\"__EMPTY\"", params, comma, prev_key, key);      
    elseif (ischar(val))
      params = sprintf("%s%s \"%s%s\":\"%s\"", params, comma, prev_key, key, val);
    elseif (isnumeric(val)||islogical(val))
      if (length(val) > 1)
	 #todo: include as list rather than as string
	 params = sprintf("%s%s \"%s%s\":\"%s\"", params, comma, prev_key, key, num2str(val));
      elseif (isnan(val))
	params = sprintf("%s%s \"%s%s\":\"NaN\"", params, comma, prev_key, key);      
      elseif (isinf(val))
	if(val > 0)
	  params = sprintf("%s%s \"%s%s\":\"Inf\"", params, comma, prev_key, key);      
	else
	  params = sprintf("%s%s \"%s%s\":\"-Inf\"", params, comma, prev_key, key);      
	endif
      elseif (isna(val))
	params = sprintf("%s%s \"%s%s\":\"NA\"", params, comma, prev_key, key);            
      else
	params = sprintf("%s%s \"%s%s\":%g", params, comma, prev_key, key, val);
      endif
    elseif (isstruct(val))
      params = sprintf("%s%s %s", params, comma, struct2params(val,sprintf("%s%s.",prev_key,key)));
    endif
    comma = ",";
  end
end

