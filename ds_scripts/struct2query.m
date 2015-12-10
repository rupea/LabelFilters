function [query] = struct2query(struct,prev_key="")
  query = "";
  andsign = "";
  for  [val, key] = struct
    if (isempty(val))
      query = sprintf("%s%s(doc.params[\"%s%s\"] == \"__EMPTY\")", query, andsign, prev_key, key);
    elseif (ischar(val))
      query = sprintf("%s%s(doc.params[\"%s%s\"] == \"%s\")", query, andsign, prev_key, key, val);
    elseif (isnumeric(val)||islogical(val))
      if (length(val)>1)
	#todo: include as list rather than as string
	query = sprintf("%s%s(doc.params[\"%s%s\"] == \"%s\")", query, andsign, prev_key, key, num2str(val));
      elseif (isnan(val))
	query = sprintf("%s%s(doc.params[\"%s%s\"] == \"NaN\")", query, andsign, prev_key, key);
      elseif (isinf(val))
	if(val > 0)
	  query = sprintf("%s%s(doc.params[\"%s%s\"] == \"Inf\")", query, andsign, prev_key, key);
	else
	  query = sprintf("%s%s(doc.params[\"%s%s\"] == \"-Inf\")", query, andsign, prev_key, key);
	endif
      elseif (isna(val))
	query = sprintf("%s%s(doc.params[\"%s%s\"] == \"NA\")", query, andsign, prev_key, key);
      else
	query = sprintf("%s%s(doc.params[\"%s%s\"] == %g)", query, andsign, prev_key, key, val);
      endif
    elseif (isstruct(val))
      query = sprintf("%s%s%s", query, andsign, struct2query(val,sprintf("%s%s.",prev_key,key)));
    endif
    andsign = "&&";
  end
