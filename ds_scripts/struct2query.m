function [query] = struct2query(struct,prev_key="")
  query = "";
  andsign = "";
  for  [val, key] = struct
    if (isempty(val))
      query = sprintf("%s%s(doc.params[\"%s%s\"] == \"__EMPTY\")", query, andsign, prev_key, key);
    elseif (ischar(val))
      query = sprintf("%s%s(doc.params[\"%s%s\"] == \"%s\")", query, andsign, prev_key, key, val);
    elseif (isnumeric(val)||islogical(val))
      query = sprintf("%s%s(doc.params[\"%s%s\"] == %g)", query, andsign, prev_key, key, val);
    elseif (isstruct(val))
      query = sprintf("%s%s%s", query, andsign, struct2query(val,sprintf("%s%s.",prev_key,key)));
    endif
    andsign = "&&";
  end
