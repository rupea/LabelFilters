function [query_return] = ds_get(query_struct, entry_type = "")
  query_return = ds_query_get(query_struct, "get", entry_type);
end
