function [query_return] = ds_query(query_struct, entry_type = "")
  query_return = ds_query_get(query_struct, "query", entry_type);
end
