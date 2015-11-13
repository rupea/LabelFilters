function name = ds_name(struct, fields)
  if (!iscell(fields))
    error("Fields must be a cell of strings");
  end
  
  name = "";

  specialfields = strncmp(fields, "__",2);
  paramfields = fields(!specialfields);
  paramvals = mygetfields(struct, paramfields, true);

  pid = 1;
  sid = 1;
  for i = 1:length(fields)
    key = fields{i};
    if (specialfields(i))
      if (strcmp(key, "__TIME"))
	name = [name "__" num2str(time)];
      elseif (strcmp(key, "__HASH"))
	oldlevels = struct_levels_to_print(100,"local");      
#	name = [name "__eca0470178275ac94e5de381969ed232"];
	name = [name "__" md5sum(cell2mat(strsplit(disp(struct),"\n")),true)];
	struct_levels_to_print(oldlevels,"local");
      else
	error(["Special name fields " key " was not implemented"]);
      endif
      sid = sid + 1;
    else
      if (!isempty(name))
	key = ["__" key];
      endif
      val = paramvals{pid};    
      if (isnumeric(val))      
	name = [name key "_" num2str(val)];
      else
	name = [name key "_" val{1}];
      endif
      pid = pid + 1;
    endif
  end
  if (isfield(struct,"type"))
    name = [name "." struct.type];
  endif
  return;
end



  #  for key = fields 
  #   key = key{1};
  #   if (strcmp(key, "__TIME"))
  #     name = [name "__" num2str(time)];
  #   elseif (strcmp(key, "__HASH"))
  #     oldlevels = struct_levels_to_print(100,"local");      
  #     name = [name "__" md5sum(disp(struct),true)];
  #     struct_levels_to_print(oldlevels,"local");
  #   else
  #     clear spkey;
  #     spkey(2,:) = strsplit(key,".");
  #     spkey(1,:) = {{1}};
  #     val = getfield(struct, spkey{:});
  #     if (!isempty(name))
  # 	key = ["__" key];
  #     endif
  #     if (isnumeric(val))      
  # 	name = [name key "_" num2str(val)];
  #     else
  # 	name = [name key "_" val];
  #     endif
  #   endif
  # end
  
