function retstruct = parse_ds_params(paramstruct)
  retparams = [];
  z = regexp(paramstruct.params, "'(?<key>[^']*)': (?<val>u?'?[^',]*)",'names');
  for i = 1:length(z.key)
    clear spkey;
    spkey(2,:) = strsplit(z.key{i},".");
    spkey(1,:) = {{1}};    
    if (strncmp(z.val{i},"u'",2))
      retparams = setfield(retparams,spkey{:},z.val{i}(3:end));
    else
      retparams = setfield(retparams,spkey{:},str2double(z.val{i}));
    endif
  end
  retstruct.params = retparams;
  retstruct.db_path = paramstruct.path;
end
