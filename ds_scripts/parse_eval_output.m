function [perfs] = parse_eval_output(output, projname)
  spout = strsplit(output,"\n");
  l = length(projname);
  spout = spout(strncmp(spout,projname,l));
  spout = strtrim(cellfun(@(x) x((l+1):end), spout, "UniformOutput", false));

#   actidx_valid = strncmp(spout,"valid_Active",12);
#   if (any(actidx_valid))
#     actstr = spout(actidx_valid);
#     actstrarr = cell2mat(regexp(actstr,'^valid_Active_\d+\s+(?<prc>\S+)\s+\((?<active>\d+)/(?<total>\d+)\)',"names"));
#     perfs.valid_prc_active_per_proj = strjoin({actstrarr.prc}," ");
#     perfs.valid_active_per_proj = strjoin({actstrarr.active}," ");
#     %%    l=(find(actidx))(end) + 1;
#   endif
    
#   actidx = strncmp(spout,"Active",6);
#   if (any(actidx))
#     actstr = spout(actidx);
#     actstrarr = cell2mat(regexp(actstr,'^Active_\d+\s+(?<prc>\S+)\s+\((?<active>\d+)/(?<total>\d+)\)',"names"));
#     perfs.prc_active_per_proj = strjoin({actstrarr.prc}," ");
#     perfs.active_per_proj = strjoin({actstrarr.active}," ");
# %    l=(find(actidx))(end) + 1;
# %  else
# %    l=1;
#   endif

#   if (!all(actidx | actidx_valid))
#     perfstr = spout(!(actidx | actidx_valid));
#     perfcell = regexp(perfstr,'^(?<name>\S+)\s+(?<val>.*)',"names");
#     for pc = perfcell 
#       perfs.(pc{1}.name) = str2num(pc{1}.val);
#     end
#   endif
  
  perfcell = regexp(spout,'^(?<name>\S+)\s+(?<val>.*)',"names");
  for pc = perfcell 
    perfs.(pc{1}.name) = str2num(pc{1}.val);
  end

  return;
end
