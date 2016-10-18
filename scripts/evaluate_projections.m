#! /usr/bin/env octave 

#  #!/home/mlshack/alex/Programs/octave-3.8.2/bin/octave -qf 

# #!/usr/bin/octave -qf

#  #!/var/run/net/mlstorage2/mlshack/alex/Programs/octave-3.8.2/bin/octave -qf


global __DS_VERBOSE = false;

srcdir = "~/Research/mcfilter/src";
bindir = [srcdir "/c"];
addpath([srcdir "/octave/"])
addpath([srcdir "/libsvm-3.17/matlab/"])
addpath([srcdir "/liblinear-1.94/matlab/"])

addpath("~/Research/mcfilter/edb_scripts")
addpath("~/Research/mcfilter/edb_scripts/jsonlab-1.2")

arg_list = argv();

input_params.data_query = "";
input_params.ova_query = "";
input_params.filter_query = "";
input_params.file_name_fields = "filter.type filter.C1 filter.C2 __HASH";
input_params.compute_full = true;
input_params.recompute = false;
input_params.same_data = true;
input_params.chunks = 1;
#input_params.ova_format = "binary";
input_params.num_threads = 0;
input_params.validation = false;
input_params.allproj = false; 

input_params = process_proj_params_args(arg_list,length(arg_list), input_params);

file_name_fields = strsplit(input_params.file_name_fields);

# full_file_name_fields = file_name_fields(!strncmp(file_name_fields,"filter.",7));
# hash_field = any(strcmp(full_file_name_fields,"__HASH"));
# full_file_name_fields = full_file_name_fields(!strcmp(full_file_name_fields,"__HASH"));
# full_file_name_fields(end+1) = "filter.type";
# if (hash_field)
#   full_file_name_fields(end+1) = "__HASH";
# endif

struct_levels_to_print(10)

input_params

db_data_entries = edb_query(input_params.data_query)

if(!input_params.same_data)
  db_ova_entries = edb_query(input_params.ova_query);
  if (!isempty(input_params.filter_query))
    db_proj_entries = edb_query(input_params.filter_query);
  else
    db_proj_entries = [];
  endif
endif

## entry in the filter field if no filter is used
fullproj = "no_filter"; 

clear perf_params;
perf_params.type = "performances";
perf_params.num_threads = input_params.num_threads;
perf_params.validation = input_params.validation;
perf_params.allproj = input_params.allproj;
for data_entry = db_data_entries  
  data_entry = data_entry{1};
  [foo, df] = fileparts(data_entry.path);
  perf_params.data = data_entry;

  if (input_params.same_data)
    clear strct;
    strct.data = data_entry;
    ova_query = [input_params.ova_query "&&" struct2query(strct)];
    db_ova_entries = edb_query(ova_query)
  endif

  for ova_entry = db_ova_entries
    ova_entry = ova_entry{1};
    ova_file = ova_entry.filename;    
    ova_format = "cellarray";
    if (isfield(ova_entry, "wfilemap") && ova_entry.wfilemap)
      ova_format = "binary";
      ova_file = [ova_file ".wmap"];
    endif
    ova_bias = 0;
    if (isfield(ova_entry, "bias"))
      ova_bias = ova_entry.bias;
    endif
    
    perf_params.ova = ova_entry;

    computed_proj = {};    
    if (!input_params.recompute)
      db_perf_entries = edb_query(perf_params);
      computed_proj = cellfun("getfield",db_perf_entries,{{1}},{"filter"},"UniformOutput",false);
    endif
    cf = input_params.compute_full;
    if (any(cellfun("isequal",computed_proj,{fullproj})))
      ## the performance with no filter has been calculated
      ## for this data and this ova model, so don't compute it again      
      cf = false
    endif
    
    if (input_params.same_data)
      clear strct;
      strct.data_params = data_entry;
      if (!isempty(input_params.filter_query))
	input_params.filter_query = [input_params.filter_query "&&" struct2query(strct)];
	db_proj_entries = edb_query(input_params.filter_query)
      else
	db_proj_entries = [];
      endif
    endif    

    proj_files = "";
    for proj_entry = db_proj_entries
      proj_entry = proj_entry{1};	
      if (!any(cellfun("isequal",computed_proj,{proj_entry})))
	if(!exist(proj_entry.path))
	  edb_download(proj_entry);
	endif
	proj_files = [proj_files " " proj_entry.path];
      endif
    end

    opt_str = "";
    if (!isempty(proj_files))
      opt_str = ["-p " proj_files];
    endif
    if (cf)
      opt_str = ["--full " opt_str];
    endif
    if (input_params.validation)
      opt_str = ["--validation " opt_str];
    endif
    if (input_params.allproj)
      opt_str = ["--allproj " opt_str];
    endif

    if (!isempty(opt_str))
      ## if there is some work to do 
      ## do this here to avoid downloading these big files if they 
      ## are not needed
      if(!strcmp(data_entry.type, "local_file"))
	edb_download(data_entry);
      endif
      if (!exist(data_entry.path,"file") )
	error(["File " data_entry.path " does not exist."])
      endif
      if (!exist(ova_file,"file"))
	error(["Ova models file " ova_file " does not exist."])
      endif
      
      ## get the ova file on /mnt/local to avoid over the network loading
      s = make_absolute_filename(ova_file);
      local_ova_file = ["/mnt/local/" substr(s,strfind(s,"alex"))]
      if (!exist(local_ova_file,"file"))	
	[status,msg] = system(["mkdir -p ", fileparts(local_ova_file)])
	if (status)
	  warning(["Copy of the ova file to the local drive failed with message: " msg "\n   Using the nfs file"]);
	  local_ova_file = ova_file;
	else	
	  [status,msg] = copyfile(ova_file,local_ova_file)
	  if (!status)
	    warning(["Copy of the ova file to the local drive failed with message: " msg "\n   Using the nfs file"]);
	    local_ova_file = ova_file;
	  endif
	endif
      endif

      eval_cmd = sprintf("%s/evaluate_projection %s --chunks %d --num_threads %d --ova_format %s -b %g -- %s %s", bindir, opt_str, input_params.chunks, input_params.num_threads, ova_format, ova_bias, data_entry.path, local_ova_file);
      
      [status,output] = system(eval_cmd, 1);
      
      disp(output)
      
      for proj_entry = db_proj_entries
	proj_entry = proj_entry{1};
	if (!any(cellfun("isequal",computed_proj,{proj_entry})))
	  perfs = parse_eval_output(output, proj_entry.path);      
	  [foo, pf] = fileparts(proj_entry.path);
	  if (!exist("perfs","dir"))
	    mkdir("perfs");
	  endif
	  
	  newperf_params = perf_params;
	  newperf_params.filter = proj_entry;
	  for [val,key] = perfs
	    newperf_params.(key) = val;
	  end
	 
	  outfile = ["perfs/" edb_name(newperf_params, file_name_fields)];
	  
	  outfid = fopen(outfile,"wt");
	  for [val,key] = perfs
	    fprintf(outfid,"%s  ", key);
	    if (ischar(val))
	      fprintf(outfid, " %s", val);
	    elseif (isnumeric(val))
	      fprintf(outfid," %g", val);
	    endif
	    fprintf(outfid,"\n");
	  end
	  fclose(outfid);
	  
	  edb_put(outfile, newperf_params);
	endif
      end

      if (cf)
	perfs = parse_eval_output(output, "full");      
	if (!exist("perfs","dir"))
	  mkdir("perfs");
	endif

	newperf_params = perf_params;
	newperf_params.filter = fullproj;
	for [val,key] = perfs
	  newperf_params.(key) = val;
	end

	outfile = ["perfs/" edb_name(newperf_params, file_name_fields)];	  

	outfid = fopen(outfile,"wt");
	for [val,key] = perfs
	  fprintf(outfid,"%s  ", key);
	  if (ischar(val))
	    fprintf(outfid, " %s", val);
	  elseif (isnumeric(val))
	    fprintf(outfid," %g", val);
	  endif
	  fprintf(outfid,"\n");
	end
	fclose(outfid);
	
	edb_put(outfile, newperf_params);
      endif
    endif
  end
end
