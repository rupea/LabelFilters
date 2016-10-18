function [nb_filename] = train_nb(params)
  
  if (!isfield(params,"bin_dir"))
    params.bin_dir = "~/Research/mcfilter/src/c";
  endif
    
  if (!isfield(params,"queue") || isempty(params.queue))
    params.queue = "batch";
  endif

  node_requests = "";
  if (isfield(params,"node_requests") && !isempty(params.node_requests))
    node_requests = [":" params.node_requests];
  endif

  [out_dir, out_file] = fileparts(params.filename);
  if (!isempty(out_dir))
    mkdir(out_dir);
  endif


  if (exist(params.filename, "file"))
    unlink(params.filename);
  endif
  

  command = [ "myqsub." params.queue " -l nodes=1:ppn=1" node_requests ",walltime=10:00:00 -o " pwd() "/" params.filename ".out -e " pwd() "/" params.filename ".err " params.bin_dir "/naive_bayes -a " num2str(params.alpha) " " params.data_file " " params.filename ".wmap >> submitted_jobs." num2str(getpid())];
 
  system(command);
  system(["qsubwait submitted_jobs." num2str(getpid())]);
  unlink(["submitted_jobs." num2str(getpid())]);
  
  #create an empty file named params.filename. This is needed to be compatible with the svm models
  # the model is in params.filename.wmap 
  fclose(fopen(params.filename,"w"));
  nb_filename = params.filename;
  return;
end
