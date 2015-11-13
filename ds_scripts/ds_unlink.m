function [] = ds_unlink(file, rm_local = true)
  global __DS_VERBOSE;
  [status,output] = system(["ds unlink " file " --y"]);
  if (status != 0 || __DS_VERBOSE)
    disp(sprintf("ds_unlink exited with status %d",status));
    disp(output);
  endif
  if (rm_local)
    unlink(file);
  endif
end
