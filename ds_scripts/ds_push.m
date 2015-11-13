function [] = ds_push()
  global __DS_VERBOSE;
  [status output]=system("ds push");
  if (status != 0 || __DS_VERBOSE)
    disp(sprintf("ds_push exited with status %d",status));
    disp(output);
  endif
end
