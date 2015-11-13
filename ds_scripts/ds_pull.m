function [] = ds_pull()
  global __DS_VERBOSE;
  [status, output]=system("ds pull");
  if (status != 0 || __DS_VERBOSE)
    disp(sprintf("ds_pull exited with status %d",status));
    disp(output);
  endif
end
