function [] = ds_connect(database)
  global __DS_VERBOSE;
  [status, output]=system(["ds connect " database],1);
  if (status != 0 || __DS_VERBOSE)
    disp(sprintf("ds_connect exited with status %d",status));
    disp(output);
  endif

end
