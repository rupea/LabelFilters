## parse the argument list 
## should only be called with an initialized proj_params
## proj_params can be initialized with devault_proj_params()
function [proj_params] = process_proj_params_args(arg_list, nargs, proj_params)
  
  arg = 1;
  while (arg <= nargs)
    argname = arg_list{arg};
    if (argname(1)!="-")
      error(["Argument name " num2str(arg) " does not start with '-'"]);
    endif
    argname = argname(2:end);
    if(!isfield(proj_params, argname))
      error(["Unrecognized argument " argname]);
    endif
    arg = arg+1;
    if (islogical(proj_params.(argname)))
      if (strcmp(arg_list{arg},"true"))
	proj_params.(argname) = true;
      elseif (strcmp(arg_list{arg},"false"))
	proj_params.(argname) = false;
      elseif (arg_list{arg}(1) == "-")
	proj_params.(argname) = true;	
	arg = arg - 1;
      else
	proj_params.(argname) = logical(str2num(arg_list{arg}));       
      endif
    elseif (isnumeric(proj_params.(argname)))
      proj_params.(argname) = str2num(arg_list{arg});
    elseif (ischar(proj_params.(argname)))
      proj_params.(argname) = arg_list{arg};
    else
      error("Argument type not recognized");
    endif
    arg = arg+1;
  end
  
  return
end

