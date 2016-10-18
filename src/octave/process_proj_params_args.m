## parse the argument list 
## should only be called with an initialized proj_params
## proj_params can be initialized with devault_proj_params()
function varargout = process_proj_params_args(arg_list, nargs, varargin) #  nargs, proj_params)
  
  arg = 1;
  varargout = varargin;
  while (arg <= nargs)
    argname = arg_list{arg};
    if (argname(1)!="-")
      error(["Argument name " num2str(arg) " does not start with '-'"]);
    endif
    argname = argname(2:end);
    found = false;
    for i = 1:length(varargin)
      if (isfield(varargin{i}, argname))	
	found = true;	
	arg = arg+1;
	if (islogical(varargin{i}.(argname)))
	  if (strcmp(arg_list{arg},"true"))
	    varargout{i}.(argname) = true;
	  elseif (strcmp(arg_list{arg},"false"))
	    varargout{i}.(argname) = false;
	  elseif (arg_list{arg}(1) == "-")
	    varargout{i}.(argname) = true;	
	    arg = arg - 1;
	  else
	    varargout{i}.(argname) = logical(str2num(arg_list{arg}));       
	  endif
	elseif (isnumeric(varargin{i}.(argname)))
	  varargout{i}.(argname) = str2num(arg_list{arg});
	elseif (ischar(varargin{i}.(argname)))
	  varargout{i}.(argname) = arg_list{arg};
	else
	  error("Argument type not recognized");
	endif
	arg = arg+1;
	break;
      endif
    end
    if !found
      error(["Unrecognized argument " argname]);
    endif
  end
  return
end

