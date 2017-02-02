function [xout] = normalize_data(xin, normalization)
  if (strcmp(normalization, "none"))
    xout = xin;
    return;
  endif
  cellinput = true;
  if (!iscell(xin))
    xin = {xin};
    cellinput = false;
  endif

  if (strcmp(normalization,"row"))
    for i = 1:length(xin)      
      xout{i} = oct_normalize_data(xin{i},1);
    end
  elseif (strcmp(normalization,"col"))
    if length(xin) > 1
      error("cloumn wise normalization is not implemented for multiple datasets")
    endif
    for i = 1:length(xin)      
      xout{i} = oct_normalize_data(xin{i},2);
    end
  else
    error(["Normlaization " normalization " is not implemented"]);
  end

  if (!cellinput)
    xout = xout{1};
  endif
end
