function [xout, idf] = code_features(xin, coding, idf_from_all_data=true);
  if (strcmp(coding, "none")) 
    ## return the original features
    xout = xin;
    idf = [];
    return;
  endif

  cellinput = true;
  if (!iscell(xin))
    xin = {xin};
    cellinput = false;
  endif

  
  if (strcmp(coding,"boolean"))
    for i = 1:length(xin)
      xout{i} = double(logical(xin{i}));
    end
    idf = [];
  elseif (strcmp(coding,"tfidf"))
    ## assumes that the input is word counts 
    if (idf_from_all_data)
      ## calculate idf using all the datasets
      x = cat(1,xin{:}); 
    else
      ## calculate idf using only the training set
      ##first one is the training set    
      x=xin{1};
    endif
    idf = log(size(x,1)./sum(logical(x),1));
    for i=1:length(xin)
      xout{i} = xin{i};
      for j=1:size(xin{i},2)	
	xout{i}(:,j) = x(:,j).*idf(j);
      end
    end
  else
    exit([coding " coding was not implemented"]);
  endif
  
  if (!cellinput)
    xout = xout{1};
  endif
end
