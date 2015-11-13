function [xout,yout,map,map_ignored] = remove_rare_classes(xin,yin,min_ex_per_class, mapin=[], min_ex_in_all_sets = true)
  cellinput = true;
  if (!iscell(xin))
    yin = {yin};
    xin = {xin};
    cellinput = false;
  endif
  assert(length(xin)==length(yin));  
  if (min_ex_in_all_sets)
    ## eliminate classes that have less than min examples over all sets
    y = cat(1,yin{:}); 
  else
    ## eliminate classes that have less than min exmaples in the training set only
    ##first one is the training set    
    y=yin{1};
  endif
  if (isvector(y))
    ## multiclass problem. y is a vector of class labels. 
    map = [];
    map_ignored = [];
    new_l = 1;
    l_ignored = 1;
    for l = unique(cat(1,yin{:}))'
      if (sum(y==l) >= min_ex_per_class)
	for i=1:length(yin)
	  yout{i}(yin{i}==l,1)=new_l;
	end	
	if (isempty(mapin))
	  map(new_l) = l;
	else
	  map(new_l)=mapin(l);
	endif
        new_l = new_l+1;	
      else
	for i=1:length(yin)
	  yout{i}(yin{i}==l,1)=-999;
	end
	if (isempty(mapin))
	  map_ignored(l_ignored) = l;
	else
	  map_ignored(l_ignored) = mapin(l);
	endif
	l_ignored = l_ignored + 1;
      endif
    end
    
    ## eliminate the examples that have had the labels removed
    for i=1:length(yout)
      yout{i} = yout{i}(yout{i} != -999);
      xout{i} = xin{i}(yout{i} != -999,:);
    end

  else
    ## multi label problem. Labels are in a matrix        
    ex_per_class=sum(y,1);
    if (isempty(mapin))
      mapin = 1:size(y,2);
    endif
    map_ignored = mapin(ex_per_class < min_ex_per_class);
    map = mapin(ex_per_class >= min_ex_per_class);
    for i=1:length(yin)
      yout{i} = yin{i}(:,ex_per_class >= min_ex_per_class);
      ## take out instances that do not have a class associated with them any more    
      n_classes=sum(y,2);
      yout{i} = yout{i}(n_classes > 0,:); 
      xout{i} = xin{i}(n_classes > 0,:);
    end
  endif
  if (!cellinput)
    xout = xout{1};
    yout = yout{1};
  endif
end
