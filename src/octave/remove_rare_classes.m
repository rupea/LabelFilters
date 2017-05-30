function [xout,yout,map,map_ignored] = remove_rare_classes(xin,yin,min_ex_per_class, max_classes = -1, mapin=[], min_ex_in_all_sets = true, eliminate_ex_with_no_labels = true)
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
    
    [ex_per_class, classes] = table(y);
    nclasses = lenght(classes);
    if (max_classes <= 0)
      max_classes = nclasses;
    endif
    rnks = ranks(ex_per_class + rand(1,nclasses)/100); # randomly break ties
    sel = (ex_per_class >= min_ex_per_class) & (rnks > nclasses - max_classes);
    for k = 1:nclasses
      l = classes(k);
      if sel(k)
	for i=1:length(yin)
	  yout{i}(yin{i}==l,1)=new_l;
	endfor	
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
    if (elminate_ex_with_no_label)
      error("Can not have examples with no label in multi-class scenario");
    endif
    for i=1:length(yout)
      yout{i} = yout{i}(yout{i} != -999);
      xout{i} = xin{i}(yout{i} != -999,:);
    end
    
  else
    ## multi label problem. Labels are in a matrix        
    ex_per_class=sum(y,1);
    nclasses = size(y,2);
    if (max_classes <= 0 )
      max_classes = nclasses;
    endif
    rnks = ranks(ex_per_class + rand(1,nclasses)/100); # randomly break ties
    sel = (ex_per_class >= min_ex_per_class) & (rnks > nclasses - max_classes);
    if (isempty(mapin))
      mapin = 1:size(y,2);
    endif    
    map = mapin(sel);
    map_ignored = mapin(!sel);
    for i=1:length(yin)
      yout{i} = yin{i}(:,sel);
      ## take out instances that do not have a class associated with them any more    
      if (eliminate_ex_with_no_labels)
	classes_per_ex=sum(yout{i},2);
	yout{i} = yout{i}(classes_per_ex > 0,:); 
	xout{i} = xin{i}(classes_per_ex > 0,:);
      else
	xout{i} = xin{i};
      endif
    end
  endif
  if (!cellinput)
    xout = xout{1};
    yout = yout{1};
  endif
end
