function [y,x, map, map_ignored, idf]=prepare_LSHTC14(fname, originalfile, min_ex_per_class=1, min_feat=1, weighting="boolean", normalization = "row")

#  fname = ["../data/LSHTC14train_minclass" num2str(min_ex_per_class) "_minfeat" num2str(min_feat) "_weighting_" weighting "_normalization_" normalization ".mat"];
  
  if exist(fname, 'file')
    load(fname);
  else
    % reading training examples
    [y,x,map] = read_sparse_ml(originalfile);  
    display("Done loading");
    ## should do this separately for train set, but I want to have the same data for multiple splits
    ##eliminate classes with fewer than min_ex_per_class examples
    ex_per_class=sum(y,1);
    y = y(:,ex_per_class >= min_ex_per_class);
    map_ignored = map(ex_per_class < min_ex_per_class);
    map = map(ex_per_class >= min_ex_per_class);
    ## take out instances that do not have a class associated with them any more
    n_classes=sum(y,2);
    y = y(n_classes > 0,:);
    
    ## eliminate features that appear in fewer than min_feat instances
    n_feat = sum(logical(x),1);
    x = x(n_classes > 0, n_feat > min_feat);

    display("Done filtering");
    idf = [];
    if (strcmp(weighting,"boolean"))
      x = double(logical(x));
    elseif (strcmp(weighting,"tfidf"))
      idf = log(size(x,1)./sum(logical(x),1));
      for j=1:size(x,2)
	x(:,j) = x(:,j).*idf(j);
      end
    else
      exit(["Weighting " weighting " was not implemented"]);
    end
    display("Done weighting");
    norms=zeros(size(x,1),1);
    if (strcmp(normalization,"row"))
      x = oct_normalize_data(x,1);
    elseif (strcmp(normalization,"col"))
      x = oct_normalize_data(x,2);
    end
    display("Done normalizing");
    save(fname,"-v6","y","x","map","map_ignored", "idf");
  end
  
end
    