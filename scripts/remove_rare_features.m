function [xout, map, map_ignored] = remove_rare_features(xin,min_word_count, mapin=[], min_word_in_all_sets=true);
  cellinput = true;
  if (!iscell(xin))    
    xin = {xin};
    cellinput = false;
  endif
  
  if (min_word_in_all_sets)
    x = cat(1,xin{:});
  else
    x = xin{1};
  endif
  
  if (isempty(mapin))
    mapin = 1:size(x,2);
  endif

  ## eliminate features that appear in fewer than min_feat instances
  word_counts = sum(logical(x),1);   
  for i = 1:length(xin)
    xout{i} = xin{i}(:, word_counts >= min_word_count);
  end
  map = mapin(word_counts >= min_word_count);
  map_ignored = mapin(word_counts < min_word_count);

  if (!cellinput)
    xout = xout{1};
  endif
end
