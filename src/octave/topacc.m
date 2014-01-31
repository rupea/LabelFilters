%%% Function that calculates the top-k error. Assumes a multi-class problem with integer labels from 1 to n 
%%% Ties are NOT randomly broken. The lower label is prefered

function topacc = topacc(labels, preds, k=5)

if (min(labels) <= 0)
  %% labels are not 1 to n
  topacc = -1;
  return; 
end

if ( max(labels) < k )
  %% there are fewer labels than k, so the topk accuracy is 1.
  topacc = 1;
  return;
end

if (size(labels,1)!=1)
  labels = labels';
end

if (size(preds,2)!=size(labels,2))
  preds = preds'
end

if (size(preds,2)!=size(labels,2))
  error("Size of the label and prediction vector does not match");
end

if (size(preds,1)<max(labels))
  error("Number of classes larger than number of predictions");
end

[foo, ind] = sort(preds, 1, "descend");
topacc=mean(max(repmat(labels,k,1)==ind(1:k,:)));
