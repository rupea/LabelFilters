function [y_tr, x_tr, y_te, x_te] = split_data(y,x,train_fraction, seed=0, stratified = false)
  
  if (stratified)
    error("Stratified splitting not implemented yet");
  endif
  
  #save the random generator state to restore it at the end
  old_state=rand("state");
  if (seed > 0)
    rand("state",seed);
  else
    rand("state","reset");
  end

  ## generate the train test split
  n=size(x,1);
  n_train = floor(train_fraction*n);
  perm=randperm(n);
  y_tr = y(perm(1:n_train),:);
  x_tr = x(perm(1:n_train),:);
  y_te = y(perm((n_train+1):end),:);
  x_te = x(perm((n_train+1):end),:);
  
  rand("state",old_state);

end