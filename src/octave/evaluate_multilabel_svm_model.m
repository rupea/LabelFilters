function [macro_acc acc macro_F1 F1 top5 top10 pred out]= ...
      evaluate_multilabel_svm_model(y_te, projected_labels, out, thresh=0)
%noties = out'; 

if (~isempty(projected_labels)&&~islogical(projected_labels))
  error("projected_labels is not an empty or a logical matrix");
end

n = size(y_te, 1);
k = size(y_te, 2);

if (issparse(out))
  ## prediction has been done when training the svms. presumably with the same threshold
  if exist('projected_labels', 'var') && ~isempty(projected_labels)
    pred = spones(out .* projected_labels);
  else
    pred = spones(out);
  end

else
  noties = (out + 1e-9 * randn(size(out)));
  
  if exist('projected_labels', 'var') && ~isempty(projected_labels)
    noties(~projected_labels) = -inf;
  end
  %%[ignore,pred] = max(noties, [],1);

  pred = sparse(noties > thresh);
end

true_pos = pred .* y_te;

sum_true_pos = sum(true_pos,1);
sum_pred = sum(pred,1);
sum_y = sum(y_te, 1);

class_precision = sum_true_pos./sum_pred;
class_precision(isnan(class_precision)) = 0; # precision of 0 when nothing was predicted positive
class_recall = sum_true_pos./sum_y;
class_F1 = 2*(class_precision .* class_recall)./(class_precision + class_recall);
class_F1(isnan(class_F1)) = 0; # F1 of 0 when nothing was predicted positive
class_acc = sum_true_pos./sum_y;

##do not average over the classes that do not exist in the test set
class_acc = class_acc(sum_y!=0);
class_F1 = class_F1(sum_y!=0);

macro_acc = mean(class_acc);
macro_F1 = mean(class_F1);

error = sum_pred + sum_y - 2.*sum_true_pos;
acc = 1 - sum(error)./(n*k);

precision = sum(sum_true_pos)/sum(sum_pred);
recall = sum(sum_true_pos)/sum(sum_y);
F1 = 2*precision*recall/(precision+recall);



# len_tr = size(out,2); 

# if len_tr >= 5, top5 = topacc(te_label, noties, 5); else top5 = 1; end;
# if len_tr >= 10, top10 = topacc(te_label, noties, 10); else top10 = 1; end;


