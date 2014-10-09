function [avg_class_acc avg_class_F1 acc F1 top5 top10 pred]= ...
      evaluate_multiclass_svm(te_label, projected_labels, out)


  if (~isempty(projected_labels)&&~islogical(projected_labels))
    error("projected_labels is not an empty or a logical matrix");
  end
  
  if (size(te_label,1)!=size(out,1))
    te_label = te_label';
  end

  if (size(te_label,1)!=size(out,1))
    error("Size different number of labels and outputs");
  end
  
  if (size(te_label,2) != 1)
    error("MultiClass evaluation has been called with sparse label indicator matrix. It only works with label vector");
  end
  
  if (issparse(out))
    warning("Max decoding is beeing used on outputs that have been thresholded");
  end
  
  noties = (out + 1e-9 * randn(size(out)))';

  if exist("projected_labels", "var") && ~isempty(projected_labels)
    noties(~projected_labels') = -inf;
  end
  [ignore,pred] = max(noties, [],1);
  
  [class_acc,class_prec,class_rec,class_F1]=get_per_class_perf(te_label,pred);
  
  avg_class_acc = mean(class_acc);
  avg_class_F1 = mean(class_F1);
  
  if size(te_label,1) == size(pred,1),
    acc = mean(te_label==(pred));
  else
    acc = mean(te_label==(pred'));
  end

  F1 = -1; # not implemented yet
  
  noClasses = size(out,1);
  
  if noClasses >= 5, top5 = topacc(te_label, noties, 5); else top5 = 1; end;
  if noClasses >= 10, top10 = topacc(te_label, noties, 10); else top10 = 1; end;
      
end