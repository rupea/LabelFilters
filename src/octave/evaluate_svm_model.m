function [macro_acc acc macro_F1 F1 top5 top10 pred out]= ...
      evaluate_multilabel_svm_model(y_te, projected_labels, out)
----------not done, work on it ----------
%noties = out'; 
noties = (out + 1e-9 * randn(size(out)))';

if exist('projected_labels', 'var') && ~isempty(projected_labels)
    noties(projected_labels' == 0) = -inf;
end
[ignore,pred] = max(noties, [],1);

[class_acc,class_prec,class_rec,class_F1]=get_per_class_multilabel_perf(y_te, pred);

avg_class_acc = mean(class_acc);
avg_class_F1 = mean(class_F1);

if size(te_label,1) == size(pred,1),
    acc = mean(te_label==(pred));
else
    acc = mean(te_label==(pred'));
end

len_tr = size(out,2); 

if len_tr >= 5, top5 = topacc(te_label, noties, 5); else top5 = 1; end;
if len_tr >= 10, top10 = topacc(te_label, noties, 10); else top10 = 1; end;


