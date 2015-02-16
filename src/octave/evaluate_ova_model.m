function [macro_acc acc macro_F1 F1 top5 top10 pred] = evaluate_ova_model(y_te, projected_labels, out, thresh=0)
  
  if ( size(y_te,2) == 1)
    %% multiclass problem with label vector
    %% evaluate assuming max decoding (predicted class is the one with 
    %% the highest output)
    [macro_acc acc macro_F1 F1 top5 top10 pred] = evaluate_multiclass_svm(y_te, projected_labels, out);
  else
    %% multilabel (or multiclass) problem with sparse label indicator matrix 
    %% class is predicted if prediction exceeds threshold. 
    [macro_acc acc macro_F1 F1 top5 top10 pred] = evaluate_multilabel_svm(y_te, projected_labels, out, thresh);
  end
