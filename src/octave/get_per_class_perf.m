%%% Function that calculates per class performance metrics.
%%% Assumes a multi-class problem with integer labels from 1 to n 
%%% preds is a vector of integers that are the class predictions

function [acc,precision,recall,F1] = get_per_class_perf(labels,preds)

if (min(labels) <= 0)
    %% labels are not 1 to n
    acc = -1;
    precision=-1;
    recall=-1;
    F1=-1;
    return; 
end

nclass = max(labels);

if (size(labels,1)!=1)
    labels = labels';
end
if (size(preds,2)!=size(labels,2))
    preds = preds';
end
if (size(preds,2)!=size(labels,2))
    error("Size of the label and prediction vector does not match");
end

acc=[];
precision=[];
recall=[];
F1=[];
for i=1:nclass
    ind_labels = find(labels==i);
    ind_preds = find(preds==i);
  
    n = length(ind_labels);
    if n > 0,
        tp = sum(labels(ind_preds)==i);
        fn = sum(preds(ind_labels)!=i);
    
        if (length(ind_preds)>0)
            precision(i)=tp/length(ind_preds);
        else
            precision(i)=0;
        end
        recall(i)=tp/n;
        if (precision(i)+recall(i)>0)
            F1(i)=2*precision(i)*recall(i)/(precision(i)+recall(i));
        else
            F1(i)=0;
        end

        acc(i)=(n-fn)/n;
    end
end

