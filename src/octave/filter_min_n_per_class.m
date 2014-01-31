function [x,tr_label,xtest,te_label] = filter_min_n_per_class(x,tr_label,xtest,te_label, min_n_per_class)

classes = unique(tr_label)';
for c = classes
    idx = find(tr_label==c);
    n = length(idx);
    if n < min_n_per_class
        tr_label(idx)=[];
        x(idx,:)=[];
        idx =  find(te_label==c);
        te_label(idx)=[];
        xtest(idx,:)=[];        
    end
end
return 