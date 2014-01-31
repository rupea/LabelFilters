function [x_orig,tr_label,xtest_orig,te_label] = filter_classes(x_orig,tr_label,xtest_orig,te_label, max_no_classes)

classes = unique(tr_label);
noClasses = length(classes);

%disp('selecting a subset ...');  
idx = ismember(tr_label,classes( 1:min(max_no_classes,noClasses)));
        
%disp('reduction ...');  
tr_label = tr_label(idx);
x_orig = x_orig(idx,:);
%disp(size(tr_label));
            
classes = unique(tr_label);
noClasses = length(classes);
      
idx = ismember(te_label,classes);
te_label = te_label(idx);
xtest_orig = xtest_orig(idx,:);
              
return ;